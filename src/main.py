"""
BankMind — AI-Powered Banking Operations Copilot
FastAPI Application Entry Point

This module sets up the FastAPI application, configures middleware,
and exposes the primary REST endpoints consumed by the frontend
and any downstream integrations (Slack bot, Teams connector, etc.).

Design decisions:
- FastAPI was chosen over Flask for its native async support (critical for
  concurrent LLM streaming calls) and automatic OpenAPI documentation.
- The /chat endpoint is intentionally thin — it delegates all business logic
  to the agent layer, keeping the API layer clean and testable.
- Graceful degradation: if no API key is configured, the app starts in
  "demo mode" and returns informative mock responses rather than crashing.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config.settings import get_settings
from src.agents.document_agent import DocQAAgent
from src.agents.report_agent import ReportGenerationAgent
from src.agents.compliance_agent import ComplianceAgent

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("bankmind.api")

# ---------------------------------------------------------------------------
# Application state (initialised at startup via lifespan)
# ---------------------------------------------------------------------------
_agents: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler — runs setup logic on startup and teardown on
    shutdown without blocking the event loop.

    We initialise agent instances here (not at import time) so that settings
    are fully loaded before any agent tries to connect to external services.
    This also makes the startup process observable and easy to debug.
    """
    settings = get_settings()
    logger.info("BankMind starting up. Mode: %s", settings.app_mode)

    # Agents are created once and reused across requests — they are stateless
    # with respect to individual conversations (session state lives in the
    # conversation memory passed per-request, not in the agent instances).
    _agents["doc_qa"] = DocQAAgent(settings=settings)
    _agents["report"] = ReportGenerationAgent(settings=settings)
    _agents["compliance"] = ComplianceAgent(settings=settings)

    logger.info("All agents initialised successfully.")
    yield  # Application runs here

    # Graceful shutdown: release any held resources (e.g., vector store connections)
    logger.info("BankMind shutting down.")
    _agents.clear()


# ---------------------------------------------------------------------------
# FastAPI application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BankMind API",
    description=(
        "Internal AI copilot for bank employees. "
        "Provides document Q&A, compliance checking, and report generation "
        "via a multi-agent LangChain architecture."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — in production this should be locked down to the specific internal
# frontend hostnames. Left open here for local development convenience.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message from a user."""
    message: str = Field(
        ...,
        description="The user's question or instruction",
        min_length=1,
        max_length=4000,
        example="What is the procedure for approving a mortgage application above 500,000 BGN?",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session identifier for conversation continuity. If omitted, a stateless single-turn response is returned.",
        example="session_abc123",
    )
    agent: Optional[str] = Field(
        default="auto",
        description="Which agent to route the request to. Options: 'auto', 'doc_qa', 'report', 'compliance'.",
        example="doc_qa",
    )


class ChatResponse(BaseModel):
    """Response from BankMind."""
    answer: str = Field(..., description="The AI-generated answer or report.")
    agent_used: str = Field(..., description="Which agent handled the request.")
    sources: list[dict] = Field(
        default_factory=list,
        description="Source documents cited in the answer (document name, page/section).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echo of the session_id from the request.",
    )
    latency_ms: Optional[float] = Field(
        default=None,
        description="Server-side processing time in milliseconds.",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    mode: str
    agents_loaded: list[str]


class ComplianceCheckRequest(BaseModel):
    """Request body for the compliance check endpoint."""
    document_text: str = Field(
        ...,
        description="The document text to check for regulatory compliance.",
        min_length=10,
        max_length=50000,
    )
    regulation_context: Optional[str] = Field(
        default="BNB",
        description="Which regulatory framework to check against. Options: 'BNB', 'ECB', 'GDPR', 'AML'.",
    )


class ReportRequest(BaseModel):
    """Request body for the report generation endpoint."""
    report_type: str = Field(
        ...,
        description="Type of report to generate.",
        example="monthly_risk_summary",
    )
    data: dict = Field(
        ...,
        description="Structured input data for the report (transaction counts, volumes, exceptions, etc.).",
    )
    period: Optional[str] = Field(
        default=None,
        description="Reporting period, e.g., '2024-Q4' or '2024-11'.",
        example="2024-11",
    )


# ---------------------------------------------------------------------------
# Request timing middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds X-Process-Time header to all responses for latency monitoring."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns application status, version, operating mode, and which agents
    are loaded. Used by load balancers, Kubernetes probes, and monitoring tools.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        mode=settings.app_mode,
        agents_loaded=list(_agents.keys()),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Conversation"])
async def chat(request: ChatRequest):
    """
    Primary chat endpoint.

    Routes the user's message to the appropriate agent based on the `agent`
    field. When set to 'auto', a lightweight classifier determines the best
    agent based on keywords in the message.

    All responses include source citations so employees can verify the
    grounding of any answer — this is non-negotiable in a banking context.
    """
    start_time = time.perf_counter()
    settings = get_settings()

    # Determine which agent to use
    agent_name = _resolve_agent(request.agent, request.message)

    if agent_name not in _agents:
        raise HTTPException(
            status_code=400,
            detail=f"Agent '{agent_name}' not found. Available: {list(_agents.keys())}",
        )

    agent = _agents[agent_name]

    try:
        result = await _dispatch_to_agent(agent_name, agent, request.message, settings)
    except Exception as exc:
        logger.error("Agent error [%s]: %s", agent_name, exc, exc_info=True)
        # Return a graceful error rather than a raw 500 — employees should
        # see a friendly message, not a stack trace.
        raise HTTPException(
            status_code=503,
            detail="The AI assistant encountered an error processing your request. Please try again.",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000

    return ChatResponse(
        answer=result["answer"],
        agent_used=agent_name,
        sources=result.get("sources", []),
        session_id=request.session_id,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/compliance/check", tags=["Compliance"])
async def check_compliance(request: ComplianceCheckRequest):
    """
    Standalone compliance check endpoint.

    Accepts a document text and returns a structured compliance assessment
    against the specified regulatory framework (default: BNB regulations).
    """
    agent = _agents.get("compliance")
    if not agent:
        raise HTTPException(status_code=503, detail="Compliance agent not available.")

    try:
        result = agent.check_document(
            document_text=request.document_text,
            regulation_context=request.regulation_context or "BNB",
        )
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("Compliance check error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Compliance check failed.")


@app.post("/reports/generate", tags=["Reports"])
async def generate_report(request: ReportRequest):
    """
    Report generation endpoint.

    Accepts structured data and a report type, and returns a drafted report
    text that can be reviewed and approved by the employee before use.
    """
    agent = _agents.get("report")
    if not agent:
        raise HTTPException(status_code=503, detail="Report agent not available.")

    try:
        if request.report_type == "monthly_risk_summary":
            result = agent.generate_monthly_report(
                data=request.data, period=request.period or "current"
            )
        elif request.report_type == "compliance_note":
            result = agent.draft_compliance_note(data=request.data)
        else:
            result = agent.generate_summary(data=request.data, report_type=request.report_type)

        return JSONResponse(content={"report": result, "report_type": request.report_type})
    except Exception as exc:
        logger.error("Report generation error: %s", exc, exc_info=True)
        raise HTTPException(status_code=503, detail="Report generation failed.")


@app.get("/agents", tags=["System"])
async def list_agents():
    """Returns information about the loaded agents and their capabilities."""
    return {
        "agents": {
            "doc_qa": {
                "description": "Document Q&A using RAG over the internal document corpus",
                "capabilities": ["policy lookup", "procedure Q&A", "regulatory document search"],
            },
            "report": {
                "description": "Structured report and summary generation",
                "capabilities": [
                    "monthly report drafting",
                    "compliance note drafting",
                    "executive summary generation",
                ],
            },
            "compliance": {
                "description": "Regulatory compliance checking against BNB/ECB/GDPR frameworks",
                "capabilities": [
                    "document compliance check",
                    "risk flagging",
                    "remediation suggestions",
                ],
            },
        }
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_agent(agent_hint: Optional[str], message: str) -> str:
    """
    Determines which agent should handle a request.

    If the caller specifies an agent explicitly, we honour it. Otherwise we
    run a lightweight keyword-based classifier. In a production system this
    classifier would be replaced with a proper intent classification model
    or an LLM-based router node in the LangGraph workflow.
    """
    if agent_hint and agent_hint != "auto":
        return agent_hint

    message_lower = message.lower()

    # Compliance and regulatory keywords
    if any(kw in message_lower for kw in ["compliance", "regulation", "bnb", "ecb", "gdpr", "aml", "risk", "check this document"]):
        return "compliance"

    # Report generation keywords
    if any(kw in message_lower for kw in ["generate report", "draft report", "monthly summary", "write a report", "create summary"]):
        return "report"

    # Default: document Q&A handles the majority of employee queries
    return "doc_qa"


async def _dispatch_to_agent(agent_name: str, agent, message: str, settings) -> dict:
    """
    Dispatches a message to the appropriate agent and normalises the response.

    Each agent has its own internal API, but this function presents a uniform
    interface to the endpoint handler. This indirection also makes it easy to
    add logging, caching, or circuit-breaking logic in one place.
    """
    if agent_name == "doc_qa":
        answer, sources = agent.query(question=message)
        return {"answer": answer, "sources": sources}

    elif agent_name == "compliance":
        # For compliance, the /chat endpoint treats the message as a document snippet
        result = agent.check_document(document_text=message)
        answer = result.get("summary", "Compliance check complete.")
        return {"answer": answer, "sources": result.get("references", [])}

    elif agent_name == "report":
        # For report generation via chat, treat the message as a natural language
        # description of what report is needed
        draft = agent.generate_summary(
            data={"user_request": message},
            report_type="ad_hoc",
        )
        return {"answer": draft, "sources": []}

    else:
        return {"answer": "Agent not implemented.", "sources": []}


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
