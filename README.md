# BankMind — AI-Powered Banking Operations Copilot

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-orange.svg)](https://platform.openai.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/yourusername/bank-ai-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/bank-ai-assistant/actions)

> An internal AI copilot for bank employees — combining Retrieval-Augmented Generation, multi-agent orchestration, and domain-specific tools to reduce manual work across document lookup, compliance review, and report drafting.

---

## The Problem

A typical bank employee at an operations or compliance desk spends a disproportionate share of their day on tasks that are information-intensive but intellectually low-value:

- Searching through hundreds of pages of internal policy manuals to answer a colleague's question
- Manually checking loan application documents against a checklist of regulatory requirements
- Drafting monthly risk summaries and compliance reports from raw transaction data
- Routing customer complaints to the right department after reading and categorising each one
- Re-reading ECB and BNB regulatory circulars to understand what changed and what action is required

These are not tasks that require strategic thinking — they require fast, accurate information retrieval and structured text generation. GenAI is exceptionally well-suited to exactly these problems.

**BankMind** addresses this gap by providing an internal AI copilot that augments bank employees with the context they need, precisely when they need it.

---

## Solution Overview

BankMind is a **multi-agent AI system** built on LangChain/LangGraph with RAG (Retrieval-Augmented Generation) at its core. Rather than relying on a single general-purpose chatbot, it routes queries to specialised agents depending on the task type:

| Task Type | Agent | Underlying Technique |
|---|---|---|
| Policy & procedure Q&A | `DocumentQAAgent` | RAG over internal document corpus |
| Loan document analysis | `ComplianceAgent` | Structured extraction + rule checking |
| Regulatory change tracking | `ComplianceAgent` | RAG over regulatory corpus + diff analysis |
| Monthly report drafting | `ReportGenerationAgent` | Template + LLM fill-in with structured data |
| Complaint summarisation | `DocumentQAAgent` + `ReportGenerationAgent` | Classification + summarisation |

The system is designed to be deployed on-premise or in a private cloud environment, respecting the strict data residency and confidentiality requirements of banking institutions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│           Streamlit Chat UI  ·  FastAPI REST Endpoints          │
│              Internal Slack Bot  ·  MS Teams Connector          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Orchestration Layer                            │
│                  LangChain / LangGraph                          │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Router     │  │  Memory     │  │  Session Manager        │ │
│  │  Agent      │  │  (conv. +   │  │  (multi-user, isolated) │ │
│  │             │  │  entity)    │  │                         │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
│         │                                                       │
│    ┌────▼────────────────────────────────────────────────┐     │
│    │                  Agent Registry                      │     │
│    │  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │     │
│    │  │  Document   │  │  Compliance  │  │  Report    │  │     │
│    │  │  QA Agent   │  │  Agent       │  │  Agent     │  │     │
│    │  └──────┬──────┘  └──────┬───────┘  └─────┬──────┘  │     │
│    └─────────┼────────────────┼────────────────┼──────────┘     │
└─────────────────────────────────────────────────────────────────┘
              │                 │                │
              ▼                 ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Tool Layer                               │
│                                                                 │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Vector Store │  │  Structured  │  │  Report             │  │
│  │  Retriever    │  │  Data Fetch  │  │  Template Engine    │  │
│  │  (Chroma /    │  │  (PostgreSQL │  │  (Jinja2 + LLM      │  │
│  │   Pinecone)   │  │   / mock)    │  │   generation)       │  │
│  └───────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
└──────────┼─────────────────┼─────────────────────┼─────────────┘
           │                 │                      │
           ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                        LLM Layer                                │
│                                                                 │
│         OpenAI GPT-4o  ·  GPT-4o-mini (cost-optimised)         │
│         Azure OpenAI (private deployment option)                │
│         Local fallback: Ollama / llama.cpp (air-gapped)         │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Sources                                │
│                                                                 │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────────┐ │
│  │ Internal Docs  │  │ Regulatory     │  │ Operational Data  │ │
│  │ Policy manuals │  │ BNB circulars  │  │ Transaction logs  │ │
│  │ HR handbooks   │  │ ECB directives │  │ Loan portfolios   │ │
│  │ Procedures     │  │ GDPR guidance  │  │ Customer records  │ │
│  │ (PDF, DOCX)    │  │ AML guidelines │  │ (mock / PII-free) │ │
│  └────────────────┘  └────────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Contextual Document Q&A** — Ask any question about internal policies, procedures, or regulations and get an answer grounded in the actual source documents, with citations
- **Loan Document Risk Analysis** — Upload a loan application package and receive an automated risk flag summary highlighting missing documents, inconsistencies, and regulatory concerns
- **Compliance Monitoring** — Check drafted communications and internal documents against BNB regulations and ECB directives before sending
- **Automated Report Drafting** — Generate structured first drafts of monthly operational reports, risk summaries, and compliance notes from structured data inputs
- **Multi-turn Conversations** — Maintains conversational context within a session, allowing follow-up questions ("What about the exception in Section 4.2?")
- **Source Attribution** — Every answer cites the specific document and section it drew from — critical for regulatory accountability
- **Privacy-First Design** — PII detection and masking before data reaches the LLM layer; fully compatible with on-premise deployment
- **Audit Logging** — All queries, responses, and retrieved sources are logged for compliance and review

---

## Tech Stack

| Layer | Technology | Rationale |
|---|---|---|
| LLM | OpenAI GPT-4o / Azure OpenAI | Best-in-class reasoning; Azure option for data residency |
| Orchestration | LangChain 0.2 + LangGraph | Mature, extensible; LangGraph adds stateful multi-agent flows |
| Vector Store | ChromaDB (dev) / Pinecone (prod) | Chroma for zero-infra local dev; Pinecone for scale |
| Embeddings | OpenAI text-embedding-3-small | Cost-efficient, high quality for European languages incl. Bulgarian |
| API Layer | FastAPI + Uvicorn | Modern async Python, automatic OpenAPI docs |
| Frontend (optional) | Streamlit | Rapid internal tool prototyping |
| Configuration | Pydantic-Settings | Type-safe config with env var injection |
| Containerisation | Docker + Docker Compose | Reproducible deployments |
| CI/CD | GitHub Actions | Automated linting, testing, Docker build |
| Data | Pandas + NumPy | Report data manipulation |
| Testing | Pytest + unittest.mock | Isolated unit tests without LLM API calls |

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- Docker and Docker Compose (optional, for containerised setup)
- An OpenAI API key (or Azure OpenAI credentials)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/bank-ai-assistant.git
cd bank-ai-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and other values
```

### 5. Run the API server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for the interactive Swagger UI.

### 6. (Optional) Run with Docker Compose

```bash
docker-compose -f docker/docker-compose.yml up --build
```

This starts:
- The BankMind API on port `8000`
- A ChromaDB vector store service on port `8001`

### 7. Index your documents

```bash
python scripts/ingest_documents.py --source-dir data/sample_docs/
```

---

## Project Structure

```
bank-ai-assistant/
├── src/
│   ├── main.py                    # FastAPI application entry point
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── document_agent.py      # RAG-based document Q&A agent
│   │   ├── report_agent.py        # Structured report generation agent
│   │   └── compliance_agent.py   # Regulatory compliance checking agent
│   └── utils/
│       ├── __init__.py
│       ├── text_processing.py     # Document chunking and preprocessing
│       └── pii_filter.py          # PII detection and masking
├── config/
│   └── settings.py                # Centralised configuration (pydantic-settings)
├── data/
│   ├── sample_docs/               # Sample banking documents for demo
│   └── mock_reports/              # Mock report templates
├── docker/
│   ├── Dockerfile                 # Application container definition
│   └── docker-compose.yml         # Multi-service orchestration
├── tests/
│   ├── __init__.py
│   └── test_agents.py             # Unit tests for agent classes
├── docs/
│   ├── architecture.md            # Detailed system design document
│   └── use-cases.md               # Worked use cases with example I/O
├── scripts/
│   └── ingest_documents.py        # Document ingestion utility
├── notebooks/
│   └── demo_rag_pipeline.ipynb    # Jupyter demo notebook
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI pipeline
├── .env.example                   # Environment variable template
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

---

## Use Cases

### 1. Employee Policy Q&A

**Scenario:** An operations employee wants to know the procedure for handling a returned international SEPA transfer.

**Interaction:**
```
User: What is the procedure for processing a returned SEPA Credit Transfer?

BankMind: According to the Payment Operations Manual (Section 7.3.2, updated Jan 2024),
a returned SEPA Credit Transfer must be:

1. Identified via the pacs.004 message in the payment gateway within T+1
2. Matched to the original pacs.008 transaction using the end-to-end transaction ID
3. Credited back to the originator account within 3 business days
4. Flagged with reason code [RC-04] in the core banking system

If the return reason is "Account Closed" (AC01), an additional notification must be
sent to the customer per the Consumer Communication Policy (Section 12.1).

Source: Payment Operations Manual v4.2, Sections 7.3.2 and cross-reference 12.1.
```

### 2. Loan Document Risk Flagging

Upload a loan application package → receive automated risk flag summary.

### 3. Regulatory Compliance Check (BNB / ECB)

Paste a drafted internal communication → receive an assessment against current BNB Ordinance No. 7 and relevant ECB guidelines.

### 4. Monthly Report Auto-Drafting

Provide structured data (transaction counts, volumes, exceptions) → receive a draft monthly operational report ready for management review.

### 5. Customer Complaint Routing

Paste a customer complaint → receive a summary, category classification, suggested handling team, and priority level.

---

## Roadmap

- [ ] **v0.2** — Streamlit chat interface for non-technical users
- [ ] **v0.3** — Bulgarian language support (BG-specific embeddings, BNB document corpus)
- [ ] **v0.4** — Integration with mock core banking API (transaction data retrieval)
- [ ] **v0.5** — LangGraph-based multi-step agentic workflows (e.g., end-to-end loan pre-check)
- [ ] **v0.6** — PII detection and masking layer (spaCy NER + regex rules for Bulgarian names/EGN)
- [ ] **v0.7** — Role-based access control (different document permissions for different employee roles)
- [ ] **v1.0** — Production hardening: rate limiting, full audit logging, SSO integration (Azure AD)

---

## Security & Compliance Notes

This project is designed for a banking environment and includes several security considerations:

- **No PII in vector store** — Document ingestion strips personal data before indexing
- **Audit logging** — All LLM interactions are logged with user ID, timestamp, and retrieved sources
- **Air-gap compatibility** — The architecture supports Ollama/llama.cpp as a fully local LLM backend, enabling deployment with no outbound internet traffic
- **Secrets management** — All credentials are loaded from environment variables, never hardcoded

> **Important:** This is a prototype / research project. Before deploying in a production banking environment, conduct a full security review, regulatory compliance assessment, and obtain approval from your institution's information security and compliance teams.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Built as a demonstration of GenAI application architecture for internal banking use cases.*
