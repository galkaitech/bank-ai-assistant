# BankMind — System Architecture

> Version 0.1 | Last updated: 2024-11 | Author: G. Karagyozov

---

## 1. Overview

BankMind is a multi-agent AI system designed for deployment within a commercial bank's internal infrastructure. Its primary purpose is to augment bank employees with AI-assisted information retrieval, compliance checking, and report generation capabilities — reducing manual effort on knowledge-intensive but analytically routine tasks.

This document describes the architectural decisions, component design, data flows, and security considerations. It is intended for technical reviewers, infrastructure teams, and senior management evaluating the system.

---

## 2. Architectural Goals

The following goals drove all design decisions:

| Goal | Implication |
|---|---|
| **Accuracy over creativity** | LLMs must be grounded in source documents (RAG); hallucination is unacceptable in banking |
| **Auditability** | Every query, response, and retrieved source must be logged |
| **Data residency** | Must be deployable without any customer data leaving the bank's network |
| **Incremental adoptability** | Must work with existing banking data formats (PDF, DOCX, CSV) without ETL transformation |
| **Graceful degradation** | If the LLM API is unavailable, the system should fail informatively, not silently |
| **Maintainability** | Non-ML engineers must be able to update documents and redeploy without ML expertise |

---

## 3. High-Level Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                 │
│                                                                           │
│   ┌────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐ │
│   │  Streamlit     │  │  REST API       │  │  Integration Adapters     │ │
│   │  Chat UI       │  │  (FastAPI)      │  │  Slack / MS Teams         │ │
│   │  (browser)     │  │  /chat, /docs   │  │  (future)                 │ │
│   └───────┬────────┘  └────────┬────────┘  └──────────────┬────────────┘ │
└───────────┼────────────────────┼──────────────────────────┼──────────────┘
            │                    │                          │
            └────────────────────┴──────────────────────────┘
                                 │
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                           ORCHESTRATION LAYER                             │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐ │
│   │                        Router Agent                                 │ │
│   │   Intent classification → routes to correct specialist agent        │ │
│   └────────────────────────────────┬────────────────────────────────────┘ │
│                                    │                                       │
│        ┌───────────────────────────┼───────────────────────┐              │
│        │                           │                       │              │
│        ▼                           ▼                       ▼              │
│   ┌──────────────┐         ┌───────────────┐       ┌─────────────────┐   │
│   │  DocQAAgent  │         │  Compliance   │       │  Report         │   │
│   │              │         │  Agent        │       │  Generation     │   │
│   │  RAG-based   │         │               │       │  Agent          │   │
│   │  Q&A over    │         │  Rule-based + │       │                 │   │
│   │  internal    │         │  LLM analysis │       │  Template +     │   │
│   │  corpus      │         │  against BNB/ │       │  LLM narrative  │   │
│   │              │         │  ECB/GDPR     │       │  generation     │   │
│   └──────┬───────┘         └──────┬────────┘       └────────┬────────┘   │
└──────────┼──────────────────────────────────────────────────┼────────────┘
           │                        │                         │
           ▼                        ▼                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              TOOL LAYER                                   │
│                                                                           │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │  Vector Retriever│   │  Rule Engine     │   │  Template Engine     │  │
│  │                  │   │                  │   │                      │  │
│  │  Chroma / Pinec. │   │  Regulatory      │   │  Jinja2 templates +  │  │
│  │  Similarity      │   │  rules corpus    │   │  structured data     │  │
│  │  search (top-k)  │   │  (BNB, ECB,      │   │  binding             │  │
│  │                  │   │   GDPR, AML)     │   │                      │  │
│  └──────────────────┘   └──────────────────┘   └──────────────────────┘  │
│                                                                           │
│  ┌──────────────────┐   ┌──────────────────┐                             │
│  │  Data Retriever  │   │  Audit Logger    │                             │
│  │  (PostgreSQL /   │   │  (JSONL append-  │                             │
│  │   mock CSV)      │   │   only log)      │                             │
│  └──────────────────┘   └──────────────────┘                             │
└───────────────────────────────────────────────────────────────────────────┘
           │                        │
           ▼                        ▼
┌─────────────────────────┐ ┌─────────────────────────────────────────────┐
│      LLM LAYER          │ │               DATA LAYER                    │
│                         │ │                                             │
│  OpenAI GPT-4o          │ │  ┌───────────────┐  ┌───────────────────┐  │
│  OpenAI GPT-4o-mini     │ │  │  Vector Store │  │  Source Documents │  │
│  Azure OpenAI (private) │ │  │  (Chroma /    │  │  (PDF, DOCX, TXT) │  │
│  Ollama (air-gapped)    │ │  │   Pinecone)   │  │                   │  │
│                         │ │  └───────────────┘  └───────────────────┘  │
└─────────────────────────┘ │                                             │
                            │  ┌───────────────┐  ┌───────────────────┐  │
                            │  │  Audit Logs   │  │  Operational Data │  │
                            │  │  (JSONL)      │  │  (PostgreSQL /    │  │
                            │  │               │  │   mock CSV)       │  │
                            │  └───────────────┘  └───────────────────┘  │
                            └─────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Router Agent

The Router Agent is a lightweight request classifier that determines which specialist agent should handle a given user query. In v0.1 it uses keyword matching; in future versions it will use an LLM-based intent classifier or LangGraph conditional edges.

**Routing logic:**
- Keywords `compliance`, `regulation`, `BNB`, `ECB`, `GDPR`, `risk` → `ComplianceAgent`
- Keywords `generate report`, `draft`, `monthly summary` → `ReportGenerationAgent`
- All other queries → `DocQAAgent` (the default handler)

### 4.2 DocQAAgent — Retrieval-Augmented Generation

The DocQAAgent answers employee questions by searching the indexed document corpus. It uses the following pipeline:

```
User question
     │
     ▼
[Embedding model]
text-embedding-3-small
     │
     ▼
[Vector similarity search]
Chroma cosine similarity, top-k=4
     │
     ▼
[Context assembly]
Retrieved chunks + metadata
     │
     ▼
[LLM synthesis]
GPT-4o-mini with "answer from context only" instruction
     │
     ▼
[Source attribution]
Extract metadata from retrieved docs
     │
     ▼
Answer + citations
```

**Why RAG rather than fine-tuning?**

Fine-tuning embeds knowledge into model weights. This makes updates expensive: when a policy changes, a new training run is required. RAG separates knowledge (the vector store) from reasoning (the LLM). Updating knowledge is as simple as re-ingesting the updated document — no ML expertise required.

Fine-tuning is better when the model needs to learn a new *reasoning style* (e.g., adopting specific formatting conventions). RAG is better when the model needs up-to-date *knowledge*. For banking policy documents, knowledge freshness dominates.

**Chunking strategy:**

Documents are split into 800-character chunks with 100-character overlap using `RecursiveCharacterTextSplitter`. This splitter tries to break at natural boundaries (paragraphs > sentences > words) before falling back to arbitrary character positions. The 800-character size was chosen to be large enough for policy paragraphs to be self-contained but small enough to return precise results.

### 4.3 ComplianceAgent

The ComplianceAgent checks documents against a curated rule corpus derived from BNB Ordinances, ECB guidelines, EU Directives, and GDPR requirements.

**Assessment pipeline:**

```
Input document
     │
     ▼
[Regulatory rule retrieval]
Select applicable rules for the specified framework (BNB/ECB/GDPR/AML/ALL)
     │
     ▼
[LLM compliance analysis]
GPT-4o with compliance specialist system prompt
Temperature = 0 (deterministic, reproducible assessments)
     │
     ▼
[Structured output parsing]
Extract findings, risk levels, regulation references
     │
     ▼
Structured compliance report
```

**False negative calibration:** The compliance prompt is calibrated to err on the side of caution (flag potential issues). This means some false positives, but in a compliance context the cost of a missed issue (a real compliance breach) far exceeds the cost of a false alarm (a human reviewer spending 10 minutes confirming a document is actually compliant).

### 4.4 ReportGenerationAgent

Uses a hybrid template + LLM approach:

```
Structured input data (numbers, metrics, lists)
     │
     ▼
[Template binding]
Jinja2-style string formatting — ensures numbers are exact
     │
     ▼
[LLM narrative generation]
GPT-4o generates commentary sections from the structured data
LLM cannot invent numbers it wasn't given
     │
     ▼
[Draft assembly]
Combine template structure with LLM narrative
     │
     ▼
Report draft with DRAFT watermark
```

---

## 5. Data Flow Diagram

### 5.1 Document Q&A Flow

```
Bank Employee                BankMind API                   Vector Store    OpenAI API
     │                            │                              │               │
     │── POST /chat ──────────────▶│                              │               │
     │   {message: "What is..."}   │                              │               │
     │                            │── embed(message) ─────────────────────────────▶│
     │                            │◀─ embedding vector ────────────────────────────│
     │                            │── similarity_search(embedding) ──▶│             │
     │                            │◀─ top-4 document chunks ─────────│             │
     │                            │── chat(system + chunks + question) ─────────────▶│
     │                            │◀─ answer text ──────────────────────────────────│
     │◀── {answer, sources} ──────│                              │               │
```

### 5.2 Document Ingestion Flow

```
Admin / Script               BankMind                      Vector Store
     │                           │                              │
     │── ingest_documents() ─────▶│                              │
     │   (source_dir path)        │                              │
     │                           │── load PDFs/TXT ─────────────│
     │                           │── chunk documents ───────────│
     │                           │── embed chunks ──────────────│ (batch embedding API call)
     │                           │── store vectors ─────────────▶│
     │                           │◀─ index built ───────────────│
     │◀── "Indexed N chunks" ────│                              │
```

---

## 6. Security Considerations

Banking environments have strict security requirements. The following considerations are built into the design:

### 6.1 Data Classification

| Data Type | Handling | Storage |
|---|---|---|
| Internal policy documents | Indexed in vector store; no customer data | ChromaDB (on-premise) |
| Regulatory documents | Public but versioned; indexed for retrieval | ChromaDB (on-premise) |
| Customer transaction data (mock) | PII-stripped before indexing; never sent to LLM raw | Structured DB only |
| LLM inputs | PII filtered before sending (if enabled) | Not stored |
| LLM outputs | Stored in audit log with session ID | JSONL audit log |

### 6.2 PII Handling

When `ENABLE_PII_FILTER=true` (the default), the system applies PII detection before sending any text to the LLM:
- Regex-based detection of Bulgarian EGN (personal identification numbers)
- Named entity recognition for person names, account numbers
- Replacement with `[REDACTED]` tokens in the LLM input

Detected PII is logged locally but never transmitted to the LLM API.

### 6.3 API Security

- All API endpoints require authentication (JWT/OAuth2) in production
- Rate limiting applied per user session to prevent abuse
- TLS 1.3 required for all connections
- No secrets in code or environment variable names in logs

### 6.4 LLM Provider Options

For maximum data security, BankMind supports three LLM backends:

1. **OpenAI API** — Default for development. Data sent to OpenAI's API. Covered by OpenAI's data processing agreement.
2. **Azure OpenAI** — Preferred for production in EU-regulated institutions. Data processed within Azure EU regions. Microsoft's data protection commitments apply. Customer data is not used for model training.
3. **Ollama (local)** — For air-gapped or maximum-security deployments. Model runs entirely on-premise. No data leaves the bank's network. Trade-off: reduced model quality compared to GPT-4o.

### 6.5 Audit Logging

All LLM interactions are logged in an append-only JSONL file with:
- `timestamp` (ISO 8601)
- `session_id` (anonymised)
- `agent_used`
- `query_hash` (SHA-256 of the query text, not the query itself, for privacy)
- `sources_retrieved` (document names and page numbers)
- `latency_ms`
- `model_used`

Audit logs are retained for a minimum of 5 years in line with banking document retention requirements.

---

## 7. Scalability Approach

### Horizontal Scaling

The application is stateless per-request (session state is client-managed). Multiple instances can run behind a load balancer.

```
Load Balancer (nginx / Azure Front Door)
        │
   ┌────┴────┐
   │         │
BankMind   BankMind        ← N stateless application instances
instance   instance
   │         │
   └────┬────┘
        │
   ChromaDB Service    ← Shared, persistent vector store
   (or Pinecone)
```

### Vector Store Scaling

- **Development / small (<100K documents):** ChromaDB embedded, single instance
- **Medium (100K–10M documents):** ChromaDB server mode with a dedicated service
- **Large (>10M documents) / multi-tenant:** Pinecone or Weaviate with namespace isolation per business unit

### LLM Cost Optimisation

| Use Case | Model | Reasoning |
|---|---|---|
| Document Q&A | GPT-4o-mini | Answer is present in retrieved context; mini model is sufficient |
| Compliance analysis | GPT-4o | Regulatory nuance requires strongest reasoning |
| Report narrative | GPT-4o | Output quality is visible to management / regulators |
| Intent classification | GPT-4o-mini | Simple binary routing decision |

---

## 8. Integration Points with Core Banking Systems

In a full production deployment, BankMind would integrate with:

| System | Integration Type | Use Case |
|---|---|---|
| Core Banking (e.g., Oracle FLEXCUBE, Temenos) | Read-only API | Transaction data for report generation |
| Document Management (SharePoint / Documentum) | File sync / webhook | Automatic re-ingestion when policy documents are updated |
| Identity Provider (Azure AD / LDAP) | SSO / OAuth2 | Employee authentication and role-based document access |
| Case Management (ServiceNow / Jira) | API write | Auto-create tickets from compliance findings |
| Email / Teams | API write | Distribute drafted reports for approval |

All integrations are read-heavy. The system does not write to core banking systems — it generates drafts and recommendations that humans review and action.

---

## 9. Technology Decision Log

| Decision | Alternatives Considered | Chosen Approach | Reason |
|---|---|---|---|
| Orchestration framework | LlamaIndex, custom | LangChain + LangGraph | Largest ecosystem, best tool integration; LangGraph adds stateful multi-agent flows |
| Vector store (dev) | FAISS, Qdrant | ChromaDB | Zero-infrastructure; can run embedded; easy to upgrade to server mode |
| Vector store (prod) | Weaviate, Qdrant | Pinecone | Managed service; no infrastructure maintenance; strong SLA |
| Embeddings | Sentence Transformers (local) | OpenAI text-embedding-3-small | Better quality; good Bulgarian/EU language support; cost-effective |
| API framework | Flask, Django | FastAPI | Native async; automatic OpenAPI docs; Pydantic integration |
| Configuration | env vars only, YAML | pydantic-settings | Type-safe; IDE support; validates at startup |
| Report approach | Pure LLM generation | Template + LLM narrative | Prevents number hallucination — critical in banking reporting |
