# BankMind Project — Build Summary

## Files Created

### Root Level
- `README.md` — Comprehensive project README with ASCII architecture diagram, badges, use cases, roadmap
- `requirements.txt` — Pinned dependencies with rationale comments
- `.env.example` — Environment template with all configuration documented
- `.gitignore` — Appropriate ignores including API keys, vector store data, logs
- `LICENSE` — MIT License
- `CONTRIBUTING.md` — Contribution guidelines with code standards, testing requirements

### Source Code (`src/`)
- `src/__init__.py`
- `src/main.py` — FastAPI application with /health, /chat, /compliance/check, /reports/generate, /agents endpoints
- `src/agents/__init__.py`
- `src/agents/document_agent.py` — DocQAAgent with RAG pipeline (ChromaDB, OpenAI embeddings)
- `src/agents/report_agent.py` — ReportGenerationAgent with template + LLM hybrid approach
- `src/agents/compliance_agent.py` — ComplianceAgent with BNB/ECB/GDPR regulatory rules corpus
- `src/utils/__init__.py`
- `src/utils/pii_filter.py` — PII detection and masking (EGN, IBAN, card PAN, email, phone)

### Configuration (`config/`)
- `config/__init__.py`
- `config/settings.py` — Pydantic-Settings with type-validated configuration for all app parameters

### Docker (`docker/`)
- `docker/Dockerfile` — Multi-stage production Dockerfile (builder + runtime, non-root user)
- `docker/docker-compose.yml` — App + ChromaDB vector store service with named volumes

### Tests (`tests/`)
- `tests/__init__.py`
- `tests/test_agents.py` — 34 unit tests across all agent classes and settings (34/34 passing)

### Documentation (`docs/`)
- `docs/architecture.md` — Full system design: component design, data flows, security, scalability, decision log
- `docs/use-cases.md` — 5 detailed banking use cases with example I/O

### CI/CD (`.github/`)
- `.github/workflows/ci.yml` — GitHub Actions: lint (flake8), test (pytest multi-version), Docker build, security scan

### Scripts & Notebooks (`scripts/`, `notebooks/`)
- `scripts/ingest_documents.py` — CLI tool for indexing documents into vector store

## Test Results

```
34 passed in 0.19s
```

All tests run without an API key using mock implementations.

## Key Architectural Decisions

1. **RAG over fine-tuning** — Policy documents update frequently (BNB circulars, internal procedures). RAG lets the knowledge base be updated by re-ingesting a document, with no ML retraining required.

2. **Hybrid template + LLM reports** — Numbers are bound into the report template before the LLM generates narrative commentary. This prevents the most dangerous failure mode in banking: hallucinated figures in reports.

3. **GPT-4o-mini for Q&A, GPT-4o for compliance/reports** — The grounded Q&A task is well-suited to the smaller model (answer is already in retrieved context). Compliance analysis and report narrative need the full model's reasoning depth.

4. **Temperature=0 for compliance** — Deterministic assessments are reproducible and defensible. A compliance finding that varies between runs is not trustworthy.

5. **Graceful mock mode** — The entire application runs without any API key in demo mode. This makes demos, CI, and testing feasible without credentials.

6. **Multi-LLM backend support** — OpenAI, Azure OpenAI (for EU data residency), and Ollama (for air-gapped). Banking institutions often cannot send data to US cloud endpoints.

7. **Pydantic-Settings configuration** — Type errors in configuration are caught at startup, not at runtime in production. The cost-model decisions (which model for which agent) are documented inline.

## Interview Talking Points

1. **Why this project?** "I built this to demonstrate how GenAI can address real inefficiencies I've observed in banking operations — specifically the time spent on document lookup, compliance pre-checks, and report drafting."

2. **Architecture choice (RAG):** "I chose RAG specifically because banking knowledge changes frequently — new BNB circulars, updated ECB guidelines. Fine-tuning would require a new training run every time a policy changes. RAG lets a non-ML team update the knowledge base by just replacing a document."

3. **The compliance agent:** "The key design decision was to use GPT-4o at temperature=0 and build in the specific BNB Ordinance and ECB article numbers. I also calibrated it to produce false positives rather than false negatives — in compliance, missing a real issue is far more costly than a false alarm."

4. **Testing without API calls:** "All 34 unit tests run without an OpenAI API key. I used the same mock-mode pattern that the production code uses for graceful degradation — the tests validate the agent interfaces and business logic, not the LLM responses."

5. **Production readiness:** "The Dockerfile uses a multi-stage build, runs as a non-root user, and has a health check. The docker-compose includes the ChromaDB service with a separate named volume so the index survives container restarts. The CI pipeline includes a Trivy security scan on PRs to main."
