"""
BankMind Application Configuration

All application configuration is managed here using pydantic-settings.
This provides:
1. Type-validated configuration (wrong types caught at startup, not at runtime)
2. Automatic loading from environment variables and .env files
3. A single source of truth for all configurable parameters
4. Clear documentation of what each setting does

Why pydantic-settings over plain os.getenv()?
---------------------------------------------
- Type coercion: LLM_TEMPERATURE="0.2" is automatically converted to float
- Validation: MIN_CHUNK_SIZE=0 raises an error before the app starts
- IDE support: Full autocompletion and type hints throughout the codebase
- Secret management: SecretStr fields prevent accidental logging of API keys

Usage:
    from config.settings import get_settings
    settings = get_settings()
    print(settings.llm_model)
"""

import logging
from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("bankmind.config")


class Settings(BaseSettings):
    """
    BankMind application settings.

    All values can be overridden via environment variables or a .env file.
    Environment variable names match the field names in uppercase.
    e.g., the field `openai_api_key` maps to the env var `OPENAI_API_KEY`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,       # OPENAI_API_KEY and openai_api_key both work
        extra="ignore",             # Ignore unknown env vars rather than raising errors
    )

    # -------------------------------------------------------------------------
    # Application metadata
    # -------------------------------------------------------------------------
    app_name: str = Field(
        default="BankMind",
        description="Application display name.",
    )
    app_mode: Literal["production", "development", "demo"] = Field(
        default="development",
        description=(
            "Operating mode. 'demo' enables mock responses when no API key is set. "
            "'production' enables stricter logging and disables debug features."
        ),
    )
    app_version: str = Field(default="0.1.0", description="Application version.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity.",
    )

    # -------------------------------------------------------------------------
    # LLM configuration
    # -------------------------------------------------------------------------
    openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description=(
            "OpenAI API key. If not set, the application runs in demo mode "
            "with mock LLM responses. Never commit this value to source control."
        ),
    )
    azure_openai_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Azure OpenAI API key (used when Azure deployment is configured).",
    )
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL (e.g., https://your-resource.openai.azure.com/).",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01",
        description="Azure OpenAI API version.",
    )

    # Model selection — different agents can use different models based on
    # the cost/quality tradeoff for each task type.
    llm_model: str = Field(
        default="gpt-4o",
        description="Default LLM model. Used for compliance and report generation (quality-critical).",
    )
    llm_model_doc_qa: str = Field(
        default="gpt-4o-mini",
        description=(
            "Model for document Q&A. GPT-4o-mini is cost-effective and performs well "
            "for grounded synthesis where the answer is already in the retrieved context."
        ),
    )
    llm_model_reports: str = Field(
        default="gpt-4o",
        description="Model for report generation. Uses the full model for quality narrative output.",
    )
    llm_model_compliance: str = Field(
        default="gpt-4o",
        description=(
            "Model for compliance checking. Uses full model due to regulatory stakes — "
            "missing a compliance issue has real consequences."
        ),
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Default LLM temperature. 0 = deterministic, higher = more creative.",
    )
    llm_max_tokens: int = Field(
        default=2000,
        ge=100,
        le=16000,
        description="Maximum tokens in LLM responses.",
    )

    # -------------------------------------------------------------------------
    # Embedding model
    # -------------------------------------------------------------------------
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description=(
            "OpenAI embedding model for vector indexing and retrieval. "
            "text-embedding-3-small: 1536 dims, cost-efficient, good multilingual quality. "
            "text-embedding-3-large: 3072 dims, higher quality, 6x more expensive."
        ),
    )

    # -------------------------------------------------------------------------
    # Vector store configuration
    # -------------------------------------------------------------------------
    vector_store_type: Literal["chroma", "pinecone", "pgvector"] = Field(
        default="chroma",
        description=(
            "Vector store backend. "
            "'chroma': embedded, zero-infrastructure, for dev/small-scale. "
            "'pinecone': managed service, for production scale. "
            "'pgvector': PostgreSQL extension, for integration with existing DB infrastructure."
        ),
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="Directory where ChromaDB persists its index to disk.",
    )
    chroma_collection_name: str = Field(
        default="bankmind_docs",
        description="ChromaDB collection name for the document index.",
    )
    pinecone_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Pinecone API key (required only when vector_store_type='pinecone').",
    )
    pinecone_index_name: str = Field(
        default="bankmind-docs",
        description="Pinecone index name.",
    )
    pinecone_environment: str = Field(
        default="gcp-starter",
        description="Pinecone environment/region.",
    )

    # -------------------------------------------------------------------------
    # Document ingestion settings
    # -------------------------------------------------------------------------
    doc_source_dir: str = Field(
        default="./data/sample_docs",
        description="Default directory to load documents from during ingestion.",
    )
    chunk_size: int = Field(
        default=800,
        ge=100,
        le=8000,
        description=(
            "Text chunk size in characters for document splitting. "
            "Smaller chunks = more precise retrieval but less context per chunk. "
            "800 chars is a good default for structured policy documents."
        ),
    )
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between consecutive text chunks to avoid losing context at boundaries.",
    )
    retrieval_top_k: int = Field(
        default=4,
        ge=1,
        le=20,
        description=(
            "Number of document chunks to retrieve for each query. "
            "Higher values = more context but higher token cost."
        ),
    )

    # -------------------------------------------------------------------------
    # API server settings
    # -------------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0", description="API server bind host.")
    api_port: int = Field(default=8000, ge=1, le=65535, description="API server port.")
    api_workers: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Number of Uvicorn worker processes.",
    )
    cors_origins: list[str] = Field(
        default=["*"],
        description=(
            "Allowed CORS origins. Should be restricted to internal hostnames in production. "
            "Example: ['https://bankmind.fibank.bg', 'https://intranet.fibank.bg']"
        ),
    )

    # -------------------------------------------------------------------------
    # Security & compliance settings
    # -------------------------------------------------------------------------
    enable_pii_filter: bool = Field(
        default=True,
        description=(
            "Enable PII detection and masking before sending text to the LLM. "
            "Should always be True in production to comply with GDPR."
        ),
    )
    enable_audit_logging: bool = Field(
        default=True,
        description=(
            "Log all LLM queries, responses, and retrieved sources to the audit log. "
            "Required for regulatory accountability."
        ),
    )
    audit_log_path: str = Field(
        default="./logs/audit.jsonl",
        description="Path to the JSONL audit log file.",
    )
    max_document_size_mb: float = Field(
        default=20.0,
        gt=0,
        description="Maximum allowed document size for upload/processing in megabytes.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk_size(cls, v: int, info) -> int:
        """Ensures chunk overlap is sensibly smaller than chunk size."""
        # We can't easily access chunk_size here in pydantic v2, so just enforce
        # an absolute maximum
        if v > 400:
            raise ValueError("chunk_overlap should not exceed 400 characters.")
        return v

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------
    @property
    def is_demo_mode(self) -> bool:
        """Returns True if the app is running in demo mode (no real LLM calls)."""
        return self.app_mode == "demo" or not self.openai_api_key

    @property
    def openai_api_key_value(self) -> Optional[str]:
        """Returns the raw API key string (use sparingly — prefer SecretStr)."""
        if self.openai_api_key:
            return self.openai_api_key.get_secret_value()
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the singleton Settings instance.

    Using lru_cache(maxsize=1) ensures settings are loaded once and reused
    across the entire application, avoiding repeated .env file reads.

    In tests, call get_settings.cache_clear() before each test that needs
    custom settings to reset the cached instance.
    """
    settings = Settings()
    logger.info(
        "Settings loaded. App mode: %s, LLM model: %s, Vector store: %s",
        settings.app_mode,
        settings.llm_model,
        settings.vector_store_type,
    )
    return settings
