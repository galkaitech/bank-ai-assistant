"""
DocQAAgent — Document Retrieval-Augmented Generation Agent

This agent answers employee questions by searching an indexed corpus of
internal bank documents (policy manuals, procedure guides, HR handbooks,
regulatory summaries) using semantic vector search, then synthesising a
grounded answer via an LLM.

Why RAG rather than fine-tuning?
----------------------------------
Fine-tuning would bake knowledge into the model weights, making updates
expensive (any policy change requires a new training run). RAG instead
treats the model as a reasoning engine and the vector store as an
updatable knowledge base — we can update a document and re-ingest it
in minutes. This matters in banking where policies change frequently in
response to BNB/ECB circulars.

Why ChromaDB?
-------------
Chroma runs embedded with no separate infrastructure, making it ideal for
local development and small-to-medium document collections (tens of thousands
of chunks). For a production deployment with millions of chunks or multi-tenant
isolation requirements, the retriever can be swapped for Pinecone, Weaviate,
or pgvector with no changes to the agent interface.
"""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bankmind.agents.doc_qa")

# ---------------------------------------------------------------------------
# Optional heavy imports — handled gracefully if dependencies aren't installed.
# This lets the test suite run without a full environment.
# ---------------------------------------------------------------------------
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.schema import Document

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning(
        "LangChain / Chroma dependencies not installed. DocQAAgent will run in mock mode."
    )


class DocQAAgent:
    """
    Document Q&A Agent using Retrieval-Augmented Generation.

    This agent maintains a vector store of indexed bank documents and
    answers employee questions by:
        1. Embedding the question with the same model used to index documents
        2. Performing a similarity search to retrieve the top-k most relevant chunks
        3. Feeding the retrieved chunks + question to the LLM with a strict prompt
           that instructs it to answer only from the provided context
        4. Returning the answer along with source citations

    The "answer only from context" constraint is critical for banking — it
    prevents the LLM from hallucinating policy details that don't exist in
    the actual documents.

    Attributes:
        settings: Application settings instance.
        vectorstore: The loaded Chroma vector store (or None if not yet initialised).
        retrieval_chain: The LangChain retrieval chain (or None in mock mode).
    """

    # Chunking parameters chosen based on experimentation:
    # - 800 tokens gives enough context for a policy paragraph to be self-contained
    # - 100-token overlap ensures we don't lose context at chunk boundaries
    # These are good defaults for structured prose documents (policy manuals, procedures).
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100
    TOP_K_RESULTS = 4  # Retrieve 4 chunks; empirically balances context quality vs. token cost

    def __init__(self, settings=None):
        """
        Initialise the DocQAAgent.

        Args:
            settings: Application settings. If None, settings are loaded from environment.
        """
        self.settings = settings
        self.vectorstore = None
        self.retrieval_chain = None
        self._mock_mode = not LANGCHAIN_AVAILABLE

        if not self._mock_mode:
            self._llm = self._build_llm()
            self._embeddings = self._build_embeddings()
        else:
            self._llm = None
            self._embeddings = None

        logger.info(
            "DocQAAgent initialised. Mock mode: %s", self._mock_mode
        )

    def _build_llm(self):
        """
        Builds the LLM instance.

        We use GPT-4o-mini for retrieval Q&A by default — it's significantly
        cheaper than GPT-4o and performs well for grounded synthesis tasks
        where the answer is already present in the context. GPT-4o is reserved
        for tasks that require more complex reasoning (compliance analysis,
        report generation).
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set — LLM calls will fail at runtime.")

        model_name = getattr(self.settings, "llm_model_doc_qa", "gpt-4o-mini") if self.settings else "gpt-4o-mini"

        return ChatOpenAI(
            model=model_name,
            temperature=0,       # Temperature 0 for factual, deterministic answers
            api_key=api_key or "placeholder",
            max_tokens=1500,
        )

    def _build_embeddings(self):
        """
        Builds the embeddings model.

        text-embedding-3-small was chosen because:
        - It's 5x cheaper than text-embedding-ada-002 with comparable quality
        - It has good multilingual performance, important for documents that
          may contain Bulgarian text or EU regulatory language
        """
        api_key = os.getenv("OPENAI_API_KEY", "")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=api_key or "placeholder",
        )

    def load_documents(self, source_dir: str) -> list:
        """
        Load documents from a directory for ingestion into the vector store.

        Supports PDF and plain text files. DOCX support can be added by
        installing unstructured and adding a DocxLoader.

        Args:
            source_dir: Path to the directory containing documents to index.

        Returns:
            A list of LangChain Document objects with page_content and metadata.

        Raises:
            FileNotFoundError: If source_dir does not exist.
            ImportError: If required document loaders are not installed.
        """
        if self._mock_mode:
            logger.warning("Running in mock mode — returning synthetic documents.")
            return self._mock_documents()

        source_path = Path(source_dir)
        if not source_path.exists():
            raise FileNotFoundError(f"Document source directory not found: {source_dir}")

        documents = []

        # Load PDFs (primary format for policy documents and regulatory circulars)
        pdf_loader = DirectoryLoader(
            str(source_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        try:
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info("Loaded %d pages from PDF files.", len(pdf_docs))
        except Exception as e:
            logger.warning("Could not load PDFs: %s", e)

        # Load plain text files (markdown docs, exported procedures, etc.)
        txt_loader = DirectoryLoader(
            str(source_path),
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )
        try:
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            logger.info("Loaded %d text documents.", len(txt_docs))
        except Exception as e:
            logger.warning("Could not load text files: %s", e)

        logger.info("Total documents loaded: %d", len(documents))
        return documents

    def create_vector_store(
        self,
        documents: Optional[list] = None,
        source_dir: Optional[str] = None,
        persist_dir: str = "./data/chroma_db",
    ) -> None:
        """
        Create (or load) the ChromaDB vector store.

        If persist_dir already contains an existing index, it is loaded
        without re-ingesting documents (saving cost and time). To force
        a full re-index, delete the persist_dir and call this method again.

        Args:
            documents: Pre-loaded Document objects. If None, source_dir must be provided.
            source_dir: Directory to load documents from (used if documents is None).
            persist_dir: Directory where ChromaDB persists its index to disk.
        """
        if self._mock_mode:
            logger.warning("Mock mode: vector store creation skipped.")
            return

        # Check if a persisted index already exists — avoid re-indexing on every startup
        if Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
            logger.info("Loading existing vector store from %s.", persist_dir)
            self.vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=self._embeddings,
                collection_name="bankmind_docs",
            )
            self._build_chain()
            return

        # No existing index — load and index documents from scratch
        if documents is None:
            if source_dir is None:
                raise ValueError("Either 'documents' or 'source_dir' must be provided.")
            documents = self.load_documents(source_dir)

        if not documents:
            logger.warning("No documents provided. Vector store will be empty.")
            return

        # Split documents into chunks
        # RecursiveCharacterTextSplitter is preferred over simple CharacterTextSplitter
        # because it tries to split at natural boundaries (paragraphs, then sentences)
        # before resorting to arbitrary character positions.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks.", len(documents), len(chunks))

        # Index chunks into ChromaDB
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings,
            persist_directory=persist_dir,
            collection_name="bankmind_docs",
        )
        logger.info("Vector store created with %d chunks at %s.", len(chunks), persist_dir)

        self._build_chain()

    def _build_chain(self) -> None:
        """
        Build the LangChain retrieval chain.

        RetrievalQAWithSourcesChain is used rather than the simpler
        RetrievalQA because it returns the source document metadata
        alongside the answer — employees need to know which document
        and section the answer came from.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store must be initialised before building the retrieval chain.")

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.TOP_K_RESULTS},
        )

        self.retrieval_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self._llm,
            chain_type="stuff",   # "stuff" works well when chunks are small enough to fit in context
            retriever=retriever,
            return_source_documents=True,
        )
        logger.info("Retrieval chain built successfully.")

    def query(self, question: str) -> tuple[str, list[dict]]:
        """
        Answer a question using the indexed document corpus.

        Args:
            question: The employee's natural language question.

        Returns:
            A tuple of (answer_text, list_of_sources) where each source is a
            dict with 'document', 'page', and 'snippet' keys.

        Example:
            >>> agent = DocQAAgent()
            >>> answer, sources = agent.query("What is the escalation procedure for AML alerts?")
            >>> print(answer)
            "According to the AML Procedures Manual (Section 5.2), all alerts above..."
        """
        if self._mock_mode or self.retrieval_chain is None:
            return self._mock_query(question)

        try:
            result = self.retrieval_chain.invoke({"question": question})
            answer = result.get("answer", "I could not find an answer in the available documents.")
            source_docs = result.get("source_documents", [])

            sources = [
                {
                    "document": Path(doc.metadata.get("source", "Unknown")).name,
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                }
                for doc in source_docs
            ]
            return answer, sources

        except Exception as exc:
            logger.error("Query failed: %s", exc, exc_info=True)
            return (
                "I encountered an error retrieving information from the document store. "
                "Please try rephrasing your question or contact the system administrator.",
                [],
            )

    # ---------------------------------------------------------------------------
    # Mock implementations for testing and demo without an API key
    # ---------------------------------------------------------------------------

    def _mock_documents(self) -> list:
        """Returns synthetic Document objects for testing."""
        # In real usage, this would never be called in production.
        # It exists to allow the test suite to run without any infrastructure.
        class MockDocument:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata

        return [
            MockDocument(
                "Section 7.3.2 — SEPA Credit Transfer Returns. A returned SEPA Credit Transfer "
                "must be processed within T+1 of the pacs.004 receipt. The originator account "
                "must be credited within 3 business days.",
                {"source": "payment_operations_manual.pdf", "page": 47},
            ),
            MockDocument(
                "Section 5.2 — AML Alert Escalation. All high-severity AML alerts (score > 85) "
                "must be escalated to the MLRO within 24 hours. Supporting documentation must "
                "be filed in the case management system within 48 hours.",
                {"source": "aml_procedures.pdf", "page": 23},
            ),
        ]

    def _mock_query(self, question: str) -> tuple[str, list[dict]]:
        """
        Returns a mock answer for demonstration and testing purposes.

        In demo mode (no LLM configured), this provides a realistic-looking
        response so the application is useful for presentations.
        """
        mock_answer = (
            f"[DEMO MODE — No API key configured]\n\n"
            f"In a live environment, I would search the indexed document corpus "
            f"for information relevant to: '{question}'\n\n"
            f"Based on a typical banking document corpus, this query would likely be answered "
            f"by retrieving sections from the relevant policy manual or regulatory document, "
            f"synthesised into a concise, cited response."
        )
        mock_sources = [
            {
                "document": "sample_policy_manual.pdf",
                "page": "12",
                "snippet": "This is a sample source excerpt that would appear in a live response...",
            }
        ]
        return mock_answer, mock_sources
