"""
Document Ingestion Script

Indexes documents from a source directory into the ChromaDB vector store.
Run this script when new documents are added or existing documents are updated.

Usage:
    python scripts/ingest_documents.py --source-dir data/sample_docs/
    python scripts/ingest_documents.py --source-dir /path/to/docs --persist-dir /path/to/chroma
    python scripts/ingest_documents.py --force-reindex  # Deletes existing index and rebuilds
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("bankmind.ingest")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into BankMind's vector store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir",
        default="data/sample_docs",
        help="Directory containing documents to index (PDFs and .txt files).",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma_db",
        help="Directory where ChromaDB stores its index.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete existing index and rebuild from scratch.",
    )
    args = parser.parse_args()

    source_path = Path(args.source_dir)
    persist_path = Path(args.persist_dir)

    if not source_path.exists():
        logger.error("Source directory does not exist: %s", source_path)
        sys.exit(1)

    # Count documents available
    pdf_count = len(list(source_path.glob("**/*.pdf")))
    txt_count = len(list(source_path.glob("**/*.txt")))
    total = pdf_count + txt_count

    if total == 0:
        logger.warning("No PDF or TXT files found in %s. Nothing to index.", source_path)
        sys.exit(0)

    logger.info("Found %d documents (%d PDFs, %d TXT) in %s.", total, pdf_count, txt_count, source_path)

    if args.force_reindex and persist_path.exists():
        logger.info("--force-reindex specified. Deleting existing index at %s.", persist_path)
        shutil.rmtree(persist_path)

    from config.settings import get_settings
    from src.agents.document_agent import DocQAAgent

    settings = get_settings()
    agent = DocQAAgent(settings=settings)

    if agent._mock_mode:
        logger.warning(
            "DocQAAgent is in mock mode (LangChain not installed). "
            "Install full requirements to enable real indexing."
        )
        sys.exit(1)

    logger.info("Starting document ingestion...")
    agent.create_vector_store(
        source_dir=str(source_path),
        persist_dir=str(persist_path),
    )
    logger.info("Ingestion complete. Vector store at: %s", persist_path)


if __name__ == "__main__":
    main()
