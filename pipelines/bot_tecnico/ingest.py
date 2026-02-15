"""
Ingestion script for technical manuals.
Parses PDFs using LlamaParse and stores embeddings in ChromaDB.

Usage:
    1. Place PDF files in the ./data/ directory
    2. Run: python ingest.py

Filename convention: [PRODUCT_CODE]_[YEAR]_[STANDARD].pdf
    Example: PCGH_2025_Eurovent.pdf
"""

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
import logfire
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent.resolve()
PIPELINES_DIR = SCRIPT_DIR.parent  # /app/pipelines
DATA_DIR = PIPELINES_DIR / "data"  # Shared data folder
CHROMA_DIR = SCRIPT_DIR / "chroma_db"  # Local to bot_tecnico (legacy)

# Load environment variables from .env file in script directory
load_dotenv(SCRIPT_DIR / ".env")

# ChromaDB server configuration
CHROMA_HOST = os.getenv("CHROMA_HOST", None)

# Configure Logfire
logfire.configure(
    token=os.getenv("LOGFIRE_TOKEN"),
    send_to_logfire=True  # Disable telemetry to Logfire servers
)
try:
    logfire.instrument_requests()
except Exception:
    pass  # Optional instrumentation


@dataclass
class ProductMetadata:
    """Structured metadata extracted from filename."""
    product_code: str   # e.g., PCGH, PDWA
    year: str           # e.g., 2025
    standard: str       # e.g., Eurovent
    product_key: str    # e.g., PCGH_2025_Eurovent (used for filtering)


def parse_product_from_filename(filename: str) -> ProductMetadata | None:
    """
    Parse product metadata from filename pattern: [PRODUCT_CODE]_[YEAR]_[STANDARD].pdf

    Examples:
        PCGH_2025_Eurovent.pdf -> ProductMetadata("PCGH", "2025", "Eurovent", "PCGH_2025_Eurovent")
        PDWA_2025_Eurovent.pdf -> ProductMetadata("PDWA", "2025", "Eurovent", "PDWA_2025_Eurovent")

    Returns None if filename doesn't match the expected pattern.
    """
    # Remove extension for matching
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename

    # Pattern: PRODUCT_YEAR_STANDARD (e.g., PCGH_2025_Eurovent)
    pattern = r"^([A-Z0-9]+)_(\d{4})_([A-Za-z0-9]+)$"
    match = re.match(pattern, stem)

    if not match:
        logfire.warn(f"Filename '{filename}' doesn't match expected pattern [PRODUCT]_[YEAR]_[STANDARD]")
        return None

    code, year, standard = match.groups()
    return ProductMetadata(
        product_code=code,
        year=year,
        standard=standard,
        product_key=f"{code}_{year}_{standard}",
    )


def get_source_files() -> list[Path]:
    """Get all Markdown files from the data directory."""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logfire.warn(f"Created empty data directory: {DATA_DIR}")
        return []

    md_files = list(DATA_DIR.glob("*.md"))

    logfire.info(f"Found {len(md_files)} Markdown files in {DATA_DIR}")
    return md_files


def parse_markdown_files(md_files: list[Path]) -> list[dict]:
    """Read Markdown files directly."""
    documents = []
    for md_path in md_files:
        with logfire.span(f"Reading Markdown: {md_path.name}"):
            logfire.info(f"Processing: {md_path.name}")
            try:
                content = md_path.read_text(encoding="utf-8")

                # Build base metadata
                metadata = {
                    "source": md_path.name,
                    "file_path": str(md_path),
                    "type": "markdown",
                }

                # Try to extract product metadata from filename
                product_meta = parse_product_from_filename(md_path.name)
                if product_meta:
                    metadata["product_code"] = product_meta.product_code
                    metadata["year"] = product_meta.year
                    metadata["standard"] = product_meta.standard
                    metadata["product_key"] = product_meta.product_key
                    logfire.info(f"Extracted product: {product_meta.product_key}")

                documents.append({
                    "content": content,
                    "metadata": metadata,
                })
                logfire.info(f"Successfully read {md_path.name}: {len(content)} characters")
            except Exception as e:
                logfire.error(f"Failed to read {md_path.name}: {e}")
                raise
    return documents




def extract_tables_and_text(content: str) -> list[dict]:
    """
    Extract tables and text separately from Markdown content.
    Tables are kept whole, text is returned for further splitting.

    Handles both:
    - Markdown tables (| header | ... | with |---| separator)
    - HTML tables (<table>...</table>) - consecutive tables are merged

    Returns list of segments with their type ("table" or "text").
    """
    segments = []

    # First, extract HTML tables using regex
    html_table_pattern = re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE)

    # Find all HTML tables and their positions
    html_tables = []
    for match in html_table_pattern.finditer(content):
        html_tables.append((match.start(), match.end(), match.group()))

    # Process content, extracting HTML tables first
    if html_tables:
        last_end = 0
        i = 0
        while i < len(html_tables):
            start, end, table_content = html_tables[i]

            # Add text before this table
            text_before = content[last_end:start]
            if text_before.strip():
                # Process this text segment for Markdown tables
                md_segments = _extract_markdown_tables(text_before)
                segments.extend(md_segments)

            # Check for consecutive HTML tables (only whitespace between them)
            merged_tables = [table_content]
            current_end = end

            while i + 1 < len(html_tables):
                next_start, next_end, next_table = html_tables[i + 1]
                # Check if only whitespace between current and next table
                between = content[current_end:next_start]
                if between.strip() == "":
                    # Merge consecutive table
                    merged_tables.append(next_table)
                    current_end = next_end
                    i += 1
                else:
                    break

            # Add merged HTML tables as single segment
            segments.append({
                "type": "table",
                "content": "\n".join(merged_tables)
            })
            last_end = current_end
            i += 1

        # Process remaining text after last HTML table
        remaining = content[last_end:]
        if remaining.strip():
            md_segments = _extract_markdown_tables(remaining)
            segments.extend(md_segments)
    else:
        # No HTML tables, just process Markdown tables
        segments = _extract_markdown_tables(content)

    return segments


def _extract_markdown_tables(content: str) -> list[dict]:
    """
    Extract Markdown tables (pipe-delimited) from content.

    Returns list of segments with type "table" or "text".
    """
    segments = []
    lines = content.split("\n")

    i = 0
    current_text = []

    while i < len(lines):
        line = lines[i]

        # Detect Markdown table start (line with pipes and following separator line)
        if "|" in line and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Check if next line is a separator (contains --- and |)
            if "|" in next_line and re.search(r':?-+:?', next_line):
                # Save accumulated text before table
                if current_text:
                    text_content = "\n".join(current_text).strip()
                    if text_content:
                        segments.append({
                            "type": "text",
                            "content": text_content
                        })
                    current_text = []

                # Extract complete table
                table_lines = [line, next_line]  # Header + separator
                i += 2

                # Continue extracting table rows
                while i < len(lines) and "|" in lines[i] and lines[i].strip():
                    table_lines.append(lines[i])
                    i += 1

                # Save table as single segment
                segments.append({
                    "type": "table",
                    "content": "\n".join(table_lines)
                })
                continue

        # Regular text line
        current_text.append(line)
        i += 1

    # Save remaining text
    if current_text:
        text_content = "\n".join(current_text).strip()
        if text_content:
            segments.append({
                "type": "text",
                "content": text_content
            })

    return segments


def split_documents(documents: list[dict]) -> list[dict]:
    """
    Split documents into chunks for embedding.
    Tables are preserved as complete chunks, text is split normally.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    for doc in documents:
        with logfire.span(f"Splitting document: {doc['metadata']['source']}"):
            # Extract tables and text segments
            segments = extract_tables_and_text(doc["content"])

            # Count tables for logging
            table_count = sum(1 for s in segments if s["type"] == "table")
            logfire.info(f"Found {table_count} tables in {doc['metadata']['source']}")

            chunk_index = 0
            table_chunks = 0
            for segment in segments:
                if not segment["content"].strip():
                    continue

                if segment["type"] == "table":
                    # Keep table as a single chunk
                    all_chunks.append({
                        "content": segment["content"],
                        "metadata": {
                            **doc["metadata"],
                            "chunk_index": chunk_index,
                            "chunk_type": "table",
                        }
                    })
                    chunk_index += 1
                    table_chunks += 1
                    logfire.debug(f"Preserved table as single chunk ({len(segment['content'])} chars)")
                else:
                    # Split text normally
                    texts = text_splitter.split_text(segment["content"])
                    for text in texts:
                        all_chunks.append({
                            "content": text,
                            "metadata": {
                                **doc["metadata"],
                                "chunk_index": chunk_index,
                                "chunk_type": "text",
                            }
                        })
                        chunk_index += 1

            logfire.info(f"Created {chunk_index} chunks ({table_chunks} tables) from {doc['metadata']['source']}")

    logfire.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def create_vector_store(chunks: list[dict]) -> chromadb.Collection:
    """Create ChromaDB vector store with default local embeddings."""
    with logfire.span("Creating embeddings and vector store"):
        # Initialize ChromaDB client
        if CHROMA_HOST:
            # Server mode: connect to ChromaDB server
            logfire.info(f"Connecting to ChromaDB server at {CHROMA_HOST}")
            host = CHROMA_HOST.replace("http://", "").split(":")[0]
            port = int(CHROMA_HOST.split(":")[-1])
            client = chromadb.HttpClient(host=host, port=port)
        else:
            # Local mode: use persistent storage (legacy)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        # Use ChromaDB default embedding (all-MiniLM-L6-v2, runs locally)
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Delete existing collection if it exists
        try:
            existing = client.get_collection("technical_manuals")
            logfire.info(f"Deleting existing collection with {existing.count()} documents")
            client.delete_collection("technical_manuals")
        except Exception:
            pass  # Collection doesn't exist

        # Create fresh collection
        collection = client.create_collection(
            name="technical_manuals",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Add documents in batches
        batch_size = 50
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                ids=[f"doc_{i + j}" for j in range(len(batch))],
                documents=[chunk["content"] for chunk in batch],
                metadatas=[chunk["metadata"] for chunk in batch],
            )
            logfire.info(f"Added batch {i // batch_size + 1}/{(len(chunks) + batch_size - 1) // batch_size}")

        storage_info = CHROMA_HOST if CHROMA_HOST else str(CHROMA_DIR)
        logfire.info(f"Vector store created at {storage_info}")
        logfire.info(f"Total documents in store: {collection.count()}")

    return collection


def main():
    """Main ingestion pipeline."""
    logfire.info("=" * 60)
    logfire.info("Starting ingestion pipeline (Markdown only)")
    logfire.info(f"Data directory: {DATA_DIR}")
    logfire.info(f"ChromaDB directory: {CHROMA_DIR}")
    logfire.info("=" * 60)

    # Get Markdown source files
    md_files = get_source_files()
    if not md_files:
        logfire.warn("No Markdown files found. Place .md files in the data directory.")
        print(f"\nNo Markdown files found in {DATA_DIR}")
        print("Please add Markdown (.md) files and run this script again.")
        sys.exit(1)

    # Parse Markdown files
    with logfire.span("Markdown Reading Phase"):
        documents = parse_markdown_files(md_files)
        logfire.info(f"Read {len(documents)} Markdown documents")

    # Split into chunks
    with logfire.span("Document Splitting Phase"):
        chunks = split_documents(documents)

    # Create vector store
    with logfire.span("Vector Store Creation Phase"):
        collection = create_vector_store(chunks)

    logfire.info("=" * 60)
    logfire.info("Ingestion complete!")
    logfire.info(f"Processed {len(md_files)} Markdown files")
    logfire.info(f"Created {len(chunks)} chunks")
    logfire.info(f"Vector store saved to {CHROMA_DIR}")
    logfire.info("=" * 60)

    print(f"\n{'=' * 60}")
    print("Ingestion complete!")
    print(f"  - Markdown files: {len(md_files)}")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Vector store: {CHROMA_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
