"""
Knowledge Base Builder

This script processes documents from a knowledge base directory and creates
an indexed embedding database for semantic search. It chunks documents,
generates embeddings using a local Hugging Face model (BAAI/bge-base-en-v1.5),
and stores them for fast retrieval.

The embedding model runs locally and is independent of the LLM provider
configuration. This allows you to build the knowledge base once in advance
before deploying the server with any LLM provider (OpenAI, Ollama, etc).

The output is used by the AI tutor server to provide context-aware responses.
"""

import os
import json
import glob
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

# Import configuration
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the local embedding model (BAAI/bge-base-en-v1.5)
# This model produces 768-dimensional embeddings and runs entirely locally
logger.info("Loading BAAI/bge-base-en-v1.5 embedding model...")
embedding_model = SentenceTransformer('BAAI/bge-base-en-v1.5')
logger.info("Embedding model loaded successfully")

# Configuration constants from Config module
KB_DIR = Config.KB_DIR  # Directory containing knowledge base documents
OUT_FILE = Config.KB_INDEX_FILE  # Output file for the indexed embeddings

def chunk_text(text: str, chunk_size: int = None, overlap: int = None):
    """
    Split text into overlapping chunks for embedding.

    Overlapping chunks help maintain context at chunk boundaries and improve
    retrieval quality for queries that span multiple chunks.

    Args:
        text: The input text to chunk
        chunk_size: Maximum size of each chunk in characters (default: from config)
        overlap: Number of overlapping characters between chunks (default: from config)

    Returns:
        List of text chunks, with whitespace stripped and empty chunks removed
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP

    logger.debug(f"Chunking text of length {len(text)} with chunk_size={chunk_size}, overlap={overlap}")

    # Normalize line endings to \n
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0

    # Slide a window across the text with overlap
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        # Move forward by (chunk_size - overlap), but ensure progress
        start = max(end - overlap, start + 1)

    # Remove empty or whitespace-only chunks
    result = [c.strip() for c in chunks if c.strip()]
    logger.debug(f"Created {len(result)} chunks from input text")
    return result

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of text chunks using the local BAAI model.

    This function uses sentence-transformers with BAAI/bge-base-en-v1.5 model
    to generate 768-dimensional embeddings. The process runs entirely locally
    and is independent of the LLM provider configuration.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each vector is a list of 768 floats)

    Raises:
        RuntimeError if embedding generation fails
    """
    logger.info(f"Generating embeddings for {len(texts)} text chunks using BAAI/bge-base-en-v1.5")

    try:
        # Generate embeddings using the local model
        # normalize_embeddings=True ensures better cosine similarity calculations
        embeddings = embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32  # Process in batches for efficiency
        )

        # Convert numpy arrays to lists for JSON serialization
        embeddings_list = embeddings.tolist()

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings (dimension: {len(embeddings_list[0])})")
        return embeddings_list
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise RuntimeError(f"Embedding generation failed: {e}")

def main():
    """
    Main function to build the knowledge base index.

    Process:
    1. Recursively find all files in the KB_DIR directory
    2. Read and chunk each document
    3. Generate embeddings for all chunks using local BAAI model
    4. Save the indexed data (chunks + embeddings) to a JSON file

    The output file can then be loaded by the server for fast semantic search.
    This process is independent of the LLM provider configuration.
    """
    logger.info(f"Starting knowledge base build from directory: {KB_DIR}")

    # Check if KB directory exists
    if not os.path.exists(KB_DIR):
        logger.error(f"Knowledge base directory '{KB_DIR}' does not exist")
        raise SystemExit(f"Error: Directory '{KB_DIR}' not found. Please create it and add documents.")

    items = []
    file_count = 0

    # Recursively process all files in the knowledge base directory
    for path in glob.glob(os.path.join(KB_DIR, "**/*.*"), recursive=True):
        # Skip directories (glob may return them)
        if not os.path.isfile(path):
            continue

        logger.info(f"Processing file: {path}")
        file_count += 1

        try:
            # Read file contents
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()

            # Split into chunks and create metadata entries
            chunks = chunk_text(raw)
            logger.debug(f"  Created {len(chunks)} chunks from {path}")

            for i, ch in enumerate(chunks):
                items.append({
                    "source": path,
                    "chunk_id": i,
                    "text": ch
                })

        except Exception as e:
            logger.error(f"Failed to process file {path}: {e}")
            # Continue processing other files instead of failing completely
            continue

    # Validate that we found documents
    if not items:
        logger.error(f"No documents found in {KB_DIR}/")
        raise SystemExit(f"Error: No docs found under {KB_DIR}/. Please add documents to index.")

    logger.info(f"Processed {file_count} files into {len(items)} chunks")

    # Generate embeddings for all chunks
    embeddings = embed_texts([it["text"] for it in items])

    # Prepare output structure
    out = {
        "items": items,
        "embeddings": embeddings,
    }

    # Write to output file
    logger.info(f"Writing index to {OUT_FILE}")
    try:
        with open(OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f)
        logger.info(f"Successfully wrote {OUT_FILE} with {len(items)} chunks")
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise

    print(f"✓ Knowledge base built successfully!")
    print(f"  Files processed: {file_count}")
    print(f"  Total chunks: {len(items)}")
    print(f"  Output: {OUT_FILE}")

if __name__ == "__main__":
    main()