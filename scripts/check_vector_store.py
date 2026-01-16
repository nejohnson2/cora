"""
Check OpenAI Vector Store Status

This script checks the status of an existing OpenAI Vector Store.

Usage:
    python check_vector_store.py [vector_store_id]

If no vector_store_id is provided, it will use OPENAI_VECTOR_STORE_ID from .env
"""

import sys
from openai import OpenAI
from config import Config

def main():
    """Check vector store status."""

    # Get vector store ID
    if len(sys.argv) > 1:
        vector_store_id = sys.argv[1]
    else:
        vector_store_id = Config.OPENAI_VECTOR_STORE_ID

    if not vector_store_id:
        print("ERROR: No vector store ID provided")
        print("\nUsage:")
        print("  python check_vector_store.py [vector_store_id]")
        print("\nOr set OPENAI_VECTOR_STORE_ID in your .env file")
        sys.exit(1)

    # Check API key
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set in .env file")
        sys.exit(1)

    print("=" * 60)
    print("OpenAI Vector Store Status Check")
    print("=" * 60)
    print(f"Vector Store ID: {vector_store_id}")
    print()

    # Initialize OpenAI client
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    try:
        # Retrieve vector store
        vs = client.vector_stores.retrieve(vector_store_id)

        print("Status Information:")
        print("-" * 60)
        print(f"Name: {vs.name}")
        print(f"Status: {vs.status}")
        print(f"Created at: {vs.created_at}")
        print()

        print("File Counts:")
        print(f"  Total: {vs.file_counts.total}")
        print(f"  Completed: {vs.file_counts.completed}")
        print(f"  In Progress: {vs.file_counts.in_progress}")
        print(f"  Failed: {vs.file_counts.failed}")
        print(f"  Cancelled: {vs.file_counts.cancelled}")
        print()

        # Additional details if available
        if hasattr(vs, 'usage_bytes'):
            print(f"Storage Used: {vs.usage_bytes:,} bytes ({vs.usage_bytes / 1024 / 1024:.2f} MB)")

        if hasattr(vs, 'expires_at') and vs.expires_at:
            print(f"Expires at: {vs.expires_at}")

        print("=" * 60)

        # Status summary
        if vs.status == 'completed':
            print("✓ Vector store is ready to use!")
        elif vs.status == 'in_progress':
            print(f"⏳ Processing... ({vs.file_counts.completed}/{vs.file_counts.total} files done)")
            print("   Check again in a few moments")
        elif vs.status == 'failed':
            print("✗ Vector store processing failed")
            if vs.file_counts.failed > 0:
                print(f"   {vs.file_counts.failed} files failed to process")
        else:
            print(f"Status: {vs.status}")

    except Exception as e:
        print(f"ERROR: Failed to retrieve vector store: {e}")
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelled by user")
        sys.exit(1)
