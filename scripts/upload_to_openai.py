"""
Upload Knowledge Base to OpenAI Vector Store

This script uploads all files from the kb directory to an OpenAI Vector Store
for use with the OpenAI RAG mode.

Usage:
    python upload_to_openai.py

The script will:
1. Read all files from the KB_DIR directory
2. Create a new OpenAI Vector Store
3. Upload all files to the vector store
4. Output the vector store ID to set in your .env file
"""

import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from config import Config

def main():
    """Upload knowledge base files to OpenAI Vector Store."""

    print("=" * 60)
    print("OpenAI Vector Store Upload")
    print("=" * 60)

    # Check API key
    if not Config.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY is not set in .env file")
        print("Please set your OpenAI API key before running this script")
        sys.exit(1)

    # Initialize OpenAI client
    client = OpenAI(api_key=Config.OPENAI_API_KEY)

    # Get KB directory
    kb_dir = Path(Config.KB_DIR)
    if not kb_dir.exists():
        print(f"ERROR: Knowledge base directory '{kb_dir}' does not exist")
        sys.exit(1)

    # Find all files in KB directory (recursive)
    print(f"\nScanning directory: {kb_dir}")
    files_to_upload = []

    for file_path in kb_dir.rglob('*'):
        if file_path.is_file():
            # Skip hidden files and common non-document files
            if not file_path.name.startswith('.'):
                files_to_upload.append(file_path)

    if not files_to_upload:
        print(f"ERROR: No files found in {kb_dir}")
        sys.exit(1)

    print(f"Found {len(files_to_upload)} files to upload:")
    for file_path in files_to_upload:
        file_size = file_path.stat().st_size
        print(f"  - {file_path.relative_to(kb_dir)} ({file_size:,} bytes)")

    # Confirm with user
    print("\n" + "=" * 60)
    response = input("Proceed with upload? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Upload cancelled")
        sys.exit(0)

    print("\n" + "=" * 60)
    print("Starting upload process...")
    print("=" * 60)

    # Upload files to OpenAI
    print("\nStep 1: Uploading files to OpenAI...")
    uploaded_file_ids = []

    for i, file_path in enumerate(files_to_upload, 1):
        print(f"  [{i}/{len(files_to_upload)}] Uploading {file_path.name}...", end=" ")

        try:
            with open(file_path, 'rb') as f:
                file_obj = client.files.create(
                    file=f,
                    purpose='assistants'
                )
            uploaded_file_ids.append(file_obj.id)
            print(f"✓ (ID: {file_obj.id})")
        except Exception as e:
            print(f"✗ Error: {e}")
            print("Cleaning up uploaded files...")
            # Clean up any uploaded files
            for file_id in uploaded_file_ids:
                try:
                    client.files.delete(file_id)
                except:
                    pass
            sys.exit(1)

    print(f"\n✓ Successfully uploaded {len(uploaded_file_ids)} files")

    # Create vector store
    print("\nStep 2: Creating vector store...")

    try:
        vector_store = client.vector_stores.create(
            name=f"AI Tutor KB - {time.strftime('%Y-%m-%d %H:%M:%S')}",
            file_ids=uploaded_file_ids
        )
        print(f"✓ Vector store created: {vector_store.id}")
    except Exception as e:
        print(f"✗ Error creating vector store: {e}")
        print("Cleaning up uploaded files...")
        for file_id in uploaded_file_ids:
            try:
                client.files.delete(file_id)
            except:
                pass
        sys.exit(1)

    # Wait for vector store to be ready
    print("\nStep 3: Waiting for vector store to process files...")
    print("This may take a few moments...")
    print("(Press Ctrl+C to skip waiting - the vector store will continue processing)")

    max_attempts = 60  # 5 minutes max
    attempt = 0
    skipped = False

    while attempt < max_attempts:
        try:
            vs_status = client.vector_stores.retrieve(vector_store.id)

            if vs_status.status == 'completed':
                print(f"\n✓ Vector store is ready!")
                print(f"  - Total files: {vs_status.file_counts.completed}")
                break
            elif vs_status.status == 'failed':
                print(f"\n✗ Vector store processing failed")
                print(f"  - Failed files: {vs_status.file_counts.failed}")
                sys.exit(1)
            else:
                # Still processing
                completed = vs_status.file_counts.completed
                in_progress = vs_status.file_counts.in_progress
                failed = vs_status.file_counts.failed
                total = vs_status.file_counts.total
                status = vs_status.status

                # Show more detailed status
                status_msg = f"  Status: {status} | {completed}/{total} complete"
                if in_progress > 0:
                    status_msg += f" | {in_progress} in progress"
                if failed > 0:
                    status_msg += f" | {failed} failed"
                status_msg += " " * 20  # Clear any leftover characters

                print(status_msg, end="\r")
                time.sleep(5)
                attempt += 1
        except KeyboardInterrupt:
            print("\n\n⚠️  Skipping wait - vector store will continue processing in the background")
            skipped = True
            break
        except Exception as e:
            print(f"\n✗ Error checking status: {e}")
            print("The vector store was created but status check failed.")
            print("It may still be processing successfully - check the OpenAI dashboard.")
            skipped = True
            break

    if attempt >= max_attempts and not skipped:
        print(f"\n⚠️  Timeout waiting for vector store to be ready")
        print("The vector store may still be processing - check the OpenAI dashboard.")
        print("You can still use the vector store ID even if processing is ongoing.")

    # Success!
    print("\n" + "=" * 60)
    print("SUCCESS! Vector store created and ready to use")
    print("=" * 60)
    print(f"\nVector Store ID: {vector_store.id}")
    print("\nNext steps:")
    print("1. Add this to your .env file:")
    print(f"   OPENAI_VECTOR_STORE_ID={vector_store.id}")
    print("2. Set RAG_MODE to 'openai' in your .env file:")
    print("   RAG_MODE=openai")
    print("3. Restart your server")
    print("\n" + "=" * 60)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
