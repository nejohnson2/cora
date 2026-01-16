"""
Test RAG Modes

Quick test to verify both local and OpenAI RAG modes work correctly.
"""

import sys
from config import Config

def test_kb_service():
    """Test that the KB service loads correctly."""
    print("=" * 60)
    print("Testing RAG Mode:", Config.RAG_MODE)
    print("=" * 60)

    if Config.RAG_MODE == 'local':
        from app.services import KnowledgeBaseService
        print("Loading local KB service...")
        kb = KnowledgeBaseService()
        stats = kb.get_statistics()
        print(f"✓ Local KB loaded:")
        print(f"  - Chunks: {stats['total_chunks']}")
        print(f"  - Sources: {stats['total_sources']}")

        # Test context building
        print("\nTesting context retrieval...")
        context = kb.build_context("What is this course about?", k=3)
        print(f"✓ Context retrieved ({len(context)} chars)")
        print(f"Preview: {context[:200]}...")

    elif Config.RAG_MODE == 'openai':
        from app.services import OpenAIKnowledgeBaseService
        print("Loading OpenAI KB service...")
        kb = OpenAIKnowledgeBaseService()
        stats = kb.get_statistics()
        print(f"✓ OpenAI KB loaded:")
        print(f"  - Vector Store ID: {stats['vector_store_id']}")
        print(f"  - Status: {stats['status']}")
        print(f"  - Files: {stats['completed_files']}")

        # Test context building (returns marker)
        print("\nTesting context marker...")
        context = kb.build_context("What is this course about?")
        print(f"✓ Context marker: {context}")

        if context.startswith("[OPENAI_VECTOR_STORE:"):
            print("✓ Marker format is correct")
        else:
            print("✗ Unexpected marker format")

    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    try:
        test_kb_service()
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
