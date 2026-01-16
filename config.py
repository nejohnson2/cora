"""
Configuration Module

Loads environment variables and provides configuration settings for the AI tutor application.
Uses python-dotenv to load variables from a .env file for local development.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
# This allows for local development without exposing secrets in code
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)


class Config:
    """
    Configuration class for the AI Tutor application.

    All settings can be overridden via environment variables.
    """

    # ==========================================
    # LLM Provider Configuration
    # ==========================================

    # LLM provider: 'openai', 'ollama_local', or 'ollama_remote'
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()

    # ==========================================
    # OpenAI API Configuration
    # ==========================================

    # OpenAI API key (REQUIRED if using OpenAI)
    # Get your API key from: https://platform.openai.com/api-keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # OpenAI model for chat completions
    # Valid models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
    OPENAI_CHAT_MODEL = os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini')

    # OpenAI model for embeddings
    OPENAI_EMBEDDING_MODEL = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')

    # ==========================================
    # Ollama Configuration
    # ==========================================

    # Ollama base URL for local instance
    OLLAMA_LOCAL_URL = os.getenv('OLLAMA_LOCAL_URL', 'http://localhost:11434')

    # Ollama base URL for remote instance
    OLLAMA_REMOTE_URL = os.getenv('OLLAMA_REMOTE_URL', '')

    # Ollama API key for remote instance (if required)
    OLLAMA_API_KEY = os.getenv('OLLAMA_API_KEY', '')

    # Ollama model for chat completions
    OLLAMA_CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'llama3.1:8b')

    # Ollama model for embeddings
    OLLAMA_EMBEDDING_MODEL = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')

    # Ollama request timeout (seconds)
    OLLAMA_TIMEOUT = int(os.getenv('OLLAMA_TIMEOUT', '120'))

    # ==========================================
    # Knowledge Base Configuration
    # ==========================================

    # RAG Mode: 'local' uses local embeddings and search, 'openai' uses OpenAI Vector Stores API
    RAG_MODE = os.getenv('RAG_MODE', 'local').lower()

    # OpenAI Vector Store ID (required if RAG_MODE=openai)
    # This is the vector store ID returned by the upload_to_openai.py script
    OPENAI_VECTOR_STORE_ID = os.getenv('OPENAI_VECTOR_STORE_ID', '')

    # Directory containing knowledge base documents
    KB_DIR = os.getenv('KB_DIR', 'kb')

    # Output file for the knowledge base index
    KB_INDEX_FILE = os.getenv('KB_INDEX_FILE', 'kb_index.json')

    # Text chunking parameters
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '900'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '150'))

    # Number of knowledge base chunks to retrieve for context
    CONTEXT_CHUNKS = int(os.getenv('CONTEXT_CHUNKS', '6'))

    # Knowledge base embedding dimension
    # The KB uses BAAI/bge-base-en-v1.5 which produces 768-dimensional embeddings
    # This is independent of the LLM provider's embedding model
    KB_EMBEDDING_DIM = 768

    # ==========================================
    # Server Configuration
    # ==========================================

    # Server host and port
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '8000'))

    # Maximum number of conversation turns to keep in memory per session
    MAX_TURNS = int(os.getenv('MAX_TURNS', '12'))

    # Session cookie name
    SESSION_COOKIE_NAME = os.getenv('SESSION_COOKIE_NAME', 'tutor_session_id')

    # ==========================================
    # Logging Configuration
    # ==========================================

    # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

    # Directory for storing chat logs
    LOG_DIR = os.getenv('LOG_DIR', 'logs')

    # Path for consented chat logs (JSONL format)
    CONSENT_LOG_FILE = os.getenv('CONSENT_LOG_FILE', 'consented_chats.jsonl')

    # Server log file
    SERVER_LOG_FILE = os.getenv('SERVER_LOG_FILE', 'tutor_server.log')

    # ==========================================
    # System Prompt Configuration
    # ==========================================

    # Path to custom system prompt file (optional)
    # If not set, uses the default prompt in server.py
    SYSTEM_PROMPT_FILE = os.getenv('SYSTEM_PROMPT_FILE', None)

    # ==========================================
    # Validation
    # ==========================================

    @classmethod
    def validate(cls):
        """
        Validate required configuration values.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        errors = []

        # Validate LLM provider
        valid_providers = ['openai', 'ollama_local', 'ollama_remote']
        if cls.LLM_PROVIDER not in valid_providers:
            errors.append(
                f"LLM_PROVIDER must be one of {valid_providers}, got '{cls.LLM_PROVIDER}'"
            )

        # Validate RAG mode
        valid_rag_modes = ['local', 'openai']
        if cls.RAG_MODE not in valid_rag_modes:
            errors.append(
                f"RAG_MODE must be one of {valid_rag_modes}, got '{cls.RAG_MODE}'"
            )

        # Check provider-specific requirements
        if cls.LLM_PROVIDER == 'openai':
            if not cls.OPENAI_API_KEY:
                errors.append(
                    "OPENAI_API_KEY is not set but LLM_PROVIDER is 'openai'. "
                    "Please set it in your .env file or as an environment variable."
                )
        elif cls.LLM_PROVIDER == 'ollama_remote':
            if not cls.OLLAMA_REMOTE_URL:
                errors.append(
                    "OLLAMA_REMOTE_URL is not set but LLM_PROVIDER is 'ollama_remote'. "
                    "Please set it in your .env file."
                )

        # Check RAG mode requirements
        if cls.RAG_MODE == 'openai':
            if not cls.OPENAI_API_KEY:
                errors.append(
                    "OPENAI_API_KEY is not set but RAG_MODE is 'openai'. "
                    "Please set it in your .env file."
                )
            if not cls.OPENAI_VECTOR_STORE_ID:
                errors.append(
                    "OPENAI_VECTOR_STORE_ID is not set but RAG_MODE is 'openai'. "
                    "Please run upload_to_openai.py to create a vector store and set the ID."
                )

        # Validate numeric parameters
        if cls.CHUNK_SIZE <= 0:
            errors.append(f"CHUNK_SIZE must be positive, got {cls.CHUNK_SIZE}")

        if cls.CHUNK_OVERLAP < 0:
            errors.append(f"CHUNK_OVERLAP must be non-negative, got {cls.CHUNK_OVERLAP}")

        if cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            errors.append(
                f"CHUNK_OVERLAP ({cls.CHUNK_OVERLAP}) must be less than "
                f"CHUNK_SIZE ({cls.CHUNK_SIZE})"
            )

        if cls.MAX_TURNS <= 0:
            errors.append(f"MAX_TURNS must be positive, got {cls.MAX_TURNS}")

        if cls.CONTEXT_CHUNKS <= 0:
            errors.append(f"CONTEXT_CHUNKS must be positive, got {cls.CONTEXT_CHUNKS}")

        if cls.PORT <= 0 or cls.PORT > 65535:
            errors.append(f"PORT must be between 1 and 65535, got {cls.PORT}")

        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if cls.LOG_LEVEL.upper() not in valid_log_levels:
            errors.append(
                f"LOG_LEVEL must be one of {valid_log_levels}, got {cls.LOG_LEVEL}"
            )

        # If there are errors, raise them all at once
        if errors:
            raise ValueError(
                "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @classmethod
    def get_consent_log_path(cls):
        """Get the full path to the consented chats log file."""
        return os.path.join(cls.LOG_DIR, cls.CONSENT_LOG_FILE)

    @classmethod
    def load_system_prompt(cls):
        """
        Load system prompt from file if configured, otherwise return None.

        Returns:
            str or None: The system prompt content, or None if no file is configured
        """
        if cls.SYSTEM_PROMPT_FILE and os.path.exists(cls.SYSTEM_PROMPT_FILE):
            with open(cls.SYSTEM_PROMPT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
        return None

    @classmethod
    def display(cls):
        """Display current configuration (with API keys masked)."""
        print("=" * 60)
        print("AI Tutor Configuration")
        print("=" * 60)
        print(f"RAG Mode: {cls.RAG_MODE}")
        print(f"LLM Provider: {cls.LLM_PROVIDER}")

        if cls.LLM_PROVIDER == 'openai':
            print(f"OpenAI API Key: {'*' * 20 if cls.OPENAI_API_KEY else 'NOT SET'}")
            print(f"OpenAI Chat Model: {cls.OPENAI_CHAT_MODEL}")
            print(f"OpenAI Embedding Model: {cls.OPENAI_EMBEDDING_MODEL}")
        elif cls.LLM_PROVIDER == 'ollama_local':
            print(f"Ollama Local URL: {cls.OLLAMA_LOCAL_URL}")
            print(f"Ollama Chat Model: {cls.OLLAMA_CHAT_MODEL}")
            print(f"Ollama Embedding Model: {cls.OLLAMA_EMBEDDING_MODEL}")
            print(f"Ollama Timeout: {cls.OLLAMA_TIMEOUT}s")
        elif cls.LLM_PROVIDER == 'ollama_remote':
            print(f"Ollama Remote URL: {cls.OLLAMA_REMOTE_URL}")
            print(f"Ollama API Key: {'*' * 20 if cls.OLLAMA_API_KEY else 'NOT SET'}")
            print(f"Ollama Chat Model: {cls.OLLAMA_CHAT_MODEL}")
            print(f"Ollama Embedding Model: {cls.OLLAMA_EMBEDDING_MODEL}")
            print(f"Ollama Timeout: {cls.OLLAMA_TIMEOUT}s")

        print(f"Knowledge Base Dir: {cls.KB_DIR}")
        if cls.RAG_MODE == 'local':
            print(f"KB Index File: {cls.KB_INDEX_FILE}")
            print(f"Chunk Size: {cls.CHUNK_SIZE}")
            print(f"Chunk Overlap: {cls.CHUNK_OVERLAP}")
            print(f"Context Chunks: {cls.CONTEXT_CHUNKS}")
        elif cls.RAG_MODE == 'openai':
            print(f"OpenAI Vector Store ID: {cls.OPENAI_VECTOR_STORE_ID[:20]}..." if len(cls.OPENAI_VECTOR_STORE_ID) > 20 else cls.OPENAI_VECTOR_STORE_ID)
        print(f"Server: {cls.HOST}:{cls.PORT}")
        print(f"Max Turns: {cls.MAX_TURNS}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print(f"Log Directory: {cls.LOG_DIR}")
        print("=" * 60)


# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    # Print validation errors but don't crash on import
    # This allows the user to see what's wrong
    print(f"\n⚠️  Configuration Error:\n{e}\n")
