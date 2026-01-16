# AI Tutor System

An AI-powered tutoring system that helps students learn through Socratic questioning and guided problem-solving. The system supports both local and OpenAI-based RAG (Retrieval-Augmented Generation) with multiple LLM provider options.

## Features

- **Dual RAG Modes**: Choose between local RAG or OpenAI Vector Stores
- **Socratic Teaching Method**: Guides students with questions rather than providing direct answers
- **Multiple LLM Providers**: OpenAI, local Ollama, or remote Ollama
- **Two Operation Modes**:
  - `/chat`: Regular tutoring without conversation logging
  - `/study`: Research mode with conversation logging (user consent)
- **Session Management**: Maintains conversation history per user session
- **Cost-Effective Options**: Use local Ollama for completely free operation

## Architecture

```
┌─────────────────┐
│   Student UI    │  (HTML/JavaScript)
└────────┬────────┘
         │
    ┌────▼─────┐
    │  FastAPI │  (app/main.py)
    │  Server  │
    └────┬─────┘
         │
         ├──────────────────────────────────┐
         │                                  │
    ┌────▼─────────────────────────┐  ┌────▼──────────────────┐
    │  LLM Client                  │  │  Knowledge Base       │
    │  ┌──────────┬──────────────┐ │  │  ┌──────────────────┐│
    │  │ OpenAI   │ Ollama       │ │  │  │ Local RAG:       ││
    │  │ API      │ (Local/Remote)│ │  │  │ BAAI Embeddings  ││
    │  └──────────┴──────────────┘ │  │  │ (kb_index.json)  ││
    └──────────────────────────────┘  │  │                  ││
                                      │  │ OpenAI RAG:      ││
                                      │  │ Vector Stores API││
                                      │  └──────────────────┘│
                                      └───────────────────────┘
```

## Project Structure

```
oora/
├── app/
│   ├── main.py                # FastAPI application entry point
│   ├── routes/
│   │   └── chat.py           # Chat endpoints and logic
│   ├── services/
│   │   ├── kb_service.py     # Local RAG knowledge base
│   │   ├── openai_kb_service.py  # OpenAI Vector Store service
│   │   └── session_service.py    # Session management
│   └── utils/
│       └── performance.py    # Performance monitoring
├── scripts/
│   ├── build_kb.py           # Build local KB index
│   ├── upload_to_openai.py   # Upload files to OpenAI Vector Store
│   ├── check_vector_store.py # Check OpenAI Vector Store status
│   └── test_rag_modes.py     # Test both RAG modes
├── static/                   # Static web assets
├── templates/                # HTML templates
├── config.py                 # Configuration management
├── llm_client.py            # LLM abstraction layer
├── run.py                   # Server startup script
├── system_prompt.txt        # Default system prompt
├── .env.example             # Example environment configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Setup

### Prerequisites

- Python 3.9+
- **For Local RAG Mode** (optional):
  - sentence-transformers library
  - BAAI/bge-base-en-v1.5 model (~400MB, downloads automatically)
- **For OpenAI RAG Mode** (optional):
  - OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- **For LLM Chat Provider** (choose one):
  - OpenAI API key, OR
  - Ollama installed locally ([Install Ollama](https://ollama.ai)), OR
  - Access to a remote Ollama instance

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd oora
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Configuration Options

The system supports two RAG modes:

#### Option 1: Local RAG (Free, Private)

```bash
# In .env
RAG_MODE=local
LLM_PROVIDER=openai  # or ollama_local/ollama_remote
OPENAI_API_KEY=your-api-key-here
```

Then build the local knowledge base:
```bash
# Add your documents to kb/ directory
mkdir -p kb
# Add .txt files to kb/

# Build the index
python scripts/build_kb.py
```

#### Option 2: OpenAI RAG (Cloud-based)

```bash
# In .env
RAG_MODE=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_VECTOR_STORE_ID=vs_xxx  # Get this after upload
```

Then upload your files to OpenAI:
```bash
# Add your documents to kb/ directory
mkdir -p kb
# Add .txt files to kb/

# Upload to OpenAI Vector Store
python scripts/upload_to_openai.py

# Copy the vector store ID to your .env file
```

## RAG Modes Comparison

| Feature | Local RAG | OpenAI RAG |
|---------|-----------|------------|
| **Cost** | Free (after initial download) | ~$0.10/GB/day storage + usage |
| **Privacy** | Fully private, runs locally | Data stored on OpenAI servers |
| **Setup** | Build index locally | Upload files to OpenAI |
| **Performance** | Fast (local lookup) | ~5-7 seconds per request |
| **Embeddings** | BAAI/bge-base-en-v1.5 (local) | OpenAI's proprietary embeddings |
| **Maintenance** | Rebuild index when docs change | Managed by OpenAI |
| **Best For** | Privacy-sensitive, cost-conscious | Convenience, managed infrastructure |

## LLM Provider Options

### OpenAI (Default)

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
OPENAI_CHAT_MODEL=gpt-4o-mini
```

### Local Ollama (Free)

```bash
LLM_PROVIDER=ollama_local
OLLAMA_CHAT_MODEL=llama3.1:8b
```

First install Ollama from https://ollama.ai, then:
```bash
ollama pull llama3.1:8b
```

### Remote Ollama

```bash
LLM_PROVIDER=ollama_remote
OLLAMA_REMOTE_URL=https://your-ollama-server.com
OLLAMA_CHAT_MODEL=llama3.1:8b
```

## Running the Server

Start the server:
```bash
python run.py
```

Or with custom host/port:
```bash
python run.py --host 0.0.0.0 --port 8000
```

The server will be available at:
- **Regular chat (no logging)**: http://localhost:8000/chat
- **Study mode (with logging)**: http://localhost:8000/study

## Usage

### Chat Mode

Navigate to `/chat` for regular tutoring sessions:
- Conversations stored in memory only
- No persistent logging
- Ideal for general tutoring

### Study Mode

Navigate to `/study` for research/assessment:
- Conversations logged to `logs/consented_chats.jsonl`
- Users are informed of logging
- Includes timestamps and session IDs

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Redirect to `/chat` |
| GET | `/chat` | Serve chat interface |
| GET | `/study` | Serve study mode interface |
| POST | `/api/chat` | Process chat message (no logging) |
| POST | `/api/study/chat` | Process chat message (with logging) |
| POST | `/api/reset` | Clear session history |
| POST | `/api/study/reset` | Clear session history (study mode) |
| GET | `/health` | Health check |
| GET | `/stats/kb` | Knowledge base statistics |
| GET | `/stats/sessions` | Session statistics |
| GET | `/config` | View current configuration |

## Scripts

### build_kb.py

Build the local knowledge base index:
```bash
python scripts/build_kb.py
```

- Processes all files in `kb/` directory
- Generates embeddings using BAAI/bge-base-en-v1.5
- Creates `kb_index.json`
- Supports `.txt` and `.md` files

### upload_to_openai.py

Upload files to OpenAI Vector Store:
```bash
python scripts/upload_to_openai.py
```

- Uploads all files from `kb/` directory
- Creates a new OpenAI Vector Store
- Displays the vector store ID to add to `.env`
- Waits for processing to complete

### check_vector_store.py

Check OpenAI Vector Store status:
```bash
python scripts/check_vector_store.py [vector_store_id]
```

- Shows vector store status
- Displays file counts and processing state
- Uses `OPENAI_VECTOR_STORE_ID` from `.env` if no ID provided

### test_rag_modes.py

Test both RAG modes:
```bash
python scripts/test_rag_modes.py
```

- Tests current RAG mode configuration
- Verifies knowledge base access
- Useful for debugging setup issues

## Environment Variables

### Core Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `RAG_MODE` | RAG mode: `local` or `openai` | `local` |
| `LLM_PROVIDER` | LLM provider: `openai`, `ollama_local`, or `ollama_remote` | `openai` |

### OpenAI Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required for OpenAI |
| `OPENAI_CHAT_MODEL` | Chat model name | `gpt-4o-mini` |
| `OPENAI_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `OPENAI_VECTOR_STORE_ID` | Vector Store ID (for OpenAI RAG) | Required for OpenAI RAG |

### Ollama Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `OLLAMA_LOCAL_URL` | Local Ollama URL | `http://localhost:11434` |
| `OLLAMA_REMOTE_URL` | Remote Ollama URL | None |
| `OLLAMA_CHAT_MODEL` | Chat model name | `llama3.1:8b` |
| `OLLAMA_TIMEOUT` | Request timeout (seconds) | `120` |

### Knowledge Base Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `KB_DIR` | Knowledge base directory | `kb` |
| `KB_INDEX_FILE` | KB index file path | `kb_index.json` |
| `CHUNK_SIZE` | Text chunk size (characters) | `900` |
| `CHUNK_OVERLAP` | Chunk overlap (characters) | `150` |
| `CONTEXT_CHUNKS` | KB chunks to retrieve | `6` |

### Server Settings

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `8000` |
| `MAX_TURNS` | Conversation turns to keep | `12` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_DIR` | Logs directory | `logs` |

### Custom System Prompt

Set a custom system prompt file:
```bash
SYSTEM_PROMPT_FILE=system_prompt.txt
```

## Teaching Philosophy

The AI tutor follows these principles:

1. **Socratic Method**: Asks guiding questions rather than giving answers
2. **No Direct Solutions**: Refuses to provide complete solutions or final answers
3. **Contextual Guidance**: Uses knowledge base materials to provide relevant hints
4. **Student-Centered**: Focuses on understanding the student's thought process
5. **Concise Communication**: Keeps responses focused and brief

## Troubleshooting

### Knowledge Base Not Found (Local RAG)

```
Error: Knowledge base file 'kb_index.json' not found
```
**Solution**: Run `python scripts/build_kb.py`

### Vector Store Not Set (OpenAI RAG)

```
Error: OPENAI_VECTOR_STORE_ID is not set
```
**Solution**: Run `python scripts/upload_to_openai.py` and add the ID to `.env`

### OpenAI API Errors

```
Error: 401 - Invalid API key
```
**Solution**: Check your `OPENAI_API_KEY` in `.env` is correct

### Port Already in Use

```
Error: Address already in use
```
**Solution**: Kill the existing process or use a different port:
```bash
# Find and kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
python run.py --port 8001
```

### Ollama Connection Issues

```
Error: Cannot connect to Ollama
```
**Solution**: Ensure Ollama is running:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
```

## Performance

### Local RAG
- Context retrieval: ~10-50ms
- Total request time: ~1-3s (depending on LLM)

### OpenAI RAG
- Context retrieval + LLM: ~5-7s
- Uses OpenAI Responses API with file_search
- Single API call includes both retrieval and generation

## Migration Notes

### OpenAI Responses API

As of January 2026, the system uses OpenAI's **Responses API** (not the deprecated Assistants API). The Assistants API will be sunset on August 26, 2026.

Key differences:
- **Responses API**: Single API call, stateless, simpler
- **Assistants API** (deprecated): Multiple API calls, stateful, complex

The migration was completed to future-proof the system.

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **User Privacy**: Inform users when conversations are logged (study mode)
- **Session Cookies**: Uses httponly and samesite flags
- **Input Validation**: Pydantic models validate all incoming data
- **Sensitive Information**: Warn users not to share personal info in study mode

## Development

### Running in Development Mode

```bash
uvicorn app.main:app --reload --log-level debug
```

### Checking Configuration

```bash
python -c "from config import Config; Config.display()"
```

### Viewing Logs

```bash
# Application logs
tail -f tutor_server.log

# Conversation logs (study mode)
tail -f logs/consented_chats.jsonl
```

## License

This project is provided as-is for educational purposes.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues or questions:
1. Check the application logs in `tutor_server.log`
2. Review this README for configuration help
3. Check the [OpenAI API documentation](https://platform.openai.com/docs)
4. Open an issue on GitHub
