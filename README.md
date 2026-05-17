# function-agent-rag

A multi-agent PDF question-answering system using LlamaIndex with function-calling agents. Features dual-retriever (BM25 + Vector) search, property graph indexing, and a Researcher → Writer agent workflow for accurate, grounded document answers.

## Features

- **Multi-Agent Workflow**: Researcher agent retrieves facts, Writer agent synthesizes answers
- **Dual Retrieval**: Combines BM25 (keyword) and Vector (semantic) search
- **Property Graph Index**: Builds knowledge graph from documents for better relationships
- **Function Calling**: Uses LlamaIndex FunctionAgent for structured tool usage
- **Persistent Storage**: Save and load document indices
- **Streamlit UI**: Simple web interface for PDF upload and querying

## Architecture

```
User Query → Researcher Agent → [graph_search, keyword_search] → Writer Agent → Answer
```

- **Researcher**: Uses two tools (graph_search + keyword_search) to find relevant passages
- **Writer**: Synthesizes facts into final answer, cites sources

## Requirements

- Python 3.10+
- Ollama running locally (for LLM + embeddings)
- Streamlit

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/function-agent-rag.git
cd function-agent-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Ollama (required)
ollama serve
# Pull required models:
# ollama pull granite4:latest
# ollama pull granite-embedding:30m
```

## Usage

```bash
# Run the app
streamlit run app.py
```

1. Open browser at `http://localhost:8501`
2. Upload PDF documents via sidebar
3. Enter a name for the embedding and click "Create Embedding"
4. Wait for indexing to complete
5. Ask questions about your documents

## Project Structure

```
function-agent-rag/
├── app.py              # Main Streamlit application
├── ARCHITECTURE.md    # Architecture documentation
├── README.md          # This file
└── index_store/       # Persisted indices (created at runtime)
```

## Configuration

Edit `app.py` to change:
- LLM model (line 78): `model="granite4:latest"`
- Embedding model (line 80): `model_name="granite-embedding:30m"`
- Ollama base URL (line 78, 81): `base_url="http://localhost:11434"`

## License

MIT License