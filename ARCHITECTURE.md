# PDF Query Agent - Architecture Documentation

## Overview

A Streamlit-based PDF document question-answering application that uses LlamaIndex with a multi-agent workflow to retrieve relevant information from uploaded documents.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   Sidebar   │  │   Query     │  │    Response Display    │  │
│  │   (Upload)  │  │   Input     │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Session State                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │   index     │  │  workflow   │  │     current_name        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Query Processing                              │
│                                                                 │
│   ┌──────────────┐      ┌──────────────┐                       │
│   │ run_async    │─────▶│ run_workflow │                      │
│   │ (threading)  │      │   _query     │                      │
│   └──────────────┘      └──────────────┘                       │
│                                 │                                │
│                                 ▼                                │
│                    ┌────────────────────────┐                   │
│                    │   AgentWorkflow        │                   │
│                    │   (LlamaIndex)          │                   │
│                    └────────────────────────┘                   │
│                               │                                  │
│              ┌────────────────┼────────────────┐                 │
│              ▼                ▼                ▼                 │
│      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│      │ Researcher  │─▶│   Writer    │  │   Writer    │          │
│      │  (Agent)    │  │   (Agent)   │  │   (Agent)   │          │
│      └─────────────┘  └─────────────┘  └─────────────┘          │
│              │                                                    │
│              ▼                                                    │
│      ┌─────────────────────────────────────────────┐            │
│      │              Query Tools                     │            │
│      │  ┌────────────────┐  ┌─────────────────┐     │            │
│      │  │ graph_search   │  │ keyword_search  │     │            │
│      │  │ (LLMSynonym    │  │ (BM25Retriever) │     │            │
│      │  │  + VectorCtx)  │  │                 │     │            │
│      │  └────────────────┘  └─────────────────┘     │            │
│      └─────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Index Storage Layer                          │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐    │
│   │              index_store/ (directory)                   │    │
│   │   ┌────────────┐  ┌────────────┐  ┌────────────┐      │    │
│   │   │  doc1/     │  │  doc2/     │  │  docN/     │      │    │
│   │   │  (persist) │  │  (persist) │  │  (persist) │      │    │
│   │   └────────────┘  └────────────┘  └────────────┘      │    │
│   └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. LLM & Embedding Setup (Lines 77-83)

```
Ollama (LLM)          → granite4:latest
OllamaEmbedding       → granite-embedding:30m
```

Global settings applied to LlamaIndex for all operations.

### 2. PDF Processing Pipeline

| Function | Purpose |
|----------|---------|
| `process_pdf` | Save uploaded file to temp directory |
| `build_index` | Create PropertyGraphIndex from documents |
| `save_index` | Persist index to disk |
| `load_index` | Load saved index from disk |

Index is built with:
- **Chunk size**: 512 characters
- **Chunk overlap**: 50 characters
- **Embed KG**: True (embeds knowledge graph)

### 3. Agent Workflow

```
┌────────────────────────────────────────────────────────────┐
│  Researcher Agent                                           │
│  ━━━━━━━━━━━━━━━━━                                          │
│  Description: Retrieves relevant document passages        │
│  Tools: graph_search, keyword_search                       │
│  Handoff to: Writer                                        │
│                                                             │
│  Process:                                                  │
│  1. Analyze user question                                  │
│  2. Query BOTH tools                                       │
│  3. Extract facts with page numbers                        │
│  4. Format as handoff message                              │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  Writer Agent                                              │
│  ━━━━━━━━━━━━━━                                            │
│  Description: Generates final answer from facts           │
│  Tools: None                                                │
│  Handoff to: Researcher (for clarification)               │
│                                                             │
│  Process:                                                  │
│  1. Parse handoff from Researcher                         │
│  2. If facts exist → answer question                       │
│  3. If "None found" → return no info message              │
│  4. Output directly without prefixes                        │
└────────────────────────────────────────────────────────────┘
```

### 4. Query Tools

| Tool | Retriever | Best For |
|------|-----------|----------|
| `graph_search` | LLMSynonymRetriever + VectorContextRetriever | Relationships, concepts, semantic search |
| `keyword_search` | BM25Retriever | Exact keywords, specific terms |

### 5. Query Execution Flow

```
User Query
    │
    ▼
build_rfp_answer_with_workflow(workflow, index, query)
    │
    ▼
run_async(run_workflow_query)  ──▶ New Thread with Event Loop
    │                                        │
    │                                        ▼
    │                               workflow.run(query)
    │                                        │
    │                                        ▼
    │                               AgentWorkflow
    │                                        │
    │                    ┌──────────────────┼──────────────────┐
    │                    ▼                  ▼                  ▼
    │              Researcher ─────▶ Writer ─────▶ Final Output
    │              (search)         (format)
    │                    │                  │
    │                    ▼                  ▼
    │              graph_search      Final Answer
    │              keyword_search
    │
    ▼
Response returned to Streamlit
```

### 6. Storage

- **Temporary**: `tempfile.mkdtemp()` - stores uploaded PDFs during processing
- **Persistent**: `./index_store/` - stores serialized indices

## Data Flow

```
PDF Upload
    │
    ▼
process_pdf() ──▶ Temp File
    │
    ▼
build_index() ──▶ PropertyGraphIndex
    │                 (chunks + embeddings + knowledge graph)
    │
    ▼
save_index() ──▶ ./index_store/{name}/
    │
    ▼
User Query
    │
    ▼
load_index() + create_workflow() ──▶ AgentWorkflow
    │
    ▼
Query → Researcher → Tools → Writer → Answer
```

## Key Design Decisions

1. **PropertyGraphIndex**: Chosen for both embedding text AND building knowledge graph relationships
2. **Dual Retrievers**: BM25 + Vector to capture both keyword and semantic matches
3. **Two-Agent Workflow**: Separation of retrieval (Researcher) and synthesis (Writer)
4. **Thread-based Async**: `run_async` runs workflow in isolated thread to avoid event loop issues in Streamlit

## File Structure

```
.
├── app.py              # Main application
├── index_store/        # Persisted indices (created at runtime)
│   └── {doc_name}/
│       ├── graph_store/
│       ├── docstore/
│       └── vector_store/
└── venv/              # Virtual environment
```

## Session State Variables

| Variable | Type | Purpose |
|----------|------|---------|
| `index` | PropertyGraphIndex | Loaded document index |
| `workflow` | AgentWorkflow | Agent workflow instance |
| `current_name` | str | Name of currently loaded index |