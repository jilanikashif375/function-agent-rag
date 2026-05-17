import os
import tempfile
from typing import List
import asyncio
import threading
import streamlit as st

from llama_index.core import (
    PropertyGraphIndex,
    SimpleDirectoryReader,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.indices.property_graph import (
    LLMSynonymRetriever,
    VectorContextRetriever,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
from llama_index.core.schema import NodeWithScore

def run_async(async_fn, *args, **kwargs):
    """Run async work on a dedicated thread with its own event loop."""
    result = {}
    error = {}

    def runner():
        loop = asyncio.new_event_loop()
        try:
            result["value"] = loop.run_until_complete(async_fn(*args, **kwargs))
        except Exception as exc:
            error["value"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]

    return result.get("value")


async def run_workflow_query(workflow, query: str):
    """Run a query through the agent workflow."""
    response = await workflow.run(query)
    return response


def build_rfp_answer_with_workflow(workflow, index, query: str) -> tuple[str, List[NodeWithScore]]:
    """Build an answer using the agent workflow."""
    try:
        response = run_async(run_workflow_query, workflow, query)

        if hasattr(response, "content") and response.content:
            response_text = str(response.content)
        else:
            response_text = str(response)

        return response_text, []
    except Exception as e:
        return f"Workflow error: {str(e)}", []

# Temp storage
DATA_DIR = tempfile.mkdtemp()

# Persistent storage for embeddings
INDEX_STORE_DIR = os.path.join(os.getcwd(), "index_store")
os.makedirs(INDEX_STORE_DIR, exist_ok=True)

# LLM setup
llm = Ollama(model="granite4:latest", base_url="http://localhost:11434")
Settings.llm = llm
Settings.embed_model = OllamaEmbedding(
    model_name="granite-embedding:30m",
    base_url="http://localhost:11434"
)

# ---------- Helpers ----------

def process_pdf(uploaded_file) -> str:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def build_index(file_paths: List[str]):
    documents = []
    for path in file_paths:
        reader = SimpleDirectoryReader(input_files=[path])
        documents.extend(reader.load_data())

    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    index = PropertyGraphIndex.from_documents(
        documents,
        embed_kg=True,
        show_progress=True,
        use_async=False,  # IMPORTANT
        llm=llm,
        node_parser=splitter
    )
    return index


def create_workflow(index):
    nodes = list(index.docstore.docs.values())

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )

    bm25_query_engine = RetrieverQueryEngine.from_args(
        retriever=bm25_retriever
    )

    graph_query_engine = index.as_query_engine(
        sub_retrievers=[
            LLMSynonymRetriever(
                index.property_graph_store,
                llm=Settings.llm,
                include_text=True
            ),
            VectorContextRetriever(
                index.property_graph_store,
                embed_model=Settings.embed_model
            )
        ]
    )

    graph_tool = QueryEngineTool(
        query_engine=graph_query_engine,
        metadata=ToolMetadata(
            name="graph_search",
            description="Best for relationships and deep context"
        )
    )

    bm25_tool = QueryEngineTool(
        query_engine=bm25_query_engine,
        metadata=ToolMetadata(
            name="keyword_search",
            description="Best for exact keyword matches"
        )
    )

    researcher = FunctionAgent(
        name="Researcher",
        description="Retrieves relevant document passages and extracts key facts",
        system_prompt=(
            "You are a research assistant. Your job is to find relevant information from the uploaded documents.\n\n"
            "Tools:\n"
            "- graph_search: For relationships, concepts, and semantic search\n"
            "- keyword_search: For exact keywords and specific terms\n\n"
            "Steps:\n"
            "1. Analyze the user's question\n"
            "2. Use BOTH tools (graph_search AND keyword_search) to find relevant passages\n"
            "3. Extract the key facts with their source page numbers\n"
            "4. Pass your findings to Writer using a clear format:\n\n"
            "--- HANDOFF TO WRITER ---\n"
            "User Question: [the original question]\n\n"
            "Retrieved Facts:\n"
            "1. [fact with page ref]\n"
            "2. [fact with page ref]\n"
            "etc.\n"
            "--- END HANDOFF ---\n\n"
            "IMPORTANT: If search results are empty or irrelevant, write:\n"
            "--- HANDOFF TO WRITER ---\n"
            "User Question: [question]\n\n"
            "Retrieved Facts: None found\n"
            "--- END HANDOFF ---"
        ),
        tools=[graph_tool, bm25_tool],
        can_handoff_to=["Writer"]
    )

    writer = FunctionAgent(
        name="Writer",
        description="Generates final answer from document facts",
        system_prompt=(
            "You are a document answer writer. Create a clear answer based on research findings.\n\n"
            "Input: You will receive a handoff from Researcher with retrieved facts or \"None found\"\n\n"
            "Rules:\n"
            "1. If facts are provided, answer the question using ONLY those facts\n"
            "2. If \"None found\" or no relevant facts, respond: 'I could not find relevant information in the uploaded documents to answer this question.'\n"
            "3. Use exact wording from documents when possible\n"
            "4. Never add information not in the retrieved facts\n\n"
            "Output: Just write the answer directly without any prefixes, labels, or citations."
        ),
        tools=[],
        can_handoff_to=["Researcher"]
    )

    workflow = AgentWorkflow(
        agents=[researcher, writer],
        root_agent="Researcher"
    )

    return workflow


def save_index(index, filename: str):
    """Save index to disk"""
    save_path = os.path.join(INDEX_STORE_DIR, filename)
    index.storage_context.persist(persist_dir=save_path)


def load_index(filename: str):
    """Load index from disk"""
    from llama_index.core import load_index_from_storage, StorageContext
    save_path = os.path.join(INDEX_STORE_DIR, filename)
    if not os.path.exists(save_path):
        return None
    storage_context = StorageContext.from_defaults(persist_dir=save_path)
    return load_index_from_storage(storage_context)


def list_saved_indexes():
    """List all saved indexes"""
    if not os.path.exists(INDEX_STORE_DIR):
        return []
    return [d for d in os.listdir(INDEX_STORE_DIR) 
            if os.path.isdir(os.path.join(INDEX_STORE_DIR, d))]


def delete_index(filename: str):
    """Delete index from disk"""
    import shutil
    save_path = os.path.join(INDEX_STORE_DIR, filename)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

# ---------- UI ----------

def main():
    st.set_page_config(page_title="PDF Query Agent", page_icon="📄")
    st.title("📄 PDF Query Agent")

    if "workflow" not in st.session_state:
        st.session_state.workflow = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "current_name" not in st.session_state:
        st.session_state.current_name = None

    # Sidebar for saved embeddings
    st.sidebar.header("Saved Embeddings")
    saved = list_saved_indexes()
    
    if saved:
        st.sidebar.write("Available:")
        for idx_name in saved:
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                if st.button(f"Load {idx_name}", key=f"load_{idx_name}"):
                    try:
                        index = load_index(idx_name)
                        if index:
                            st.session_state.index = index
                            st.session_state.workflow = create_workflow(index)
                            st.session_state.current_name = idx_name
                            st.success(f"Loaded {idx_name}")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error loading: {str(e)}")
            with col2:
                if st.button("🗑️", key=f"del_{idx_name}"):
                    delete_index(idx_name)
                    st.rerun()
    else:
        st.sidebar.info("No saved embeddings")

    st.sidebar.divider()

    # Upload new PDFs
    st.sidebar.subheader("Upload New PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        file_name = st.sidebar.text_input("Name for this embedding", 
                                          value=uploaded_files[0].name.replace(".pdf", ""))
        
        if st.sidebar.button("Create Embedding", type="primary"):
            with st.spinner("Processing..."):
                try:
                    file_paths = [process_pdf(f) for f in uploaded_files]
                    index = build_index(file_paths)
                    save_index(index, file_name)
                    st.session_state.index = index
                    st.session_state.workflow = create_workflow(index)
                    st.session_state.current_name = file_name
                    st.success(f"Created embedding: {file_name}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.current_name:
        st.success(f"Active: {st.session_state.current_name}")

    st.divider()

    query = st.text_input("Ask a question:")

    if st.button("Submit Query") and query:
        workflow = st.session_state.workflow
        index = getattr(st.session_state, "index", None)
        if index is None:
            st.warning("Upload PDFs first")
        elif workflow is None:
            st.warning("Workflow not initialized. Please load or create embeddings first.")
        else:
            with st.spinner("Searching..."):
                try:
                    response_text, evidence_nodes = build_rfp_answer_with_workflow(workflow, index, query)

                    if response_text:
                        st.write(response_text)
                    else:
                        st.warning("The workflow completed but returned an empty response.")

                    if evidence_nodes:
                        with st.expander("Retrieved Evidence"):
                            for item in evidence_nodes:
                                page = item.node.metadata.get("page_label", "?")
                                st.markdown(f"**Page {page}**")
                                st.write(format_node_excerpt(item.node.text, limit=1200))

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
