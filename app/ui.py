import streamlit as st
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd

# Core Base Imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingest import run_ingestion

st.set_page_config(page_title="Selective Local RAG Engine", page_icon="🧠", layout="wide")

DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"
UPLOAD_BASE_DIR = "./upload"

def get_compact_ingest_metrics():
    """Queries pgvector directly to extract aggregated file types using standard regex replacements."""
    try:
        engine = create_engine(DB_URL)
        query = text("""
            WITH extracted_data AS (
                SELECT 
                    cmetadata->>'source' as file_path,
                    CASE 
                        WHEN cmetadata->>'source' LIKE '%%.%%' 
                        THEN '.' || LOWER(REGEXP_REPLACE(cmetadata->>'source', '^.*\\.', ''))
                        ELSE '.no-ext'
                    END as extension
                FROM langchain_pg_embedding
            )
            SELECT 
                extension as file_ext,
                COUNT(DISTINCT file_path) as unique_files,
                COUNT(*) as cumulative_chunks
            FROM extracted_data
            GROUP BY file_ext
            ORDER BY cumulative_chunks DESC;
        """)
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        if not result:
            return None
        data = []
        for row in result:
            data.append({
                "🧩 Extension": str(row[0]),
                "📁 Files Count": int(row[1]),
                "✂️ Total Chunks": int(row[2])
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error querying compact metrics: {e}")
        return None

def get_all_distinct_sources():
    """Fetches a clean list of all unique file paths currently indexed in the vector store."""
    try:
        engine = create_engine(DB_URL)
        query = text("SELECT DISTINCT cmetadata->>'source' FROM langchain_pg_embedding ORDER BY 1;")
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return [row[0] for row in result if row[0]]
    except Exception as e:
        print(f"Error fetching unique sources: {e}")
        return []

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def get_vector_store():
    """Cache-managed global access to the persistent PGVector index."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    return PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=DB_URL)

@st.cache_resource
def load_llm_and_prompts():
    """Initializes LLM engine and prompt components once for high performance."""
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, base_url="http://ollama:11434", timeout=300.0, num_ctx=2048)
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it exactly as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    system_prompt = (
        "You are an expert technical analysis assistant. Answer the question using the provided context fragments below. "
        "If you do not know the answer or if the context does not contain it, state clearly that the information is "
        "missing from local logs. Keep your answers concise, clear, and professional.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    response_generation_chain = qa_prompt | llm | StrOutputParser()
    
    return llm, contextualize_chain, response_generation_chain

# Initialize core variables
vector_store = get_vector_store()
llm, contextualize_chain, response_generation_chain = load_llm_and_prompts()

# Sidebar Document Upload, Ingest Dashboard, and Context Targeting Controls
with st.sidebar:
    st.header("🧹 Session Operations")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")

    # ─── TARGETED CONTEXT FILTER SELECTION SWITCH ───
    st.header("🎯 Context Targeting Scope")
    all_available_paths = get_all_distinct_sources()
    
    # Human-friendly dictionary labels tracking actual systemic string directories
    path_display_map = {p: f"📂 {os.path.dirname(p)} ({os.path.basename(p)})" if "sub" in p else f"📄 {os.path.basename(p)}" for p in all_available_paths}
    
    selected_display_names = st.multiselect(
        "Select target files/folders to search against (Leave blank to search everything):",
        options=list(path_display_map.values()),
        default=[]
    )
    
    # Reverse map chosen items back to original raw file path arrays
    selected_raw_paths = [path for path, label in path_display_map.items() if label in selected_display_names]
    # ────────────────────────────────────────────────

    st.markdown("---")
    st.header("📥 Ingest Subsystem")
    uploaded_files = st.file_uploader("Upload files to context database", type=["txt", "pdf"], accept_multiple_files=True)
    
    if st.button("🚀 Process & Ingest Uploads"):
        if uploaded_files:
            date_folder = datetime.now().strftime("%Y%m%d")
            target_dir = os.path.join(UPLOAD_BASE_DIR, date_folder)
            os.makedirs(target_dir, exist_ok=True)
            
            for f in uploaded_files:
                save_path = os.path.join(target_dir, f.name)
                with open(save_path, "wb") as buffer:
                    buffer.write(f.getbuffer())
                with st.spinner(f"Indexing {f.name}..."):
                    run_ingestion(single_file_path=save_path)
            st.success("✅ Selected files uploaded and vector indexed!")
            st.rerun()
            
    st.markdown("---")
    st.subheader("📊 Ingest Summary")
    df_metrics = get_compact_ingest_metrics()
    if df_metrics is not None:
        st.table(df_metrics)
        total_unique_files = df_metrics["📁 Files Count"].sum()
        total_global_chunks = df_metrics["✂️ Total Chunks"].sum()
        st.caption(f"📁 Global Files Tracked: {total_unique_files}")
        st.caption(f"✂️ Total Database Chunks: {total_global_chunks}")
    else:
        st.info("No documents indexed inside pgvector database yet.")

# Main Application Conversational Interface Layout
st.title("🧠 Conversational Local RAG Engine")
st.caption("Secure local inference engine featuring scoped metadata file filtering pipelines.")

# Handle current scope alert banner placements
if selected_raw_paths:
    st.info(f"🎯 **Target Active:** Search limited to **{len(selected_raw_paths)}** targeted selection constraint rules.")
else:
    st.success("🌐 **Global Active:** Search scanning all available documents in the vector database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── CRITICAL FIX: ALL EXECUTION LINES MUST SIT INSIDE THIS INDENT BLOCK ───
if user_query := st.chat_input("Ask a question about your files..."):
    # 1. Add user query to state and display it
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Reconstruct history objects cleanly
    chat_history_objects = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history_objects.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history_objects.append(AIMessage(content=m["content"]))

    # 3. Generate assistant answer response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        with st.spinner("🤖 Analyzing prompt context vectors..."):
            # Contextualize/rewrite question if history exists
            if chat_history_objects:
                standalone_question = contextualize_chain.invoke({
                    "input": user_query,
                    "chat_history": chat_history_objects
                })
            else:
                standalone_question = user_query

            # Build metadata filter block for pgvector targeting
            search_kwargs = {"k": 3}
            if selected_raw_paths:
                # Injects LangChain's strict dictionary filtering constraints mapping directly into SQL clauses
                search_kwargs["filter"] = {"source": {"$in": selected_raw_paths}}
                
            # Query the vector index
            retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
            retrieved_docs = retriever.invoke(standalone_question)
            
            # Synthesize final answer string response
            full_res = response_generation_chain.invoke({
                "context": format_docs(retrieved_docs),
                "chat_history": chat_history_objects,
                "input": standalone_question
            })
            
        placeholder.markdown(full_res)
        
        # Display references
        if retrieved_docs:
            with st.expander("📋 Verified Sources Referenced"):
                seen = set()
                for idx, doc in enumerate(retrieved_docs, start=1):
                    fname = os.path.basename(doc.metadata.get("source", "Unknown"))
                    pnum = doc.metadata.get("page", None)
                    loc = f"Page {pnum + 1}" if pnum is not None else "Text Line"
                    if f"{fname}|{loc}" not in seen:
                        st.markdown(f"**[{idx}] File:** `{fname}` — *{loc}*")
                        st.caption(f'*"...{doc.page_content.strip()[:140]}..."*')
                        seen.add(f"{fname}|{loc}")
                        
    # ✅ INDENTED PRECISELY: This line now sits inside the if block!
    # It will ONLY run when a new message is actively being generated.
    st.session_state.messages.append({"role": "assistant", "content": full_res})
