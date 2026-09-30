import streamlit as st
import os
from datetime import datetime
from sqlalchemy import create_engine, text
import pandas as pd

# Core Base Imports (Available across both older and modern versions)
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from ingest import run_ingestion

st.set_page_config(page_title="Conversational RAG Engine", page_icon="🧠", layout="wide")

DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"
UPLOAD_BASE_DIR = "./upload"

def get_compact_ingest_metrics():
    """Queries pgvector directly to extract aggregated file types and chunk sizes."""
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
                "🧩 Extension": row[0],
                "📁 Files Count": row[1],
                "✂️ Total Chunks": row[2]
            })
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error querying compact metrics: {e}")
        return None

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def init_conversational_rag():
    """Initializes a context-aware RAG chain using raw LCEL pipe operators."""
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.1, base_url="http://ollama:11434", timeout=300.0, num_ctx=2048)
    
    vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=DB_URL)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # 1. Query Reformulation Layer
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
    
    # Contextualization chain: takes input + history -> outputs a clear standalone question string
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # 2. Main Q&A Synthesizer Layer
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

    # 3. Dynamic Orchestration Branch Function
    def contextualized_input(x):
        """If history exists, pass the query through the LLM rewrite step first."""
        if x.get("chat_history"):
            return contextualize_chain
        return RunnablePassthrough() | (lambda inputs: inputs["input"])

    # 4. Final Combined LCEL Processing Map Pipeline
    lcel_rag_pipeline = (
        RunnablePassthrough.assign(
            standalone_question=contextualized_input
        )
        | RunnablePassthrough.assign(
            context=lambda x: base_retriever.invoke(x["standalone_question"])
        )
        | RunnablePassthrough.assign(
            answer=lambda x: response_generation_chain.invoke({
                "context": format_docs(x["context"]),
                "chat_history": x["chat_history"],
                "input": x["standalone_question"]
            })
        )
    )
    
    return lcel_rag_pipeline

rag_pipeline = init_conversational_rag()

# Sidebar Document Upload & Dashboard System
with st.sidebar:
    st.header("📂 Document Management")

    # ─── Chat reset switch ───
    if st.button("🧹 Clear Chat History", use_container_width=True):
        # Wipe the persistent message array cache completely
        st.session_state.messages = []
        # Force a native layout re-run to present a clean workspace view instantly
        st.rerun()
    st.markdown("---")

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
            st.sidebar.markdown("---")
            st.rerun()
        else:
            st.warning("Please choose files first.")
            
    # Compact Ingest Status Summary Table
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
st.caption("Chat with local Windows directories securely and completely offline with full contextual memory recall.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your files..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Reconstruct history into explicit object states
    chat_history_objects = []
    for m in st.session_state.messages[:-1]:
        if m["role"] == "user":
            chat_history_objects.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history_objects.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
        # Invoke the custom version-agnostic pipeline mapping
        with st.spinner("🧠 Analyzing history and consulting local vector index..."):
            # Execute the unified LCEL pipeline mapping explicitly
            response = rag_pipeline.invoke({
                "input": user_query,
                "chat_history": chat_history_objects
            })
        
        full_res = response["answer"]
        placeholder.markdown(full_res)
        
        if "context" in response and response["context"]:
            with st.expander("📋 Verified Sources Referenced"):
                seen = set()
                for idx, doc in enumerate(response["context"], start=1):
                    fname = os.path.basename(doc.metadata.get("source", "Unknown"))
                    pnum = doc.metadata.get("page", None)
                    loc = f"Page {pnum + 1}" if pnum is not None else "Text Line"
                    if f"{fname}|{loc}" not in seen:
                        st.markdown(f"**[{idx}] File:** `{fname}` — *{loc}*")
                        st.caption(f'*"...{doc.page_content.strip()[:140]}..."*')
                        seen.add(f"{fname}|{loc}")
                        
    st.session_state.messages.append({"role": "assistant", "content": full_res})
