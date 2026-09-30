import streamlit as st
import os
from datetime import datetime
from sqlalchemy import create_engine, text 
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate, message
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from ingest import run_ingestion, get_relative_source_path

# 1. Page Configuration Setup
st.set_page_config(page_title="Local Conversational RAG", page_icon="🧠", layout="wide")
#st.title("🧠 Local RAG Enterprise Engine")
#st.caption("Powered by LangChain, PostgreSQL pgvector, and Ollama (qwen2.5:1.5b)")

DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"
UPLOAD_BASE_DIR = "./upload"

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 2. Cache heavy AI models in resource memory to prevent reload lag on every query button click
@st.cache_resource
def load_rag_system():
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    # Using ultra-lightweight qwen 1.5b model optimized for 8GB RAM laptop performance
    llm = ChatOllama(
        model="qwen2.5:1.5b", 
        temperature=0.1, 
        base_url="http://ollama:11434",
        timeout=300.0,
        num_ctx=2048
    )
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DB_URL,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})


    
    rag_prompt_template = """Use the provided context fragments to answer the user question.

Context:
{context}

Question: 
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(rag_prompt_template)
    llm_chain = prompt | llm | StrOutputParser()
    
    return retriever, llm_chain

# Initialize RAG background layers
try:
    retriever, llm_chain = load_rag_system()
except Exception as e:
    st.error(f"Could not connect to Docker services: {e}")
    st.info("💡 Ensure your Docker containers are active (`docker-compose up -d`) and the models are fully downloaded.")
    st.stop()

def get_compact_ingest_metrics():
    """Queries pgvector directly to extract aggregated file types and chunk sizes."""
    import pandas as pd
    try:
        engine = create_engine(DB_URL)
        # Advanced SQL regex to isolate extension tags out of LangChain source paths
        query = text("""
            WITH extracted_data AS (
                SELECT 
                    cmetadata->>'source' as file_path,
                    LOWER(SUBSTRING(cmetadata->>'source' FROM '\.[^.]+$')) as extension
                FROM langchain_pg_embedding
            )
            SELECT 
                COALESCE(extension, '.unknown') as file_ext,
                COUNT(DISTINCT file_path) as unique_files,
                COUNT(*) as cumulative_chunks
            FROM extracted_data
            GROUP BY extension
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

def get_indexed_files_dataframe():
    """Queries pgvector directly to extract a clean list of files and chunk counts."""
    import pandas as pd
    try:
        engine = create_engine(DB_URL)
        # Query to parse the file paths out of LangChain's metadata JSONB structure
        query = text("""
            SELECT 
                cmetadata->>'source' as file_path, 
                COUNT(*) as total_chunks
            FROM langchain_pg_embedding
            GROUP BY cmetadata->>'source'
            ORDER BY total_chunks DESC;
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
            
        if not result:
            return None
            
        # Clean up absolute system paths into readable file names for the UI table
        data = []
        for row in result:
            path = row[0] if row[0] else "Unknown Source"
            data.append({
                "📄 File Name": os.path.basename(path),
                "✂️ Total Chunks": row[1]
            })
            
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error querying dashboard metrics: {e}")
        return None
    
# Sidebar Document Upload System
with st.sidebar:
    st.header("📂 Document Management")
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
                
                # Ingest immediately on file write
                with st.spinner(f"Indexing {f.name}..."):
                    run_ingestion(single_file_path=save_path)
            st.success("✅ Selected files uploaded and vector indexed!")
        else:
            st.warning("Please choose files first.")
    
    # --- ADD THE INGEST STATUS DASHBOARD HERE ---
    st.markdown("---")
    st.subheader("📊 Ingest Status Dashboard")

    df_metrics = get_compact_ingest_metrics()
    if df_metrics is not None:
        # Use clean static rendering to completely avoid JavaScript dynamic import errors
        st.table(df_metrics)
        
        # Calculate totals dynamically out of the matrix dataframe
        total_unique_files = df_metrics["📁 Files Count"].sum()
        total_global_chunks = df_metrics["✂️ Total Chunks"].sum()
        
        st.caption(f"📁 Global Files Tracked: {total_unique_files}")
        st.caption(f"✂️ Total Database Chunks: {total_global_chunks}")
    else:
        st.info("No documents indexed inside pgvector database yet.")
    
    
    #df_files = get_indexed_files_dataframe()
    #if df_files is not None:
    #    # Render a native, clean, scrollable Streamlit Data Table widget
    #    st.dataframe(
    #        df_files, 
    #        use_container_width=True, 
    #        hide_index=True
    #    )
    #    st.caption(f"Total Unique Files: {len(df_files)}")
    #else:
    #    st.info("No documents indexed inside pgvector database yet.")
    
# Chat Interface Logic
st.title("🧠 Local RAG Enterprise Engine")
st.caption("Powered by LangChain, PostgreSQL pgvector, and Ollama (qwen2.5:1.5b)\nChat with local Windows folders or runtime files completely offline.")

# 3. Maintain Chat History Session Arrays (Loop-back continuity query loop)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat elements cleanly
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle Incoming Query Inputs
if user_query := st.chat_input("Ask a question about your folder documents..."):
    # Append user question to visible container
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Execute vector retrieval processing parameters
    with st.spinner("Searching pgvector clusters and analyzing context documents..."):
        retrieved_docs = retriever.invoke(user_query)
        context_content = format_docs(retrieved_docs)

    # Stream final response tokens directly into user layout box
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Execute the optimized decoupled stream pipeline
        for chunk in llm_chain.stream({"context": context_content, "question": user_query}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        
        # 5. Display Precise Sources inside an expanding block
        with st.expander("📋 Verified Sources Referenced"):
            seen_sources = set()
            for idx, doc in enumerate(retrieved_docs, start=1):
                metadata = doc.metadata
                source_path = metadata.get("source", "Unknown Document")
                file_name = os.path.basename(source_path)
                
                page_number = metadata.get("page", None)
                page_str = f"Page {page_number + 1}" if page_number is not None else "Text Snippet"
                
                source_identifier = f"{file_name} | {page_str}"
                
                if source_identifier not in seen_sources:
                    st.markdown(f"**[{idx}] File:** `{file_name}` — *{page_str}*")
                    st.caption(f'*"...{doc.page_content.strip()[:140]}..."*')
                    st.markdown("---")
                    seen_sources.add(source_identifier)
                    
    # Log the complete assistant answer to the session state loop back historical memory
    st.session_state.messages.append({"role": "assistant", "content": full_response})
