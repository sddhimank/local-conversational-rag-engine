import os
# ✅ THE HARDWARE ACCELERATION FIX: Force PyTorch to utilize all available CPU execution cores 
# and switch to highly optimized parallel OpenMP/MKL computing backends.
'''
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
'''

import glob
import io
from celery import Celery, group
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
#from langchain_huggingface import HuggingFaceEmbeddings

REDIS_URL = os.getenv("REDIS_URL") or "redis://localhost:6379/0"
DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"

# Initialize the Celery asynchronous task instance configuration
celery_app = Celery("rag_tasks", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    worker_prefetch_multiplier=1,  # Pre-fetches tasks to keep CPU cores 100% saturated
    task_acks_late=True
)

# This serves both our manual metadata SQL operations and the main LangChain vector store!
print("🐘 Global Init: Creating dedicated, thread-safe SQLAlchemy engine pooling architecture...")
_SHARED_DB_ENGINE = create_engine(
    DB_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800
)

# ─── THE SINGLETON OPTIMIZATION FIX ───
# Instantiating this globally ensures that threads share a single model weight array in RAM
#print("🧠 Global Init: Loading thread-safe HuggingFace embedding engine into baseline memory layer...")
'''
_GLOBAL_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="/workspace/app/model/bge-large-en-v1.5",
    encode_kwargs={'normalize_embeddings': True}
)
'''
print("🧠 Global Init: Registering high-speed Ollama BGE-Large embedding connector...")
_GLOBAL_EMBEDDINGS = OllamaEmbeddings(
    model="bge-large:latest", 
    base_url="http://ollama:11434"
)

# Initialize the PGVector instance globally EXACTLY ONCE.
# This ensures shared memory threads never attempt to redefine SQLAlchemy tables.
print("🐘 Global Init: Pre-binding thread-safe persistent PGVector store reference layer...")
_GLOBAL_VECTOR_STORE = PGVector(
    embeddings=_GLOBAL_EMBEDDINGS,
    collection_name=COLLECTION_NAME,
    connection=_SHARED_DB_ENGINE, # Pass the pre-warmed shared engine reference directly!
    use_jsonb=True
)

def get_relative_source_path(filepath):
    path_str = os.path.normpath(filepath)
    for prefix in ["/workspace/data/", "/workspace/upload/", "./data/", "./upload/", "data/", "upload/"]:
        if path_str.startswith(os.path.normpath(prefix)):
            return os.path.relpath(path_str, os.path.normpath(prefix))
    return os.path.basename(path_str)

def get_embedded_metadata_registry():
    #engine = create_engine(DB_URL,pool_size=5, max_overflow=5, pool_pre_ping=True)
    query = text("SELECT DISTINCT cmetadata->>'source' as src, cmetadata->>'file_size' as sz, cmetadata->>'file_mtime' as mt FROM langchain_pg_embedding;")
    registry = {}
    try:
        with _SHARED_DB_ENGINE.connect() as conn:
            result = conn.execute(query).fetchall()
        for row in result:
            if row:
                rel_path = get_relative_source_path(row[0])
                registry[rel_path] = {
                    "file_size": int(row[1]) if row[1] is not None else 0,
                    "file_mtime": float(row[2]) if row[2] is not None else 0.0
                }
        return registry
    except Exception as e:
        print(f"⚠️ Metadata registry fetch warning: {e}")
        return {}
    #finally:
    #    engine.dispose()

def delete_vectors_for_file(filepath):
    #engine = create_engine(DB_URL)
    query = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' LIKE :source_suffix;")
    try:
        filename = os.path.basename(filepath)
        with _SHARED_DB_ENGINE.begin() as conn:
            conn.execute(query, {"source_suffix": f"%{filename}"})
        print(f"🗑️ Background Worker: Purged old vectors for: {filename}")
    except Exception as e:
        print(f"❌ Error deleting vectors: {e}")
    #finally:
    #    engine.dispose()

@celery_app.task(name="app.tasks.async_process_file")
def async_process_file(filepath, force_update=False):
    """Asynchronous background worker execution task to ingest and embed a single file."""
    print(f"⚙️ Celery Background Task: Starting ingestion for {os.path.basename(filepath)}")
    
    live_size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    live_mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0
    rel_path = get_relative_source_path(filepath)
    
    metadata_registry = get_embedded_metadata_registry()
    if rel_path in metadata_registry:
        stored = metadata_registry[rel_path]
        if not force_update and stored["file_size"] == live_size and abs(stored["file_mtime"] - live_mtime) <= 1.0:
            print(f"⏭️ Background Worker: Skipping unchanged file {rel_path}")
            return f"Skipped {rel_path}"
        delete_vectors_for_file(filepath)

    ext = os.path.splitext(filepath)[1].lower()
    docs = []
    
    if ext == ".txt":
        encodings = ["utf-8", "utf-16", "utf-16le", "cp1252", "latin-1"]
        content = None
        for enc in encodings:
            try:
                with open(filepath, "r", encoding=enc, errors="ignore") as f:
                    content = f.read().replace("\x00", "")
                break
            except UnicodeDecodeError:
                continue
        if content:
            docs.append(Document(page_content=content, metadata={"source": filepath}))
            
    elif ext == ".pdf":
        try:
            with open(filepath,"rb") as f:
                file_bytes = f.read()

            # Create a unique tracking filename inside the container's local Linux RAM workspace
            temp_filename = f"/tmp/worker_{os.path.basename(filepath)}"
            
            # Write the raw bytes directly to container local /tmp storage
            with open(temp_filename, "wb") as f_temp:
                f_temp.write(file_bytes)
            
            #blob_stream=io.BytesIO(file_bytes)
            #loader = PyPDFLoader(blob_stream)

            # Feed the native local file path cleanly to PyPDFLoader
            loader = PyPDFLoader(temp_filename)
            docs.extend(loader.load())

            # Clean up the virtual temp asset instantly to preserve container space
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        except Exception as e:
            # Cleanup fallback in case an extraction error takes place
            if 'temp_filename' in locals() and os.path.exists(temp_filename):
                os.remove(temp_filename)
            return f"Error parsing PDF {filepath}: {str(e)}"

    if not docs:
        return f"No content collected for {filepath}"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
        chunk.metadata.update({
            "source": filepath,
            "file_size": live_size,
            "file_mtime": live_mtime
        })

    # ✅ THE PERFORMANCE FIX: Swap Ollama HTTP calls for local, fast HuggingFace in-memory vector calculations
    # BGE-Small-EN-v1.5 is an industry-tier, lightweight, ultra-fast embedding model
    # embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name="/workspace/app/model/bge-small-en-v1.5",
        encode_kwargs={'normalize_embeddings': True}
    )
    '''
    '''
    #vector_store = PGVector(embeddings=embeddings, collection_name=COLLECTION_NAME, connection=DB_URL)
    # ✅ FIX: Prevent background celery worker processes from crashing if the UI container is initializing tables simultaneously
    from sqlalchemy.exc import IntegrityError
    try:
        vector_store = PGVector(
            embeddings=_GLOBAL_EMBEDDINGS, 
            collection_name=COLLECTION_NAME, 
            connection=DB_URL,
            engine_args={"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}
        )
    except IntegrityError:
        vector_store = PGVector(
            embeddings=_GLOBAL_EMBEDDINGS, 
            collection_name=COLLECTION_NAME, 
            connection=DB_URL,
            engine_args={"pool_size": 5, "max_overflow": 5, "pool_pre_ping": True}
        )
        
    vector_store.add_documents(chunks, batch_size=100)
    print(f"✅ Background Worker: Successfully embedded {rel_path} ({len(chunks)} chunks)")
    return f"Indexed {rel_path}"
    '''
    # ✅ FIX: Reuse the global vector store singleton instance instead of redefining it inline
    #with torch.inference_mode():
    #    _GLOBAL_VECTOR_STORE.add_documents(chunks, batch_size=100)

    # ✅ THE CONCURRENCY SPEED FIX: Extract text strings to run an optimized batch operation
    texts_to_embed = [chunk.page_content for chunk in chunks]
    metadatas_to_bind = [chunk.metadata for chunk in chunks]

    # Calculate all embeddings concurrently inside Ollama in a single network transmission pass
    print(f"⚡ Ingest Engine: Sending {len(texts_to_embed)} chunks to Ollama in an optimized parallel batch...")
    try:
        embeddings_arrays = _GLOBAL_EMBEDDINGS.embed_documents(texts_to_embed)
    except Exception as e:
        return f"Error computing batch embeddings via Ollama for {filepath}: {str(e)}"

    # Write the calculated high-density vector matrices directly to pgvector in stable batches
    _GLOBAL_VECTOR_STORE.add_embeddings(
        texts=texts_to_embed,
        embeddings=embeddings_arrays,
        metadatas=metadatas_to_bind,
        batch_size=100
    )

    print(f"✅ Background Worker: Successfully embedded {rel_path} ({len(chunks)} chunks)")
    return f"Indexed {rel_path}"

@celery_app.task(name="app.tasks.async_scan_directory")
def async_scan_directory(target_directory, force_update=False):
    """Asynchronous background task to scan an entire directory structure and enqueue items."""
    print(f"📂 Celery Scanning Directory: Triggered baseline scan for '{target_directory}'")
    if not os.path.exists(target_directory):
        return f"Directory {target_directory} does not exist."
        
    txt_files = glob.glob(os.path.join(target_directory, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(target_directory, "**/*.pdf"), recursive=True)
    all_files = txt_files + pdf_files

    if not all_files:
        return "No files found to process."
    '''
    for filepath in txt_files + pdf_files:
        # Chain separate sub-tasks onto the Celery task queue queue channel loop
        async_process_file.delay(filepath, force_update=force_update)
    '''
    # ✅ THE CRITICAL PERFORMANCE FIX: 
    # Group all file subtasks together and send them to Redis in a single, high-speed batch broadcast
    job_group = group(async_process_file.s(filepath, force_update) for filepath in all_files)
    job_group.apply_async()
        
    return f"🚀 Super-Batch: Successfully enqueued {len(txt_files) + len(all_files)} files into Redis concurrently!"
