import os
import glob
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from langchain_core.documents import Document

# Configuration Constants
# Verify this line is exact in app/ingest.py
DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"

# NEW Configuration toggles mapped from environment parameters
STORAGE_TYPE = os.getenv("STORAGE_TYPE", "LOCAL").upper() # Options: LOCAL, S3, AZURE, GCS
DATA_DIR = "./data"

def prepare_database():
    """Ensure the pgvector extension is explicitly active before LangChain connects."""
    print("🛠️ Checking database and activating pgvector extension...")
    engine = create_engine(DB_URL)
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    engine.dispose()
    print("✅ Database extension verified.")

def init_vector_store(embeddings):
    """Initialise or connect to the pgvector database collection."""
    from sqlalchemy.exc import IntegrityError
    try:
        return PGVector(
            embeddings=embeddings,
            collection_name=COLLECTION_NAME,
            connection=DB_URL,
            use_jsonb=True,
        )
    except IntegrityError as e:
        # Catch and handle multi-container startup race conditions cleanly
        if "already exists" in str(e) or "UniqueViolation" in str(e):
            print("🔄 Database Engine Notice: Detected parallel table creation trace. Re-binding to existing collection references safely...")
            # Re-attempt connection bind now that the table asset is guaranteed to exist on disk
            return PGVector(
                embeddings=embeddings,
                collection_name=COLLECTION_NAME,
                connection=DB_URL,
                use_jsonb=True,
            )
        raise e

def read_text_file(filepath):
    """Robust loop to handle problematic Windows/Linux text encodings."""
    encodings = ["utf-8", "utf-16", "utf-16le", "utf-16be", "cp1252","latin-1"]
    for encoding in encodings:
        try:
            with open(filepath, "r", encoding=encoding, errors="ignore") as f:
                content = f.read()

            # CRITICAL: Clean raw binary null bytes from memory string buffer 
            # without stripping your hex memory addresses (0x0000...)
            content = content.replace("\x00", "")
            return content
        except UnicodeDecodeError:
            continue
    return None

# --- NEW: Helper to extract clean relative path from any baseline workspace directory ---
def get_relative_source_path(filepath):
    """Converts absolute or prefixed paths into clean relative execution paths."""
    # Removes absolute container prefixes like /workspace/data/ or /workspace/upload/
    # turning paths into clean roots like 'test_log.txt' or 'sub/test_log.txt'
    path_str = os.path.normpath(filepath)
    for prefix in ["/workspace/data/", "/workspace/upload/", "./data/", "./upload/", "data/", "upload/"]:
        if path_str.startswith(os.path.normpath(prefix)):
            return os.path.relpath(path_str, os.path.normpath(prefix))
    return os.path.basename(path_str)

def get_embedded_metadata_registry():
    """Queries pgvector to extract file paths along with their stored sizes and modification times."""
    engine = create_engine(DB_URL)
    query = text("""
        SELECT DISTINCT 
            cmetadata->>'source' as src, 
            cmetadata->>'file_size' as sz, 
            cmetadata->>'file_mtime' as mt 
        FROM langchain_pg_embedding;
    """)
    registry = {}
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        for row in result:
            if row[0]:
                rel_path = get_relative_source_path(row[0])
                registry[rel_path] = {
                    "file_size": int(row[1]) if row[1] is not None else 0,
                    "file_mtime": float(row[2]) if row[2] is not None else 0.0
                }
        return registry
    except Exception as e:
        print(f"⚠️ Metadata registry fetch warning: {e}")
        return {}
    finally:
        engine.dispose()

def get_already_embedded_files():
    """Queries pgvector metadata jsonb to extract already parsed source paths."""
    engine = create_engine(DB_URL)
    query = text("SELECT DISTINCT cmetadata->>'source' FROM langchain_pg_embedding;")
    try:
        with engine.connect() as conn:
            result = conn.execute(query).fetchall()
        return {get_relative_source_path(row[0]) for row in result if row[0]}
    except Exception as e:
        print(f"⚠️ Metadata fetch warning: {e}")
        return set()
    finally:
        engine.dispose()

def delete_vectors_for_file(filepath):
    """Purges all historical vector chunks matching a specific source path filename."""
    engine = create_engine(DB_URL)
    query = text("DELETE FROM langchain_pg_embedding WHERE cmetadata->>'source' LIKE :source_suffix;")
    try:
        filename = os.path.basename(filepath)
        with engine.begin() as conn:
            conn.execute(query, {"source_suffix": f"%{filename}"})
        print(f"🗑️ Purged old vectors for: {filename}")
    except Exception as e:
        print(f"❌ Error deleting vectors for {filepath}: {e}")
    finally:
        engine.dispose()

def process_and_embed_file(filepath, vector_store, force_update=False, metadata_registry=None):
    """Processes a single file and adds it immediately to the vector DB."""
    """Processes a single file. Automatically clears old states if updated."""
    # Convert incoming path to absolute format
    #abs_filepath = os.path.abspath(filepath)
    rel_path = get_relative_source_path(filepath)
    filename = os.path.basename(filepath)

    if metadata_registry is None:
        metadata_registry = get_embedded_metadata_registry()

    # Get live metrics from the laptop disk file
    live_size = 0
    live_mtime = 0.0
    if os.path.exists(filepath):
        live_size = os.path.getsize(filepath)
        live_mtime = os.path.getmtime(filepath)

    # Check if file has an existing state registry entry
    if rel_path in metadata_registry:
        stored = metadata_registry[rel_path]
        
        # FIX: Trigger update if flagged by watcher OR if disk metrics don't match database records
        if force_update or stored["file_size"] != live_size or abs(stored["file_mtime"] - live_mtime) > 1.0:
            print(f"🔄 Outdated file or modification detected. Re-indexing: {rel_path}")
            delete_vectors_for_file(filepath)
        else:
            print(f"⏭️ Skipping (Unchanged & Embedded): {rel_path}")
            return

    """
    # --- UPDATED CONTEXT OVERWRITE LOGIC ---
    # If the file path exists but the file watcher flags a modification change event:
    if rel_path in existing_files and force_update:
        print(f"🔄 Content modification detected. Re-indexing: {rel_path}")
        delete_vectors_for_file(filepath)
    # If it exists on a standard baseline container startup boot, skip it safely:
    elif rel_path in existing_files and not force_update:
        print(f"⏭️ Skipping (Already Embedded): {rel_path}")
        return
    # ---------------------------------------
    """
    """
    if existing_files is None:
        existing_files = get_already_embedded_files()

    if filename in existing_files and not force_update:
        print(f"⏭️ Skipping (Already Embedded): {rel_path}")
        return

    if rel_path in existing_files and force_update:
        delete_vectors_for_file(filepath)
    """
    
    ext = os.path.splitext(filepath)[1].lower()
    docs = []

    if ext == ".txt":
        content = read_text_file(filepath)
        if content:
            docs.append(Document(page_content=content, metadata={"source": filepath}))
    elif ext == ".pdf":
        try:
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
        except Exception as e:
            print(f"❌ Error reading PDF {filepath}: {e}")

    if not docs:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)

    # Clean NUL bytes
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
        # Keep the full path inside metadata so citations and lookups work perfectly
        chunk.metadata["source"] = filepath
        # CRITICAL PERSISTENCE FIX: Save live file fingerprints into DB fields
        chunk.metadata["file_size"] = live_size
        chunk.metadata["file_mtime"] = live_mtime

    if chunks:
        vector_store.add_documents(chunks)
        print(f"✅ Ingested: {rel_path} ({len(chunks)} chunks)")

def ingest_directory(target_dir, vector_store, force_update=False, metadata_registry=None):
    """Scans a full directory path recursively."""
    print(f"📂 Scanning target directory: '{target_dir}'...")
    if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    
    txt_files = glob.glob(os.path.join(target_dir, "**/*.txt"), recursive=True)
    pdf_files = glob.glob(os.path.join(target_dir, "**/*.pdf"), recursive=True)

    for filepath in txt_files + pdf_files:
        process_and_embed_file(filepath, vector_store, force_update=force_update, metadata_registry=metadata_registry)

def run_ingestion(single_file_path=None, target_directory=None, force_update=False):
    try:
        prepare_database()
    except Exception as e:
        print(f"❌ Database error: {e}")
        return

    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")
    vector_store = init_vector_store(embeddings)
    #existing_files = get_already_embedded_files()
    metadata_registry=get_embedded_metadata_registry()

    if single_file_path:
        process_and_embed_file(single_file_path, vector_store, force_update=force_update, metadata_registry=metadata_registry)
    elif target_directory:
        ingest_directory(target_directory, vector_store, force_update=force_update, metadata_registry=metadata_registry)
'''
def fetch_documents():
    """Loads documents dynamically depending on configured source path targets."""
    docs = []
    
    if STORAGE_TYPE == "LOCAL":
        print(f"📂 Scanning local container directory path: '{DATA_DIR}' for documents...")
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        
        # 1. Process structured PDF documents normally
        pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
        try:
            docs.extend(pdf_loader.load())
        except Exception as e:
            print(f"⚠️ Warning loading PDFs: {e}")

        # 2. Robust Encoding Rotation Loop to handle problematic text files securely
        import glob
        from langchain_core.documents import Document
        
        txt_files = glob.glob(os.path.join(DATA_DIR, "**/*.txt"), recursive=True)
        for filepath in txt_files:
            text_content = None
            # Explicit list of encodings to check (handles Linux, Windows UTF-16, and legacy files)
            encodings_to_try = ["utf-8", "utf-16", "utf-16le", "utf-16be", "cp1252"]
            
            for encoding in encodings_to_try:
                try:
                    with open(filepath, "r", encoding=encoding) as f:
                        text_content = f.read()
                    print(f"✅ Successfully loaded '{filepath}' using encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"⚠️ Error opening file '{filepath}': {e}")
                    break
        
            if text_content is not None:
                # Manually wrap the text into a standard LangChain Document object structure
                docs.append(Document(page_content=text_content, metadata={"source": filepath}))
            else:
                print(f"❌ Failed to load '{filepath}' after trying all available encodings.")


    elif STORAGE_TYPE == "S3":
        # Connect to online cloud storage systems using LangChain S3 wrapper hooks
        from langchain_community.document_loaders import S3DirectoryLoader
        bucket_name = os.getenv("S3_BUCKET_NAME")
        prefix = os.getenv("S3_PREFIX", "")
        print(f"☁️ Synchronizing and fetching records from AWS S3 Bucket: '{bucket_name}/{prefix}'...")
        s3_loader = S3DirectoryLoader(bucket_name=bucket_name, prefix=prefix)
        docs.extend(s3_loader.load())

    elif STORAGE_TYPE == "AZURE":
        from langchain_community.document_loaders import AzureBlobStorageDirectoryLoader
        conn_str = os.getenv("AZURE_CONN_STR")
        container = os.getenv("AZURE_CONTAINER")
        print(f"☁️ Fetching file blobs from Azure Storage Container: '{container}'...")
        azure_loader = AzureBlobStorageDirectoryLoader(conn_str=conn_str, container=container)
        docs.extend(azure_loader.load())

    return docs

def ingest_documents():
     # Preparing databse
    try:
        prepare_database()
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")
        return
    
    print("🔄 Initialising open-source embedding model via Ollama...")
    # Using a high-performing open-source text embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://ollama:11434"
    )

    docs=fetch_documents()
    if not docs:
        print(f"⚠️ No active source documents identified inside target storage pool location ({STORAGE_TYPE}).")
        return 

    # Logic moved to fetch_documents()
    
    # 1. Load documents from directory
    print(f"📂 Scanning directory: '{DATA_DIR}' for documents...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created '{DATA_DIR}' folder. Please place files in it and rerun.")
        return

    # Dynamic loader supporting both flat text and PDFs
    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    docs.extend(txt_loader.load())
    docs.extend(pdf_loader.load())
    
    if not docs:
        print("⚠️ No matching documents (.txt or .pdf) found in data directory.")
        return
 

    print(f"📄 Loaded {len(docs)} files. Splitting into text chunks...")

    # 2. Split documents into chunk vectors
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    # --- ADD THIS CLEANING STEP TO STRIP NUL BYTES ---
    for chunk in chunks:
        chunk.page_content = chunk.page_content.replace("\x00", "")
    # -------------------------------------------------
    print(f"✂️ Created {len(chunks)} contextual text chunks.")

    # 3. Store text chunks inside Postgres Vector table
    print("📥 Syncing embeddings to PostgreSQL pgvector database...")
    vector_store = init_vector_store(embeddings)
    vector_store.add_documents(chunks)
    print("✅ Ingestion successfully completed!")
'''

if __name__ == "__main__":
    #ingest_documents()
    # Standard startup baseline scan directories
    run_ingestion(target_directory="./data", force_update=False)
    if os.path.exists("./upload"):
        run_ingestion(target_directory="./upload", force_update=False)
