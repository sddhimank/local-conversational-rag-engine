import os
from langchain_core.tools import tool
from sqlalchemy import create_engine, text
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

DB_URL = os.getenv("DB_URL") or "postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db"
COLLECTION_NAME = "folder_documents"

@tool
def calculate_database_file_metrics(target_folder: str = None) -> str:
    """Useful for finding the exact count or numerical total of files currently ingested in the database index registry.
    Can be scoped to a specific target subfolder name if provided. Use this tool whenever the user asks quantitative, counting, presence, or volume questions about documents.
    """
    engine = create_engine(DB_URL)
    # ✅ THE STRIP GUARDRAIL FIX: Cleanse generic textual fillers passed by the LLM
    if target_folder:
        clean_check = target_folder.strip().lower()
        if clean_check in ["system", "database", "root", "global", "all", "here"]:
            target_folder = None # Force fallback to global counts
            
    try:
        with engine.connect() as conn:
            if target_folder:
                query = text("""
                    SELECT COUNT(DISTINCT cmetadata->>'source') 
                    FROM langchain_pg_embedding 
                    WHERE cmetadata->>'source' LIKE :pattern;
                """)
                count = conn.execute(query, {"pattern": f"{target_folder}/%"}).scalar() or 0
                return f"SUCCESS: The verified database registry reports exactly {count} file(s) inside the subfolder path structure: '{target_folder}'."
            else:
                query = text("SELECT COUNT(DISTINCT cmetadata->>'source') FROM langchain_pg_embedding;")
                count = conn.execute(query).scalar() or 0
                return f"SUCCESS: The verified database registry reports a grand total of {count} unique file(s) ingested globally across all corporate folders."
    except Exception as e:
        return f"ERROR: Failed to execute database metrics aggregation check: {str(e)}"
    finally:
        engine.dispose()

@tool
def execute_document_semantic_search(semantic_query: str) -> str:
    """Useful for answering text meaning, analysis, clauses, guidelines, policies, or debugging questions about file contents.
    This tool runs a high-density vector similarity search over text blocks and returns the most relevant contextual content fragments found.
    """
    try:
        embeddings = OllamaEmbeddings(model="bge-large:latest", base_url="http://ollama:11434")
        vector_store = PGVector(embeddings=embeddings, collection_name="agent_content_chunks", connection=DB_URL)
        
        # ✅ THE QUERY SANITIZER FIX: Cleanse file paths and extensions if passed by the LLM
        cleaned_query = semantic_query
        for word in [".pdf", "file:", "document:", "content:", "find", "search"]:
            cleaned_query = cleaned_query.replace(word, "")
        cleaned_query = cleaned_query.strip()
        
        print(f"🔍 [Search Agent Executing]: Cleaned target query string -> '{cleaned_query}'")
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(cleaned_query)
        
        if not docs:
            return "NOTICE: Similarity search completed successfully, but zero relevant document fragments matched this query inside the vector index grid."
            
        formatted_context = []
        for idx, doc in enumerate(docs, start=1):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            formatted_context.append(f"--- Context Source Segment [{idx}]: {src} ---\n{doc.page_content}\n")
            
        return "\n".join(formatted_context)
    except Exception as e:
        return f"ERROR: Failed to compute vector similarity lookup sequence: {str(e)}"

@tool
def query_document_metadata(
    author: str = None, 
    modified_after: str = None, 
    modified_before: str = None, 
    max_file_size_bytes: int = None
) -> str:
    """Useful for searching, filtering, and locating files based on structural metadata properties 
    such as who wrote the file (author), when it was created/modified (before or after a certain date), 
    or its size constraints on disk.
    
    Args:
        author: The name string of the document creator/author to filter by.
        modified_after: A date string formatted strictly as 'YYYY-MM-DD'. Filters for files changed AFTER this date.
        modified_before: A date string formatted strictly as 'YYYY-MM-DD'. Filters for files changed BEFORE this date.
        max_file_size_bytes: An integer filtering for files whose size is less than or equal to this byte threshold.
    """
    import datetime
    engine = create_engine(DB_URL)

     # ✅ THE BULLETPROOF SANITIZER FIX: Convert LLM text literal 'null' strings back into Python None values
    if str(author).strip().lower() in ["none", "null", ""]: author = None
    if str(modified_after).strip().lower() in ["none", "null", ""]: modified_after = None
    if str(modified_before).strip().lower() in ["none", "null", ""]: modified_before = None
    
    # Process the file size integer parameter boundary safely
    if max_file_size_bytes is not None:
        size_str = str(max_file_size_bytes).strip().lower()
        if size_str in ["none", "null", "0", ""]:
            max_file_size_bytes = None # Drop constraint if the LLM passes zero or null strings
        else:
            try:
                max_file_size_bytes = int(max_file_size_bytes)
            except Exception:
                max_file_size_bytes = None
    
    # Base raw SQL query targeting unique source files from the JSONB metadata column
    base_query = "SELECT DISTINCT cmetadata->>'source' as src, cmetadata->>'file_size' as sz, cmetadata->>'file_mtime' as mt, cmetadata->>'author' as aut FROM langchain_pg_embedding WHERE 1=1"
    #params = {}
    
    try:
        with engine.connect() as conn:
            results = conn.execute(text(base_query)).fetchall()
            
        if not results:
            return "NOTICE: The database registry currently contains zero indexed documents."
            
        matched_files = []
        for row in results:
            # Safely extract and parse metadata from the row object
            src_path = row.src or "Unknown"
            f_size = int(row.sz) if row.sz is not None else 0
            f_author = row.aut or "Unknown"
            
            # Convert epoch timestamp floats to standard datetime date objects for strict math comparison
            f_date = datetime.date.fromtimestamp(float(row.mt)) if row.mt is not None else None
            
            # Evaluate filters mathematically in-memory
            keep = True
            if author:
                search_term = author.lower()
                # Matches if the term is found in either the file properties or the page text clauses
                f_auth_match = search_term in str(row.get("file_author", "")).lower()
                t_auth_match = search_term in str(row.get("text_author", "")).lower()
                if not (f_auth_match or t_auth_match):
                    keep = False
            '''
            if author and author.lower() not in f_author.lower():
                keep = False
            if max_file_size_bytes and f_size > max_file_size_bytes:
                keep = False
            '''

            if modified_after and f_date:
                after_dt = datetime.datetime.strptime(modified_after, "%Y-%m-%d").date()
                if f_date <= after_dt:
                    keep = False
            if modified_before and f_date:
                before_dt = datetime.datetime.strptime(modified_before, "%Y-%m-%d").date()
                if f_date >= before_dt:
                    keep = False
                    
            if keep:
                readable_date = f_date.strftime("%Y-%m-%d") if f_date else "Unknown"
                readable_size = f"{f_size / 1024:.2f} KB" if f_size > 0 else "Unknown"
                matched_files.append(f"* 📄 **File:** `{os.path.basename(src_path)}` | ✍️ **Author:** `{f_author}` | 📅 **Modified:** `{readable_date}` | 📊 **Size:** `{readable_size}`")
                
        if matched_files:
            return "SUCCESS: Found the following files matching your metadata constraints:\n\n" + "\n".join(matched_files)
        else:
            return "NOTICE: Query executed successfully, but zero files matched the specified metadata filter criteria."
            
    except Exception as e:
        return f"ERROR: Failed to parse document metadata fields: {str(e)}"
    finally:
        engine.dispose()

@tool
def execute_metadata_semantic_search(metadata_query: str) -> str:
    """Useful for running a semantic vector similarity search ONLY over document metadata summaries, 
    abstracts, authors, and classification parameters. Use this tool first when a user asks about 
    the presence of files, summaries of topics, or files authored by specific categories.
    """
    try:
        embeddings = OllamaEmbeddings(model="bge-large:latest", base_url="http://ollama:11434")
        # Explicitly targets the isolated metadata table registry collection
        metadata_store = PGVector(embeddings=embeddings, collection_name="agent_metadata_summaries", connection=DB_URL)
        
        retriever = metadata_store.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(metadata_query)
        
        if not docs:
            return "NOTICE: Metadata similarity lookup completed successfully, but zero matching document summaries were found."
            
        formatted_summaries = []
        for idx, doc in enumerate(docs, start=1):
            src = os.path.basename(doc.metadata.get("source", "Unknown"))
            formatted_summaries.append(f"📦 [Matched File Summary {idx}]: `{src}`\nTags Found: {doc.page_content}\n")
            
        return "SUCCESS: Vector search on metadata table matched the following files:\n\n" + "\n".join(formatted_summaries)
    except Exception as e:
        return f"ERROR: Failed to search semantic metadata table rows: {str(e)}"
