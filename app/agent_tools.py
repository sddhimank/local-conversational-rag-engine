import os
import re
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
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://ollama:11434")
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
            
        #return "\n".join(formatted_context)

        if len(formatted_context) > 1:
            combined_context = "\n".join(formatted_context)
            # Hard character ceiling to prevent CPU context inflation
            if len(combined_context) > 3000:
                combined_context = combined_context[:3000] + "\n...[Context truncated for token efficiency]..."
            return combined_context
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
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://ollama:11434")
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

@tool
def fallback_regex_keyword_search(search_keyword: str) -> str:
    """Useful fallback tool for locating specific tabular structural data, 
    version matrices, or exact matching text phrases across document layouts 
    when conversational vector similarity searches fail to yield accurate hits.
    """
    engine = create_engine(DB_URL)
    
    query_pieces = re.split(r'[\s_\-\.]', search_keyword.lower())
    pipeline_verbs = {"list", "show", "history", "revision", "document", "for", "the", "pdf", "", "a", "of", "in", "to", "tabular", "table"}
    file_target_tokens = list(dict.fromkeys([p for p in query_pieces if p not in pipeline_verbs and len(p) >= 3]))
    
    raw_tokens = re.findall(r'\b\w{3,}\b', search_keyword.lower())
    content_tokens = [t for t in raw_tokens if t not in file_target_tokens and t not in ["pdf"]]
    
    expanded_text_tokens = []
    for token in content_tokens:
        expanded_text_tokens.append(token)
        if len(token) > 3 and token not in pipeline_verbs:
            expanded_text_tokens.append(token[:3])
    expanded_text_tokens = list(dict.fromkeys(expanded_text_tokens))

    query_params = {}
    where_clauses = ["col.name = 'agent_content_chunks'"]
    
    if file_target_tokens:
        file_conditions = []
        for i, f_token in enumerate(file_target_tokens):
            file_conditions.append(f"LOWER(emb.cmetadata->>'source') LIKE :f_{i}")
            query_params[f"f_{i}"] = f"%{f_token}%"
        where_clauses.append(f"({' AND '.join(file_conditions)})")
        
    if expanded_text_tokens:
        text_conditions = []
        for j, t_token in enumerate(expanded_text_tokens):
            text_conditions.append(f"LOWER(emb.document) LIKE :t{j}")
            query_params[f"t{j}"] = f"%{t_token}%"
        where_clauses.append(f"({' OR '.join(text_conditions)})")

    sql_string = f"""
        SELECT emb.cmetadata->>'source' as src, emb.document 
        FROM langchain_pg_embedding emb
        JOIN langchain_pg_collection col ON emb.collection_id = col.uuid
        WHERE { ' AND '.join(where_clauses) }
        ORDER BY (CASE WHEN emb.document LIKE '%|%' THEN 0 ELSE 1 END) ASC, emb.id ASC
        LIMIT 25;
    """
    
    try:
        with engine.connect() as conn:
            results = conn.execute(text(sql_string), query_params).fetchall()
            
        if not results:
            return f"NOTICE: No rows matched target parameters: {file_target_tokens}"
            
        raw_table_rows = []
        matched_filename = "Unknown"
        
        # 1. First Pass: Collect all lines across database chunks that belong to a table layout
        for row in results:
            current_src = os.path.basename(row.src or "Unknown")
            if file_target_tokens and not any(tok in current_src.lower() for tok in file_target_tokens):
                continue
                
            matched_filename = current_src
            raw_lines = row.document.split('\n')
            
            for line in raw_lines:
                line_str = line.strip()
                if not line_str or "rev#" in line_str.lower() or "version" in line_str.lower() or re.match(r'^\|[\s\-:|]+$', line_str):
                    continue
                if line_str not in raw_table_rows:
                    raw_table_rows.append(line_str)

        # 2. 🌟 ADAPTIVE ROW ACCUMULATOR & PARSER ENGINE
        consolidated_table_rows = []
        current_version_row = None
        seen_versions = set()
        
        for line in raw_table_rows:
            # Detect version strings (e.g., 1.0, 4.5)
            version_match = re.search(r'\b\d\.\d\b', line)
            
            if version_match:
                v_num = version_match.group(0)
                if v_num in seen_versions:
                    continue
                    
                # If we have an existing row accumulated, commit it before starting a new one
                if current_version_row:
                    consolidated_table_rows.append(current_version_row)
                    
                seen_versions.add(v_num)
                
                if '|' in line:
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if len(cells) >= 3:
                        current_version_row = f"| {cells[0]} | {cells[1]} | {' '.join(cells[2:])} |"
                    else:
                        current_version_row = None
                        seen_versions.remove(v_num)
                else:
                    # 🌟 INDEX SYNTAX FIX FOR 1.0 AND 1.1
                    segments = re.split(r'\s{2,}', line)
                    if len(segments) >= 3:
                        details_payload = " ".join(segments[2:])
                        current_version_row = f"| {segments[0]} | {segments[1]} | {details_payload} |"
                    else:
                        current_version_row = None
                        seen_versions.remove(v_num)
            else:
                # 🌟 MULTI-LINE DESCRIPTION BINDING: If the line doesn't start with a number
                # and contains column data, append it to the active row's description field
                if current_version_row and '|' in line:
                    clean_append_text = line.replace('|', ' ').strip()
                    # Clean out formatting artifacts
                    clean_append_text = re.sub(r'\s+', ' ', clean_append_text)
                    if clean_append_text:
                        current_version_row = current_version_row[:-2] + " • " + clean_append_text + " |"

        # Catch the final row in the buffer pipeline
        if current_version_row:
            consolidated_table_rows.append(current_version_row)

        if consolidated_table_rows:
            # Sort chronologically by version number
            try:
                consolidated_table_rows.sort(key=lambda x: [float(i) for i in re.findall(r'\d+\.\d+', x)[:1]])
            except:
                pass
                
            table_header = "| Revision Version | Effective Date | Author & Change Modification Details |"
            separator = "|---|---|---|"
            clean_table_output = f"{table_header}\n{separator}\n" + "\n".join(consolidated_table_rows)
            
            return f"--- Verified Table Extraction for File '{matched_filename}' ---\n\n{clean_table_output}"
        else:
            return "NOTICE: Structural matrices could not be consolidated from text layers."
            
    except Exception as e:
        return f"ERROR: Dynamic stream consolidator failed: {str(e)}"
    finally:
        engine.dispose()
