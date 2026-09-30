# Enterprise Local Conversational RAG Engine 🧠🤖

A secure, high-performance, completely offline Retrieval-Augmented Generation (RAG) ecosystem engineered to index, scope, and troubleshoot enterprise logs and technical documentation. This platform operates entirely on local hardware using containerized services, featuring real-time file-system synchronization, session memory continuity, and granular context-targeting filters.

---

## 🏗️ Architecture Topology Overview

The application is deployed as an automated, multi-container orchestration cluster via Docker Compose:

*   **`rag_streamlit_app` (Python 3.11 Runtime)**
    *   **Thread 1 (Core UI):** A responsive Streamlit dashboard featuring multi-turn chat memory, granular document/folder selection scopes, and database chunk metrics.
    *   **Thread 2 (Background Daemon):** A type-safe `watchdog` engine utilizing a `PollingObserver` to track local Windows file-system actions (creations, content modifications, renames, and deletions) over mapped virtualization volumes.
*   **`rag_postgres` (Data Store Layer):** PostgreSQL 16 engine optimized with the `pgvector` spatial index extension to store and fetch high-density text embeddings.
*   **`rag_ollama` (Local Inference Node):** Accelerates open-source tokenizations via `nomic-embed-text` and streams CPU-bound text generation using the `qwen2.5:1.5b` model layout.

---

## 🛠️ System Features

1. **Incremental Ingestion Pipeline:** Auto-detects local file changes offline during boot cycles by comparing file size and timestamp fingerprints against metadata cached inside the database. Unchanged files are skipped instantly (`⏭️ Skipping`).
2. **Context Targeting Scope:** Dynamically injects strict metadata filtering queries (`$in` logic arrays) into the retriever loop. This isolates similarity searches to selected files or nested subfolders, preventing cross-contamination during root-cause log analysis.
3. **Conversational LCEL Core Memory:** Implements a version-agnostic LangChain Expression Language (LCEL) routing branch pipeline that contextualizes vague follow-up questions (e.g., matching *"who signed it?"* to its historical document entity) before querying the database index.
4. **CI/CD Guardrails:** Integrated GitHub Actions pipeline that auto-provisions testing VMs, runs structural style lints (`flake8`), and executes python unit tests (`pytest` with mocks) on every codebase push.

---

## 🚀 Installation & Local Deployment

### 1. Prerequisites (Windows 11 Laptop)
*   Docker Desktop installed with the **WSL2 backend framework** active.
*   Git command-line interface tracking tools.
*   VS Code (Ensure file line-endings dropdown in the bottom status margin bar is set strictly to **`LF`** to prevent execution scripting failures inside Linux containers).

### 2. Clone and Initialize the Project Workspace
```powershell
git clone https://github.com
cd local-conversational-rag-engine
```

### 3. Establish the Environment Variables Configuration (`.env`)
Create a `.env` file in the root project directory to manage system parameters:
```env
DB_URL=postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
STORAGE_TYPE=LOCAL
```

### 4. Build and Launch the Docker Infrastructure Cluster
Run the initial clean-slate compilation loop:
```powershell
# Disable automatic CRLF conversions to protect Linux shell script formatting
git config core.autocrlf false

# Compile container layers and launch the cluster in the background
docker-compose down --volumes
docker-compose up --build -d
```

### 5. Access the Platform Engine Interface
Open your web browser and navigate directly to:
👉 **`http://localhost:8501`**

---

## 📈 Operational Commands & Cheat Sheet

### Monitor Application Startup & Processing Logs
```powershell
docker logs -f rag_streamlit_app
```

### Inspect the Exact Total Vector Chunk Capacity inside PostgreSQL
```powershell
docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM langchain_pg_embedding;"
```

### Verify Distinct Log File Paths Indexed within the Metadata JSONB Store
```powershell
docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT DISTINCT cmetadata->>'source' FROM langchain_pg_embedding;"
```

### Force a Manual Command-Line Directory Directory Ingestion Sweep
```powershell
docker exec -it rag_streamlit_app python app/ingest.py
```

### Stop the Service Infrastructure Stack securely without data loss
```powershell
docker-compose down
```

---

## ⚙️ Performance Tuning & Optimization

### Core Document Splitting Parameters (`app/ingest.py`)
To alter how large system logs or documents are parsed into vectors, update the `RecursiveCharacterTextSplitter` configuration parameters:
*   `chunk_size` (Default: `1000` tokens): Determines the maximum token size per database vector block. For long stack traces, increase this to `2000` to keep continuous multi-line exception sequences within a single database chunk.
*   `chunk_overlap` (Default: `150` tokens): Preserves contextual continuity along splitting boundaries.

### LLM Inference Context Window Limits (`app/ui.py`)
The system loads `ChatOllama` configured to target CPU stability on an 8 GB/16 GB RAM workspace laptop:
*   `num_ctx=2048`: Allocates a maximum context window of 2,048 tokens.
*   `temperature=0.1`: Kept close to zero to ensure maximum analytical accuracy and completely suppress creative hallucination paths.
*   `search_kwargs={"k": 3}`: Pulls exactly the top 3 most relevant vector chunks to limit CPU memory load during real-time token streaming.
