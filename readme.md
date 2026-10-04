# Enterprise Local Conversational RAG Engine 🧠🤖

A high-performance, completely offline, production-grade Retrieval-Augmented Generation (RAG) platform engineered to ingest, index, and analyze dense technical documentation, standard operating procedures (SOPs), and system logs. 

This architecture leverages a **fully decoupled multi-container design** optimized for local hardware deployments. It ensures a 100% responsive user search window by offloading all heavy file parsing, chunking, and embedding processes to an asynchronous background processing cluster.

---

## 🏗️ System Architecture & Lifecycle Topology

The application stack is orchestrated via Docker Compose as an isolated, multi-service ecosystem. It cleanly splits the user interface layer from the compute-heavy ingestion pipeline:

```text
 💻 WINDOWS 11 LAPTOP HOST DISK ENVIRONMENT (WSL2 Architecture)
 │
 ├── 📂 C:\...\rag-postgres-app\data\ ───► [ Mapped Directory Volume Mount ] ──┐
 └── 📂 C:\...\rag-postgres-app\upload\ ◄─ [ Mapped User Upload Storage ] ───┐  │
                                                                             │  │
┌────────────────────────────────────────────────────────────────────────────┼──┼┐
│ 🐳 DOCKER COMPOSE MULTI-CONTAINER INFRASTRUCTURE CLUSTER                   │  │
│                                                                            │  │
│  ┌──────────────────────────────┐          📦 [ rag_redis ]                │  │
│  │ 🚀 [ rag_streamlit_app ]     │          │  (Redis Task Queue Buffer)    │  │
│  │  (Streamlit Core UI Layer)   │          ▲                               │  │
│  │                              │          │ (Instant Handoff)             │  │
│  │  ├── app/ui.py Chat Window   ├──────────┼───────────────────────────────┘  │
│  │  └── app/watcher.py Daemon   ├──────────┘                                  │
│  └──────────────┬───────────────┘                                             │
│                 │                                                             │
│                 │ [HTTP Port 11434]                                           │
│                 ▼                                                             │
│  ┌──────────────────────────────┐          ⚙️ [ rag_celery_worker ]           │
│  │ 🧠 [ rag_ollama ]             │ ◄────────  (Celery Thread Pool, -P threads)│
│  │  (Accelerated C++ Runtime)  │             │                                │
│  │                              │             ├──► app/tasks.py Ingestion Engine
│  │  ├── bge-large:latest        │ ◄───────────┤    (Local /tmp file caching)  │
│  │  └── qwen2.5:1.5b (LLM)      │             │                                │
│  └──────────────────────────────┘             ▼ [Psycopg Pool Shared Engine]  │
│                                            ┌────────────────────────────────┐ │
│                                            │ 🐘 [ rag_postgres ]            │ │
│                                            │  (PostgreSQL 16 Data Store)    │ │
│                                            │                                │ │
│                                            │  └── 🧮 pgvector (1024-Dim)   │ ◄┘
│                                            └────────────────────────────────┘
└────────────────────────────────────────────────────────────────────────────────┘
```

### ⚙️ Component Blueprint Matrix

*   **`rag_streamlit_app` (Python 3.11 Runtime Container):** Serves the conversational user dashboard (`app/ui.py`) and maintains chat state parameters. It spawns a background filesystem watchdog (`app/watcher.py`) that instantly intercepts directory changes and offloads task payloads to Redis, keeping the UI entirely freeze-free.
*   **`rag_redis` (Redis 7-Alpine Message Broker):** Functions as an asynchronous network memory buffer. It absorbs burst ingestion requests (such as dropping hundreds of files simultaneously) and distributes them sequentially to the background cluster.
*   **`rag_celery_worker` (Asynchronous Worker Node):** Executes out-of-process data operations using a thread-pool worker strategy (`-P threads --concurrency=4`). This locks the model weight footprint down to a safe singleton state, preventing system-level memory crashes (`SIGKILL`).
*   **`rag_ollama` (Accelerated Inference Node):** Runs a native C++ compiled framework (`llama.cpp`) that utilizes host-level execution optimizations to compute dense vector shapes and generate streamed, token-by-token textual answers.
*   **`rag_postgres` (PostgreSQL 16 Data Store + `pgvector`):** Persists text chunks mapped against high-density vector rows and structured `JSONB` file metadata paths.

---

## 🔄 End-to-End Ingestion & Processing Flow

```text
📁 Local Folder ──► [ watcher.py ] ──► 📦 Redis ──► ⚙️ Celery Thread ──► 🧮 Metadata Check ──► 💾 /tmp Buffer ──► 🧠 Ollama Batch ──► 🐘 pgvector
```

### 1. Ingestion Initialization (`app/app_init.sh` & `app/init_scan.py`)
*   Upon container startup, `app_init.sh` waits for the PostgreSQL sockets to report healthy conditions.
*   It fires an isolated bootstrap script `python app/init_scan.py` which crawls the input storage volumes.
*   `init_scan.py` features an atomic `try...except` block designed to intercept parallel multi-container DDL race conditions. If it encounters a `UniqueViolation` (table already exists), it safely falls back, binds to the active index, and triggers:
    ```python
    async_scan_directory.delay("./data")
    ```

### 2. Task Queuing & Batch Distribution (`app/tasks.py`)
*   The folder target is serialized into an event token and pushed to `rag_redis`. 
*   The `rag_celery_worker` pulls the task and runs an asynchronous directory scan.
*   Instead of looping through documents sequentially, it deploys a Celery **`group` orchestration matrix**, packaging all file operations together to broadcast the entire workload tree concurrently back to the task queue.

### 3. Smart Skipping & Incremental Sync Guardrails
*   An available worker thread grabs a single file task. It extracts the file’s absolute size on disk and its last modification date timestamp.
*   The thread uses a shared, thread-safe database connection pool engine (`_SHARED_DB_ENGINE`) to cross-reference these attributes with the database cache registry.
*   **The Decision:** If the size and timestamp match existing records, the worker logs `⏭️ Background Worker: Skipping unchanged file` and finishes in under **5 milliseconds**. If a modification or a new file is detected, it invokes `delete_vectors_for_file()`, dropping stale data matrices to prevent vector pollution.

### 4. Local `/tmp` File Buffering (gRPC Volume Deadlock Bypass)
*   To bypass Windows-to-Linux virtualization filesystem freezes (`gRPC FUSE` deadlocks) during dense read-seek cycles, the worker copies the file's raw binary data array into the container's native Linux RAM space in a single I/O sweep.
*   It writes the bytes into a temporary local variable file path (`/tmp/worker_filename.pdf`).
*   LangChain’s **`PyPDFLoader`** reads from this local `/tmp` buffer, extracts layout text data, and passes strings to the `RecursiveCharacterTextSplitter` to yield uniform chunks of 1,000 characters with a 150-character overlap boundary. The temporary asset is instantly deleted via `os.remove()` upon completion.

### 5. High-Speed Parallel Ollama Embedding Batching
*   The text chunks are converted into flat string arrays. Instead of executing hundreds of sequential HTTP calls one-by-one, LangChain's optimized **`.embed_documents()`** batch mechanism is used:
    ```python
    embeddings_arrays = _GLOBAL_EMBEDDINGS.embed_documents(texts_to_embed)
    ```
*   This compiles all chunks inside the Ollama container (`bge-large:latest`) in a single network transmission pass, reducing embedding calculation time from minutes down to seconds.
*   The resulting 1,024-dimensional floating-point vectors are written to the database using the optimized store connector method, applying a strict **`batch_size=100`** constraint block to protect PostgreSQL from memory spikes.

---

## 🎯 Conversational Search & Analytics Routing Guardrail

When an input question is submitted to the **Streamlit Web Application UI**, the system routes it through a specialized dual-channel path inside `app/ui.py` to maximize accuracy:

```text
                              ┌──► [ True Math Query ] ──► Executes SQL Direct Count ──► 100% Precise Metrics
                              │
🧑 Input Prompt ──► [ui.py] ──┤
                              │
                              └──► [ Semantic Query ] ──► LCEL Contextualizer ──► Filtered Retriever ──► LLM Synthesis
```

### Channel A: The Structural Analytics Route (Hallucination Suppression)
*   If a user asks statistical or quantitative counting questions (e.g., *"How many files in this folder?"*, *"Total number of PDFs"*), the system completely **bypasses the unstructured vector similarity search**.
*   The interface intercepts the query using a keyword match matrix and pulls metadata metrics straight from your SQL tables, calculating counts for actively checked sidebar folder selections (`selected_folders`) using a strict `LIKE` wildcard schema.
*   **Result:** Delivers 100% precise file counts instantly, completely eliminating LLM quantitative mathematical hallucinations.

### Channel B: The Conversational Vector Route (LCEL Chain Synthesis)
For standard semantic inquiries, LangChain Expression Language (LCEL) chains handle multi-turn history-aware lookups:
1.  **Contextualizer Chain:** Pulls past chat history logs via a `MessagesPlaceholder` and uses `qwen2.5:1.5b` to transform vague queries (e.g., *"What does it say about security?"*) into standalone, fully explicit search strings.
2. **Filter-Scoped Retriever:** Injects strict metadata targeting dictionary keys ({"source": {"$like": "Folder/%"}}) matching the sidebar scope directly into the vector retriever pipeline.
3. **QA Synthesis Chain:** Fetches the top 3 most relevant context blocks, aggregates them into a clean string, inserts them into the expert system template, and prompts the LLM to stream verified answers word-by-word onto the dashboard interface.

##  🚀 Installation, Tuning & Operations

### 1. Prerequisites (Windows 11 Host Laptop)
* Docker Desktop configured with the WSL2 backend framework active.
* Ensure the file line-endings configuration dropdown in your code editor is explicitly set to LF for all .sh and .py files inside the app/ folder to prevent Linux execution errors.

### 2. Environment Variables Configuration (`.env`)
Create a `.env` file in the root project folder to declare connection variables:
```env
DB_URL=postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password
POSTGRES_DB=rag_db
REDIS_URL=redis://redis:6379/0
```

### 3. Deploy the Multi-Container Cluster Infrastructure
Execute these sequential commands in your terminal to initialize your local model weights and spin up your runtime cluster:
```powershell
# 1. Start the Ollama service container first
docker-compose up -d ollama

# 2. Force the Ollama node to pull the high-accuracy models completely offline
docker exec -it rag_ollama ollama pull bge-large
docker exec -it rag_ollama ollama pull qwen2.5:1.5b

# 3. Launch the full decoupled background ingestion stack
docker-compose up -d
```

### 4. Operations Cheat Sheet & Diagnostics

*   **Access the Streamlit Dashboard UI:** Open your web browser directly to 👉 **`http://127.0.0.1:8501`**
*   **Trace Real-Time Parsing and Embedding Progress:** 
    ```powershell
    docker logs -f rag_celery_worker
    ```
*   **Audit Web Application Routing and Socket Logs:** 
    ```powershell
    docker logs -f rag_streamlit_app
    ```
*   **Verify Vector Array Chunks Count inside PostgreSQL:** 
    ```powershell
    docker exec -it rag_postgres psql -U rag_user -d rag_db -c "SELECT COUNT(*) FROM langchain_pg_embedding;"
    ```
*   **Wipe Database Records for a Clean Ingestion Test:**
    ```powershell
    docker exec -it rag_postgres psql -U rag_user -d rag_db -c "DROP TABLE IF EXISTS langchain_pg_embedding CASCADE; DROP TABLE IF EXISTS langchain_pg_collection CASCADE;"
    docker-compose restart rag-web-app rag-celery-worker
    ```

---

## 🛠️ Production Performance Tuning Reference

The platform features pre-configured hardware allocation profiles fine-tuned specifically for local laptop deployment stability:

| Parameter Key | Active Default Location | Functional Intent | Tuning Guardrail Strategy |
| :--- | :--- | :--- | :--- |
| `--concurrency=1` | `docker-compose.yml` | Sets active worker threads. | Prevents CPU thread contention so a single task can utilize full processing power. |
| `batch_size=100` | `app/tasks.py` | Controls database insertion passes. | Prevents high-density memory spikes or Docker OOM container crashes (`SIGKILL`). |
| `worker_prefetch_multiplier=1` | `app/tasks.py` | Sets memory task allocations. | Keeps text document string arrays from piling up in background RAM before processing. |
| `num_ctx=2048` | `app/ui.py` | Allocates chat context windows. | Protects local laptop hardware memory boundaries during streaming token inference. |
