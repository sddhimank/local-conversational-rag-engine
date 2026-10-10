
# Autonomous Agentic RAG Platform 🧠🤖

A production-grade, highly optimized Autonomous Agentic Retrieval-Augmented Generation (RAG) system built with Python, LangChain, Ollama, and pgvector. This platform is specially engineered to run efficiently on resource-constrained host machines (e.g., a 4 CPU / 16GB RAM Azure Linux VM) by utilizing asynchronous task routing, programmatic metadata caching, and high-speed database short-circuit bypasses.


## 🏗️ System Architecture & Design Matrix

The system separates user interaction threads from heavy mathematical and language-model evaluation workloads using a specialized multi-container topology:
*  Frontend Dashboard: Streamlit UI running an integrated, real-time background task monitoring callback trace widget.
*  Orchestration Layer: Custom version-independent LangChain State Management engine (CustomAgentExecutor).
*  Asynchronous Processing: Redis acting as a transient message broker pairing tasks directly to a background Celery Prefork worker pipeline.
*  Persistent Vector Core: PostgreSQL equipped with the pgvector extension utilizing optimized HNSW indexes over dual table collections (agent_content_chunks and agent_metadata_summaries).
*  Local LLM Engine: Ollama serving quantized (Q4_K_M) models via internal loopback network requests (llama3.1:latest and nomic-embed-text:latest).

### Block Diagram
```text
    ========================================================================================
                              SYSTEM CORE WORKFLOW BLOCK DIAGRAM
    ========================================================================================

                         +───────────────────────────────+
                         │     Streamlit User Interface  │
                         +───────────────┬───────────────+
                                         │
                    📄 File Uploads     │ 💬 User Queries
                    (Async Path)        │ (Real-Time Path)
                                         ▼
            +─────────────────────────────────────────────────────────+
            │               Redis Cache & Message Broker              │
            +────────────┬───────────────────────────────┬────────────+
                         │                               │
       🔄 Ingestion Job  │                               │ 🧠 Agentic API Call
       (Concurrency = 1) │                               │ (Config: Callback UI)
                         ▼                               ▼
            +─────────────────────────+     +─────────────────────────+
            │ Celery Ingestion Worker │     │ LangChain Agent Engine  │
            │  (pdfplumber Parser)    │     │  (CustomAgentExecutor)  │
            +────────────┬────────────+     +────────────┬────────────+
                         │                               │
                         │ Injects Layout                │ Reads Data Matrix via
                         │ Markdown Matrix               │ Advanced Token Filters
                         ▼                               ▼
            +─────────────────────────────────────────────────────────+
            │             PostgreSQL Vector Database DB               │
            │   - Table 1: agent_content_chunks (Raw Text / Tables)   │
            │   - Table 2: agent_metadata_summaries (Programmatic)    │
            +────────────────────────────┬────────────────────────────+
                                         │
                                         │ Binds 768-Dim Indexing Math
                                         ▼
            +─────────────────────────────────────────────────────────+
            │               Ollama Local Inference Node               │
            │      - Model A: llama3.1 (Core Tool Decision Brain)     │
            │      - Model B: nomic-embed-text (CPU Embedding)        │
            +─────────────────────────────────────────────────────────+

    ========================================================================================
                       INSIDE THE LANGCHAIN ENGINE: CHAT PROCESSING LOOP
    ========================================================================================

        [ User Question ]
           │
           ▼
        [ CustomAgentExecutor.invoke() ] ──► (Rule 5: Resolves ambiguous pronouns via history)
           │
           ▼
        [ Ollama: Llama 3.1 Turn 1 Analysis ]
           │
      ┌────┴───────────────────────────────────────┐
      │ (Tabular / Version Log Query)              │ (Conversational / Clause Query)
      ▼                                            ▼
    [ Tool: fallback_regex_keyword_search ]      [ Tool: execute_document_semantic_search ]
      │                                            │
      ├───► Splits Alphanumeric Filename Tokens    ├───► Extracts 768-Dimension Dense Vector
      ├───► Enforces Strict AND Target Join        ├───► Queries HNSW Index Cache
      ├───► Filters out structural watermarks      │
      ├───► Groups rows via Adaptive Accumulator   │
      │                                            ▼
      ▼                                      [ CPU Optimization Guardrail Check ]
    [ Check: '--- Verified Table Extraction' ]         │
      │                                            │ (Bypasses Turn 2 CPU Loop)
      ├───► (YES) ──► [ EXPRESS SHORT-CIRCUIT ] ───┼─► [ Returns Clean Answer Text ]
      │                     (Returns pristine      │
      │                      Markdown Table)       │
      │                                            ▼
      └───► (NO)  ──► [ Append Scratchpad ] ───────┴─► [ Renders to Streamlit UI ]
```
## 🔄 Cognitive Execution Flow Paths

The system utilizes two distinct execution flow paths depending on the structural density of the data requested:

### 1. Ingestion Flow Pipeline (Zero LLM Overhead): 
To prevent host CPU starvation (100% core locks), heavy text parsing and extraction loops bypass sequential LLM calls entirely:
*  File Upload: User uploads a PDF manual via the Streamlit workspace or drops an item into the watch folder directory layout.
*  Worker Handoff: Streamlit writes the binary block to a shared volume path and throws a slim metadata payload notification directly to Redis.
*  Parsing & Markdown Consolidation: A single-concurrency Celery worker processes the document line by line via pdfplumber, reconstructing data grids programmatically into valid Markdown table strings (|).
*  Vector Commits: Structural blocks are converted instantly using the fast CPU-friendly model nomic-embed-text (768 dimensions), then committed to PostgreSQL in batched pipelines (batch_size=50).

### 2. Query Execution Loop (With Short-Circuit Latency Bypass)

*  Turn 1 Prompt Execution: The user inputs an explicit or conversational document directive. The query is intercepted by the StreamlitAgentCallbackHandler to render live engine operations on-screen.
*  High-Priority Structural Routing: If the prompt targets matrices (e.g., "revision history"), the engine routes directly to fallback_regex_keyword_search.
*  Token Partial Filename Intersection: The tool splits the query using path delimiters and builds a strict SQL metadata clause using an AND operation. This isolates the query space (e.g., separating SDLC-00240 from SOP-00471) and prevents cross-file data pollution.
*  Fuzzy Expansion & Row Accumulation: Text filters look for fuzzy 3-letter roots (e.g., matching %rev% to find | Rev# |). The tool aggregates split rows, deletes duplicate entries via a custom verification set, and stitches multi-line text descriptions back to their parent index cells.
*  Express Short-Circuit Bypass: If a valid table block is successfully compiled, the python tool returns a unique tracking header flag. The CustomAgentExecutor catches this flag, interrupts the slow LLM token-generation prefill loop on Turn 1, and pushes the raw markdown table straight to Streamlit in under 2 seconds.
*  CPU Context Guardrail: If a conversational query invokes execute_document_semantic_search, the engine captures the initial lookup context and uses a single-turn synthesis bypass. This stops the model from entering multi-turn tool-calling loops that would overload the 4 CPU cores.


## 🧠 Configuration & System Optimization Settings


### 1. Ollama Host Thread Constraints (agent_engine.py): 
To prevent the model from over-allocating worker thread paths and freezing database transactions, threads are limited to match the machine's exact physical properties:
*  Model Parameter Profile: ChatOllama(model="llama3.1:latest", temperature=0.0)
*  Context Window Token Limit (num_ctx): 4096 (Caps memory footprint and prevents RAM thrashing)
*  Core Hardware Affinity (num_thread): 4 (Locks compilation directly to your 4 physical CPU cores)
*  Memory Optimization Mode (low_vram): True (Forces aggressive thread clean-up routines)

### 2. Postgres Resource Allocation Matrix (docker-compose.yml)

*  Database Memory Buffers (shared_buffers): 1GB (Allows caching frequent vectors in RAM)
*  Operation Sorting Space (work_mem): 32MB (Prevents disk sorting for vector lookups)
*  Index Construction Space (maintenance_work_mem): 256MB (Allocates memory for quick HNSW index builds)

### 3. Celery Concurrency Guardrails: 
Because running heavy vector computations concurrently on a CPU triggers system lockups, the worker container execution runtime is locked to sequential queue checking:
*  Worker Deployment Command: 
```text
celery -A app.tasks.celery_app worker --loglevel=info --concurrency=1
```