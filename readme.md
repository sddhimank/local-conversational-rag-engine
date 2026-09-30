# Local Conversational Enterprise RAG Engine
## System Architecture & Technical Specification Blueprint

This document details the production-ready, fully automated local Retrieval-Augmented Generation (RAG) platform optimized for offline troubleshooting and context-isolated log analysis.

---

## 🌐 System Architecture Topology Diagram

```text
 💻 WINDOWS 11 LAPTOP HOST DISK ENVIRONMENT (WSL2 Architecture)
 │
 ├── 📂 C:\...\rag-postgres-app\data\ ───► [ Live Docker Volume Mount Linked ] ───┐
 └── 📂 C:\...\rag-postgres-app\upload\ ◄─ [ Persistent Local Asset Backups  ]─┐  │
                                                                               │  │
┌──────────────────────────────────────────────────────────────────────────────┼──┼┐
│ 🐳 DOCKER COMPOSE MULTI-CONTAINER INFRASTRUCTURE CLUSTER                     │  │
│                                                                              │  │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │  │
│ │ 🚀 rag_streamlit_app CONTAINER (Python 3.11 Runtime Workspace Environment)  │  │  │
│ │                                                                         │  │  │
│ │  [ Thread 1: app/watcher.py Background Daemon Observer ]                 │  │  │
│ │    └── PollingObserver ──► Intercepts File System Events (Create/Move/Delete) ◄┘
│ │                                                                         │     │
│ │  [ Thread 2: app/ui.py Streamlit Interactive Frontend Webserver App ]    │     │
│ │    ├── 🎯 Context Targeting Sidebar Widget (Granular Scope Filter)      │     │
│ │    ├── 📥 Drag-and-Drop Ingest Subsystem ──► Writes to Local Disk Directory ──┘
│ │    └── 💬 Conversational History-Aware LCEL Core Memory Execution Pipe  │
│ └─────────┬───────────────────────────────────┬───────────────────────────┘
│           │ [ HTTP Routing: Port 11434 ]       │ [ Psycopg DB Driver API ]
│           ▼                                   ▼
│ ┌───────────────────────────────────────┐ ┌─────────────────────────────────────┐
│ │ 🧠 rag_ollama INFERENCE CONTAINER     │ │ 🐘 rag_postgres DATA STORE CONTAINER│
│ │                                       │ │                                     │
│ │ ├── 📦 nomic-embed-text (Embeddings)  │ │ ├── 🛠️ PostgreSQL 16 DB Core Engine │
│ │ └── 📦 qwen2.5:1.5b (LLM Synthesis)   │ │ └── 🧮 pgvector Spatial Index Vector │
│ └───────────────────────────────────────┘ └─────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Execution Pipeline Lifecycle Flow Walkthrough

The application operations are orchestrated across three core native programmatic lifecycle paths:

### 1. The Startup Baseline Sync Sequence (`app/ingest.py`)
* **Service Interlocking:** The web workspace node delays script calls during the boot loop sequence until both the `postgres` extension sockets and the `ollama` model layer assets report structural readiness via Docker health-check handles.
* **Fingerprint Auditing:** The script scans target local host paths recursively, evaluating active local documents across both text and unstructured PDF vectors.
* **Incremental Mapping:** Instead of dropping records or rewriting arrays blindly, the code extracts compound file metadata properties (File Size + Last Modified Unix Timestamp).
* **Sync Decisions:** It queries the pgvector JSONB metadata fields (`cmetadata`) to check for changes:
  * **Unchanged:** If both the file size and timestamp match database indices exactly, it issues a quick terminal message: `⏭️ Skipping (Unchanged & Embedded)` and moves to the next file in under a second.
  * **Offline Modifications:** If size or timestamp metrics differ, it recognizes that an off-line document edit took place, instantly invokes a clean targeted purge transaction, splits the new data stream, and commits the fresh vector embeddings.

### 2. The Runtime Watchdog Observer Loop (`app/watcher.py`)
Running as an isolated worker on Thread 1, the `PollingObserver` daemon monitors your laptop storage workspace in real-time, matching modifications seamlessly across virtual virtualization filesystems:
* **Additions & Content Appends (`on_created` / `on_modified`):** When you save changes to a text file or append system trace configurations, the daemon catches the modification stream. It applies a time-gated debounce loop to let Windows finish writing the buffer before pushing the target file down to the ingestion engine with `force_update=True`.
* **File Renames & Relocations (`on_moved`):** The thread captures shifts across directories, extracts path variables from tuple objects cleanly, and re-indexes nested folder trees without throwing exceptions.
* **Pruning and Removals (`on_deleted`):** When a file is removed from the host laptop, the watcher triggers a strict parameterized SQL `DELETE ... LIKE` statement, immediately freeing up space and ensuring that reference citations drop right back down automatically.

### 3. Isolated Context Targeting Prompt Flow (`app/ui.py`)
* **State Management:** Streamlit naturally drops script caches and re-runs files from line 1 on every widget change or keypress. Chat history states are cleanly preserved globally within `st.session_state.messages`.
* **Query Contextualization:** Vague follow-up prompts (e.g., *"Can it be extended?"*) are caught by an independent LCEL prompt pipe. The LLM reads past message items to formulate an explicit standalone query statement (*"Can the maternity leave policy be extended?"*) behind the scenes.
* **Granular Context Isolation:** If specific subfolders or documents are checked inside the sidebar Targeting Control panel, the script generates a clean metadata dict mapping criteria object (`{"source": {"$in": selected_raw_paths}}`) and injects it straight to the retriever engine.
* **Strict SQL Scoping:** The `pgvector` engine computes spatial cosine similarities *only* across chunks matching those source indices, entirely blocking outside cross-contamination from other logs or data folders.
* **Response Generation:** The context-matched text blocks are fed into the final prompt synthesizer alongside the chat history objects, streaming the final troubleshoot summaries onto your browser screen.

---

## 📋 Technology Stack Specifications

* **Frontend UI Dashboard:** Streamlit v1.35.0 (Custom LCEL Pipeline Interface)
* **Orchestration Framework:** LangChain / LangChain-Core / LangChain-Postgres (v0.3.x Architecture Architecture)
* **Local Embedding Engine:** Ollama / `nomic-embed-text`
* **Local Language Inference:** Ollama / `qwen2.5:1.5b` (2048 Context Window Capacity)
* **Relational Vector Database:** PostgreSQL 16 paired with the `pgvector` Spatial Extension
* **Background Daemon Worker:** Watchdog v4.0.0 (`PollingObserver` Engine Tuning)
* **Database Driver Connectivity:** Psycopg 3 (Binary Pool Infrastructure)