# 🛠️ THE AGENTIC RAG DEPLOYMENT RUNBOOK

This guide details how to set up, resource-constrain, and launch your Agentic RAG pipeline from a clean install on your VM (4 CPU / 16GB RAM).


## 📋 Prerequisites

Before launching the application, ensure the host environment meets the following baseline criteria:
• Docker Engine: Version 20.10.0 or higher installed and running.
• Docker Compose Plugin: Version 2.0.0 or higher active.
• Git CLI: Installed and authenticated to pull from your repository.
• System Hardware Minimum: 4 CPU Cores and 8 GB of unallocated system RAM (16 GB highly recommended to support concurrent model executions).

## 🚀 Step-by-Step Deployment Execution

### 1. Clone the Active Project Repository

Navigate to your working directory on the host environment and pull the codebase straight from GitHub:
```text
# Clone the repository
git clone https://github.com/sddhimank/local-conversational-rag-engine/tree/feature/agentic-rag-loop

# Move into the project root directory
cd agentic-rag-loop
```

### 2. Synchronize Your Configuration Matrix

Verify that your root directory contains your optimized, resource-constrained docker-compose.yml configuration:
```text
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: rag_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: rag_user
      POSTGRES_PASSWORD: rag_password
      POSTGRES_DB: rag_db
    command: >
      postgres
      -c shared_buffers=1GB
      -c work_mem=32MB
      -c maintenance_work_mem=256MB
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - rag_network

  redis:
    image: redis:7-alpine
    container_name: rag_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    networks:
      - rag_network

  ollama:
    image: ollama/ollama:latest
    container_name: rag_ollama
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_NUM_PARALLEL=1
      - OLLAMA_MAX_LOADED_MODELS=1
    deploy:
      resources:
        limits:
          cpus: '2.5'
          memory: 8G
    networks:
      - rag_network

  rag-web-app:
    build: .
    container_name: rag_streamlit_app
    restart: unless-stopped
    ports:
      - "8502:8502"
    environment:
      - DB_URL=postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./app:/workspace/app
      - ./agent_data:/workspace/agent_data
    command: streamlit run app/agent_ui.py --server.port=8501 --server.address=0.0.0.0
    depends_on:
      - postgres
      - redis
      - ollama
    networks:
      - rag_network

  rag-celery-worker:
    build: .
    container_name: rag_celery_worker
    restart: unless-stopped
    environment:
      - DB_URL=postgresql+psycopg://rag_user:rag_password@postgres:5432/rag_db
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./app:/workspace/app
      - ./agent_data:/workspace/agent_data
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=1
    depends_on:
      - redis
      - postgres
    networks:
      - rag_network

volumes:
  pgdata:
  ollama_data:

networks:
  rag_network:
    driver: bridge
```

### 3. Compile the Application Stack

Execute the following OS-agnostic terminal commands to build your Docker images and boot the microservices network:
```text
# Compile custom images from source and launch services in background/detached mode
docker compose up -d --build

# Verify that all 5 microservice containers are healthy and running
docker compose ps
```