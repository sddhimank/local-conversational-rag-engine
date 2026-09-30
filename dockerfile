# Use a lightweight official Python runtime base image
FROM python:3.11-slim

# Set system environment variables to prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Establish the working directory inside the container
WORKDIR /workspace

# Install system dependencies needed for compiling certain packages if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to take advantage of Docker's layer caching mechanics
COPY requirements.txt .

# Upgrade pip and install the complete RAG python library stack
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application source code and the source data documents directory
COPY app/ ./app/
COPY data/ ./data/

# Expose Streamlit's default networking port to the internal container network
EXPOSE 8501

# Fix Windows CRLF line endings, give execution rights to the initialization script
RUN sed -i 's/\r$//' /workspace/app/app_init.sh && \
    chmod +x /workspace/app/app_init.sh

#1. Healthcheck configuration parameters to ensure Docker knows if the web server is healthy
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 2. Set the startup orchestration script as the absolute entrypoint
# (This script handles ingestion FIRST, then executes Streamlit)
ENTRYPOINT ["/workspace/app/app_init.sh"]

#2. Execute Streamlit, disabling CORS restrictions for standard local network binding
#CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
