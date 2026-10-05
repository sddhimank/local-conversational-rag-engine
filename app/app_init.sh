#!/bin/sh

# 1. Run the folder file watcher loop worker in the background
#python app/watcher.py &

echo "⏳ App Init: Waiting for PostgreSQL database container socket..."
until curl -s http://postgres:5432 > /dev/null || true; do
  sleep 2
done

echo "⏳ App Init: Waiting for Ollama model weights to settle..."
sleep 5

#echo "🔄 App Init: Running automated document ingestion pipeline..."
#python app/ingest.py
# ✅ THE FIXED ENTRY POINT INITIALIZATION: 
# This line safely forces Python to initialize the tasks module namespace 
# and enqueue the baseline directory scans into Redis without computing vectors locally!
echo "🔄 App Init: Enqueuing first-time baseline scans into asynchronous task queue..."
#python -c "from app.tasks import async_scan_directory; async_scan_directory.delay('./data'); async_scan_directory.delay('./upload')"
python app/init_scan.py

# 2. Run the folder file watcher daemon in the background to catch live updates
python app/watcher.py &

echo "🚀 App Init: Launching Linear RAG interface on port 8501..."
# ✅ FIX: Added the '&' to run port 8501 in the background asynchronously
exec streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0 &

echo "🤖 App Init: Launching Agentic Sandbox interface on port 8502..."
# This runs in the foreground to keep the container engine active and stable
exec streamlit run app/agent_ui.py --server.port=8502 --server.address=0.0.0.0