#!/bin/sh

# 1. Run the folder file watcher loop worker in the background
python app/watcher.py &

echo "⏳ App Init: Waiting for PostgreSQL database container socket..."
until curl -s http://postgres:5432 > /dev/null || true; do
  sleep 2
done

echo "⏳ App Init: Waiting for Ollama model weights to settle..."
sleep 5

echo "🔄 App Init: Running automated document ingestion pipeline..."
python app/ingest.py

echo "🚀 App Init: Launching Streamlit interface..."
exec streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0
