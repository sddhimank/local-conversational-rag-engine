#!/bin/bash

# 1. Start the Ollama engine process in the background explicitly
/bin/ollama serve &

# 2. Wait for the engine to initialize its network listening socket
echo "⏳ Ollama Init: Waiting for server port socket (11434) to initialize..."
while ! exec 3<>/dev/tcp/127.0.0.1/11434 2>/dev/null; do
    sleep 2
done
exec 3>&- # Clean up file descriptor trace

# 3. Pull models hands-free
echo "📥 Ollama Init: Pre-loading embedding model: nomic-embed-text..."
/bin/ollama pull nomic-embed-text
/bin/ollama pull bge-large

echo "📥 Ollama Init: Pre-loading LLM engine: qwen2.5:1.5b..."
/bin/ollama pull qwen2.5:1.5b

echo "✅ Ollama Init: All AI models successfully loaded!"

# 4. Hold the process alive so the container doesn't shut down
wait
