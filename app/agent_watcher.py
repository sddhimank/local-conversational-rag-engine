import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from agent_ingest import process_file_agentically

class AgenticFolderWatchHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        
        filepath = event.src_path
        ext = os.path.splitext(filepath)[1].lower()
        if ext != ".pdf":
            return
            
        print(f"\n📂 [AGENT WATCHER DETECTED FILE]: Found new asset entry -> {filepath}")
        # Allow file write stream buffer arrays to stabilize on disk before reading
        time.sleep(2)
        
        try:
            process_file_agentically(filepath)
        except Exception as e:
            print(f"❌ Critical system failure inside Agent Ingestion processing loops: {e}")

if __name__ == "__main__":
    WATCH_DIR = "./agent_data"
    os.makedirs(WATCH_DIR, exist_ok=True)
    
    event_handler = AgenticFolderWatchHandler()
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    
    print(f"🚀 [AGENTIC WATCHER STARTED]: Monitoring directory folder '{WATCH_DIR}' for incoming PDFs...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
