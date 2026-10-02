import time
import os
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from tasks import async_process_file,async_scan_directory, delete_vectors_for_file


WATCH_DIR = "./data"

class NewFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.last_processed={}
    
    def process_file(self, filepath, force_update=False):
        # FIX FOR WSL/DOCKER BYTES ERROR: Natively decode filepath if it arrives as bytes
        if isinstance(filepath, bytes):
            filepath = filepath.decode('utf-8', errors='ignore')
        
        if os.path.isdir(filepath):
            return
            
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".txt", ".pdf"]:
            # Ignore hidden system temporary files created by Windows/Word/Excel
            if os.path.basename(filepath).startswith("~$") or os.path.basename(filepath).startswith("."):
                return

            current_time = time.time()
            # Windows file save debounce logic
            if filepath in self.last_processed and not force_update:
                if current_time - self.last_processed[filepath] < 3:
                    return
            
            self.last_processed[filepath] = current_time
            print(f"⚡ Watcher: Intercepting system change. Handing off to Celery queue: {filepath}")
            
            # ✅ HIGH SCALE FIX: Dispatch task immediately to Redis message broker channel
            async_process_file.delay(filepath, force_update=force_update)
            '''   
            print(f"⚡ Watcher: Intercepted system change for file: {filepath}")
            # Give Windows ample time to completely finish writing the file payload to disk
            time.sleep(2)
            
            if os.path.exists(filepath):
                run_ingestion(single_file_path=filepath,force_update=force_update)
            else:
                print(f"⚠️ Watcher: File {filepath} disappeared before ingestion.")
            '''

    def on_created(self, event):
        self.process_file(event.src_path, force_update=False)
        
    # FIX FOR WINDOWS: Catch content stream shifts if creation event was skipped by Docker
    def on_modified(self, event):
        self.process_file(event.src_path, force_update=True)

    def on_moved(self, event):
        dest_path = event.dest_path
        if isinstance(dest_path, bytes):
            dest_path = dest_path.decode('utf-8', errors='ignore')
        print(f"🔄 Watcher: File renamed to: {os.path.basename(dest_path)}")
        self.process_file(dest_path, force_update=True)

    def on_deleted(self, event):
        filepath = event.src_path
        if isinstance(filepath, bytes):
            filepath = filepath.decode('utf-8', errors='ignore')
        
        ext = os.path.splitext(filepath)[1].lower()
        if ext in [".txt", ".pdf"]:
            print(f"🗑️ Watcher: Detected file removal: {filepath}")
            delete_vectors_for_file(filepath)

if __name__ == "__main__":
    event_handler = NewFileHandler()
    # PollingObserver is used for stable virtualization folder bindings on Windows WSL
    observer = PollingObserver()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=True)
    
    print(f"🚀 Background Worker Active: Watching '{WATCH_DIR}' for new files...")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
