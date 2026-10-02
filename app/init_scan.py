import os
import sys
import time
from sqlalchemy.exc import IntegrityError

# Force the current workspace context directory into Python's active system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def safe_enqueue_baseline_scans():
    """Initializes the background directory scanning pipeline, safely catching database collisions."""
    try:
        # Explicitly import your tasks module
        from tasks import async_scan_directory
        
        print("📂 Ingest Initializer: Dispatching baseline data scan to Redis...")
        async_scan_directory.delay("./data")
        
        if os.path.exists("./upload"):
            print("📂 Ingest Initializer: Dispatching baseline upload scan to Redis...")
            async_scan_directory.delay("./upload")
            
        print("✅ Ingest Initializer: Startup tasks successfully enqueued!")
        
    except IntegrityError as e:
        # Intercept parallel initialization unique violations from PostgreSQL
        if "already exists" in str(e) or "UniqueViolation" in str(e):
            print("🔄 Ingest Initializer Notice: Parallel DDL collision detected. Retrying connection bind...")
            time.sleep(2)
            # Re-attempt the task handoff now that the tables are fully built by the other thread
            from tasks import async_scan_directory
            async_scan_directory.delay("./data")
            if os.path.exists("./upload"):
                async_scan_directory.delay("./upload")
            print("✅ Ingest Initializer: Startup tasks successfully enqueued on fallback retry!")
        else:
            raise e
    except Exception as e:
        print(f"❌ Ingest Initializer Critical Error: Failed to enqueue baseline scan: {e}")

if __name__ == "__main__":
    safe_enqueue_baseline_scans()