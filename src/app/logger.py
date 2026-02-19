from pathlib import Path
import time

def new_session_path():
    Path("logs").mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"session_{ts}.csv"

def write_header(path):
    path.write_text("ts,event,value\n", encoding="utf-8")

def log_event(path, event, value=""):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{ts},{event},{value}\n")