"""
One-command launcher script for Disaster Response System Demo.
Launches FastAPI backend (serving embedded React bundle at http://localhost:8000) and optionally Vite dev server.
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("==========================================================")
    print("🚀 Starting Multi-Agent Disaster Response System Demo")
    print("==========================================================")
    
    # Ensure frontend is built
    if not os.path.exists("frontend/dist"):
        print("Building React frontend...")
        subprocess.run(["npm", "run", "build"], cwd="frontend", shell=True, check=True)

    print("Launching FastAPI Backend Server at http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=os.getcwd()
    )

    time.sleep(2)
    webbrowser.open("http://localhost:8000")

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down demo server...")
        backend_proc.terminate()

if __name__ == "__main__":
    main()
