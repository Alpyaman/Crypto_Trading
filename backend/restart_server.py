"""
Quick server restart utility
"""
import subprocess
import time
import sys
import psutil
import os

def find_and_kill_server():
    """Find and kill any running FastAPI server on port 8000"""
    print("🔍 Looking for existing server processes...")
    
    killed_any = False
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline and any('app.main' in str(cmd) for cmd in cmdline):
                print(f"  🔪 Killing process {proc.info['pid']}: {proc.info['name']}")
                proc.kill()
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if killed_any:
        print("  ⏱️ Waiting for processes to stop...")
        time.sleep(2)
    else:
        print("  ✅ No existing server processes found")

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting Crypto Trading AI server...")
    
    # Change to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(backend_dir)
    
    # Start the server
    try:
        subprocess.run([sys.executable, "-m", "app.main"], check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Server failed to start: {e}")

def main():
    print("🔄 Crypto Trading AI Server Restart Utility")
    print("=" * 50)
    
    # Kill existing servers
    find_and_kill_server()
    
    # Start new server
    start_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        input("Press Enter to continue...")