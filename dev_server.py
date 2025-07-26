#!/usr/bin/env python3
"""
Development server with enhanced auto-reload capabilities
"""
import os
import sys
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask
import webbrowser
import subprocess

class FileChangeHandler(FileSystemEventHandler):
    def __init__(self, app_path):
        self.app_path = app_path
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.is_directory:
            return
            
        # Only watch Python, HTML, CSS, JS files
        if not any(event.src_path.endswith(ext) for ext in ['.py', '.html', '.css', '.js']):
            return
            
        # Debounce rapid changes
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < 1:
                return
                
        self.last_modified[event.src_path] = current_time
        print(f"\n🔄 File changed: {event.src_path}")
        print("💡 Flask debug mode will auto-reload the server")
        print("🌐 Refresh your browser to see changes\n")

def start_flask_dev_server():
    """Start Flask development server"""
    print("🚀 Starting Flask development server...")
    print("📝 Debug mode is enabled - Python files will auto-reload")
    print("🌐 Open http://localhost:5000 in your browser")
    print("🔄 Browser will auto-refresh when files change\n")
    
    # Import and run the Flask app
    from app import app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)

def setup_file_watcher():
    """Setup file watching for development"""
    event_handler = FileChangeHandler('.')
    observer = Observer()
    observer.schedule(event_handler, '.', recursive=True)
    observer.start()
    return observer

def main():
    print("🎯 Cryptolyzer Development Server")
    print("=" * 40)
    
    # Check if required packages are installed
    try:
        import flask
    except ImportError:
        print("❌ Flask not found. Installing requirements...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Setup file watcher
    observer = setup_file_watcher()
    
    try:
        # Start Flask server
        start_flask_dev_server()
    except KeyboardInterrupt:
        print("\n👋 Shutting down development server...")
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    main() 