#!/usr/bin/env python3
"""
Auto-push script for Cryptolyzer
Automatically commits and pushes changes after file modifications
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(command, description):
    """Run a git command and return success status"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(f"✅ {description}: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr.strip()}")
        return False

def auto_push():
    """Automatically commit and push changes"""
    print("🔄 Auto-pushing changes...")
    
    # Check if there are changes to commit
    status_result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
    if not status_result.stdout.strip():
        print("📝 No changes to commit")
        return True
    
    # Add all changes
    if not run_command("git add .", "Added changes"):
        return False
    
    # Create commit message with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit_message = f"Auto-update: {timestamp}"
    
    # Commit changes
    if not run_command(f'git commit -m "{commit_message}"', "Committed changes"):
        return False
    
    # Push to remote
    if not run_command("git push", "Pushed to remote"):
        return False
    
    print("🎉 Auto-push completed successfully!")
    return True

if __name__ == "__main__":
    success = auto_push()
    sys.exit(0 if success else 1) 