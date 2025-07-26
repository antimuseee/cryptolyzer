#!/usr/bin/env python3
"""
Setup script for CoinGecko API key
This script helps you set up your API key for local development
"""

import os
from pathlib import Path

def setup_api_key():
    """Set up the API key for local development"""
    print("🔑 CoinGecko API Key Setup")
    print("=" * 40)
    
    # Check if .env file exists
    env_file = Path(".env")
    
    if env_file.exists():
        print("📁 .env file already exists")
        with open(env_file, 'r') as f:
            content = f.read()
            if 'COINGECKO_API_KEY' in content:
                print("✅ API key is already configured in .env file")
                return
            else:
                print("⚠️  .env file exists but doesn't contain API key")
    
    print("\n📝 To use your CoinGecko API key locally:")
    print("1. Create a file named '.env' in this directory")
    print("2. Add this line to the file:")
    print("   COINGECKO_API_KEY=CG-YourActualAPIKeyHere")
    print("3. Replace 'CG-YourActualAPIKeyHere' with your actual API key")
    print("\n💡 Example .env file content:")
    print("   COINGECKO_API_KEY=CG-abc123def456ghi789")
    
    # Try to create a template .env file
    try:
        with open(env_file, 'w') as f:
            f.write("# CoinGecko API Key for local development\n")
            f.write("# Replace with your actual API key\n")
            f.write("COINGECKO_API_KEY=CG-YourAPIKeyHere\n")
        print(f"\n✅ Created template .env file at {env_file.absolute()}")
        print("📝 Please edit the file and replace 'CG-YourAPIKeyHere' with your actual API key")
    except Exception as e:
        print(f"\n❌ Could not create .env file: {e}")
        print("📝 Please create the .env file manually")

if __name__ == "__main__":
    setup_api_key() 