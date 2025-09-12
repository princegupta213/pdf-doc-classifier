#!/usr/bin/env python3
"""
Setup script for OpenAI API key configuration.
Run this script to help configure your API key for the PDF Document Classifier.
"""

import os
import sys
from pathlib import Path

def setup_api_key():
    """Interactive setup for OpenAI API key."""
    print("🔑 OpenAI API Key Setup for PDF Document Classifier")
    print("=" * 50)
    
    # Check if API key is already set
    current_key = os.getenv("OPENAI_API_KEY")
    if current_key:
        print(f"✅ API key is already set: {current_key[:10]}...")
        choice = input("Do you want to update it? (y/n): ").lower()
        if choice != 'y':
            return
    
    print("\n📋 To get your OpenAI API key:")
    print("1. Go to: https://platform.openai.com/api-keys")
    print("2. Sign in or create an account")
    print("3. Click 'Create new secret key'")
    print("4. Copy the key (starts with 'sk-')")
    
    print("\n🔑 Enter your OpenAI API key:")
    api_key = input("API Key: ").strip()
    
    if not api_key:
        print("❌ No API key provided. Exiting.")
        return
    
    if not api_key.startswith("sk-"):
        print("⚠️  Warning: API key should start with 'sk-'")
        confirm = input("Continue anyway? (y/n): ").lower()
        if confirm != 'y':
            return
    
    # Set environment variable for current session
    os.environ["OPENAI_API_KEY"] = api_key
    print("✅ API key set for current session")
    
    # Create .env file for permanent storage
    env_file = Path(".env")
    if env_file.exists():
        print(f"📄 .env file already exists")
        overwrite = input("Overwrite it? (y/n): ").lower()
        if overwrite != 'y':
            return
    
    with open(env_file, "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
        f.write("# OpenAI API Configuration\n")
        f.write("# Get your API key from: https://platform.openai.com/api-keys\n")
    
    print("✅ API key saved to .env file")
    
    # Test the API key
    print("\n🧪 Testing API key...")
    try:
        import openai
        openai.api_key = api_key
        
        # Simple test call
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        print("✅ API key is working!")
        
    except ImportError:
        print("⚠️  OpenAI package not installed. Install with: pip install openai")
    except Exception as e:
        print(f"❌ API key test failed: {e}")
        print("Please check your API key and try again.")
    
    print("\n🎉 Setup complete!")
    print("You can now run your Streamlit app with LLM features enabled.")
    print("Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    setup_api_key()
