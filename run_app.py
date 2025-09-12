#!/usr/bin/env python3
"""
Demo script to run the enhanced PDF Document Classifier Streamlit app.
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app with proper configuration."""
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    
    # Check if all dependencies are available
    required_packages = [
        "sentence_transformers",
        "torch", 
        "numpy",
        "pymupdf",
        "pdf2image",
        "pytesseract",
        "pandas",
        "plotly"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    streamlit_app_path = os.path.join(script_dir, "streamlit_app.py")
    
    if not os.path.exists(streamlit_app_path):
        print(f"❌ streamlit_app.py not found at {streamlit_app_path}")
        return 1
    
    print(f"\n🚀 Starting PDF Document Classifier...")
    print(f"📁 App location: {streamlit_app_path}")
    print(f"🌐 The app will open in your browser at http://localhost:8501")
    print(f"⏹️  Press Ctrl+C to stop the server")
    
    # Run streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            streamlit_app_path,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down PDF Document Classifier...")
        return 0
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
