#!/bin/bash

# PDF Document Classifier - Run with Gemini AI
# This script sets the GEMINI_API_KEY and runs the Streamlit app

echo "🚀 Starting PDF Document Classifier with Gemini AI..."

# Check and install dependencies if needed
echo "📦 Checking dependencies..."
if ! python3 -c "import plotly" 2>/dev/null; then
    echo "📥 Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Set the Gemini API key
export GEMINI_API_KEY="AIzaSyDsziZRc_ND5qnFQ0RtqLpxcAoqjqzR6ms"

# Verify the API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY not set"
    exit 1
else
    echo "✅ GEMINI_API_KEY is set"
fi

# Run the Streamlit app
echo "🌐 Starting Streamlit app..."
echo "📱 The app will open at: http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app.py
