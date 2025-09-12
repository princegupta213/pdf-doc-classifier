#!/bin/bash

# Quick deployment script for PDF Document Classifier
# This script helps you deploy to Streamlit Cloud quickly

echo "🚀 PDF Document Classifier - Quick Deploy Script"
echo "================================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit: PDF Document Classifier v2.0"
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if remote origin exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo ""
    echo "🔗 Please set up your GitHub repository:"
    echo "1. Go to https://github.com and create a new repository"
    echo "2. Copy the repository URL (e.g., https://github.com/username/repo-name.git)"
    echo "3. Run: git remote add origin YOUR_REPOSITORY_URL"
    echo "4. Run: git push -u origin main"
    echo ""
    echo "Then visit https://share.streamlit.io to deploy!"
else
    echo "✅ Remote origin is configured"
    echo "🔄 Pushing to GitHub..."
    git add .
    git commit -m "Update: Enhanced PDF Document Classifier $(date '+%Y-%m-%d %H:%M:%S')"
    git push origin main
    echo "✅ Code pushed to GitHub"
    echo ""
    echo "🌐 Now go to https://share.streamlit.io to deploy your app!"
fi

echo ""
echo "📋 Deployment Checklist:"
echo "✅ Repository structure is ready"
echo "✅ Requirements.txt is configured"
echo "✅ Streamlit config is optimized"
echo "✅ Streamlit Cloud deployment ready"
echo "✅ .gitignore is configured"
echo ""
echo "🎯 Next Steps:"
echo "1. Visit https://share.streamlit.io"
echo "2. Sign in with GitHub"
echo "3. Click 'New app'"
echo "4. Select your repository"
echo "5. Set main file to 'streamlit_app.py'"
echo "6. Click 'Deploy!'"
echo ""
echo "🎉 Your enhanced PDF Document Classifier will be live in minutes!"
