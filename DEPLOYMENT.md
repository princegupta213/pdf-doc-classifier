# 🚀 Streamlit Cloud Deployment Guide

This guide will help you deploy your PDF Document Classifier to Streamlit Cloud.

## 📋 Prerequisites

1. **GitHub Account** - Your code must be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [share.streamlit.io](https://share.streamlit.io)
3. **Repository Structure** - Ensure your project has the correct structure

## 🏗️ Repository Structure

Your repository should look like this:

```
pdf_doc_classifier/
├── streamlit_app.py          # Main Streamlit application
├── extract_and_classify.py   # Core classification logic
├── field_extraction.py       # Field extraction utilities
├── requirements.txt          # Python dependencies
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── .gitignore               # Git ignore file
├── class_examples/          # Training examples (optional)
│   ├── invoice/
│   ├── bank_statement/
│   ├── resume/
│   ├── ITR/
│   └── government_id/
└── README.md                # Project documentation
```

## 🚀 Deployment Steps

### Step 1: Prepare Your Repository

1. **Initialize Git** (if not already done):
   ```bash
   cd /Users/princekumar/Downloads/pdf_doc_classifier
   git init
   git add .
   git commit -m "Initial commit: PDF Document Classifier"
   ```

2. **Create GitHub Repository**:
   - Go to [GitHub](https://github.com) and create a new repository
   - Name it `pdf-doc-classifier` or similar
   - Don't initialize with README (you already have files)

3. **Push to GitHub**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/pdf-doc-classifier.git
   git branch -M main
   git push -u origin main
   ```

### Step 2: Deploy on Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/pdf-doc-classifier`
   - Set the main file path: `streamlit_app.py`
   - Choose branch: `main`

3. **Configure App**:
   - **App URL**: Choose a unique URL (e.g., `your-name-pdf-classifier`)
   - **Python version**: 3.9 or 3.10 (recommended)
   - **Dependencies**: Streamlit will auto-detect `requirements.txt`

4. **Deploy**:
   - Click "Deploy!"
   - Wait for the build process (5-10 minutes)

## ⚙️ Configuration Options

### Environment Variables (Optional)

If you need to set environment variables:

1. Go to your app's settings in Streamlit Cloud
2. Add secrets in the "Secrets" section:

```toml
# .streamlit/secrets.toml
[general]
openai_api_key = "your-api-key-here"  # If using LLM fallback
```

### Custom Configuration

The app includes optimized settings in `.streamlit/config.toml`:

- **Headless mode** for cloud deployment
- **CORS disabled** for better performance
- **Custom theme** matching your app design
- **Usage stats disabled** for privacy

## 🔧 Troubleshooting

### Common Issues

1. **Build Failures**:
   - Check `requirements.txt` for correct package versions
   - Ensure all dependencies are listed
   - Check the build logs in Streamlit Cloud

2. **Import Errors**:
   - Verify all Python files are in the repository
   - Check file paths in imports
   - Ensure `class_examples/` folder exists

3. **Memory Issues**:
   - The app uses caching to optimize memory usage
   - Large PDFs might cause timeouts (Streamlit Cloud has limits)

4. **OCR Dependencies**:
   - Tesseract and Poppler are not available on Streamlit Cloud
   - OCR functionality will be limited to PyMuPDF extraction

### Performance Tips

1. **File Size Limits**:
   - Streamlit Cloud has file upload limits
   - Large PDFs may timeout during processing

2. **Caching**:
   - The app uses `@st.cache_resource` and `@st.cache_data`
   - This improves performance for repeated operations

3. **Model Loading**:
   - The SentenceTransformer model is cached
   - First load may take longer

## 📊 Monitoring

### App Metrics

Your deployed app will show:
- **Processing history** in the sidebar
- **Real-time statistics**
- **Error handling** with user-friendly messages

### Logs

- Check Streamlit Cloud logs for debugging
- Monitor app performance and usage

## 🔄 Updates

To update your deployed app:

1. Make changes to your local code
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: Add new features"
   git push origin main
   ```
3. Streamlit Cloud will automatically redeploy

## 🌐 Your App URL

Once deployed, your app will be available at:
```
https://YOUR_APP_NAME.streamlit.app
```

## 📞 Support

- **Streamlit Cloud Docs**: [docs.streamlit.io/streamlit-community-cloud](https://docs.streamlit.io/streamlit-community-cloud)
- **Community Forum**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Create an issue in your repository

## 🎉 Success!

Your PDF Document Classifier is now live on Streamlit Cloud! Share the URL with others to test your AI-powered document classification system.

---

**Note**: This deployment guide assumes you have the enhanced version of the app with all the new features including batch processing, advanced visualizations, and cloud optimizations.
