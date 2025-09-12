"""Streamlit web app for classifying uploaded PDF documents."""

import os
import io
import json
import tempfile
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import hashlib
from typing import Dict

from extract_and_classify import (
    extract_text_with_pymupdf,
    extract_text_with_ocr,
    build_class_centroids,
    classify_text,
    _get_embedding_model,
)

# Import Gemini LLM functionality
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
# Field extraction removed - not used in current version

# Page configuration
st.set_page_config(
    page_title="PDF Document Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud deployment optimizations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_cloud_info():
    """Get deployment information for cloud environments."""
    return {
        "deployment": "Streamlit Cloud",
        "version": "2.0",
        "features": ["batch_processing", "advanced_visualizations", "caching"]
    }

def enhance_with_gemini(text: str, classification: dict, model) -> dict:
    """Enhance classification using Gemini AI."""
    if not model or not text.strip():
        return classification
    
    try:
        prompt = f"""
        Analyze this document text and provide classification insights:
        
        Text: {text[:2000]}
        
        Current classification: {classification}
        
        Provide:
        1. Document type confirmation
        2. Key fields that could be extracted
        3. Confidence assessment
        4. Any additional insights
        
        Respond in a concise format.
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=300,
                temperature=0.1
            )
        )
        
        enhanced = classification.copy()
        enhanced["gemini_enhanced"] = True
        enhanced["gemini_insights"] = response.text
        enhanced["ai_provider"] = "Gemini"
        
        return enhanced
        
    except Exception as e:
        print(f"Gemini enhancement failed: {e}")
        return classification


def enhance_with_gemini_fallback(text: str, result: Dict, model) -> Dict:
    """Use Gemini AI to re-classify unknown documents with good confidence."""
    try:
        confidence = result.get("confidence", 0.0)
        
        prompt = f"""
        You are a document classification expert. This document was initially classified as "unknown" but has a confidence score of {confidence:.2f}, suggesting it might be classifiable.

        Document text: {text[:2000]}

        Please analyze this document and determine what type it is. Choose from these categories:
        - invoice: Bills, receipts, payment requests, commercial invoices, tax invoices, service bills, proforma invoices
        - bank_statement: Bank account statements, transaction records, monthly statements, credit card statements
        - resume: CV, job applications, professional profiles, curriculum vitae, bio-data, career summaries
        - ITR: Income tax returns, tax documents, assessment forms, tax filing documents
        - government_id: Passport, driver's license, national ID, Aadhaar, voter ID, PAN card, government issued documents
        - unknown: If truly unclassifiable

        Look for key indicators:
        - Invoice: "invoice", "bill", "total due", "payment terms", "GST", "amount", "subtotal"
        - Bank Statement: "account", "transaction", "debit", "credit", "balance", "statement", "bank"
        - Resume: "experience", "education", "skills", "projects", "career", "professional", "work"
        - ITR: "income tax", "PAN", "assessment year", "tax paid", "deductions", "ITR"
        - Government ID: "government", "ID card", "issued by", "date of birth", "passport", "license"

        Respond with:
        1. Document type (one of the categories above)
        2. Confidence level (0.0 to 1.0)
        3. Brief explanation of why you classified it this way
        
        Format: TYPE: [category] | CONFIDENCE: [0.0-1.0] | REASON: [explanation]
        """
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=300,
            )
        )
        
        if response and response.text:
            # Parse the response
            response_text = response.text.strip()
            result["ai_insights"] = f"LLM Fallback Analysis: {response_text}"
            result["ai_enhanced"] = True
            
            # Try to extract new classification from response
            if "TYPE:" in response_text and "CONFIDENCE:" in response_text:
                try:
                    lines = response_text.split('\n')
                    for line in lines:
                        if "TYPE:" in line:
                            new_type = line.split("TYPE:")[1].split("|")[0].strip().lower()
                            if new_type in ["invoice", "bank_statement", "resume", "itr", "government_id"]:
                                result["label"] = new_type
                                result["llm_fallback"] = True
                                result["original_label"] = "unknown"
                                break
                        if "CONFIDENCE:" in line:
                            conf_part = line.split("CONFIDENCE:")[1].split("|")[0].strip()
                            try:
                                new_confidence = float(conf_part)
                                result["confidence"] = new_confidence
                            except ValueError:
                                pass
                except Exception:
                    pass
        else:
            result["ai_insights"] = "LLM fallback analysis unavailable"
            result["ai_enhanced"] = False
            
    except Exception as e:
        result["ai_insights"] = f"LLM fallback failed: {str(e)}"
        result["ai_enhanced"] = False
    
    return result

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    /* Custom metric styling */
    .metric-label {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        color: #666 !important;
    }
    .metric-value {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }
    /* Override Streamlit metric styling */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    [data-testid="metric-container"] > div {
        font-size: 0.9rem !important;
    }
    [data-testid="metric-container"] > div > div {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>PDF Document Classifier</h1>
    <p>Upload PDFs to automatically classify them into categories like invoice, bank statement, resume, ITR, or government ID</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_model_and_centroids(class_examples_folder: str):
    """Cache the model and centroids to avoid reloading on every interaction."""
    model = _get_embedding_model()
    centroids = build_class_centroids(class_examples_folder, model=model)
    return model, centroids

def get_centroids_hash(centroids: dict) -> str:
    """Generate a hash for centroids to use as cache key."""
    centroids_str = json.dumps({k: v.tolist() for k, v in centroids.items()}, sort_keys=True)
    return hashlib.md5(centroids_str.encode()).hexdigest()

@st.cache_data(show_spinner=True, ttl=3600)  # Cache for 1 hour
def process_single_pdf(file_content: bytes, centroids_hash: str, ocr_dpi: int = 300, ocr_lang: str = 'eng+hin') -> dict:
    """Cache individual PDF processing  based on file content and centroids."""
    # Get the model and centroids (these are cached separately)
    model, centroids = get_model_and_centroids(examples_dir)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name
    
    try:
        # Extract text with PyMuPDF first
        text, method = extract_text_with_pymupdf(tmp_path)
    except Exception:
        text, method = "", ""
    
    # Enhanced OCR fallback with user settings
    if len(text.strip()) < 100:
        try:
            text, method = extract_text_with_ocr(tmp_path, dpi=ocr_dpi, lang=ocr_lang)
        except Exception:
            pass
    
    # Classify
    result = classify_text(text, centroids, model=model)
    result["method"] = "embedding"
    if method and "ocr" in method.lower():
        result["method"] += "+OCR"
    result["extracted_chars"] = len(text)
    result["extracted_text"] = text  # Include the actual extracted text
    result["ocr_settings"] = {"dpi": ocr_dpi, "language": ocr_lang}
    
    # Enhance with Gemini AI for medium confidence cases (0.3-0.7) and unknown fallback
    confidence = result.get("confidence", 0.0)
    label = result.get("label", "unknown")
    
    # Use LLM for medium confidence OR for unknown documents with medium/high confidence
    should_use_llm = (
        enable_llm_enhancement and gemini_model and len(text.strip()) > 50 and (
            (0.3 <= confidence <= 0.7) or  # Medium confidence range
            (label == "unknown" and confidence >= 0.3) or  # Unknown but medium/high confidence fallback
            (label == "unknown" and "ambiguous" in result.get("rationale", ""))  # Ambiguous cases with good confidence
        )
    )
    
    if should_use_llm:
        if label == "unknown" and confidence >= 0.3:
            # Special fallback for unknown documents with good confidence
            result = enhance_with_gemini_fallback(text, result, gemini_model)
            result["llm_reason"] = f"Unknown classification with {confidence:.2f} confidence - using LLM fallback"
        else:
            # Regular medium confidence enhancement
            result = enhance_with_gemini(text, result, gemini_model)
            result["llm_reason"] = f"Medium confidence ({confidence:.2f}) - using LLM for enhancement"
        
        # Mark LLM involvement
        result["method"] += "+LLM"
        result["ai_provider"] = "Gemini"
    elif confidence > 0.7 and label != "unknown":
        result["llm_reason"] = f"High confidence ({confidence:.2f}) - LLM not needed"
    elif confidence < 0.3:
        result["llm_reason"] = f"Low confidence ({confidence:.2f}) - LLM not used (unknown classification)"
    elif label == "unknown" and confidence < 0.3:
        result["llm_reason"] = f"Unknown classification with low confidence ({confidence:.2f}) - LLM not used"
    
    # Cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    
    return result

# Initialize session state for processing history
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

# Sidebar configuration
st.sidebar.header("Configuration")

# Class examples folder configuration (hidden from UI)
default_examples = os.path.join(os.path.dirname(__file__), "class_examples")
examples_dir = default_examples

# LLM Configuration
st.sidebar.header("AI Features")

# Check Gemini availability - support both local env and Streamlit secrets
gemini_api_key = None

# Try to get API key from Streamlit secrets first (for cloud deployment)
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
except (KeyError, FileNotFoundError):
    # Fallback to environment variable (for local development)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

if GEMINI_AVAILABLE and gemini_api_key:
    st.sidebar.success("AI Powered LLM Fallback")
    # Configure Gemini
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    enable_llm_enhancement = True
    # enable_llm_fields removed - not used in current version
else:
    st.sidebar.warning("AI Not Available")
    enable_llm_enhancement = False
    # enable_llm_fields removed - not used in current version
    gemini_model = None

# OCR Configuration
st.sidebar.header("OCR Settings")
ocr_dpi = st.sidebar.slider("OCR DPI", min_value=150, max_value=600, value=300, help="Higher DPI = better quality but slower")
ocr_language = st.sidebar.selectbox("OCR Language", ["eng+hin", "eng", "hin"], help="Language for OCR processing")

# Load model and centroids
with st.spinner("Loading AI model and building class centroids..."):
    model, centroids = get_model_and_centroids(examples_dir)

# Help section
st.sidebar.header("How to Use")
st.sidebar.info("""
**Quick Start:**
1. Upload a PDF file
2. Get instant classification
3. View analysis details

**Supported Documents:**
• Invoices • Bank Statements • Resumes • ITR Forms • Government IDs
""")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload PDF Document")
    uploaded = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="Upload a PDF document to classify it automatically"
    )

with col2:
    st.header("ℹ️ About")
    st.info("""
    **Supported Document Types:**
    - 📄 Invoice
    - 🏦 Bank Statement  
    - 👤 Resume/CV
    - 📊 ITR (Income Tax Return)
    - 🆔 Government ID
    
    **Features:**
    - AI-powered classification
    - LLM fallback for medium confidence
    - OCR fallback support
    - Confidence scoring
    - Batch processing
    - Multilingual support (English + Hindi)
    - Real-time processing
    - JSON export
    """)

if uploaded is not None:
    # Processing with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Processing PDF...")
        progress_bar.progress(25)
        
        # Use cached processing function with OCR settings
        centroids_hash = get_centroids_hash(centroids)
        result = process_single_pdf(uploaded.read(), centroids_hash, ocr_dpi, ocr_language)
        # fields extraction removed - not used in current version
        
        status_text.text("Classification complete!")
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        # Visuals
        label = result.get("label", "unknown")
        confidence = float(result.get("confidence", 0.0))
        rationale = result.get("rationale", "")
        top_scores = result.get("top_scores", {})
        
        # Add to processing history
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": uploaded.name,
            "classification": label,
            "confidence": confidence,
            "method": result.get("method", ""),
            "text_length": result.get("extracted_chars", 0)
        }
        st.session_state.processing_history.append(history_entry)
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Error processing PDF: {str(e)}")
        st.info("Please try uploading a different PDF file or check if the file is not corrupted.")
        st.stop()

    # Results section
    st.header("Classification Results")
    
    # Main result display
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Badge color by confidence bucket
        if label == "unknown" or confidence < 0.30:
            color = "#dc3545"
            conf_class = "confidence-low"
        elif confidence <= 0.70:
            color = "#ffc107"
            conf_class = "confidence-medium"
        else:
            color = "#28a745"
            conf_class = "confidence-high"
        
        # Check if this was an LLM fallback classification
        fallback_indicator = ""
        if result.get("llm_fallback"):
            fallback_indicator = " (LLM Fallback)"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: {color};">{label.title()}{fallback_indicator}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Document Classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Confidence", f"{confidence:.1%}", delta=None)
        st.progress(min(max(confidence, 0.0), 1.0))
        
        # Show LLM reasoning if available
        llm_reason = result.get("llm_reason", "")
        if llm_reason:
            if "High confidence" in llm_reason:
                st.info(f"Note: {llm_reason}")
            elif "Medium confidence" in llm_reason:
                st.success(f"AI Enhancement: {llm_reason}")
            elif "Low confidence" in llm_reason:
                st.warning(f"Warning: {llm_reason}")
    
    with col3:
        # Custom styled metrics with controlled font sizes
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 1rem;">
            <div class="metric-label">Extracted Text</div>
            <div class="metric-value">{result.get('extracted_chars',0):,} chars</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Method</div>
            <div class="metric-value">{result.get('method','').title()}</div>
        </div>
        """, unsafe_allow_html=True)

    # Skip detailed visualizations - user requested removal

    # Rationale and details
    st.subheader("🔍 Analysis Details")
    st.info(f"**Rationale:** {rationale}")
    
    # Extracted fields section removed - user requested removal
    
    # AI Insights section removed - user requested removal

    # Tabs for additional information
    tab1, tab2, tab3 = st.tabs(["Extracted Text", "Raw Data", "Download"])
    
    with tab1:
        # Get text from result or show placeholder
        extracted_text = result.get("extracted_text", "No text available")
        st.text_area("Full extracted text", value=extracted_text[:100000], height=300, help="This is the text extracted from your PDF document")
    
    with tab2:
        st.json(result)
    
    with tab3:
        st.download_button(
            label="Download JSON Result",
            data=json.dumps(result, indent=2).encode("utf-8"),
            file_name=f"classification_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download the complete classification results as a JSON file"
        )

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

# Batch processing section
st.header("Batch Processing")
st.write("Upload multiple PDFs for batch classification")

uploaded_files = st.file_uploader(
    "Choose multiple PDF files", 
    type=["pdf"],
    accept_multiple_files=True,
    help="Select multiple PDF files to process them all at once"
)

if uploaded_files:
    st.write(f"📊 Processing {len(uploaded_files)} files...")
    
    # Create a container for batch results
    batch_results = []
    
    # Process each file with progress tracking
    progress_container = st.container()
    with progress_container:
        batch_progress = st.progress(0)
        batch_status = st.empty()
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        batch_status.text(f"Processing {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
        batch_progress.progress((i + 1) / len(uploaded_files))
        
        with st.expander(f"📄 {uploaded_file.name}", expanded=False):
            try:
                # Use cached processing function with OCR settings
                centroids_hash = get_centroids_hash(centroids)
                result = process_single_pdf(uploaded_file.read(), centroids_hash, ocr_dpi, ocr_language)
                result["filename"] = uploaded_file.name
                
                batch_results.append(result)
                
                # Display result for this file
                label = result.get("label", "unknown")
                confidence = float(result.get("confidence", 0.0))
                text_length = result.get("extracted_chars", 0)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    fallback_text = " (LLM Fallback)" if result.get("llm_fallback") else ""
                    st.metric("Classification", f"{label.title()}{fallback_text}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col3:
                    st.metric("Text Length", f"{text_length:,} chars")
                
                # Show success indicator
                st.success("Processed successfully")
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                # Add error result to batch
                error_result = {
                    "filename": uploaded_file.name,
                    "label": "error",
                    "confidence": 0.0,
                    "rationale": f"Processing error: {str(e)}",
                    "method": "error",
                    "extracted_chars": 0
                }
                batch_results.append(error_result)
    
    # Clear progress indicators
    batch_progress.empty()
    batch_status.empty()
    
    # Batch summary
    if batch_results:
        st.subheader("📈 Batch Summary")
        
        # Create summary DataFrame
        summary_data = []
        for result in batch_results:
            summary_data.append({
                'Filename': result['filename'],
                'Classification': result['label'].title(),
                'Confidence': f"{result['confidence']:.1%}",
                'Method': result['method'].title(),
                'Text Length': result['extracted_chars']
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Classification distribution
        classification_counts = pd.DataFrame(summary_data)['Classification'].value_counts()
        if len(classification_counts) > 0:
            fig_dist = px.pie(
                values=classification_counts.values,
                names=classification_counts.index,
                title="Classification Distribution"
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Download batch results
        batch_json = {
            "batch_summary": {
                "total_files": len(batch_results),
                "processed_at": datetime.now().isoformat(),
                "results": batch_results
            }
        }
        
        st.download_button(
            label="Download Batch Results",
            data=json.dumps(batch_json, indent=2).encode("utf-8"),
            file_name=f"batch_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            help="Download all batch classification results as a JSON file"
        )

# Footer with deployment info
cloud_info = get_cloud_info()
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🤖 Powered by AI Document Classification | Built with Streamlit</p>
    <p>☁️ Deployed on {cloud_info['deployment']} | Version {cloud_info['version']}</p>
    <p>📧 For support or feature requests, please contact the development team</p>
</div>
""", unsafe_allow_html=True)


