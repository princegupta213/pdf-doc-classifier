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
from typing import Dict, List

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


def detect_document_language(text: str) -> str:
    """Detect the primary language of the document text."""
    if not text or len(text.strip()) < 10:
        return "eng"  # Default to English
    
    # Simple language detection based on character patterns
    text_lower = text.lower()
    
    # Hindi detection (Devanagari script)
    hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
    total_chars = len([c for c in text if c.isalpha()])
    
    if total_chars > 0 and (hindi_chars / total_chars) > 0.1:
        return "hin"
    elif total_chars > 0 and (hindi_chars / total_chars) > 0.05:
        return "eng+hin"  # Mixed content
    else:
        return "eng"  # Primarily English

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
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 1.8rem;
    }
    .main-header p {
        color: #f0f0f0;
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
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
    <p>AI-powered document classification for invoices, bank statements, resumes, ITR, and government IDs</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def get_model_and_centroids(class_examples_folder: str, custom_training_examples: Dict[str, List[str]] = None):
    """Cache the model and centroids to avoid reloading on every interaction."""
    model = _get_embedding_model()
    centroids = build_class_centroids(class_examples_folder, model=model)
    
    # Add custom centroids if provided
    if custom_training_examples:
        custom_centroids = build_custom_centroids(custom_training_examples, model=model)
        centroids.update(custom_centroids)
    
    return model, centroids

def get_centroids_hash(centroids: dict) -> str:
    """Generate a hash for centroids to use as cache key."""
    centroids_str = json.dumps({k: v.tolist() for k, v in centroids.items()}, sort_keys=True)
    return hashlib.md5(centroids_str.encode()).hexdigest()

@st.cache_data(show_spinner=True, ttl=3600)  # Cache for 1 hour
def process_single_pdf(file_content: bytes, centroids_hash: str, ocr_dpi: int = 300, ocr_lang: str = 'eng+hin', custom_training_examples: Dict[str, List[str]] = None) -> dict:
    """Cache individual PDF processing  based on file content and centroids."""
    # Get the model and centroids (these are cached separately)
    model, centroids = get_model_and_centroids(examples_dir, custom_training_examples)
    
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

# Help section - moved to top
st.sidebar.header("How to Use")
st.sidebar.info("""
**Quick Start:**
1. Upload a PDF file
2. Get instant classification
3. View analysis details
""")

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
    
    # LLM Fallback Toggle
    enable_llm_enhancement = st.sidebar.checkbox(
        "Enable AI LLM Fallback", 
        value=True, 
        help="Use AI to enhance classification for medium confidence cases and unknown documents"
    )
else:
    st.sidebar.warning("AI Not Available")
    enable_llm_enhancement = False
    gemini_model = None

# OCR Configuration
st.sidebar.header("OCR Settings")
ocr_dpi = st.sidebar.slider("OCR DPI", min_value=150, max_value=600, value=300, help="Higher DPI = better quality but slower")
ocr_language = st.sidebar.selectbox("OCR Language", ["eng+hin", "eng", "hin", "auto-detect"], help="Language for OCR processing")

# Custom Categories
st.sidebar.header("Custom Categories")
if st.sidebar.button("Add New Category"):
    st.session_state.show_add_category = True

if st.session_state.get("show_add_category", False):
    new_category = st.sidebar.text_input("Category Name", placeholder="e.g., contract, receipt")
    if st.sidebar.button("Save Category"):
        if new_category and new_category.strip():
            # Add to session state
            if 'custom_categories' not in st.session_state:
                st.session_state.custom_categories = []
            st.session_state.custom_categories.append(new_category.strip().lower())
            st.sidebar.success(f"Added category: {new_category}")
            st.session_state.show_add_category = False
            st.rerun()

# Display custom categories with training examples management
if 'custom_categories' in st.session_state and st.session_state.custom_categories:
    st.sidebar.write("**Your Categories:**")
    for cat in st.session_state.custom_categories:
        col1, col2, col3 = st.sidebar.columns([2, 1, 1])
        with col1:
            st.sidebar.write(f"• {cat.title()}")
        with col2:
            if st.sidebar.button("📚", key=f"train_{cat}", help="Add training examples"):
                st.session_state.selected_custom_category = cat
                st.session_state.show_training_upload = True
        with col3:
            if st.sidebar.button("×", key=f"del_{cat}"):
                st.session_state.custom_categories.remove(cat)
                # Also remove training examples if they exist
                if 'custom_training_examples' in st.session_state:
                    st.session_state.custom_training_examples.pop(cat, None)
                st.rerun()

# Training examples upload for custom categories
if st.session_state.get("show_training_upload", False):
    st.sidebar.header("📚 Add Training Examples")
    selected_cat = st.session_state.get("selected_custom_category", "")
    st.sidebar.write(f"**Category:** {selected_cat.title()}")
    
    # Upload training examples
    training_files = st.sidebar.file_uploader(
        "Upload training examples (PDF files)",
        type=['pdf'],
        accept_multiple_files=True,
        key="custom_training_upload",
        help="Upload 5-10 PDF examples of this document type for better classification"
    )
    
    if training_files:
        if st.sidebar.button("Process Training Examples"):
            with st.sidebar.spinner("Processing training examples..."):
                # Initialize custom training examples storage
                if 'custom_training_examples' not in st.session_state:
                    st.session_state.custom_training_examples = {}
                
                # Process each uploaded file
                examples_text = []
                for file in training_files:
                    try:
                        # Extract text from PDF
                        text = extract_text_from_pdf(file, ocr_dpi, "eng+hin")
                        if text.strip():
                            examples_text.append(text.strip())
                    except Exception as e:
                        st.sidebar.error(f"Error processing {file.name}: {str(e)}")
                
                if examples_text:
                    # Store training examples
                    st.session_state.custom_training_examples[selected_cat] = examples_text
                    st.sidebar.success(f"Added {len(examples_text)} training examples for {selected_cat}")
                    st.session_state.show_training_upload = False
                    st.rerun()
                else:
                    st.sidebar.error("No text extracted from uploaded files")
    
    if st.sidebar.button("Cancel"):
        st.session_state.show_training_upload = False
        st.rerun()

# Display training examples count
if 'custom_training_examples' in st.session_state and st.session_state.custom_training_examples:
    st.sidebar.write("**Training Examples:**")
    for cat, examples in st.session_state.custom_training_examples.items():
        st.sidebar.write(f"• {cat.title()}: {len(examples)} examples")

# Classification History Dashboard
st.sidebar.header("📊 History Dashboard")
if st.sidebar.button("View Processing History"):
    st.session_state.show_history = True

if st.session_state.get("show_history", False):
    if st.session_state.processing_history:
        st.sidebar.write("**Recent Classifications:**")
        for i, entry in enumerate(st.session_state.processing_history[-5:]):  # Show last 5
            timestamp = entry['timestamp'][:19].replace('T', ' ')
            st.sidebar.write(f"**{entry['filename']}**")
            st.sidebar.write(f"→ {entry['classification'].title()}")
            st.sidebar.write(f"Confidence: {entry['confidence']:.1%}")
            st.sidebar.write(f"Time: {timestamp}")
            st.sidebar.write("---")
    else:
        st.sidebar.info("No processing history yet")
    
    if st.sidebar.button("Clear History"):
        st.session_state.processing_history = []
        st.rerun()

# Load model and centroids
with st.spinner("Loading AI model and building class centroids..."):
    model, centroids = get_model_and_centroids(examples_dir)


# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Upload PDF Document")
    uploaded = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"],
        help="Upload a PDF document to classify it automatically"
    )
    
    # Add visual spacing
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    
    # Batch processing section - moved here
    st.header("Batch Processing")
    st.write("Upload multiple PDFs for batch classification")
    
    uploaded_files = st.file_uploader(
        "Choose multiple PDF files", 
        type=["pdf"],
        accept_multiple_files=True,
        help="Select multiple PDF files to process them all at once"
    )
    
    # Batch processing logic
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
                    # Store PDF content for potential review queue use
                    file_pdf_content = uploaded_file.read()
                    uploaded_file.seek(0)  # Reset file pointer for processing
                    
                    # Use cached processing function with OCR settings
                    centroids_hash = get_centroids_hash(centroids)
                    result = process_single_pdf(file_pdf_content, centroids_hash, ocr_dpi, ocr_language, st.session_state.get('custom_training_examples', {}))
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
                    
                    # Add to review queue if ambiguous classification (same logic as single upload)
                    rationale = result.get("rationale", "")
                    is_ambiguous = "ambiguous: margin < 0.10" in rationale or "margin < 0.10" in rationale
                    
                    if is_ambiguous:  # Only ambiguous classification (margin < 10%)
                        # Check if this document is already in the review queue to prevent duplicates
                        if 'review_queue' not in st.session_state:
                            st.session_state.review_queue = []
                        
                        # Check if document is already in queue
                        already_in_queue = any(item['filename'] == uploaded_file.name for item in st.session_state.review_queue)
                        
                        if not already_in_queue:
                            review_item = {
                                "filename": uploaded_file.name,
                                "classification": label,
                                "confidence": confidence,
                                "rationale": rationale,
                                "timestamp": datetime.now().isoformat(),
                                "method": result.get("method", ""),
                                "pdf_content": file_pdf_content  # Use stored PDF content for preview
                            }
                            st.session_state.review_queue.append(review_item)
                            st.warning(f"⚠️ Ambiguous classification (margin < 10%) added to review queue")
                    
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

    # Manual Review Queue
    st.markdown("<br>", unsafe_allow_html=True)
    st.header("🔍 Manual Review Queue")
    
    # Initialize review queue in session state
    if 'review_queue' not in st.session_state:
        st.session_state.review_queue = []
    
    # Debug: Show current review queue status
    st.write(f"🔍 **Review Queue Status:** {len(st.session_state.review_queue)} documents")
    
    # Show review queue
    if st.session_state.review_queue:
        st.write(f"**{len(st.session_state.review_queue)} documents need review:**")
        
        # Create a copy of the queue to avoid index issues when removing items
        queue_items = list(st.session_state.review_queue)
        
        for i, item in enumerate(queue_items):
            with st.expander(f"📄 {item['filename']} - {item['classification'].title()} ({item['confidence']:.1%})", expanded=False):
                # Create tabs for different views
                tab1, tab2 = st.tabs(["📋 Review Details", "👁️ PDF Preview"])
                
                with tab1:
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Current Classification:** {item['classification'].title()}")
                        st.write(f"**Confidence:** {item['confidence']:.1%}")
                        st.write(f"**Reason:** {item.get('rationale', 'No reason provided')}")
                        
                        # Manual correction
                        st.write("**Correct Classification:**")
                        corrected_class = st.selectbox(
                            "Select correct category:",
                            ["invoice", "bank_statement", "resume", "ITR", "government_id", "unknown"] + 
                            (st.session_state.get('custom_categories', [])),
                            key=f"correct_{item['filename']}_{i}"  # Use filename to make key unique
                        )
                        
                        if st.button(f"Update Classification", key=f"update_{item['filename']}_{i}"):
                            # Update the classification
                            item['corrected_classification'] = corrected_class
                            item['manually_reviewed'] = True
                            item['review_timestamp'] = datetime.now().isoformat()
                            
                            # Remove from queue by finding the item
                            st.session_state.review_queue = [x for x in st.session_state.review_queue if x != item]
                            st.success(f"Updated classification to: {corrected_class.title()}")
                            st.rerun()
                    
                    with col2:
                        if st.button(f"Remove from Queue", key=f"remove_{item['filename']}_{i}"):
                            # Remove from queue by finding the item
                            st.session_state.review_queue = [x for x in st.session_state.review_queue if x != item]
                            st.rerun()
                
                with tab2:
                    # PDF Preview
                    if 'pdf_content' in item and item['pdf_content']:
                        try:
                            # Display PDF using Streamlit's PDF viewer
                            st.pdf(item['pdf_content'])
                        except Exception as e:
                            # Fallback: Show PDF info and download option
                            st.warning("PDF preview not available. PDF viewer component not installed.")
                            st.info("To enable PDF preview, install: `pip install streamlit[pdf]`")
                            
                            # Show PDF information
                            st.write(f"**PDF File:** {item['filename']}")
                            st.write(f"**File Size:** {len(item['pdf_content']):,} bytes")
                            
                            # Provide download option
                            st.download_button(
                                label="📥 Download PDF",
                                data=item['pdf_content'],
                                file_name=item['filename'],
                                mime="application/pdf",
                                help="Download the PDF to view it locally"
                            )
                    else:
                        st.warning("PDF content not available for preview.")
    else:
        st.info("No documents in review queue. Ambiguous classifications (margin < 10%) will appear here automatically.")

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
    # Store PDF content for potential review queue use
    pdf_content = uploaded.read()
    uploaded.seek(0)  # Reset file pointer for processing
    
    # Processing with progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🔄 Processing PDF...")
        progress_bar.progress(25)
        
        # Handle auto-detect language
        actual_ocr_language = ocr_language
        if ocr_language == "auto-detect":
            # First extract text to detect language
            temp_result = process_single_pdf(uploaded.read(), get_centroids_hash(centroids), ocr_dpi, "eng", st.session_state.get('custom_training_examples', {}))
            detected_lang = detect_document_language(temp_result.get("extracted_text", ""))
            actual_ocr_language = detected_lang
            st.info(f"🔍 Auto-detected language: {detected_lang}")
        
        # Use cached processing function with OCR settings
        centroids_hash = get_centroids_hash(centroids)
        result = process_single_pdf(uploaded.read(), centroids_hash, ocr_dpi, actual_ocr_language, st.session_state.get('custom_training_examples', {}))
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
        
        # Add to review queue if ambiguous classification
        is_ambiguous = "ambiguous: margin < 0.10" in rationale or "margin < 0.10" in rationale
        
        
        if is_ambiguous:  # Only ambiguous classification (margin < 10%)
            # Check if this document is already in the review queue to prevent duplicates
            if 'review_queue' not in st.session_state:
                st.session_state.review_queue = []
            
            # Check if document is already in queue
            already_in_queue = any(item['filename'] == uploaded.name for item in st.session_state.review_queue)
            
            if not already_in_queue:
                review_item = {
                    "filename": uploaded.name,
                    "classification": label,
                    "confidence": confidence,
                    "rationale": rationale,
                    "timestamp": datetime.now().isoformat(),
                    "method": result.get("method", ""),
                    "pdf_content": pdf_content  # Use stored PDF content for preview
                }
                st.session_state.review_queue.append(review_item)
                st.warning(f"⚠️ Ambiguous classification (margin < 10%) added to review queue")
        
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


