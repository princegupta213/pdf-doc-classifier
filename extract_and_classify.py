"""Core utilities to extract text from PDFs and classify documents.

Workflow: extract -> clean -> embed -> compare -> classify

- Extraction methods: PyMuPDF (native text) and OCR fallback (pdf2image + Tesseract)
- Cleaning: normalize whitespace and remove non-printable characters, lowercase
- Embeddings: SentenceTransformer "all-mpnet-base-v2"
- Class comparison: cosine similarity vs. centroid per class
- Heuristics: keyword boosting and confidence thresholds
"""

import os
import re
import json
from typing import Dict, List, Tuple

import numpy as np

from field_extraction import (
    extract_invoice_number,
    extract_pan_number,
    extract_account_number,
    extract_dob,
    extract_gov_id,
)

# Import LLM functionality
try:
    from llm_prompts import (
        get_llm_enhanced_classification,
        get_llm_field_extraction,
        enhance_with_llm,
        llm_manager
    )
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM prompts module not available. Install openai package for LLM features.")

def _lazy_imports():
    import importlib
    modules = {}
    modules["fitz"] = importlib.import_module("fitz")  # PyMuPDF
    modules["pdf2image"] = importlib.import_module("pdf2image")
    modules["pytesseract"] = importlib.import_module("pytesseract")
    modules["SentenceTransformer"] = getattr(importlib.import_module("sentence_transformers"), "SentenceTransformer")
    return modules


_MODEL_CACHE = {"model": None, "name": None}


def _get_embedding_model(model_name: str = "all-mpnet-base-v2"):
    """Load or reuse the sentence embedding model.

    Uses a simple module-level cache to avoid re-loading.
    """
    if _MODEL_CACHE["model"] is not None and _MODEL_CACHE["name"] == model_name:
        return _MODEL_CACHE["model"]
    modules = _lazy_imports()
    model = modules["SentenceTransformer"](model_name)
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["name"] = model_name
    return model


def extract_text_with_pymupdf(path: str) -> Tuple[str, str]:
    """Extract text using PyMuPDF.

    Returns a tuple of (text, method_name).
    """
    modules = _lazy_imports()
    fitz = modules["fitz"]
    text_chunks: List[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            text_chunks.append(page.get_text("text"))
    text = "\n".join(text_chunks).strip()
    return text, "pymupdf"


def extract_text_with_ocr(path: str, dpi: int = 300, lang: str = 'eng') -> Tuple[str, str]:
    """Enhanced OCR-based extraction using pdf2image + pytesseract.

    Returns a tuple of (text, method_name).
    """
    modules = _lazy_imports()
    pdf2image = modules["pdf2image"]
    pytesseract = modules["pytesseract"]
    
    try:
        # Convert PDF to images with higher quality
        images = pdf2image.convert_from_path(
            path, 
            dpi=dpi,
            first_page=None,
            last_page=None,
            fmt='jpeg',
            jpegopt={'quality': 95}
        )
        
        ocr_text: List[str] = []
        ocr_config = '--oem 3 --psm 6'  # OCR Engine Mode 3, Page Segmentation Mode 6
        
        for i, image in enumerate(images):
            try:
                # Enhanced OCR with better configuration
                page_text = pytesseract.image_to_string(
                    image, 
                    lang=lang,
                    config=ocr_config
                )
                ocr_text.append(page_text)
            except Exception as e:
                print(f"OCR error on page {i+1}: {e}")
                # Fallback to basic OCR
                try:
                    page_text = pytesseract.image_to_string(image)
                    ocr_text.append(page_text)
                except:
                    ocr_text.append("")
        
        text = "\n".join(ocr_text).strip()
        
        # Post-process OCR text
        text = _post_process_ocr_text(text)
        
        return text, "enhanced_ocr"
        
    except Exception as e:
        print(f"OCR extraction failed: {e}")
        return "", "ocr_failed"


def _post_process_ocr_text(text: str) -> str:
    """Post-process OCR text to improve quality."""
    if not text:
        return text
    
    # Fix common OCR errors
    replacements = {
        '|': 'I',  # Common OCR mistake
        '0': 'O',  # In certain contexts
        '5': 'S',  # In certain contexts
        '8': 'B',  # In certain contexts
    }
    
    # Apply replacements carefully (only in specific contexts)
    for old, new in replacements.items():
        # Only replace in specific patterns to avoid over-correction
        if old == '|' and '|' in text:
            # Replace | with I only when it looks like a letter
            import re
            text = re.sub(r'\b\|\b', 'I', text)
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


_SPACE_RE = re.compile(r"\s+")
_NON_PRINTABLE_RE = re.compile(r"[^\x09\x0A\x0D\x20-\x7E]")


def clean_text(text: str) -> str:
    """Normalize text: spaces, non-printable characters, casing."""
    if not text:
        return ""
    # Normalize whitespace and remove non-printable chars; lower for normalization
    text = text.replace("\u00a0", " ")
    text = _NON_PRINTABLE_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text)
    return text.strip().lower()


def _read_text_files(directory: str) -> List[str]:
    texts: List[str] = []
    if not os.path.isdir(directory):
        return texts
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".txt"):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                        txt = f.read()
                        texts.append(clean_text(txt))
                except Exception:
                    continue
    return texts


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_class_centroids(class_examples_folder: str, model=None) -> Dict[str, np.ndarray]:
    """Compute centroid embedding for each class from example `.txt` files.

    Expects structure: `class_examples/<class>/*.txt`.
    """
    if model is None:
        model = _get_embedding_model()
    centroids: Dict[str, np.ndarray] = {}
    if not os.path.isdir(class_examples_folder):
        return centroids
    for cls_name in sorted(os.listdir(class_examples_folder)):
        cls_dir = os.path.join(class_examples_folder, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        samples = _read_text_files(cls_dir)
        if not samples:
            continue
        embeddings = model.encode(samples, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        centroid = np.mean(embeddings, axis=0)
        # Normalize centroid
        norm = np.linalg.norm(centroid)
        if norm != 0:
            centroid = centroid / norm
        centroids[cls_name] = centroid.astype(np.float32)
    return centroids


def _keyword_boosts() -> Dict[str, List[str]]:
    return {
        "invoice": ["invoice", "total due", "subtotal", "bill to", "invoice #", "gst", "vat"],
        "bank_statement": ["account statement", "transaction", "debit", "credit", "balance", "ifsc", "swift", "account number"],
        "resume": ["resume", "curriculum vitae", "experience", "education", "skills", "projects", "summary"],
        "ITR": ["income tax return", "itr", "pan", "assessment year", "tax paid", "gross total income"],
        "government_id": ["government id", "id card", "issuing authority", "dob", "date of birth", "aadhaar", "passport", "driver", "voter"]
    }


def _apply_keyword_boosts(cleaned_text: str, raw_scores: Dict[str, float]) -> Dict[str, float]:
    boosts = _keyword_boosts()
    boosted = dict(raw_scores)
    for cls, words in boosts.items():
        count = 0
        for w in words:
            if w in cleaned_text:
                count += 1
        if count > 0:
            # Scale boost modestly to avoid overpowering embeddings
            boost = min(0.05 * count, 0.15)
            boosted[cls] = min(1.0, boosted.get(cls, 0.0) + boost)
    return boosted


def _confidence_bucket(score: float) -> str:
    if score > 0.70:
        return "high"
    if 0.45 <= score <= 0.70:
        return "medium"
    return "unknown"


def classify_text(text: str, centroids: Dict[str, np.ndarray], model=None) -> Dict:
    """Classify text against class centroids and return a result dict.

    Returns keys: `label`, `confidence`, `rationale`, `top_scores`, `method`.
    """
    if model is None:
        model = _get_embedding_model()
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "rationale": "Empty or unreadable text",
            "top_scores": {},
            "method": ""
        }

    embedding = model.encode([cleaned], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)[0]

    # Base scores via cosine similarity
    scores: Dict[str, float] = {}
    for cls, centroid in centroids.items():
        scores[cls] = _cosine_similarity(embedding, centroid)

    # Keyword boosting
    boosted_scores = _apply_keyword_boosts(cleaned, scores)

    # Ranking
    sorted_items = sorted(boosted_scores.items(), key=lambda kv: kv[1], reverse=True)
    if not sorted_items:
        return {
            "label": "unknown",
            "confidence": 0.0,
            "rationale": "No classes available",
            "top_scores": {},
            "method": ""
        }

    best_label, best_score = sorted_items[0]
    second_score = sorted_items[1][1] if len(sorted_items) > 1 else 0.0

    rationale_parts = [
        f"best={best_label}:{best_score:.3f}",
        f"second={second_score:.3f}",
    ]

    # Apply thresholds and tie-break
    if best_score < 0.45:
        label = "unknown"
        conf_bucket = "unknown"
        rationale_parts.append("below low threshold")
    elif (best_score - second_score) < 0.10:
        label = "unknown"
        conf_bucket = "unknown"
        rationale_parts.append("ambiguous: margin < 0.10")
    else:
        label = best_label
        conf_bucket = _confidence_bucket(best_score)

    result = {
        "label": label,
        "confidence": float(best_score),
        "rationale": "; ".join(rationale_parts),
        "top_scores": {k: float(v) for k, v in sorted_items[:5]},
        "method": ""
    }
    
    # LLM Enhancement for low confidence or unknown classifications
    if LLM_AVAILABLE and (label == "unknown" or best_score < 0.6):
        try:
            llm_result = enhance_with_llm(cleaned, result)
            if llm_result and llm_result.confidence > best_score:
                result.update({
                    "label": llm_result.label,
                    "confidence": llm_result.confidence,
                    "rationale": llm_result.rationale,
                    "llm_enhanced": True,
                    "llm_insights": llm_result.llm_insights,
                    "suggested_actions": llm_result.suggested_actions
                })
        except Exception as e:
            print(f"LLM enhancement failed: {e}")
            result["llm_error"] = str(e)
    
    return result


def extract_and_classify(pdf_path: str, class_examples_folder: str) -> Dict:
    """High-level helper: extract text (PDF), build/load centroids, classify."""
    model = _get_embedding_model()
    centroids = build_class_centroids(class_examples_folder, model=model)

    # Try PyMuPDF first
    text, method = "", ""
    try:
        text, method = extract_text_with_pymupdf(pdf_path)
    except Exception:
        text, method = "", ""

    # Fallback to OCR if needed
    if len(text.strip()) < 100:
        try:
            text, method = extract_text_with_ocr(pdf_path)
        except Exception:
            pass

    # Optional LLM fallback (disabled if no API key provided)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        # Noop: We simply log via rationale later that LLM fallback is disabled
        llm_note = "LLM fallback disabled"
    else:
        llm_note = "LLM fallback available (not invoked by default)"

    result = classify_text(text, centroids, model=model)
    result["method"] = method
    result["extracted_chars"] = len(text)
    # Add note about LLM fallback availability into rationale to surface clearly
    if result.get("rationale"):
        result["rationale"] = f"{result['rationale']}; {llm_note}"
    else:
        result["rationale"] = llm_note
    # Enhanced field extraction with LLM fallback
    fields: Dict[str, str] = {}
    label = result.get("label", "unknown")
    cleaned_text = clean_text(text)
    
    try:
        # Traditional regex-based extraction
        if label == "invoice":
            inv = extract_invoice_number(cleaned_text)
            if inv:
                fields["invoice_number"] = inv
        elif label == "ITR":
            pan = extract_pan_number(cleaned_text)
            if pan:
                fields["pan"] = pan
        elif label == "bank_statement":
            acc = extract_account_number(cleaned_text)
            if acc:
                fields["account_number"] = acc
        elif label == "government_id":
            dob = extract_dob(cleaned_text)
            gov = extract_gov_id(cleaned_text)
            if dob:
                fields["dob"] = dob
            if gov:
                fields["gov_id"] = gov
        
        # LLM-powered field extraction for better results
        if LLM_AVAILABLE and label != "unknown":
            try:
                llm_fields = get_llm_field_extraction(cleaned_text, label)
                if llm_fields:
                    # Merge LLM fields with regex fields (LLM takes precedence)
                    fields.update(llm_fields)
                    result["llm_fields_extracted"] = True
            except Exception as e:
                print(f"LLM field extraction failed: {e}")
                result["llm_field_error"] = str(e)
                
    except Exception as e:
        print(f"Field extraction error: {e}")
        result["field_extraction_error"] = str(e)
    
    result["fields"] = fields
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract and classify a PDF document.")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--class-examples", default=os.path.join(os.path.dirname(__file__), "class_examples"), help="Path to class_examples folder")
    args = parser.parse_args()
    out = extract_and_classify(args.file, args.class_examples)
    print(json.dumps(out, indent=2))


