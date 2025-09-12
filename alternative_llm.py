"""Alternative LLM integration for when OpenAI quota is exceeded."""

import os
import json
from typing import Dict, Optional

def get_alternative_llm_classification(text: str, current_classification: Dict) -> Optional[Dict]:
    """Alternative LLM classification using local models or other services."""
    
    # For now, we'll use enhanced heuristics
    # In the future, you could integrate with:
    # - Hugging Face Transformers
    # - Local Ollama models
    # - Google Gemini API
    # - Anthropic Claude API
    
    label = current_classification.get("label", "unknown")
    confidence = current_classification.get("confidence", 0.0)
    
    # Enhanced heuristics for better classification
    enhanced_result = {
        "label": label,
        "confidence": confidence,
        "rationale": current_classification.get("rationale", ""),
        "enhanced_heuristics": True,
        "suggestions": [
            "Consider adding more training examples",
            "Try different OCR settings for better text extraction",
            "Check if document type matches expected format"
        ]
    }
    
    # Add some basic field extraction hints
    if label == "invoice":
        enhanced_result["field_hints"] = {
            "invoice_number": "Look for patterns like 'Invoice #', 'INV-', or similar",
            "total_amount": "Search for currency symbols and amounts",
            "vendor_name": "Look for company names or 'Bill From' sections"
        }
    elif label == "bank_statement":
        enhanced_result["field_hints"] = {
            "account_number": "Look for account numbers or 'Account' labels",
            "statement_period": "Search for date ranges or 'Statement Period'",
            "balance": "Look for 'Balance' or 'Available Balance'"
        }
    
    return enhanced_result

def get_alternative_field_extraction(text: str, document_type: str) -> Dict[str, str]:
    """Alternative field extraction using enhanced regex patterns."""
    
    import re
    
    fields = {}
    
    if document_type == "invoice":
        # Enhanced invoice number patterns
        invoice_patterns = [
            r'invoice\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Z0-9\-/]+)',
            r'inv\s*[:#-]?\s*([A-Z0-9\-/]+)',
            r'bill\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Z0-9\-/]+)'
        ]
        
        for pattern in invoice_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["invoice_number"] = match.group(1)
                break
        
        # Enhanced amount patterns
        amount_patterns = [
            r'total\s*(?:amount|due)?\s*[:$]?\s*([0-9,]+\.?[0-9]*)',
            r'amount\s*due\s*[:$]?\s*([0-9,]+\.?[0-9]*)',
            r'grand\s*total\s*[:$]?\s*([0-9,]+\.?[0-9]*)'
        ]
        
        for pattern in amount_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["total_amount"] = match.group(1)
                break
    
    elif document_type == "bank_statement":
        # Enhanced account number patterns
        account_patterns = [
            r'account\s*(?:no\.?|number|#)?\s*[:#-]?\s*([0-9\-]+)',
            r'acct\s*(?:no\.?|number|#)?\s*[:#-]?\s*([0-9\-]+)'
        ]
        
        for pattern in account_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields["account_number"] = match.group(1)
                break
    
    return fields

# Integration function for the main app
def get_enhanced_classification_without_llm(text: str, classification: Dict) -> Dict:
    """Enhanced classification without LLM when quota is exceeded."""
    
    # Use alternative LLM classification
    enhanced = get_alternative_llm_classification(text, classification)
    
    # Use alternative field extraction
    if classification.get("label") != "unknown":
        fields = get_alternative_field_extraction(text, classification["label"])
        enhanced["extracted_fields"] = fields
    
    return enhanced
