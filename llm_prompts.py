"""LLM prompts and integration for enhanced document classification."""

import os
import json
from typing import Dict, List, Optional, Tuple
import openai
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Enhanced classification result with LLM insights."""
    label: str
    confidence: float
    rationale: str
    llm_insights: Optional[str] = None
    extracted_fields: Optional[Dict[str, str]] = None
    suggested_actions: Optional[List[str]] = None


class LLMPromptManager:
    """Manages LLM prompts for document classification and analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with OpenAI API key."""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        self.model = "gpt-3.5-turbo"  # Can be upgraded to gpt-4
    
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return bool(self.api_key)
    
    def get_classification_prompt(self, text: str, current_classification: Dict) -> str:
        """Generate prompt for document classification."""
        return f"""
You are an expert document classifier. Analyze the following document text and provide insights about its classification.

CURRENT CLASSIFICATION:
- Label: {current_classification.get('label', 'unknown')}
- Confidence: {current_classification.get('confidence', 0.0):.2f}
- Rationale: {current_classification.get('rationale', '')}

DOCUMENT TEXT (first 2000 characters):
{text[:2000]}

Please provide:
1. Document type classification (invoice, bank_statement, resume, ITR, government_id, or other)
2. Confidence level (0.0 to 1.0)
3. Key evidence supporting this classification
4. Any important fields or information you can extract
5. Suggested next steps or actions

Respond in JSON format:
{{
    "classification": "document_type",
    "confidence": 0.85,
    "evidence": ["key evidence 1", "key evidence 2"],
    "extracted_fields": {{"field_name": "value"}},
    "suggestions": ["suggestion 1", "suggestion 2"]
}}
"""
    
    def get_field_extraction_prompt(self, text: str, document_type: str) -> str:
        """Generate prompt for field extraction based on document type."""
        field_templates = {
            "invoice": {
                "fields": ["invoice_number", "total_amount", "vendor_name", "date", "due_date"],
                "description": "Extract invoice-specific fields like invoice number, total amount, vendor details, and dates."
            },
            "bank_statement": {
                "fields": ["account_number", "bank_name", "statement_period", "opening_balance", "closing_balance"],
                "description": "Extract bank statement fields like account number, bank name, statement period, and balances."
            },
            "resume": {
                "fields": ["name", "email", "phone", "experience_years", "education", "skills"],
                "description": "Extract resume fields like contact information, experience, education, and skills."
            },
            "ITR": {
                "fields": ["pan_number", "assessment_year", "total_income", "tax_paid", "filing_date"],
                "description": "Extract ITR fields like PAN number, assessment year, income details, and tax information."
            },
            "government_id": {
                "fields": ["id_number", "name", "date_of_birth", "issuing_authority", "validity_date"],
                "description": "Extract government ID fields like ID number, name, date of birth, and issuing authority."
            }
        }
        
        template = field_templates.get(document_type, {
            "fields": ["key_information"],
            "description": "Extract any important information from the document."
        })
        
        return f"""
You are an expert at extracting structured information from {document_type} documents.

DOCUMENT TEXT:
{text[:3000]}

Please extract the following fields: {', '.join(template['fields'])}

{template['description']}

Respond in JSON format with only the fields you can identify:
{{
    "field_name": "extracted_value",
    "another_field": "another_value"
}}

If a field is not found, omit it from the response. Be precise and accurate.
"""
    
    def get_enhancement_prompt(self, text: str, classification: Dict) -> str:
        """Generate prompt for enhancing low-confidence classifications."""
        return f"""
You are an expert document analyst. The following document has been classified with low confidence.

CURRENT CLASSIFICATION:
- Label: {classification.get('label', 'unknown')}
- Confidence: {classification.get('confidence', 0.0):.2f}

DOCUMENT TEXT:
{text[:2500]}

Please provide:
1. A more confident classification
2. Detailed reasoning for your classification
3. Key indicators that led to your decision
4. Any potential alternative classifications
5. Suggestions for improving classification accuracy

Respond in JSON format:
{{
    "enhanced_classification": "document_type",
    "confidence": 0.90,
    "reasoning": "detailed explanation",
    "key_indicators": ["indicator 1", "indicator 2"],
    "alternatives": ["alt1", "alt2"],
    "improvements": ["suggestion 1", "suggestion 2"]
}}
"""
    
    def classify_with_llm(self, text: str, current_classification: Dict) -> Optional[ClassificationResult]:
        """Use LLM to enhance classification."""
        if not self.is_available():
            return None
        
        try:
            prompt = self.get_classification_prompt(text, current_classification)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document classifier. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                result = json.loads(result_text)
                return ClassificationResult(
                    label=result.get("classification", current_classification.get("label", "unknown")),
                    confidence=float(result.get("confidence", current_classification.get("confidence", 0.0))),
                    rationale=current_classification.get("rationale", "") + f" | LLM: {result.get('evidence', [])}",
                    llm_insights=json.dumps(result, indent=2),
                    extracted_fields=result.get("extracted_fields", {}),
                    suggested_actions=result.get("suggestions", [])
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return ClassificationResult(
                    label=current_classification.get("label", "unknown"),
                    confidence=current_classification.get("confidence", 0.0),
                    rationale=current_classification.get("rationale", "") + f" | LLM Response: {result_text[:200]}",
                    llm_insights=result_text
                )
                
        except Exception as e:
            print(f"LLM classification error: {e}")
            return None
    
    def extract_fields_with_llm(self, text: str, document_type: str) -> Optional[Dict[str, str]]:
        """Use LLM to extract fields from document."""
        if not self.is_available():
            return None
        
        try:
            prompt = self.get_field_extraction_prompt(text, document_type)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from documents. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                return {"llm_extraction": result_text[:200]}
                
        except Exception as e:
            print(f"LLM field extraction error: {e}")
            return None
    
    def enhance_low_confidence(self, text: str, classification: Dict) -> Optional[ClassificationResult]:
        """Use LLM to enhance low-confidence classifications."""
        if not self.is_available() or classification.get("confidence", 0) > 0.6:
            return None
        
        try:
            prompt = self.get_enhancement_prompt(text, classification)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1200,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result = json.loads(result_text)
                return ClassificationResult(
                    label=result.get("enhanced_classification", classification.get("label", "unknown")),
                    confidence=float(result.get("confidence", classification.get("confidence", 0.0))),
                    rationale=result.get("reasoning", classification.get("rationale", "")),
                    llm_insights=json.dumps(result, indent=2),
                    suggested_actions=result.get("improvements", [])
                )
            except json.JSONDecodeError:
                return ClassificationResult(
                    label=classification.get("label", "unknown"),
                    confidence=classification.get("confidence", 0.0),
                    rationale=classification.get("rationale", "") + f" | LLM Enhancement: {result_text[:200]}",
                    llm_insights=result_text
                )
                
        except Exception as e:
            print(f"LLM enhancement error: {e}")
            return None


# Global LLM manager instance
llm_manager = LLMPromptManager()


def get_llm_enhanced_classification(text: str, classification: Dict) -> Optional[ClassificationResult]:
    """Get LLM-enhanced classification."""
    return llm_manager.classify_with_llm(text, classification)


def get_llm_field_extraction(text: str, document_type: str) -> Optional[Dict[str, str]]:
    """Get LLM field extraction."""
    return llm_manager.extract_fields_with_llm(text, document_type)


def enhance_with_llm(text: str, classification: Dict) -> Optional[ClassificationResult]:
    """Enhance low-confidence classification with LLM."""
    return llm_manager.enhance_low_confidence(text, classification)
