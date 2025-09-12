"""Regex-based field extraction utilities for common document fields.

All matchers are case-insensitive and attempt to handle common variations.
Each function returns the first matched value as a string, or None if not found.
"""

import re
from typing import Optional


def _search(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    # Return last capturing group if present, else the whole match
    if match.lastindex:
        return match.group(match.lastindex).strip()
    return match.group(0).strip()


def extract_invoice_number(text: str) -> Optional[str]:
    """Extract invoice number after phrases like 'Invoice No', 'Invoice Number', 'Invoice', etc."""
    # Examples: Invoice No: ABC123, Invoice #: INV-2023-01, Invoice Number - A1B2C3
    pattern = r"\binvoice\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Z0-9\-/]+)\b"
    return _search(pattern, text)


def extract_pan_number(text: str) -> Optional[str]:
    """Extract Indian PAN format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)."""
    pattern = r"\b([A-Z]{5}[0-9]{4}[A-Z])\b"
    return _search(pattern, text)


def extract_account_number(text: str) -> Optional[str]:
    """Extract 9–18 digit account number following 'Account' or 'Account Number'."""
    pattern = r"\baccount(?:\s*number)?\s*[:#-]?\s*([0-9]{9,18})\b"
    return _search(pattern, text)


def extract_dob(text: str) -> Optional[str]:
    """Extract common date of birth formats: DD-MM-YYYY, DD/MM/YYYY, DD Mon YYYY."""
    # Try numeric formats first
    for pat in [
        r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b",
        r"\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{4})\b",
    ]:
        value = _search(pat, text)
        if value:
            return value
    return None


def extract_gov_id(text: str) -> Optional[str]:
    """Extract generic government ID references: ID No, Voter ID, Passport No, DL No."""
    patterns = [
        r"\b(?:id\s*(?:no\.?|number)?)\s*[:#-]?\s*([A-Z0-9\-]+)\b",
        r"\b(?:voter\s*id)\s*[:#-]?\s*([A-Z0-9\-]+)\b",
        r"\b(?:passport\s*(?:no\.?|number)?)\s*[:#-]?\s*([A-Z0-9\-]+)\b",
        r"\b(?:dl\s*(?:no\.?|number)?|driving\s*license\s*(?:no\.?|number)?)\s*[:#-]?\s*([A-Z0-9\-]+)\b",
    ]
    for pat in patterns:
        value = _search(pat, text)
        if value:
            return value
    return None


