# Configuration file for PDF Document Classifier
# Copy this file to config.py and add your API keys

# OpenAI API Configuration
# Get your API key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "sk-your-actual-api-key-here"

# Optional: Customize LLM model
OPENAI_MODEL = "gpt-3.5-turbo"  # or "gpt-4" for better results

# OCR Configuration
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_LANGUAGE = "eng"

# Classification thresholds
LOW_CONFIDENCE_THRESHOLD = 0.6
HIGH_CONFIDENCE_THRESHOLD = 0.7
