# PDF Document Classifier

Classify PDFs into categories like invoice, bank statement, resume, ITR, and government_id using text extraction, cleaning, sentence embeddings, and cosine similarity with keyword boosting.

**Language Support:** English interface with Hindi OCR support

## Approach

1. Extract: Try PyMuPDF; fallback to OCR (pdf2image + Tesseract) if text is too short.
2. Clean: Normalize whitespace, strip non-printable characters, lowercase.
3. Embed: Use SentenceTransformer `all-mpnet-base-v2` to encode text.
4. Compare: Compute cosine similarity to each class centroid (average of example embeddings).
5. Classify: Apply keyword boosting and thresholds to choose label.
   - Score > 0.70 → high confidence
   - 0.45–0.70 → medium
   - < 0.45 → unknown
   - If (best − second_best) < 0.10 → unknown (ambiguous)

## Project Structure

```
pdf_doc_classifier/
├─ extract_and_classify.py       # Core extraction + classification
├─ app_demo.py                   # CLI tool to classify a PDF
├─ streamlit_app.py              # Web UI for uploads and visualization
├─ field_extraction.py           # Field extraction utilities
├─ llm_prompts.py                # LLM functionality
├─ alternative_llm.py            # Alternative LLM implementation
├─ requirements.txt              # Python dependencies
├─ README.md                     # This file
├─ class_examples/
│  ├─ invoice/*.txt
│  ├─ bank_statement/*.txt
│  ├─ resume/*.txt
│  ├─ ITR/*.txt
│  └─ government_id/*.txt
└─ run_app.py                    # Application launcher
```

## Setup

1. System dependencies (for OCR path):
   - macOS (Homebrew): `brew install tesseract poppler tesseract-lang`
   - Linux: `sudo apt-get install tesseract-ocr poppler-utils tesseract-ocr-hin`
   - For Hindi support: Install Hindi language pack for Tesseract
2. Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Language Support

The application provides English interface with Hindi OCR support:
- **OCR Processing**: English + Hindi (`eng+hin`) by default
- **UI Interface**: English
- **Document Classification**: Works with documents in both languages

### OCR Language Options:
- `eng+hin` - English + Hindi (Default)
- `eng` - English only
- `hin` - Hindi only
- `eng+fra` - English + French
- `eng+spa` - English + Spanish
- `eng+deu` - English + German

## Run Instructions

- CLI:

```bash
python app_demo.py --file samples/sample.pdf
```

- Streamlit app:

```bash
streamlit run streamlit_app.py
```

- Run with launcher script:

```bash
python run_app.py
```

## AI Integration / AI एकीकरण

### Google Gemini AI (Free)
- **Free tier**: 15 requests/minute, 1M tokens/day
- **High quality**: Google's latest AI model
- **Bilingual support**: Works great with English + Hindi documents
- **Setup**: Set `GEMINI_API_KEY` environment variable
- **Get API key**: https://makersuite.google.com/app/apikey

### Setup Gemini AI
```bash
# Get your API key from: https://makersuite.google.com/app/apikey
export GEMINI_API_KEY="your-gemini-api-key-here"

# Install dependencies
pip install google-generativeai

# Run the app
streamlit run streamlit_app.py
```

## Deployment

- Streamlit Cloud (Recommended):
  - Push repo to GitHub
  - Visit [share.streamlit.io](https://share.streamlit.io)
  - Deploy directly from GitHub repository
  - Add environment variable in Streamlit Cloud secrets:
    - `GEMINI_API_KEY` (for Google Gemini AI)

- Heroku Deployment:
  - `heroku login`
  - `heroku create`
  - `git push heroku main`
  - App auto-starts with Procfile
  - Open the deployed Heroku URL

## How to Add New Categories

1. Create a new subfolder under `class_examples/` with your category name, e.g. `class_examples/shipping_label/`.
2. Add 10–15 short `.txt` snippets representing that category (diverse examples help).
3. Re-run the CLI or Streamlit app. Centroids are built on startup, so the new class will be included automatically.

## Notes

- The classifier returns a JSON with: `label`, `confidence`, `rationale`, `top_scores`, and `method` (extraction method used).
- If very little text is extractable, OCR is attempted. For best results ensure Tesseract and Poppler are installed.
- The thresholds and keyword lists are simple heuristics; you can adjust them in `extract_and_classify.py`.
