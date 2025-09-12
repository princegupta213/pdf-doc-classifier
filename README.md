# PDF Document Classifier

Classify PDFs into categories like invoice, bank statement, resume, ITR, and government_id using text extraction, cleaning, sentence embeddings, and cosine similarity with keyword boosting.

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
├─ evaluate.py                   # Evaluate PDFs under tests/<class>/
├─ requirements.txt              # Python dependencies
├─ README.md                     # This file
├─ class_examples/
│  ├─ invoice/*.txt
│  ├─ bank_statement/*.txt
│  ├─ resume/*.txt
│  ├─ ITR/*.txt
│  └─ government_id/*.txt
└─ tests/
   ├─ invoice/*.pdf
   ├─ bank_statement/*.pdf
   ├─ resume/*.pdf
   ├─ ITR/*.pdf
   └─ government_id/*.pdf
```

## Setup

1. System dependencies (for OCR path):
   - macOS (Homebrew): `brew install tesseract poppler`
   - Linux: `sudo apt-get install tesseract-ocr poppler-utils`
2. Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Instructions

- CLI:

```bash
python app_demo.py --file samples/sample.pdf
```

- Streamlit app:

```bash
streamlit run streamlit_app.py
```

- Evaluation (place PDFs in `tests/<class>/`):

```bash
python evaluate.py
```

## Deployment

- Local Docker Run:

```
docker build -t doc-classifier .
docker run -p 8501:8501 doc-classifier
```

Open http://localhost:8501

- Render Deployment:
  - Push repo to GitHub
  - Create a new Web Service on Render
  - Select Docker as environment
  - Expose port 8501
  - Add environment variable `OPENAI_API_KEY` in Render dashboard if LLM fallback is enabled

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
