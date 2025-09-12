# PDF Document Classifier

**Live Demo**: [PDF Document Classifier](https://pdf-classifier-idfy.streamlit.app)

## Assignment Submission

### Deliverables
- **Runnable Prototype**: [PDF Document Classifier](https://pdf-classifier-idfy.streamlit.app)
- **Source Code**: This GitHub repository
- **Brief README**: This file (≤1 page)

### Chosen Categories
1. **Invoice** - Business invoices, receipts, bills
2. **Bank Statement** - Financial statements, transaction records
3. **Resume** - CVs, professional profiles, job applications
4. **ITR** - Income Tax Returns, tax documents
5. **Government ID** - Passports, driver's licenses, identity cards

### Approach (5-8 lines)
This application uses a hybrid approach combining text embedding similarity with keyword-based classification. Document text is extracted using PyMuPDF and OCR (Tesseract), then converted to sentence embeddings using SentenceTransformers. The system calculates cosine similarity against pre-built centroids from training samples for each document type. Keyword boosting enhances classification accuracy by identifying domain-specific terms. For ambiguous cases (confidence 0.3-0.7), Google Gemini AI provides intelligent fallback classification. The system supports both English and Hindi documents through multilingual OCR processing.

### How to Add New Categories
1. Create a new subfolder under `class_examples/` with your category name (e.g., `class_examples/shipping_label/`)
2. Add 10-15 short `.txt` snippets representing that category (diverse examples help)
3. Re-run the application - centroids are built automatically on startup, so the new class will be included

## How to Run

### Option 1: Run with provided script (includes Gemini AI)
```bash
./run_with_gemini.sh
```

### Option 2: Run manually with API key
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
streamlit run streamlit_app.py
```

### Option 3: Run without API key (limited features)
```bash
streamlit run streamlit_app.py
```

**Note**: For full AI features, get a free Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)