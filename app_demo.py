"""Command-line interface to classify a PDF document.

Example:
    python app_demo.py --file samples/sample.pdf
"""

import os
import json
import argparse

from extract_and_classify import extract_and_classify, build_class_centroids, classify_text, extract_text_with_pymupdf, extract_text_with_ocr, _get_embedding_model


def main():
    """Parse arguments, extract text, classify, and print JSON result."""
    parser = argparse.ArgumentParser(description="CLI to classify a PDF document")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--class-examples", default=os.path.join(os.path.dirname(__file__), "class_examples"), help="Path to class_examples folder")
    args = parser.parse_args()

    model = _get_embedding_model()
    centroids = build_class_centroids(args.class_examples, model=model)

    # Extract via PyMuPDF then fallback to OCR if text too short
    method = ""
    try:
        text, method = extract_text_with_pymupdf(args.file)
    except Exception as e:
        text, method = "", ""

    if len(text.strip()) < 100:
        try:
            text, method = extract_text_with_ocr(args.file)
        except Exception:
            pass

    result = classify_text(text, centroids, model=model)
    result["method"] = method
    result["extracted_chars"] = len(text)
    print(json.dumps(result, indent=2))
    # Also print extracted fields separately for quick view
    fields = result.get("fields", {})
    if fields:
        print("\nExtracted Fields:")
        for k, v in fields.items():
            print(f"- {k}: {v}")


if __name__ == "__main__":
    main()


