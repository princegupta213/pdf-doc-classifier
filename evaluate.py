"""Evaluate the classifier on PDFs located under tests/<class>/ folders."""

import os
import argparse
from typing import List, Tuple

from sklearn.metrics import classification_report

from extract_and_classify import (
    build_class_centroids,
    classify_text,
    extract_text_with_pymupdf,
    extract_text_with_ocr,
    _get_embedding_model,
)


def collect_pdfs(tests_dir: str) -> List[Tuple[str, str]]:
    """Return list of (pdf_path, class_label) pairs from tests directory."""
    pairs = []
    if not os.path.isdir(tests_dir):
        return pairs
    for cls in sorted(os.listdir(tests_dir)):
        cls_dir = os.path.join(tests_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith(".pdf"):
                pairs.append((os.path.join(cls_dir, fname), cls))
    return pairs


def main():
    """Build centroids, run classification over tests, and print report."""
    parser = argparse.ArgumentParser(description="Evaluate classifier on tests/<class> PDFs")
    parser.add_argument("--tests", default=os.path.join(os.path.dirname(__file__), "tests"), help="Path to tests folder")
    parser.add_argument("--class-examples", default=os.path.join(os.path.dirname(__file__), "class_examples"), help="Path to class_examples folder")
    args = parser.parse_args()

    model = _get_embedding_model()
    centroids = build_class_centroids(args.class_examples, model=model)

    items = collect_pdfs(args.tests)
    y_true: List[str] = []
    y_pred: List[str] = []

    for pdf_path, true_label in items:
        try:
            text, method = extract_text_with_pymupdf(pdf_path)
        except Exception:
            text, method = "", ""
        if len(text.strip()) < 100:
            try:
                text, method = extract_text_with_ocr(pdf_path)
            except Exception:
                pass
        result = classify_text(text, centroids, model=model)
        pred_label = result.get("label", "unknown")
        y_true.append(true_label)
        y_pred.append(pred_label)

    if not y_true:
        print("No test PDFs found in tests/<class>/.")
        return

    print(classification_report(y_true, y_pred, labels=sorted(set(y_true + y_pred))))


if __name__ == "__main__":
    main()


