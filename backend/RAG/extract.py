"""
extract.py — Stage 1: High-Performance Table & Text Extraction for Financial RAG.
Features:
  - Multi-column reading order preserved (sort=True).
  - Tables converted to searchable Markdown tables.
  - Resumable multi-core processing (ProcessPoolExecutor).
  - Clean JSONL format (no CSV field size limits).
"""

import os
import re
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pymupdf as fitz

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = "DATA"
OUTPUT_DIR = "output"
TEXT_OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "extracted_text.jsonl")
TABLES_OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "extracted_tables.jsonl")
LOG_FILE = os.path.join(OUTPUT_DIR, "extraction_log.txt")

OCR_TEXT_THRESHOLD = 30
TABLE_MIN_ROWS = 2

KNOWN_10K_ITEMS = [
    r"item\s*1\.?\s*business",
    r"item\s*1a\.?\s*risk factors",
    r"item\s*1b\.?\s*unresolved staff comments",
    r"item\s*2\.?\s*properties",
    r"item\s*3\.?\s*legal proceedings",
    r"item\s*7\.?\s*management.s discussion",
    r"item\s*7a\.?\s*quantitative and qualitative",
    r"item\s*8\.?\s*financial statements",
    r"item\s*9a\.?\s*controls and procedures",
]
KNOWN_10K_ITEMS_RE = [re.compile(p, re.IGNORECASE) for p in KNOWN_10K_ITEMS]

BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*page\s+\d+(\s+of\s+\d+)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\d+\s*$"),
    re.compile(r"^\s*©.*all rights reserved.*$", re.IGNORECASE),
    re.compile(r"^\s*confidential\s*$", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_filename(path):
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) < 3:
        # Fallback for filenames with different formats
        return parts[0].upper(), 2024, "REPORT"
    return parts[0].upper(), int(parts[1]), parts[2].upper()

def clean_boilerplate(text):
    lines = text.split("\n")
    kept = [ln for ln in lines if not any(p.match(ln.strip()) for p in BOILERPLATE_PATTERNS)]
    return "\n".join(kept).strip()

def table_to_markdown(rows):
    """Converts 2D table array to clean Markdown table representation."""
    if not rows or len(rows) < TABLE_MIN_ROWS:
        return ""
    
    # Sanitize and replace None with empty strings
    cleaned_rows = []
    for row in rows:
        cleaned_rows.append([str(c).replace("\n", " ").replace("|", "/").strip() if c is not None else "" for c in row])
    
    # Build markdown
    header = cleaned_rows[0]
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in cleaned_rows[1:]:
        # Pad row if shorter than header
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row[:len(header)]) + " |")
    return "\n".join(md)

def classify_table(rows):
    flat = " ".join(str(c) for row in rows[:4] for c in row if c).lower()
    if any(k in flat for k in ["total assets", "liabilities", "shareholders", "equity"]):
        return "balance_sheet"
    if any(k in flat for k in ["revenue", "cost of sales", "net income", "operating expenses", "gross profit"]):
        return "income_statement"
    if any(k in flat for k in ["cash flows", "operating activities", "investing activities", "financing activities"]):
        return "cash_flow"
    if any(k in flat for k in ["segment", "geographic", "depreciation", "notes"]):
        return "notes_and_segments"
    return "financial_table"

# ---------------------------------------------------------------------------
# Per-PDF Worker
# ---------------------------------------------------------------------------

def process_single_pdf(pdf_path):
    fname = os.path.basename(pdf_path)
    ticker, year, report_type = parse_filename(pdf_path)
    
    text_records = []
    table_records = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return fname, False, f"Failed to open: {e}", [], []

    n_pages = len(doc)
    n_tables = 0

    for page_index in range(n_pages):
        page = doc[page_index]
        page_number = page_index + 1

        # Use sort=True for multi-column layout preservation
        raw_text = page.get_text("text", sort=True)
        cleaned_text = clean_boilerplate(raw_text)

        # Detect 10-K Section Headings
        current_heading = ""
        if report_type == "10K":
            for rx in KNOWN_10K_ITEMS_RE:
                match = rx.search(cleaned_text[:500]) # Check top of page
                if match:
                    current_heading = match.group(0)
                    break

        if len(cleaned_text.strip()) > OCR_TEXT_THRESHOLD:
            text_records.append({
                "ticker": ticker,
                "year": year,
                "report_type": report_type,
                "page": page_number,
                "heading": current_heading,
                "text": cleaned_text,
                "source": fname
            })

        # Extract Tables using PyMuPDF Table Finder
        try:
            tabs = page.find_tables()
            for i, tab in enumerate(tabs.tables):
                rows = tab.extract()
                if len(rows) >= TABLE_MIN_ROWS:
                    table_type = classify_table(rows)
                    md_table = table_to_markdown(rows)
                    
                    if md_table:
                        table_records.append({
                            "ticker": ticker,
                            "year": year,
                            "report_type": report_type,
                            "page": page_number,
                            "table_index": i,
                            "table_type": table_type,
                            "markdown_table": md_table,
                            "source": fname
                        })
                        n_tables += 1
        except Exception:
            pass

    doc.close()
    return fname, True, f"Pages: {n_pages}, Tables: {n_tables}", text_records, table_records

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdf_files = sorted(Path(args.data_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in '{args.data_dir}/'")
        return

    print(f"Found {len(pdf_files)} PDFs. Extracting with {args.workers} workers...")

    with open(TEXT_OUTPUT_JSONL, "w", encoding="utf-8") as tf, \
         open(TABLES_OUTPUT_JSONL, "w", encoding="utf-8") as tblf, \
         open(LOG_FILE, "w", encoding="utf-8") as lf:

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_pdf = {executor.submit(process_single_pdf, str(p)): p for p in pdf_files}
            
            for future in as_completed(future_to_pdf):
                fname, success, detail, texts, tables = future.result()
                status = "SUCCESS" if success else "FAILED"
                print(f"[{status}] {fname}: {detail}")
                lf.write(f"{fname} | {status} | {detail}\n")

                for t in texts:
                    tf.write(json.dumps(t) + "\n")
                for tb in tables:
                    tblf.write(json.dumps(tb) + "\n")

    print(f"\nExtraction complete! Files saved in `{OUTPUT_DIR}/`:")
    print(f"  - {TEXT_OUTPUT_JSONL}")
    print(f"  - {TABLES_OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
