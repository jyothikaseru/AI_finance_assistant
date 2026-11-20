"""
extract.py — Stage 1: High-Performance Table & Text Extraction for Financial RAG.
Features:
  - Real-time live visual progress bar (tqdm) — never appears frozen or stuck.
  - Multi-column reading order preserved (sort=True).
  - Fast table extraction converted to clean Markdown.
  - Heading detection for both US GAAP (10-K) and Indian SEBI (Integrated Reports).
  - Incremental & resumable: skips already extracted PDFs.
  - Clean JSONL format (no CSV field size limits).
  - Supports `--limit N` to test on a small number of PDFs.
"""

import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pymupdf as fitz

# ---------------------------------------------------------------------------
# Config & Defaults
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = str(BASE_DIR / "DATA")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "output")

OCR_TEXT_THRESHOLD = 30
TABLE_MIN_ROWS = 2

# 10-K Section Headings (US GAAP)
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

# SEBI Integrated Report Section Headings (Indian Standards)
KNOWN_SEBI_ITEMS = [
    r"management\s+discussion\s+and\s+analysis",
    r"business\s+responsibility\s+and\s+sustainability\s+report",
    r"standalone\s+financial\s+statements",
    r"consolidated\s+financial\s+statements",
    r"directors?'\s+report",
    r"board'?s\s+report",
    r"corporate\s+governance\s+report",
    r"notes\s+to\s+(the\s+)?financial\s+statements",
    r"independent\s+auditors?'\s+report",
    r"balance\s+sheet",
    r"statement\s+of\s+profit\s+and\s+loss",
    r"cash\s+flow\s+statement",
]
KNOWN_SEBI_ITEMS_RE = [re.compile(p, re.IGNORECASE) for p in KNOWN_SEBI_ITEMS]

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
        match = re.match(r"([A-Za-z]+)(\d{4})", stem)
        if match:
            return match.group(1).upper(), int(match.group(2)), "REPORT"
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
    
    cleaned_rows = []
    for row in rows:
        cleaned_rows.append([str(c).replace("\n", " ").replace("|", "/").strip() if c is not None else "" for c in row])
    
    header = cleaned_rows[0]
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in cleaned_rows[1:]:
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

def get_already_processed_files(log_path, text_output_path=None):
    """Detects which PDFs have already been successfully processed."""
    processed = set()
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" | ")
                if len(parts) >= 2 and parts[1] == "SUCCESS":
                    processed.add(parts[0])

    if not processed and text_output_path and os.path.exists(text_output_path):
        try:
            with open(text_output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "source" in item:
                            processed.add(item["source"])
        except Exception:
            pass

    return processed

# ---------------------------------------------------------------------------
# Per-PDF Worker with Real-Time Progress
# ---------------------------------------------------------------------------

def process_single_pdf(pdf_path, pbar=None):
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

    heading_candidates = KNOWN_10K_ITEMS_RE if report_type == "10K" else KNOWN_SEBI_ITEMS_RE

    for page_index in range(n_pages):
        page = doc[page_index]
        page_number = page_index + 1

        # Multi-column reading order preservation
        raw_text = page.get_text("text", sort=True)
        cleaned_text = clean_boilerplate(raw_text)

        # Section Heading Detection
        current_heading = ""
        for rx in heading_candidates:
            match = rx.search(cleaned_text[:500])
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

        # Fast table extraction with 'lines' strategy
        try:
            tabs = page.find_tables(strategy="lines")
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

        if pbar:
            pbar.update(1)
            pbar.set_postfix({"tables": n_tables})

    doc.close()
    return fname, True, f"Pages: {n_pages}, Tables: {n_tables}", text_records, table_records

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract text and tables from corporate filings (10-K & SEBI).")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Directory containing PDF files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save extracted outputs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to extract in this run")
    parser.add_argument("--force", action="store_true", help="Force re-extraction of all PDFs (overwrites previous runs)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    text_output_jsonl = os.path.join(args.output_dir, "extracted_text.jsonl")
    tables_output_jsonl = os.path.join(args.output_dir, "extracted_tables.jsonl")
    log_file = os.path.join(args.output_dir, "extraction_log.txt")

    all_pdfs = sorted(Path(args.data_dir).glob("*.pdf"))
    if not all_pdfs:
        print(f"No PDFs found in '{args.data_dir}'")
        return

    # Check for already-extracted files
    if args.force:
        already_processed = set()
        file_mode = "w"
        print("[INFO] --force specified: Overwriting and re-extracting all PDFs from scratch.")
    else:
        already_processed = get_already_processed_files(log_file, text_output_jsonl)
        file_mode = "a"

    pending_pdfs = [p for p in all_pdfs if p.name not in already_processed]

    if args.limit and args.limit > 0:
        pending_pdfs = pending_pdfs[:args.limit]

    print("=" * 68)
    print("  FinRAG Document Extraction Pipeline (Stage 1)")
    print("=" * 68)
    print(f"  Total PDFs in folder:   {len(all_pdfs)}")
    print(f"  Already extracted:      {len(already_processed)}")
    print(f"  Queued for this run:    {len(pending_pdfs)}")
    print(f"  Output directory:       {args.output_dir}")
    print("=" * 68 + "\n")

    if not pending_pdfs:
        print("All PDFs have already been extracted! Nothing to do.")
        print("To force re-extraction from scratch, run: python extract.py --force\n")
        return

    with open(text_output_jsonl, file_mode, encoding="utf-8") as tf, \
         open(tables_output_jsonl, file_mode, encoding="utf-8") as tblf, \
         open(log_file, file_mode, encoding="utf-8") as lf:

        for idx, pdf_path in enumerate(pending_pdfs, 1):
            fname = pdf_path.name
            try:
                doc_preview = fitz.open(str(pdf_path))
                total_pages = len(doc_preview)
                doc_preview.close()
            except Exception:
                total_pages = 0

            with tqdm(
                total=total_pages,
                desc=f"[{idx}/{len(pending_pdfs)}] {fname[:24]:<24}",
                unit="page",
                dynamic_ncols=True,
                leave=True
            ) as pbar:
                fname, success, detail, texts, tables = process_single_pdf(str(pdf_path), pbar=pbar)

            status = "SUCCESS" if success else "FAILED"
            lf.write(f"{fname} | {status} | {detail}\n")
            lf.flush()

            for t in texts:
                tf.write(json.dumps(t) + "\n")
            tf.flush()

            for tb in tables:
                tblf.write(json.dumps(tb) + "\n")
            tblf.flush()

    print(f"\nExtraction complete! Data appended to `{args.output_dir}/`:")
    print(f"  - {text_output_jsonl}")
    print(f"  - {tables_output_jsonl}")
    print(f"  - {log_file}\n")

if __name__ == "__main__":
    main()
