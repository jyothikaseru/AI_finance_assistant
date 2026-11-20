"""
chunk.py — Stage 2: Context-Enriched Chunking for Financial RAG.

Reads:
  - output/extracted_text.jsonl
  - output/extracted_tables.jsonl

Produces:
  - output/chunked_data.jsonl

Key Enhancements & Bug Fixes:
  1. Exact Page Citation Tracking: Chunks now record the exact page range
     (e.g. Page 74 or Pages 74-75) rather than coarse chapter boundaries.
  2. Table Splitting with Repeated Headers: Large tables (>420 tokens) are split row-wise
     with the column header row repeated on every split, eliminating silent BERT vector truncation.
  3. Abbreviation-Safe Sentence Splitting: Protects financial abbreviations ('U.S.', 'Inc.', 'vs.',
     'approx.', '$391.03') from being fragmented mid-sentence.
  4. Safe Token Limits: Target 420 tokens leaves safe headroom for metadata headers under the 512-token limit.
  5. Live Progress Bar (tqdm): Displays real-time progress for both text and table chunking.
"""

import os
import re
import json
import argparse
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "output")

TARGET_CHUNK_TOKENS = 420     # Target token count (leaves room for header under 512 limit)
CHUNK_OVERLAP_TOKENS = 50     # Overlap between consecutive chunks
MIN_CHUNK_TOKENS = 30         # Filter out tiny trailing fragments

COMMON_ABBREVS = [
    "U.S.", "Inc.", "Ltd.", "Corp.", "Co.", "vs.", "approx.", "e.g.", "i.e.",
    "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."
]

try:
    import tiktoken
    ENCODING = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_OK = True
except Exception:
    ENCODING = None
    _TIKTOKEN_OK = False


def count_tokens(text: str) -> int:
    """Calculates token count. Falls back safely to BERT/WordPiece approximation if tiktoken is offline."""
    if _TIKTOKEN_OK and ENCODING:
        try:
            return len(ENCODING.encode(text))
        except Exception:
            pass
    # WordPiece/BERT tokenization approximation: ~1.35 tokens per whitespace-separated word
    return max(1, int(len(text.split()) * 1.35))


def split_sentences_safely(text: str):
    """
    Splits text into sentences while strictly protecting financial abbreviations
    (e.g., 'U.S.', 'Inc.', 'vs.', 'approx.', decimals like '$391.03').
    """
    temp = text
    for i, abb in enumerate(COMMON_ABBREVS):
        temp = temp.replace(abb, f"__ABBR_{i}__")

    # Split on period/question/exclamation followed by whitespace and a capital letter or digit
    raw_sents = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', temp)

    final_sents = []
    for s in raw_sents:
        restored = s
        for i, abb in enumerate(COMMON_ABBREVS):
            restored = restored.replace(f"__ABBR_{i}__", abb)
        if restored.strip():
            final_sents.append(restored.strip())

    return final_sents if final_sents else [text]


# ---------------------------------------------------------------------------
# Step 1: Text Document Loading & Exact Page Paragraph Tracking
# ---------------------------------------------------------------------------

def load_text_documents(path):
    """Groups extracted pages by document: (ticker, fiscal_year, report_type)."""
    docs = defaultdict(list)
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            key = (item["ticker"], item["year"], item["report_type"])
            docs[key].append(item)

    for key in docs:
        docs[key].sort(key=lambda x: x["page"])
    return docs


def extract_section_paragraph_items(pages):
    """
    Extracts individual paragraphs from pages, preserving their exact page number
    and grouping by section headings.
    """
    sections = []
    current_section = None

    for page_data in pages:
        heading = page_data.get("heading", "").strip()
        raw_text = page_data.get("text", "")
        page_num = page_data.get("page", 1)
        source = page_data.get("source", "")

        starts_new = bool(heading) and (current_section is None or heading != current_section["heading"])

        if current_section is None:
            current_section = {
                "heading": heading if heading else "Overview / Front Matter",
                "source": source,
                "items": []
            }

        elif starts_new:
            sections.append(current_section)
            current_section = {
                "heading": heading,
                "source": source,
                "items": []
            }

        # Extract paragraphs with their exact page number
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        for p in paragraphs:
            current_section["items"].append({
                "text": p,
                "page": page_num
            })

    if current_section and current_section["items"]:
        sections.append(current_section)

    return sections


def chunk_section_items(items, target_tokens, overlap_tokens):
    """
    Chunks paragraph/sentence items while dynamically computing the exact
    page_start and page_end for each chunk.
    """
    # Break down oversized paragraphs into sentences while preserving page numbers
    atomic_units = []
    for it in items:
        p_text = it["text"]
        p_page = it["page"]
        tok = count_tokens(p_text)

        if tok > target_tokens:
            sents = split_sentences_safely(p_text)
            for s in sents:
                atomic_units.append({
                    "text": s,
                    "page": p_page,
                    "tokens": count_tokens(s)
                })
        else:
            atomic_units.append({
                "text": p_text,
                "page": p_page,
                "tokens": tok
            })

    chunks = []
    current_window = []
    current_tokens = 0

    for unit in atomic_units:
        u_tok = unit["tokens"]

        if current_tokens + u_tok > target_tokens and current_window:
            p_start = current_window[0]["page"]
            p_end = current_window[-1]["page"]
            c_text = "\n\n".join(u["text"] for u in current_window)

            chunks.append({
                "text": c_text,
                "page_start": p_start,
                "page_end": p_end
            })

            # Calculate rolling overlap
            overlap_buffer = []
            overlap_count = 0
            for item in reversed(current_window):
                if overlap_count + item["tokens"] <= overlap_tokens:
                    overlap_buffer.insert(0, item)
                    overlap_count += item["tokens"]
                else:
                    break

            current_window = overlap_buffer
            current_tokens = overlap_count

        current_window.append(unit)
        current_tokens += u_tok

    if current_window and current_tokens >= MIN_CHUNK_TOKENS:
        p_start = current_window[0]["page"]
        p_end = current_window[-1]["page"]
        c_text = "\n\n".join(u["text"] for u in current_window)
        chunks.append({
            "text": c_text,
            "page_start": p_start,
            "page_end": p_end
        })

    return chunks


# ---------------------------------------------------------------------------
# Step 2: Table Chunking with Header-Preserving Sub-Table Splitting
# ---------------------------------------------------------------------------

def chunk_table(table_record, target_tokens):
    """
    Processes a Markdown table into one or more chunks.
    If the table exceeds target_tokens, it splits the rows into sub-tables
    and repeats the column header row on every split to prevent BERT truncation.
    """
    ticker = table_record["ticker"]
    year = table_record["year"]
    report_type = table_record["report_type"]
    page = table_record["page"]
    table_index = table_record.get("table_index", 0)
    table_type = table_record.get("table_type", "financial_table")
    table_type_clean = table_type.replace("_", " ").title()
    md_table = table_record.get("markdown_table", "").strip()

    if not md_table:
        return []

    lines = [ln.strip() for ln in md_table.split("\n") if ln.strip()]

    # If table is less than 3 lines, emit as single chunk
    if len(lines) < 3:
        context_header = (
            f"[Table: {table_type_clean} | Company: {ticker} | Year: {year} | Page: {page}]"
        )
        full_content = f"{context_header}\n\n{md_table}"
        chunk_uid = f"{ticker}_{year}_{report_type}_t{table_index}_p{page}"
        return [{
            "chunk_uid": chunk_uid,
            "ticker": ticker,
            "fiscal_year": year,
            "report_type": report_type,
            "chunk_type": "table",
            "section": table_type,
            "page_start": page,
            "page_end": page,
            "token_count": count_tokens(full_content),
            "content": full_content,
            "source": table_record.get("source", "")
        }]

    header_lines = lines[:2]
    data_rows = lines[2:]
    header_str = "\n".join(header_lines)
    header_tokens = count_tokens(header_str)

    # Check if table fits comfortably in a single chunk
    single_context = f"[Table: {table_type_clean} | Company: {ticker} | Year: {year} | Page: {page}]"
    single_content = f"{single_context}\n\n{md_table}"

    if count_tokens(single_content) <= target_tokens:
        chunk_uid = f"{ticker}_{year}_{report_type}_t{table_index}_p{page}"
        return [{
            "chunk_uid": chunk_uid,
            "ticker": ticker,
            "fiscal_year": year,
            "report_type": report_type,
            "chunk_type": "table",
            "section": table_type,
            "page_start": page,
            "page_end": page,
            "token_count": count_tokens(single_content),
            "content": single_content,
            "source": table_record.get("source", "")
        }]

    # Table is oversized (> target_tokens): split rows with repeated header
    table_parts = []
    current_rows = []
    current_tokens = header_tokens + 35  # Allowance for contextual metadata header

    for row in data_rows:
        r_tok = count_tokens(row)
        if current_tokens + r_tok > target_tokens and current_rows:
            table_parts.append(current_rows)
            current_rows = [row]
            current_tokens = header_tokens + 35 + r_tok
        else:
            current_rows.append(row)
            current_tokens += r_tok

    if current_rows:
        table_parts.append(current_rows)

    total_parts = len(table_parts)
    chunks = []

    for part_idx, part_rows in enumerate(table_parts, 1):
        sub_md = "\n".join(header_lines + part_rows)
        part_context = (
            f"[Table: {table_type_clean} (Part {part_idx}/{total_parts}) | "
            f"Company: {ticker} | Year: {year} | Page: {page}]"
        )
        full_content = f"{part_context}\n\n{sub_md}"
        chunk_uid = f"{ticker}_{year}_{report_type}_t{table_index}_p{page}_part{part_idx}"

        chunks.append({
            "chunk_uid": chunk_uid,
            "ticker": ticker,
            "fiscal_year": year,
            "report_type": report_type,
            "chunk_type": "table",
            "section": table_type,
            "page_start": page,
            "page_end": page,
            "token_count": count_tokens(full_content),
            "content": full_content,
            "source": table_record.get("source", "")
        })

    return chunks


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Context-Enriched Chunking for Financial RAG.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Path to output directory")
    parser.add_argument("--target-tokens", type=int, default=TARGET_CHUNK_TOKENS, help="Target tokens per chunk")
    parser.add_argument("--overlap-tokens", type=int, default=CHUNK_OVERLAP_TOKENS, help="Overlap tokens")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    text_jsonl = os.path.join(args.output_dir, "extracted_text.jsonl")
    tables_jsonl = os.path.join(args.output_dir, "extracted_tables.jsonl")
    chunked_jsonl = os.path.join(args.output_dir, "chunked_data.jsonl")

    print("=" * 68)
    print("  FinRAG Document Chunking Pipeline (Stage 2)")
    print("=" * 68)

    print("\n[1/2] Loading and chunking narrative text records...")
    documents = load_text_documents(text_jsonl)
    print(f"      Loaded text records for {len(documents)} document(s).")

    total_text_chunks = 0
    total_table_chunks = 0

    with open(chunked_jsonl, "w", encoding="utf-8") as out_f:
        # 1. Process Narrative Text
        with tqdm(total=len(documents), desc="Chunking Text", unit="doc") as pbar:
            for key, pages in documents.items():
                ticker, fiscal_year, report_type = key
                sections = extract_section_paragraph_items(pages)
                chunk_seq = 0

                for sec in sections:
                    sub_chunks = chunk_section_items(
                        sec["items"], args.target_tokens, args.overlap_tokens
                    )

                    for c in sub_chunks:
                        chunk_uid = f"{ticker}_{fiscal_year}_{report_type}_c{chunk_seq}"
                        page_start = c["page_start"]
                        page_end = c["page_end"]

                        page_str = (
                            f"Page: {page_start}"
                            if page_start == page_end
                            else f"Pages: {page_start}-{page_end}"
                        )
                        context_header = (
                            f"[Company: {ticker} | Year: {fiscal_year} | Type: {report_type} | "
                            f"Section: {sec['heading']} | {page_str}]"
                        )
                        full_content = f"{context_header}\n\n{c['text']}"

                        record = {
                            "chunk_uid": chunk_uid,
                            "ticker": ticker,
                            "fiscal_year": fiscal_year,
                            "report_type": report_type,
                            "chunk_type": "text",
                            "section": sec["heading"],
                            "page_start": page_start,
                            "page_end": page_end,
                            "token_count": count_tokens(full_content),
                            "content": full_content,
                            "source": sec.get("source", "")
                        }
                        out_f.write(json.dumps(record) + "\n")
                        chunk_seq += 1
                        total_text_chunks += 1

                pbar.update(1)

        # 2. Process Tables with Header-Preserving Sub-Table Splitting
        if os.path.exists(tables_jsonl):
            print("\n[2/2] Processing and chunking Markdown tables (with header preservation)...")
            
            # Count total tables for progress bar
            with open(tables_jsonl, "r", encoding="utf-8") as f:
                table_count = sum(1 for line in f if line.strip())

            with open(tables_jsonl, "r", encoding="utf-8") as tbl_f, \
                 tqdm(total=table_count, desc="Chunking Tables", unit="table") as pbar:

                for line in tbl_f:
                    if not line.strip():
                        continue
                    tbl = json.loads(line.strip())
                    table_chunks = chunk_table(tbl, args.target_tokens)

                    for tc in table_chunks:
                        out_f.write(json.dumps(tc) + "\n")
                        total_table_chunks += 1

                    pbar.update(1)

    print("\n" + "=" * 68)
    print("  Chunking complete!")
    print(f"  - Text Chunks Generated:  {total_text_chunks}")
    print(f"  - Table Chunks Generated: {total_table_chunks} (Oversized tables safely split)")
    print(f"  - Total Indexed Chunks:   {total_text_chunks + total_table_chunks}")
    print(f"  - Output File:            {chunked_jsonl}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
