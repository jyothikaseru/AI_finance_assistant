


"""
chunk.py — Stage 2: Context-Enriched Chunking for Financial RAG.

Reads:
  - output/extracted_text.jsonl
  - output/extracted_tables.jsonl

Produces:
  - output/chunked_data.jsonl

Key Features:
  1. Section-aware text chunking with contextual header injection.
  2. Tables indexed as atomic chunks with full Markdown structure.
  3. Sentence/paragraph-safe token splitting (prevents breaking numbers).
  4. Structured JSONL output ready for Supabase / pgvector indexing.

Requires: pip install tiktoken --break-system-packages
"""

import os
import json
import argparse
from collections import defaultdict
import tiktoken

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
TEXT_JSONL = os.path.join(OUTPUT_DIR, "extracted_text.jsonl")
TABLES_JSONL = os.path.join(OUTPUT_DIR, "extracted_tables.jsonl")
CHUNKED_JSONL = os.path.join(OUTPUT_DIR, "chunked_data.jsonl")

TARGET_CHUNK_TOKENS = 450     # Target token count per text chunk
CHUNK_OVERLAP_TOKENS = 60     # Overlap between consecutive chunks
MIN_CHUNK_TOKENS = 30         # Filter out tiny trailing fragments

try:
    ENCODING = tiktoken.get_encoding("cl100k_base")
    _TIKTOKEN_OK = True
except Exception as e:
    print(f"[WARN] tiktoken unavailable ({e}). Using word-count approximation.")
    ENCODING = None
    _TIKTOKEN_OK = False


def count_tokens(text: str) -> int:
    if _TIKTOKEN_OK:
        return len(ENCODING.encode(text))
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Step 1: Text Section Grouping & Paragraph-Aware Splitting
# ---------------------------------------------------------------------------

def load_text_documents(path):
    """Groups extracted pages by document: (ticker, fiscal_year, report_type)."""
    docs = defaultdict(list)
    if not os.path.exists(path):
        print(f"[WARN] {path} not found.")
        return docs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            key = (item["ticker"], item["year"], item["report_type"])
            docs[key].append(item)

    for key in docs:
        docs[key].sort(key=lambda x: x["page"])
    return docs


def group_pages_into_sections(pages):
    """Groups consecutive pages by section headings."""
    sections = []
    current = None

    for page_data in pages:
        heading = page_data.get("heading", "").strip()
        text = page_data["text"]
        page_num = page_data["page"]

        starts_new = bool(heading) and (current is None or heading != current["heading"])

        if current is None:
            current = {
                "heading": heading if heading else "Overview / Front Matter",
                "page_start": page_num,
                "page_end": page_num,
                "text_parts": [text],
                "source": page_data.get("source", ""),
            }
            continue

        if starts_new:
            sections.append(current)
            current = {
                "heading": heading,
                "page_start": page_num,
                "page_end": page_num,
                "text_parts": [text],
                "source": page_data.get("source", ""),
            }
        else:
            current["text_parts"].append(text)
            current["page_end"] = page_num

    if current is not None:
        sections.append(current)

    for s in sections:
        s["text"] = "\n\n".join(s.pop("text_parts")).strip()

    return sections


def split_text_cleanly(text, target_tokens, overlap_tokens):
    """
    Splits text by paragraphs/sentences to avoid cutting numbers mid-digit,
    maintaining a sliding window with overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        # If a single paragraph is larger than target, split by sentences
        if para_tokens > target_tokens:
            sentences = [s.strip() + "." for s in para.split(". ") if s.strip()]
            for sent in sentences:
                sent_tokens = count_tokens(sent)
                if current_tokens + sent_tokens > target_tokens and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Keep rolling overlap
                    overlap_buffer = []
                    overlap_count = 0
                    for item in reversed(current_chunk):
                        item_tokens = count_tokens(item)
                        if overlap_count + item_tokens <= overlap_tokens:
                            overlap_buffer.insert(0, item)
                            overlap_count += item_tokens
                        else:
                            break
                    current_chunk = overlap_buffer
                    current_tokens = overlap_count

                current_chunk.append(sent)
                current_tokens += sent_tokens
        else:
            if current_tokens + para_tokens > target_tokens and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Keep rolling overlap
                overlap_buffer = []
                overlap_count = 0
                for item in reversed(current_chunk):
                    item_tokens = count_tokens(item)
                    if overlap_count + item_tokens <= overlap_tokens:
                        overlap_buffer.insert(0, item)
                        overlap_count += item_tokens
                    else:
                        break
                current_chunk = overlap_buffer
                current_tokens = overlap_count

            current_chunk.append(para)
            current_tokens += para_tokens

    if current_chunk and current_tokens >= MIN_CHUNK_TOKENS:
        chunks.append("\n\n".join(current_chunk))

    return chunks


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-tokens", type=int, default=TARGET_CHUNK_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=CHUNK_OVERLAP_TOKENS)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading extracted text records...")
    documents = load_text_documents(TEXT_JSONL)
    print(f"  Found text records for {len(documents)} document(s)")

    total_text_chunks = 0
    total_table_chunks = 0

    with open(CHUNKED_JSONL, "w", encoding="utf-8") as out_f:
        # 1. Process Narrative / Section Text
        for key, pages in documents.items():
            ticker, fiscal_year, report_type = key
            sections = group_pages_into_sections(pages)
            chunk_seq = 0

            for section in sections:
                sub_chunks = split_text_cleanly(
                    section["text"], args.target_tokens, args.overlap_tokens
                )
                for chunk_body in sub_chunks:
                    chunk_uid = f"{ticker}_{fiscal_year}_{report_type}_c{chunk_seq}"
                    
                    # Contextual Header Injection
                    context_header = (
                        f"[Company: {ticker} | Year: {fiscal_year} | Type: {report_type} | "
                        f"Section: {section['heading']} | Pages: {section['page_start']}-{section['page_end']}]"
                    )
                    full_content = f"{context_header}\n\n{chunk_body}"

                    record = {
                        "chunk_uid": chunk_uid,
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "report_type": report_type,
                        "chunk_type": "text",
                        "section": section["heading"],
                        "page_start": section["page_start"],
                        "page_end": section["page_end"],
                        "token_count": count_tokens(full_content),
                        "content": full_content,
                        "source": section.get("source", "")
                    }
                    out_f.write(json.dumps(record) + "\n")
                    chunk_seq += 1
                    total_text_chunks += 1

        # 2. Process Tables (Track A: Atomic Markdown Table-as-a-Chunk)
        if os.path.exists(TABLES_JSONL):
            print("Processing extracted Markdown tables...")
            with open(TABLES_JSONL, "r", encoding="utf-8") as tbl_f:
                for line in tbl_f:
                    tbl = json.loads(line.strip())
                    ticker = tbl["ticker"]
                    year = tbl["year"]
                    report_type = tbl["report_type"]
                    page = tbl["page"]
                    table_type = tbl.get("table_type", "financial_table")
                    md_table = tbl.get("markdown_table", "")

                    if not md_table:
                        continue

                    chunk_uid = f"{ticker}_{year}_{report_type}_t{tbl.get('table_index', 0)}_p{page}"
                    
                    # Contextual Header Injection for Tables
                    context_header = (
                        f"[Table: {table_type.replace('_', ' ').title()} | "
                        f"Company: {ticker} | Year: {year} | Page: {page}]"
                    )
                    full_content = f"{context_header}\n\n{md_table}"

                    record = {
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
                        "source": tbl.get("source", "")
                    }
                    out_f.write(json.dumps(record) + "\n")
                    total_table_chunks += 1

    print(f"\nChunking complete!")
    print(f"  - Text Chunks Generated:  {total_text_chunks}")
    print(f"  - Table Chunks Generated: {total_table_chunks}")
    print(f"  - Output File:            {CHUNKED_JSONL}")


if __name__ == "__main__":
    main()
