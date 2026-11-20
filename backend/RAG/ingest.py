"""
ingest.py — Stage 3: High-Performance Embedding Generation & PostgreSQL Ingestion.

Features:
  1. Auto-credential loading from chat/.env.local (no manual environment variable exports required).
  2. BAAI/bge-base-en-v1.5 embedding generation with GPU/CPU auto-detection and normalization.
  3. Resumable Ingestion Checkpoint: Tracks progress so you can resume without re-embedding if interrupted.
  4. Robust Upload with Retries: 3 attempts with exponential backoff and automatic sub-batch splitting.
  5. Optimized Batch Sizing (default 50): Prevents PostgREST HTTP timeouts while updating HNSW & tsvector indexes.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Config & Auto-Credential Loading
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKED_FILE = str(BASE_DIR / "output" / "chunked_data.jsonl")
CHECKPOINT_FILE = str(BASE_DIR / "output" / "ingestion_checkpoint.txt")
FAILED_CHUNKS_FILE = str(BASE_DIR / "output" / "failed_chunks.jsonl")

# Automatically search for .env.local in chat or root
possible_env_paths = [
    BASE_DIR / "chat" / ".env.local",
    BASE_DIR.parent.parent / ".env",
    BASE_DIR / ".env",
]

for env_path in possible_env_paths:
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

# Also support Kaggle secrets if running on Kaggle
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    os.environ.setdefault("SUPABASE_URL", user_secrets.get_secret("SUPABASE_URL"))
    os.environ.setdefault("SUPABASE_KEY", user_secrets.get_secret("SUPABASE_KEY"))
except Exception:
    pass

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def upload_batch_with_retry(supabase_client, payload: list, max_retries: int = 3):
    """
    Uploads a batch to Supabase with exponential backoff.
    If a large batch times out, automatically splits it in half and tries sub-batches.
    """
    for attempt in range(max_retries):
        try:
            supabase_client.table("financial_chunks").upsert(
                payload, on_conflict="chunk_uid"
            ).execute()
            return True, None
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                last_err = str(e)

    # Fallback: if batch > 25, attempt splitting into two smaller sub-batches
    if len(payload) > 25:
        mid = len(payload) // 2
        ok1, err1 = upload_batch_with_retry(supabase_client, payload[:mid], max_retries=2)
        ok2, err2 = upload_batch_with_retry(supabase_client, payload[mid:], max_retries=2)
        if ok1 and ok2:
            return True, None
        return False, err1 or err2

    return False, last_err


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed financial chunks and upload to Supabase pgvector.")
    parser.add_argument("--batch-size", type=int, default=50, help="Number of chunks per upload batch (default: 50)")
    parser.add_argument("--input-file", default=DEFAULT_CHUNKED_FILE, help="Path to chunked_data.jsonl")
    parser.add_argument("--force", action="store_true", help="Start from chunk 0 (clears previous checkpoint)")
    args = parser.parse_args()

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("\n[ERROR] Missing Supabase credentials!")
        print("Please ensure `backend/RAG/chat/.env.local` exists with:")
        print("  NEXT_PUBLIC_SUPABASE_URL=https://<your-project>.supabase.co")
        print("  SUPABASE_SERVICE_ROLE_KEY=ey...\n")
        sys.exit(1)

    print("=" * 68)
    print("  FinRAG Database Ingestion Pipeline (Stage 3)")
    print("=" * 68)
    print(f"  Target Supabase:  {SUPABASE_URL[:35]}...")
    print(f"  Batch Size:       {args.batch_size}")
    print(f"  Input File:       {args.input_file}")
    print("=" * 68 + "\n")

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file '{args.input_file}' not found. Please run chunk.py first.")
        sys.exit(1)

    # Initialize Supabase client
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Load Model
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading embedding model 'BAAI/bge-base-en-v1.5' on {device.upper()}...")
    embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

    # Load Chunks
    print("\nReading chunk records from disk...")
    records = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    total_chunks = len(records)
    print(f"Loaded {total_chunks} total chunks to index.")

    # Checkpoint management
    start_index = 0
    if args.force:
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)
        print("[INFO] --force specified: Starting fresh from chunk 0.")
    elif os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                saved_idx = int(f.read().strip())
                if 0 < saved_idx < total_chunks:
                    start_index = saved_idx
                    print(f"[RESUME] Resuming from checkpoint: chunk {start_index:,}/{total_chunks:,}")
        except Exception:
            start_index = 0

    if start_index >= total_chunks:
        print(f"\nAll {total_chunks:,} chunks have already been uploaded according to checkpoint!")
        print("To force re-ingestion, run: python ingest.py --force\n")
        return

    failed_chunks = []
    batch_size = args.batch_size

    print(f"\nEmbedding & Uploading {total_chunks - start_index:,} remaining chunks...\n")

    with tqdm(
        total=total_chunks,
        initial=start_index,
        desc="Ingesting to pgvector",
        unit="chunk",
        dynamic_ncols=True
    ) as pbar:

        for i in range(start_index, total_chunks, batch_size):
            batch = records[i:i + batch_size]
            texts_to_embed = [item["content"] for item in batch]

            # Generate 768-dim normalized embeddings
            embeddings = embed_model.encode(
                texts_to_embed,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            ).tolist()

            # Format payload for PostgreSQL table schema
            payload = []
            for idx, item in enumerate(batch):
                payload.append({
                    "chunk_uid": item["chunk_uid"],
                    "ticker": item["ticker"],
                    "fiscal_year": int(item["fiscal_year"]),
                    "report_type": item["report_type"],
                    "chunk_type": item["chunk_type"],
                    "section": item.get("section", ""),
                    "page_start": int(item.get("page_start", 0)),
                    "page_end": int(item.get("page_end", 0)),
                    "token_count": int(item.get("token_count", 0)),
                    "content": item["content"],
                    "source": item.get("source", ""),
                    "embedding": embeddings[idx]
                })

            # Upload with retry & sub-batch fallback
            ok, err = upload_batch_with_retry(supabase, payload)

            if ok:
                pbar.update(len(batch))
                # Update checkpoint
                with open(CHECKPOINT_FILE, "w") as f:
                    f.write(str(i + len(batch)))
            else:
                print(f"\n[ERROR] Failed to upload batch at index {i}-{i+len(batch)}: {err}")
                failed_chunks.extend(payload)
                pbar.update(len(batch))

    # Log any failed chunks for safety
    if failed_chunks:
        with open(FAILED_CHUNKS_FILE, "w", encoding="utf-8") as f:
            for fc in failed_chunks:
                f.write(json.dumps(fc) + "\n")
        print(f"\n[WARN] {len(failed_chunks)} chunks failed to upload. Logged to `{FAILED_CHUNKS_FILE}`.")
    else:
        # Clean up checkpoint on 100% completion
        if os.path.exists(CHECKPOINT_FILE):
            os.remove(CHECKPOINT_FILE)

    # Fast count check
    print("\n" + "=" * 68)
    try:
        count_res = supabase.table("financial_chunks").select("id", count="exact").limit(1).execute()
        print(f"  Ingestion Complete! Total rows in Supabase: {count_res.count:,}")
    except Exception as e:
        print(f"  Ingestion Complete! (Count check: {e})")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
