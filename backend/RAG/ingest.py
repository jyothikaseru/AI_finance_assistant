
import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from kaggle_secrets import UserSecretsClient

# 1. Fetch credentials securely
user_secrets = UserSecretsClient()
SUPABASE_URL = user_secrets.get_secret("SUPABASE_URL")
SUPABASE_KEY = user_secrets.get_secret("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# 2. Load BAAI BGE model on GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading embedding model 'BAAI/bge-base-en-v1.5' on {device.upper()}...")
embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)

# 3. Read chunks
CHUNKED_FILE = "output/chunked_data.jsonl"
records = []
with open(CHUNKED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line.strip()))

total_chunks = len(records)
print(f"Total chunks to embed and upload: {total_chunks}")

# 4. Batch Embed & Upload
BATCH_SIZE = 100

for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Uploading to Supabase"):
    batch = records[i:i + BATCH_SIZE]
    texts_to_embed = [item["content"] for item in batch]
    
    # Generate 768-dim embeddings
    embeddings = embed_model.encode(
        texts_to_embed, 
        normalize_embeddings=True, 
        batch_size=len(texts_to_embed),
        show_progress_bar=False
    ).tolist()

    # Prepare payload matching table schema
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

    # Upload batch
    try:
        supabase.table("financial_chunks").upsert(payload, on_conflict="chunk_uid").execute()
    except Exception as e:
        print(f"\n[ERROR] on batch {i}-{i+BATCH_SIZE}: {e}")

# 5. Verify total uploaded
count_res = supabase.table("financial_chunks").select("id", count="exact").execute()
print(f"\nALL DONE! Total rows now in Supabase database: {count_res.count}")
