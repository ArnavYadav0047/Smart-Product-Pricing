# embed_text.py
# Purpose: Convert catalog_content to 768-d pooled embeddings using HF encoders.
# Recommended: roberta-base with slow tokenizer on Windows/Python 3.13 for stability.

import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

def build_text_embeddings(csv_path, model_name, cache_dir, max_len=256, device=None, batch_size=32):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[TEXT] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Force slow tokenizer to avoid Windows fast-tokenizer/tiktoken conversion issues
    print(f"[TEXT] Loading tokenizer/model (slow tokenizer): {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModel.from_pretrained(model_name)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    texts = df["catalog_content"].fillna("").astype(str).tolist()
    all_vecs = []

    print(f"[TEXT] Encoding {len(texts)} samples (batch={batch_size}, max_len={max_len}) on {device}")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            vec = mean_pool(out.last_hidden_state, enc["attention_mask"])
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_vecs.append(vec.cpu())

    embs = torch.cat(all_vecs, dim=0)
    out_path = os.path.join(cache_dir, "text_emb.pt")
    torch.save({"ids": df["sample_id"].tolist(), "emb": embs}, out_path)
    print(f"[TEXT] Saved embeddings to: {out_path} | shape={tuple(embs.shape)}")
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to train.csv or test.csv")
    # Default to roberta-base for Windows stability; you can still override via CLI
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--cache_dir", default="caches/text_train")
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    build_text_embeddings(
        csv_path=args.csv,
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        max_len=args.max_len,
        batch_size=args.batch_size
    )
