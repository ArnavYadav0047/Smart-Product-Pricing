# prepare_data.py
# Purpose: Build consolidated train/test tensors: text_emb, image_emb, tabular, price.

import os
import torch
import pandas as pd
import numpy as np
from features import build_tabular

def join_embeddings(df, text_cache_path, image_cache_path):
    print(f"[PREP] Loading text cache: {text_cache_path}")
    text_obj = torch.load(text_cache_path)
    print(f"[PREP] Loading image cache: {image_cache_path}")
    image_obj = torch.load(image_cache_path)
    tid2idx = {sid:i for i,sid in enumerate(text_obj["ids"])}
    iid2idx = {sid:i for i,sid in enumerate(image_obj["ids"])}
    tmat = text_obj["emb"]
    imat = image_obj["emb"]
    tvecs = []
    ivecs = []
    for sid in df["sample_id"]:
        tvecs.append(tmat[tid2idx[sid]].numpy())
        ivecs.append(imat[iid2idx[sid]].numpy())
    return np.vstack(tvecs), np.vstack(ivecs)

def build_dataset(csv_path, text_cache, image_cache, out_path):
    print(f"[PREP] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print("[PREP] Building tabular features...")
    tab = build_tabular(df)
    print("[PREP] Joining text & image embeddings...")
    X_text, X_img = join_embeddings(df, text_cache, image_cache)
    if "brand" in tab.columns:
        tab = tab.drop(columns=["brand"])
    X_tab = tab.to_numpy(dtype=float)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "sample_id": df["sample_id"].tolist(),
        "X_text": X_text,
        "X_img": X_img,
        "X_tab": X_tab,
        "price": df["price"].values if "price" in df.columns else None
    }
    torch.save(payload, out_path)
    print(f"[PREP] Saved dataset tensors to: {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text_cache", required=True)
    ap.add_argument("--image_cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build_dataset(args.csv, args.text_cache, args.image_cache, args.out)
