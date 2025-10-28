# embed_image.py
# Purpose: Convert images to 768-d pooled embeddings using timm ViT-Base.
# Supports separate image folders for train/test and flexible filename resolution.

import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import timm
import torchvision.transforms as T

def default_transform(img_size=224):
    return T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def resolve_path(images_dir, sample_id, exts=(".jpg", ".jpeg", ".png", ".webp", ".bmp")):
    """Try common extensions for a given sample_id; return path or None."""
    for ext in exts:
        p = os.path.join(images_dir, f"{sample_id}{ext}")
        if os.path.exists(p):
            return p
    return None

def build_image_embeddings(csv_path, images_dir, model_name, cache_dir, img_size=224, device=None, batch_size=32):
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[IMG] Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IMG] Loading vision model: {model_name} on {device}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()

    tfm = default_transform(img_size)
    print(f"[IMG] Resolving image paths from: {images_dir}")
    paths = []
    missing = 0
    for sid in df["sample_id"]:
        p = resolve_path(images_dir, sid)
        if p is None:
            missing += 1
            p = ""  # placeholder -> zero tensor
        paths.append(p)
    print(f"[IMG] Path resolution done. Missing files (will use zero tensor): {missing}")

    print(f"[IMG] Encoding {len(paths)} images (batch={batch_size}, size={img_size})")
    tensors = []
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(len(paths))):
            p = paths[i]
            try:
                if p and os.path.exists(p):
                    img = Image.open(p).convert("RGB")
                    x = tfm(img)
                else:
                    raise FileNotFoundError
            except Exception:
                x = torch.zeros(3, img_size, img_size)
            tensors.append(x)

            if len(tensors) == batch_size:
                batch = torch.stack(tensors).to(device)
                vec = model(batch)
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
                all_embs.append(vec.cpu())
                tensors = []

        if tensors:
            batch = torch.stack(tensors).to(device)
            vec = model(batch)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            all_embs.append(vec.cpu())

    embs = torch.cat(all_embs, dim=0) if len(all_embs) else torch.empty(0, 768)
    out_path = os.path.join(cache_dir, "image_emb.pt")
    torch.save({"ids": df["sample_id"].tolist(), "emb": embs}, out_path)
    print(f"[IMG] Saved embeddings to: {out_path} | shape={tuple(embs.shape)}")
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--model_name", default="vit_base_patch16_224")
    ap.add_argument("--cache_dir", default="caches/image_train")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    build_image_embeddings(args.csv, args.images_dir, args.model_name, args.cache_dir, args.img_size, batch_size=args.batch_size)
