# blend_predict.py
# Blend MLP and LGBM predictions using OOF-optimized weight; write final submission.

import os
import json
import numpy as np
import pandas as pd
from utils_metrics import smape, validate_submission_schema

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def search_best_weight(y, oof_mlp, oof_lgbm):
    print("[BLEND] Searching best blend weight w in [0,1] to minimize SMAPE...")
    ws = np.linspace(0.0, 1.0, 101)
    best_w, best_s = 0.5, 1e9
    for w in ws:
        s = smape(y, w*oof_mlp + (1-w)*oof_lgbm)
        if s < best_s:
            best_s, best_w = s, w
    print(f"[BLEND] Best w={best_w:.2f} | OOF SMAPE={best_s:.4f}%")
    return float(best_w), float(best_s)

def apply_affine_if_available(yhat, affine_path):
    if os.path.exists(affine_path):
        try:
            cfg = load_json(affine_path)
            a, b = float(cfg.get("a", 1.0)), float(cfg.get("b", 0.0))
            print(f"[BLEND] Applying affine correction in log-space: a={a:.4f}, b={b:.4f}")
            yhat = np.exp(a*np.log(np.clip(yhat, 1e-6, None)) + b)
        except Exception as e:
            print(f"[BLEND] Affine correction load failed: {e}")
    return yhat

def blend_and_write(
    cv_summary_dir,           # outputs/fusion (for MLP CV and optional affine.json)
    stacker_dir,              # outputs/stacker (for LGBM predictions)
    test_csv, sample_out_csv, out_csv
):
    # Load OOF and test predictions
    print("[BLEND] Loading artifacts...")
    mlp_cv = load_json(os.path.join(cv_summary_dir, "cv_summary.json"))
    oof_mlp = np.array(mlp_cv.get("oof_pred", []), dtype=float) if "oof_pred" in mlp_cv else None

    # If MLP OOF not stored, reconstruct from training run not available; fail gracefully
    if oof_mlp is None or len(oof_mlp) == 0:
        # Ask user to dump OOF in train_fusion next time; meanwhile set w=0.0 (use LGBM only)
        print("[BLEND] MLP OOF not found in cv_summary.json; defaulting to LGBM-only blend (w=0.0).")
        best_w = 0.0
    else:
        y = np.array(load_json(os.path.join(cv_summary_dir, "cv_summary.json")).get("oof_true", []), dtype=float)
        if y.size == 0:
            print("[BLEND] True OOF targets missing; cannot optimize weight. Defaulting to w=0.5.")
            best_w = 0.5
        else:
            oof_lgbm = np.load(os.path.join(stacker_dir, "oof_lgbm.npy"))
            best_w, _ = search_best_weight(y, oof_mlp, oof_lgbm)

    test_mlp = None
    # Prefer to use predict.py output if available; otherwise cannot reconstruct MLP test preds here
    # For simplicity, expect predict.py already created outputs/test_out.csv with MLP predictions
    mlp_out_path = os.path.join(os.path.dirname(out_csv), "test_out.csv")
    if os.path.exists(mlp_out_path):
        df_mlp = pd.read_csv(mlp_out_path)
        test_mlp = df_mlp["price"].values
        print(f"[BLEND] Loaded existing MLP test predictions from: {mlp_out_path}")
    else:
        print("[BLEND] MLP test predictions not found; using LGBM-only (w=0.0).")
        best_w = 0.0

    test_lgbm = np.load(os.path.join(stacker_dir, "test_lgbm.npy"))

    if test_mlp is None:
        blended = test_lgbm
    else:
        blended = best_w * test_mlp + (1.0 - best_w) * test_lgbm

    # Optional affine correction if saved by training (affine.json)
    affine_path = os.path.join(cv_summary_dir, "affine.json")
    blended = apply_affine_if_available(blended, affine_path)

    # Positivity and numeric safety
    blended = np.clip(blended, 1e-6, None)
    blended = np.nan_to_num(blended, nan=1.0, posinf=1e6, neginf=1.0)

    # Write validated submission
    test_df = pd.read_csv(test_csv)
    sub = pd.DataFrame({"sample_id": test_df["sample_id"], "price": blended})
    sample_sub = pd.read_csv(sample_out_csv)

    print("[BLEND] Validating submission schema...")
    validate_submission_schema(test_df, sub, sample_sub)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"[BLEND] Submission ready: {out_csv} (rows={len(sub)})")

    # Save blend weight for record
    with open(os.path.join(stacker_dir, "blend_weight.json"), "w", encoding="utf-8") as f:
        json.dump({"w_mlp": float(best_w), "w_lgbm": float(1.0 - best_w)}, f, indent=2)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv_summary_dir", default="outputs/fusion")
    ap.add_argument("--stacker_dir", default="outputs/stacker")
    ap.add_argument("--test_csv", default="dataset/test.csv")
    ap.add_argument("--sample_out_csv", default="dataset/sample_test_out.csv")
    ap.add_argument("--out_csv", default="outputs/final/test_out_blended.csv")
    args = ap.parse_args()
    blend_and_write(args.cv_summary_dir, args.stacker_dir, args.test_csv, args.sample_out_csv, args.out_csv)
