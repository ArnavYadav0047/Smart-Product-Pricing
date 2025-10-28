# predict.py
# Purpose: Predict on test, ensemble folds, optionally apply isotonic calibration,
# validate submission schema, and write test_out.csv.
# Updated: Safe torch.load for PyTorch 2.6+ (weights_only=False + allowlist).

import os
import json
import numpy as np
import pandas as pd
import torch
from train_fusion import FusionMLP
from utils_metrics import validate_submission_schema
import joblib

def safe_load_pt(path):
    """Load cached dataset tensors with weights_only=False and a safe allowlist (PyTorch 2.6+)."""
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            import numpy as _np
            add_safe_globals([_np.core.multiarray._reconstruct])
        except Exception:
            pass
        return torch.load(path, weights_only=False, map_location="cpu")

def load_dataset(pt_path):
    print(f"[PRED] Loading test tensors: {pt_path}")
    return safe_load_pt(pt_path)

def predict_test(train_summary_dir, test_pt, test_csv, sample_out_csv, out_csv):
    data = load_dataset(test_pt)
    X_text = torch.tensor(data["X_text"], dtype=torch.float32)
    X_img  = torch.tensor(data["X_img"], dtype=torch.float32)
    X_tab  = torch.tensor(data["X_tab"], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_text = X_text.to(device); X_img = X_img.to(device); X_tab = X_tab.to(device)

    print(f"[PRED] Loading CV summary: {os.path.join(train_summary_dir, 'cv_summary.json')}")
    with open(os.path.join(train_summary_dir, "cv_summary.json")) as f:
        summ = json.load(f)
    cfg = summ["config"]

    d_text = int(X_text.shape[1]); d_img = int(X_img.shape[1]); d_tab = int(X_tab.shape[1])
    preds = []
    k = 0
    while True:
        mpath = os.path.join(train_summary_dir, f"fusion_fold{k}.pt")
        if not os.path.exists(mpath): break
        print(f"[PRED] Loading fold model: {mpath}")
        model = FusionMLP(d_text, d_img, d_tab, hidden=cfg["hidden"], dropout=cfg["dropout"], positivity=cfg["positivity"]).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device))
        model.eval()
        with torch.no_grad():
            yp = model(X_text, X_img, X_tab).cpu().numpy().ravel()
        preds.append(yp)
        k += 1
    if len(preds) == 0:
        raise RuntimeError("No fold models found.")

    P = np.stack(preds, axis=0).mean(axis=0)

    if summ.get("calibration_used", False) and os.path.exists(os.path.join(train_summary_dir, "isotonic.pkl")):
        print("[PRED] Applying isotonic calibration...")
        ir = joblib.load(os.path.join(train_summary_dir, "isotonic.pkl"))
        P = ir.predict(P)

    # Enforce positivity and numerical safety
    P = np.clip(P, 1e-6, None)
    P = np.nan_to_num(P, nan=1.0, posinf=1e6, neginf=1.0)

    # Build and validate submission
    test_df = pd.read_csv(test_csv)
    sub = pd.DataFrame({"sample_id": test_df["sample_id"], "price": P})
    sample_sub = pd.read_csv(sample_out_csv)

    print("[PRED] Validating submission schema against sample_test_out.csv...")
    validate_submission_schema(test_df, sub, sample_sub)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Submission ready: {out_csv} (rows={len(sub)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_summary_dir", default="outputs/fusion")
    ap.add_argument("--test_pt", default="caches/test_data.pt")
    ap.add_argument("--test_csv", default="dataset/test.csv")
    ap.add_argument("--sample_out_csv", default="dataset/sample_test_out.csv")
    ap.add_argument("--out_csv", default="outputs/test_out.csv")
    args = ap.parse_args()
    predict_test(args.train_summary_dir, args.test_pt, args.test_csv, args.sample_out_csv, args.out_csv)
# predict.py
# Purpose: Predict on test, ensemble folds, optionally apply isotonic calibration,
# validate submission schema, and write test_out.csv.
# Updated: Safe torch.load for PyTorch 2.6+ (weights_only=False + allowlist).

import os
import json
import numpy as np
import pandas as pd
import torch
from train_fusion import FusionMLP
from utils_metrics import validate_submission_schema
import joblib

def safe_load_pt(path):
    """Load cached dataset tensors with weights_only=False and a safe allowlist (PyTorch 2.6+)."""
    try:
        return torch.load(path, weights_only=False, map_location="cpu")
    except Exception:
        try:
            from torch.serialization import add_safe_globals
            import numpy as _np
            add_safe_globals([_np.core.multiarray._reconstruct])
        except Exception:
            pass
        return torch.load(path, weights_only=False, map_location="cpu")

def load_dataset(pt_path):
    print(f"[PRED] Loading test tensors: {pt_path}")
    return safe_load_pt(pt_path)

def predict_test(train_summary_dir, test_pt, test_csv, sample_out_csv, out_csv):
    data = load_dataset(test_pt)
    X_text = torch.tensor(data["X_text"], dtype=torch.float32)
    X_img  = torch.tensor(data["X_img"], dtype=torch.float32)
    X_tab  = torch.tensor(data["X_tab"], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_text = X_text.to(device); X_img = X_img.to(device); X_tab = X_tab.to(device)

    print(f"[PRED] Loading CV summary: {os.path.join(train_summary_dir, 'cv_summary.json')}")
    with open(os.path.join(train_summary_dir, "cv_summary.json")) as f:
        summ = json.load(f)
    cfg = summ["config"]

    d_text = int(X_text.shape[1]); d_img = int(X_img.shape[1]); d_tab = int(X_tab.shape[1])
    preds = []
    k = 0
    while True:
        mpath = os.path.join(train_summary_dir, f"fusion_fold{k}.pt")
        if not os.path.exists(mpath): break
        print(f"[PRED] Loading fold model: {mpath}")
        model = FusionMLP(d_text, d_img, d_tab, hidden=cfg["hidden"], dropout=cfg["dropout"], positivity=cfg["positivity"]).to(device)
        model.load_state_dict(torch.load(mpath, map_location=device))
        model.eval()
        with torch.no_grad():
            yp = model(X_text, X_img, X_tab).cpu().numpy().ravel()
        preds.append(yp)
        k += 1
    if len(preds) == 0:
        raise RuntimeError("No fold models found.")

    P = np.stack(preds, axis=0).mean(axis=0)

    if summ.get("calibration_used", False) and os.path.exists(os.path.join(train_summary_dir, "isotonic.pkl")):
        print("[PRED] Applying isotonic calibration...")
        ir = joblib.load(os.path.join(train_summary_dir, "isotonic.pkl"))
        P = ir.predict(P)

    # Enforce positivity and numerical safety
    P = np.clip(P, 1e-6, None)
    P = np.nan_to_num(P, nan=1.0, posinf=1e6, neginf=1.0)

    # Build and validate submission
    test_df = pd.read_csv(test_csv)
    sub = pd.DataFrame({"sample_id": test_df["sample_id"], "price": P})
    sample_sub = pd.read_csv(sample_out_csv)

    print("[PRED] Validating submission schema against sample_test_out.csv...")
    validate_submission_schema(test_df, sub, sample_sub)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    sub.to_csv(out_csv, index=False)
    print(f"Submission ready: {out_csv} (rows={len(sub)})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_summary_dir", default="outputs/fusion")
    ap.add_argument("--test_pt", default="caches/test_data.pt")
    ap.add_argument("--test_csv", default="dataset/test.csv")
    ap.add_argument("--sample_out_csv", default="dataset/sample_test_out.csv")
    ap.add_argument("--out_csv", default="outputs/test_out.csv")
    args = ap.parse_args()
    predict_test(args.train_summary_dir, args.test_pt, args.test_csv, args.sample_out_csv, args.out_csv)
