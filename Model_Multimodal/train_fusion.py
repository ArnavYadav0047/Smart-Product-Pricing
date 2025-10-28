# train_fusion.py
# Late-fusion MLP on cached embeddings + tabular with 5-fold CV.
# Updates:
# - Safe torch.load for PyTorch 2.6+ (weights_only=False + allowlist)
# - Robust --config parsing (raw JSON or path)
# - Saves OOF arrays (oof_pred and oof_true) into cv_summary.json for blending

import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from utils_metrics import smape, decile_smape, calibration_bins, quantile_error_bands, residual_stats, secondary_metrics

def set_seed(sd=42):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

def safe_load_pt(path):
    # Prefer weights_only=False for cached dicts created by our scripts (PyTorch 2.6+ default changed)
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

class FusionMLP(nn.Module):
    def __init__(self, d_text, d_img, d_tab, hidden=(512,256), dropout=0.2, positivity="softplus"):
        super().__init__()
        layers = []
        in_d = d_text + d_img + d_tab
        for h in hidden:
            layers += [nn.Linear(in_d, h), nn.GELU(), nn.Dropout(dropout)]
            in_d = h
        self.backbone = nn.Sequential(*layers)
        self.out = nn.Linear(in_d, 1)
        self.positivity = positivity
        self.softplus = nn.Softplus()

    def forward(self, x_text, x_img, x_tab):
        x = torch.cat([x_text, x_img, x_tab], dim=1)
        h = self.backbone(x)
        z = self.out(h)
        return self.softplus(z) + 1e-6 if self.positivity == "softplus" else torch.exp(z)

def train_fold(data, idx_tr, idx_va, cfg, device):
    X_text = torch.tensor(data["X_text"][idx_tr], dtype=torch.float32).to(device)
    X_img  = torch.tensor(data["X_img"][idx_tr], dtype=torch.float32).to(device)
    X_tab  = torch.tensor(data["X_tab"][idx_tr], dtype=torch.float32).to(device)
    y_tr   = torch.tensor(data["price"][idx_tr], dtype=torch.float32).to(device).view(-1,1)

    X_text_v = torch.tensor(data["X_text"][idx_va], dtype=torch.float32).to(device)
    X_img_v  = torch.tensor(data["X_img"][idx_va], dtype=torch.float32).to(device)
    X_tab_v  = torch.tensor(data["X_tab"][idx_va], dtype=torch.float32).to(device)
    y_va     = torch.tensor(data["price"][idx_va], dtype=torch.float32).to(device).view(-1,1)

    d_text = int(X_text.shape[1]); d_img = int(X_img.shape[1]); d_tab = int(X_tab.shape[1])
    model = FusionMLP(d_text, d_img, d_tab, hidden=cfg["hidden"], dropout=cfg["dropout"], positivity=cfg["positivity"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    if cfg["loss"] == "log_mae":
        def loss_fn(pred, target):
            return torch.mean(torch.abs(torch.log(pred+1e-6) - torch.log(target+1e-6)))
    else:
        def loss_fn(pred, target):
            denom = (torch.abs(pred) + torch.abs(target) + 1e-3) / 2.0
            return torch.mean(torch.abs(pred - target) / denom)

    best_smape = float("inf"); best_state = None; wait = 0
    print("[TRAIN] Starting epochs...")
    for epoch in range(cfg["epochs"]):
        model.train()
        idx = torch.randperm(X_text.size(0))
        for i in range(0, len(idx), cfg["batch"]):
            b = idx[i:i+cfg["batch"]]
            yp = model(X_text[b], X_img[b], X_tab[b])
            loss = loss_fn(yp, y_tr[b])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            yp_v = model(X_text_v, X_img_v, X_tab_v).cpu().numpy().ravel()
            yv = y_va.cpu().numpy().ravel()
            val_smape = smape(yv, yp_v)
        print(f"[TRAIN] Epoch {epoch+1}/{cfg['epochs']} | Val SMAPE: {val_smape:.4f}% (best {best_smape:.4f}%)")
        if val_smape < best_smape - 1e-6:
            best_smape = val_smape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                print("[TRAIN] Early stopping triggered.")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        yp_tr = model(X_text, X_img, X_tab).cpu().numpy().ravel()
        yp_v  = model(X_text_v, X_img_v, X_tab_v).cpu().numpy().ravel()
    return model, yp_tr, yp_v

def run_cv(train_pt, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"[CV] Loading dataset tensors: {train_pt}")
    data = safe_load_pt(train_pt)
    n = len(data["price"]) if data["price"] is not None else 0

    kf = KFold(n_splits=5, shuffle=True, random_state=cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CV] Device: {device}")

    oof_pred = np.zeros(n, dtype=float)
    folds_info = []

    for k, (tr, va) in enumerate(kf.split(np.arange(n))):
        print(f"\n[CV] ===== Fold {k} / 5 =====")
        model, yp_tr, yp_va = train_fold(data, tr, va, cfg, device)
        oof_pred[va] = yp_va

        yv = data["price"][va]
        fold_metrics = {
            "fold": int(k),
            "smape": float(smape(yv, yp_va)),
            **secondary_metrics(yv, yp_va),
            "decile_smape": decile_smape(yv, yp_va),
            "residual_stats": residual_stats(yv, yp_va),
            "calibration_bins": calibration_bins(yv, yp_va),
            "quantile_error_bands": quantile_error_bands(yv, yp_va),
        }
        print(f"[CV] Fold {k} SMAPE: {fold_metrics['smape']:.4f}%")
        torch.save(model.state_dict(), os.path.join(out_dir, f"fusion_fold{k}.pt"))
        folds_info.append(fold_metrics)

    y = np.asarray(data["price"], dtype=float)
    oof_sm = smape(y, oof_pred)
    print(f"\n[CV] OOF SMAPE (ensemble): {oof_sm:.4f}%")
    print(f"FINAL VALIDATION SMAPE: {oof_sm:.2f}%")

    # Optional isotonic calibration on OOF
    print("[CV] Fitting isotonic calibration on OOF predictions...")
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_pred, y)
    oof_cal = ir.predict(oof_pred)
    oof_cal_sm = smape(y, oof_cal)
    cal_used = False
    if oof_cal_sm + 1e-6 < oof_sm:
        cal_used = True
        print(f"[CV] Calibration improved OOF SMAPE to {oof_cal_sm:.4f}% (used)")
        import joblib
        joblib.dump(ir, os.path.join(out_dir, "isotonic.pkl"))
    else:
        print(f"[CV] Calibration did not improve (cal={oof_cal_sm:.4f}% vs base={oof_sm:.4f}%). Not used.")

    # Save summary INCLUDING OOF arrays for blending
    summary = {
        "oof_smape": float(oof_sm),
        "oof_smape_calibrated": float(oof_cal_sm),
        "calibration_used": cal_used,
        "folds": folds_info,
        "config": cfg,
        "oof_pred": oof_pred.tolist(),   # Added: OOF predictions for blending
        "oof_true": y.tolist()           # Added: True targets for OOF weight search
    }
    with open(os.path.join(out_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[CV] Saved CV summary and fold models to: {out_dir}")

    return [os.path.join(out_dir, f"fusion_fold{k}.pt") for k in range(5)], cal_used

if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", default="caches/train_data.pt")
    ap.add_argument("--out_dir", default="outputs/fusion")
    ap.add_argument("--config", default="", help="Raw JSON or path to JSON file")
    args = ap.parse_args()

    cfg = {
        "seed": 42,
        "hidden": [512,256],
        "dropout": 0.2,
        "positivity": "softplus",
        "loss": "log_mae",
        "lr": 1e-3,
        "wd": 1e-2,
        "epochs": 30,
        "batch": 256,
        "patience": 5
    }

    if args.config:
        try:
            if os.path.exists(args.config):
                with open(args.config, "r", encoding="utf-8") as f:
                    cfg.update(_json.load(f))
            else:
                cfg.update(_json.loads(args.config))
        except Exception as e:
            print(f"[CFG] Warning: Failed to parse --config. Using defaults. Error: {e}")

    set_seed(cfg["seed"])
    run_cv(args.train_pt, cfg, args.out_dir)
