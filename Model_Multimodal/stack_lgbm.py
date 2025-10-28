# stack_lgbm.py
# Train LightGBM on frozen features and export OOF + test predictions for blending.

import os
import json
import numpy as np
import torch
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from utils_metrics import smape, secondary_metrics


def load_pt(path):
    print(f"[LGBM] Loading tensors: {path}")
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


def build_cv_strata(y, n_bins=10, min_count=5):
    """Build decile-like bins on log-price; merge rare bins to satisfy min_count."""
    y = np.asarray(y, float)
    ylog = np.log(np.clip(y, 1e-6, None))
    # Initial equal-width bins
    raw = np.floor(np.interp(ylog, (ylog.min(), ylog.max()), (0, n_bins - 1))).astype(int)
    # Merge sparse bins into neighbors until each has >= min_count
    bins = raw.copy()
    changed = True
    while changed:
        changed = False
        vals, counts = np.unique(bins, return_counts=True)
        sparse = vals[counts < min_count]
        if len(sparse) == 0:
            break
        for v in sparse:
            # Merge to nearest neighbor bin
            candidates = vals[vals != v]
            if len(candidates) == 0:
                continue
            tgt = candidates[np.argmin(np.abs(candidates - v))]
            bins[bins == v] = tgt
            changed = True
    # Reindex bins to 0..K-1
    uniq = np.unique(bins)
    remap = {u: i for i, u in enumerate(uniq)}
    bins = np.vectorize(remap.get)(bins).astype(int)
    return bins


def train_lgbm(train_pt, test_pt, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    tr = load_pt(train_pt)
    te = load_pt(test_pt)

    X_tr = np.hstack([tr["X_text"], tr["X_img"], tr["X_tab"]]).astype(np.float32)
    y_tr = np.asarray(tr["price"], dtype=np.float32)
    X_te = np.hstack([te["X_text"], te["X_img"], te["X_tab"]]).astype(np.float32)

    print(f"[LGBM] Train shape: {X_tr.shape}, Test shape: {X_te.shape}")

    strata = build_cv_strata(y_tr, n_bins=10, min_count=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    params = {
        "objective": "mae",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 40,
        "max_depth": -1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "n_estimators": 4000,
        "force_row_wise": True
    }

    oof = np.zeros(len(y_tr), dtype=np.float32)
    test_preds = []
    folds_summary = []

    for k, (tr_idx, va_idx) in enumerate(skf.split(X_tr, strata)):
        print(f"\n[LGBM] ===== Fold {k} / 5 =====")
        dtr = lgb.Dataset(X_tr[tr_idx], label=y_tr[tr_idx])
        dva = lgb.Dataset(X_tr[va_idx], label=y_tr[va_idx], reference=dtr)

        callbacks = [
            lgb.early_stopping(stopping_rounds=200, verbose=True),
            lgb.log_evaluation(period=200)
        ]

        model = lgb.train(
            params=params,
            train_set=dtr,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            num_boost_round=4000,
            callbacks=callbacks
        )

        model.save_model(os.path.join(out_dir, f"lgbm_fold{k}.pkl"))
        with open(os.path.join(out_dir, f"lgbm_fold{k}.txt"), "w", encoding="utf-8") as f:
            f.write(model.model_to_string())

        oof[va_idx] = model.predict(X_tr[va_idx], num_iteration=model.best_iteration)
        test_preds.append(model.predict(X_te, num_iteration=model.best_iteration))

        sm = float(smape(y_tr[va_idx], oof[va_idx]))
        sec = secondary_metrics(y_tr[va_idx], oof[va_idx])
        folds_summary.append({"fold": int(k), "smape": sm, **sec})
        print(f"[LGBM] Fold {k} SMAPE: {sm:.4f}%")

    oof_sm = float(smape(y_tr, oof))
    print(f"\n[LGBM] OOF SMAPE (ensemble): {oof_sm:.4f}%")

    P_te = np.mean(np.stack(test_preds, axis=0), axis=0)

    np.save(os.path.join(out_dir, "oof_lgbm.npy"), oof)
    np.save(os.path.join(out_dir, "test_lgbm.npy"), P_te)

    with open(os.path.join(out_dir, "cv_summary_lgbm.json"), "w", encoding="utf-8") as f:
        json.dump({"oof_smape": oof_sm, "folds": folds_summary, "params": params}, f, indent=2)

    print(f"[LGBM] Saved OOF/test predictions and summary to: {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_pt", default="caches/train_data.pt")
    ap.add_argument("--test_pt", default="caches/test_data.pt")
    ap.add_argument("--out_dir", default="outputs/stacker")
    args = ap.parse_args()
    train_lgbm(args.train_pt, args.test_pt, args.out_dir)
