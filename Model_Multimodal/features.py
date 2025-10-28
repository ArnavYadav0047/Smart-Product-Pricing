# features.py
# Purpose: Convert catalog_content into robust structured/tabular signals.

import re
import numpy as np
import pandas as pd

_UNIT_MAP = {
    "ml": ("ml", 1.0),
    "l": ("ml", 1000.0),
    "g": ("g", 1.0),
    "kg": ("g", 1000.0),
    "oz": ("g", 28.3495),
    "lb": ("g", 453.592),
}

KEYWORDS = ["wireless","stainless","organic","premium","refurbished","pro","ultra","mini","max"]

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def parse_brand(text: str):
    m = re.search(r"\bby\s+([a-z0-9\-\_ ]{2,30})", text)
    if m:
        b = m.group(1).strip()
        return re.sub(r"[^a-z0-9 ]","",b)[:30]
    toks = re.sub(r"[^a-z0-9 ]"," ",text).split()
    if not toks:
        return "other"
    cand = " ".join(toks[:2])
    return cand[:30]

def parse_pack_and_size(text: str):
    pack_qty = 1
    total_ml = 0.0
    total_g = 0.0
    m = re.search(r"(\d+)\s*[x×]\s*(\d+(\.\d+)?)(ml|l|g|kg|oz|lb)", text)
    if m:
        pack_qty = int(m.group(1))
        val = float(m.group(2))
        unit = m.group(4)
        base, scale = _UNIT_MAP.get(unit, (unit, 1.0))
        if base == "ml":
            total_ml = pack_qty * val * scale
        elif base == "g":
            total_g = pack_qty * val * scale
    else:
        m2 = re.search(r"(\d+(\.\d+)?)(ml|l|g|kg|oz|lb)", text)
        if m2:
            val = float(m2.group(1))
            unit = m2.group(3)
            base, scale = _UNIT_MAP.get(unit, (unit, 1.0))
            if base == "ml":
                total_ml = val * scale
            elif base == "g":
                total_g = val * scale
    return pack_qty, total_ml, total_g

def build_tabular(df: pd.DataFrame) -> pd.DataFrame:
    txt = df["catalog_content"].fillna("").apply(clean_text)
    brand = txt.apply(parse_brand)
    pack, vol_ml, wt_g = zip(*txt.apply(parse_pack_and_size))
    out = pd.DataFrame({
        "pack_qty": np.array(pack, dtype=float),
        "total_volume_ml": np.array(vol_ml, dtype=float),
        "total_weight_g": np.array(wt_g, dtype=float),
        "len_chars": txt.apply(len).astype(float),
        "len_words": txt.apply(lambda s: len(s.split())).astype(float),
        "digit_count": txt.apply(lambda s: sum(ch.isdigit() for ch in s)).astype(float),
        "brand": brand.astype(str),
    })
    for kw in KEYWORDS:
        out[f"kw_{kw}"] = txt.str.contains(rf"\b{kw}\b").astype(float)
    num_cols = ["pack_qty","total_volume_ml","total_weight_g","len_chars","len_words","digit_count"] + [f"kw_{k}" for k in KEYWORDS]
    for c in num_cols:
        q1, q99 = out[c].quantile(0.01), out[c].quantile(0.99)
        out[c] = out[c].clip(q1, q99)
    top_brands = out["brand"].value_counts().head(1000).index
    out["brand_enc"] = out["brand"].where(out["brand"].isin(top_brands), "other")
    brand_dummies = pd.get_dummies(out["brand_enc"], prefix="brand", sparse=False)
    out = pd.concat([out.drop(columns=["brand_enc"]), brand_dummies], axis=1)
    return out
