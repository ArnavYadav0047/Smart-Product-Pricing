import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
import lightgbm as lgb


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    return 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-6))


def extract_tsq(text):
    text = str(text).lower()
    OZ_PER_KG = 35.274
    OZ_PER_G = 0.035274
    OZ_PER_L = 33.814

    m_mul = re.findall(r'(\d+)\s*[x×*]\s*(\d+\.?\d*)\s*(oz|g|kg|ml|l|ct|pcs|pack|bottle|lb|lbs)', text)
    if m_mul:
        total_quantity = 0.0
        for count, qty, unit in m_mul:
            qty = float(qty)
            count = float(count)
            if unit in ["lb", "lbs"]:
                qty *= 16.0
            elif unit == "kg":
                qty *= OZ_PER_KG
            elif unit == "g":
                qty *= OZ_PER_G
            elif unit == "l":
                qty *= OZ_PER_L
            total_quantity += count * qty
        if total_quantity > 1.0:
            return total_quantity

    m_count = re.search(r'(?:pack of\s*|case of\s*|^)(\d+)\s*(ct|count|pcs|pack|bottle)?', text)
    if m_count:
        return float(m_count.group(1))

    m_unit = re.search(r'(\d+\.?\d*)\s*(oz|ounce|lb|lbs|fl oz|g|kg|ml|l)', text)
    if m_unit:
        val = float(m_unit.group(1))
        unit = m_unit.group(2)
        if unit in ["lb", "lbs"]:
            return val * 16.0
        if unit == "kg":
            return val * OZ_PER_KG
        if unit == "g":
            return val * OZ_PER_G
        if unit == "l":
            return val * OZ_PER_L
        if unit in ["fl oz", "ounce", "oz", "ml"]:
            return val

    return 1.0


def fast_nup_feature_engineering(df, is_train=True):
    df_fe = df.copy()
    df_fe["catalog_content"] = df_fe["catalog_content"].fillna("")
    df_fe["TSQ"] = df_fe["catalog_content"].progress_apply(extract_tsq)
    df_fe["log_TSQ"] = np.log1p(df_fe["TSQ"])
    if is_train:
        df_fe["log_price"] = np.log1p(df_fe["price"])
    df_fe["text_len"] = df_fe["catalog_content"].apply(len)
    df_fe["word_count"] = df_fe["catalog_content"].apply(lambda x: len(str(x).split()))

    flags_raw = [
        "vegan", "gluten free", "non-gmo", "peanut free", "dairy free",
        "organic", "kosher", "egg free", "natural", "gluten-free",
        "no sugar", "sugar free"
    ]
    flags = list({flag.replace(' ', '_').replace('-', '_') for flag in flags_raw})
    for flag in flags:
        df_fe[f"{flag}_flag"] = df_fe["catalog_content"].str.contains(flag.replace('_', ' '), case=False, na=False).astype(int)

    df_fe["premium_flag"] = df_fe["catalog_content"].str.contains("gourmet|premium|luxury|artisanal", case=False, na=False).astype(int)

    def extract_brand(txt):
        brands = re.findall(r"\b[A-Z][A-Za-z0-9\-&]+\b", str(txt))
        return brands[0] if brands else "unknown"
    df_fe["brand"] = df_fe["catalog_content"].apply(extract_brand)
    return df_fe


def train_lgbm(X_train, y_train, X_val, y_val, params, categorical_features):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=True)],
        categorical_feature=categorical_features
    )
    return model


def main():
    tqdm.pandas()

    train_csv = "dataset/train.csv"
    test_csv = "dataset/test.csv"
    output_file = "results/test_out.csv"
    os.makedirs("results", exist_ok=True)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print("Feature engineering...")
    train_features = fast_nup_feature_engineering(train_df, True)
    test_features = fast_nup_feature_engineering(test_df, False)

    vectorizer = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 3), token_pattern=r"\b[a-zA-Z]{3,}\b")
    X_text_train = vectorizer.fit_transform(train_features["catalog_content"])
    X_text_test = vectorizer.transform(test_features["catalog_content"])

    combined_brands = pd.concat([train_features["brand"], test_features["brand"]]).astype(str)
    brand_encoder = combined_brands.factorize()
    train_brand_enc = brand_encoder[0][:len(train_df)]
    test_brand_enc = brand_encoder[0][len(train_df):]

    flags_raw = [
        "vegan", "gluten free", "non-gmo", "peanut free", "dairy free",
        "organic", "kosher", "egg free", "natural", "gluten-free",
        "no sugar", "sugar free"
    ]
    flags = list({f"{flag.replace(' ', '_').replace('-', '_')}_flag" for flag in flags_raw})

    train_flags = [col for col in train_features.columns if col in flags]
    test_flags = [col for col in test_features.columns if col in flags]

    num_feat = ["log_TSQ", "text_len", "word_count", "premium_flag"]

    train_tab_features = train_features[num_feat + train_flags].copy()
    test_tab_features = test_features[num_feat + test_flags].copy()

    train_tab_features["brand_enc"] = train_brand_enc
    test_tab_features["brand_enc"] = test_brand_enc

    X_text_train_df = pd.DataFrame(X_text_train.toarray(), columns=[f"tfidf_{i}" for i in range(X_text_train.shape[1])])
    X_text_test_df = pd.DataFrame(X_text_test.toarray(), columns=[f"tfidf_{i}" for i in range(X_text_test.shape[1])])

    X_train = pd.concat([train_tab_features.reset_index(drop=True), X_text_train_df.reset_index(drop=True)], axis=1)
    X_test = pd.concat([test_tab_features.reset_index(drop=True), X_text_test_df.reset_index(drop=True)], axis=1)

    if X_train.columns.duplicated().sum() > 0:
        X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    if X_test.columns.duplicated().sum() > 0:
        X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    y_train = train_features["log_price"].values
    y_train = np.clip(y_train, np.percentile(y_train, 1), np.percentile(y_train, 99))

    params1 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 6000,
        "learning_rate": 0.012,
        "num_leaves": 256,
        "max_depth": 14,
        "min_child_samples": 30,
        "lambda_l1": 0.3,
        "lambda_l2": 0.4,
        "subsample": 0.85,
        "feature_fraction": 0.8,
        "bagging_freq": 5,
        "bagging_fraction": 0.8,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 42,
    }
    params2 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 5000,
        "learning_rate": 0.01,
        "num_leaves": 200,
        "max_depth": 12,
        "min_child_samples": 40,
        "lambda_l1": 0.4,
        "lambda_l2": 0.5,
        "subsample": 0.9,
        "feature_fraction": 0.85,
        "bagging_freq": 5,
        "bagging_fraction": 0.85,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 24,
    }
    params3 = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": 4000,
        "learning_rate": 0.015,
        "num_leaves": 150,
        "max_depth": 10,
        "min_child_samples": 50,
        "lambda_l1": 0.5,
        "lambda_l2": 0.6,
        "subsample": 0.95,
        "feature_fraction": 0.7,
        "bagging_freq": 5,
        "bagging_fraction": 0.9,
        "n_jobs": -1,
        "verbose": -1,
        "seed": 10,
    }

    categorical_feats = ["brand_enc"]

    print("Running 5-fold cross-validation with three-model ensemble...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_smapes = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        print(f"Fold {fold} training:")
        Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
        ytr, yval = y_train[train_idx], y_train[val_idx]

        model1 = train_lgbm(Xtr, ytr, Xval, yval, params1, categorical_feats)
        model2 = train_lgbm(Xtr, ytr, Xval, yval, params2, categorical_feats)
        model3 = train_lgbm(Xtr, ytr, Xval, yval, params3, categorical_feats)

        pred_val1 = np.expm1(model1.predict(Xval))
        pred_val2 = np.expm1(model2.predict(Xval))
        pred_val3 = np.expm1(model3.predict(Xval))
        pred_val = (pred_val1 + pred_val2 + pred_val3) / 3
        pred_val[pred_val < 0] = 0.01

        smape_cv = smape(np.expm1(yval), pred_val)
        cv_smapes.append(smape_cv)
        print(f"Fold {fold} SMAPE: {smape_cv:.3f}")

    print(f"Mean CV SMAPE: {np.mean(cv_smapes):.3f} ± {np.std(cv_smapes):.3f}")

    print("Training final three-model ensemble on all training data...")
    final_model1 = lgb.LGBMRegressor(**params1)
    final_model1.fit(X_train, y_train, categorical_feature=categorical_feats)

    final_model2 = lgb.LGBMRegressor(**params2)
    final_model2.fit(X_train, y_train, categorical_feature=categorical_feats)

    final_model3 = lgb.LGBMRegressor(**params3)
    final_model3.fit(X_train, y_train, categorical_feature=categorical_feats)

    preds1_log = final_model1.predict(X_test)
    preds2_log = final_model2.predict(X_test)
    preds3_log = final_model3.predict(X_test)
    preds = (np.expm1(preds1_log) + np.expm1(preds2_log) + np.expm1(preds3_log)) / 3
    preds[preds < 0] = 0.01

    output_df = pd.DataFrame({"sample_id": test_df["sample_id"], "price": preds})
    output_df.to_csv(output_file, index=False)

    print(f"Test predictions saved to {output_file}")
    print(f"Total features used: {X_train.shape[1]}")
    print("Three-model ensemble pipeline completed successfully!")


if __name__ == "__main__":
    main()
