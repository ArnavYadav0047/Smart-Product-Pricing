# Smart Product Pricing (Multimodal, Ensemble, SMAPE-optimized)

![Status](https://img.shields.io/badge/Status-Active-brightgreen) ![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🧠 Overview

**Smart Product Pricing** is a dual-model machine learning system designed to predict retail product prices from multimodal data — combining text, image, and tabular features. The repository hosts two complementary solutions:

1. **Model 1: LightGBM Ensemble** — a feature-engineered regression model using textual and tabular attributes.
2. **Model 2: Multimodal Fusion Model** — a gated neural fusion architecture combining frozen RoBERTa and ViT embeddings with tabular inputs.

Both models aim to minimize **SMAPE (Symmetric Mean Absolute Percentage Error)** for more stable price forecasting in e-commerce settings.

**Team Name:** Mystique  
**Authors:** Arnav Yadav, Soumadeep Samanta, Aditya Bhattacharya  
**Challenge:** ML Challenge 2025 – Product Price Forecasting  
**Status:** Active  
**Main Result:** Achieved a mean cross-validation SMAPE of **43%**

---

## 📚 Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset](#dataset)
3. [Getting Started & Installation](#getting-started--installation)
4. [Model 1: LightGBM Ensemble](#model-1-lightgbm-ensemble)
5. [Model 2: Multimodal Fusion Model](#model-2-multimodal-fusion-model)
6. [Results](#results)
7. [Visualization & Evaluation](#visualization--evaluation)
8. [Reproducibility](#reproducibility)

---

## 📁 Project Structure

```bash
smart-product-pricing/
│
├── README.md
│
├── dataset/
│   └── Train.csv
│   └── Test.csv
│
├── utils/
│   ├── download_images.py
│   └── run_download_commands.txt
│
├── Model_LightGBM/
│   ├── main.py
│   └── requirements.txt
│
└── Model_Multimodal/
    ├── blend_predict.py
    ├── embed_image.py
    ├── embed_text.py
    ├── features.py
    ├── predict.py
    ├── prepare_data.py
    ├── stack_lgbm.py
    ├── train_fusion.py
    ├── utils_metrics.py
    ├── requirements.txt
```

---

## 🗂️ Dataset

All training and testing CSV files are available at the following Google Drive link:
🔗 [Dataset Folder](https://drive.google.com/drive/folders/1NLHao1ennJhPiHT8YRp9qTqxVwGpzOFl)

Each CSV contains structured product data such as product title, brand, category, pack quantity, and image links.

---

## ⚙️ Getting Started & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/smart-product-pricing.git
cd smart-product-pricing
```

### 2. Setup Python Environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install Dependencies

```bash
pip install -r Model_LightGBM/requirements.txt
pip install -r Model_Multimodal/requirements.txt
```

### 4. Download Datasets

All required CSV files are hosted on [Google Drive](https://drive.google.com/drive/folders/1NLHao1ennJhPiHT8YRp9qTqxVwGpzOFl?usp=sharing).

### 5. Download Images

Use the helper script:

```bash
cd utils
python download_images.py --csv ../dataset/train.csv
python download_images.py --csv ../dataset/test.csv
```

Images will be saved automatically in corresponding `train/` and `test/` folders.

---

## ⚡ Model 1: LightGBM Ensemble

**Goal:** Baseline regression using structured tabular and textual data.

### Features:

* TF-IDF and categorical encoding
* Product quantity and brand normalization
* Ensemble of LightGBM regressors using 5-fold CV
* SMAPE-optimized post-processing

### Run Instructions:

```bash
cd Model_LightGBM
python main.py
```

---

## 🔮 Model 2: Multimodal Fusion Model

**Goal:** Combine text, image, and tabular embeddings for improved SMAPE and robustness.

### Architecture Overview:

* Frozen **RoBERTa** text and **ViT** image embeddings (cached as `.pt` files)
* Engineered tabular matrix (brand, pack qty, capacity, category)
* **Gated Fusion Block** for adaptive modality weighting
* 3-block residual MLP (LayerNorm + GELU + Dropout)
* Auxiliary decile-classification loss for multi-task regularization

### Training Objective:

$$Loss = 0.85 \times logMAE + 0.15 \times CrossEntropy$$

### Workflow:

```bash
# 1. Generate image embeddings
# Terminal:
python embed_image.py

# 2. Generate text embeddings
python embed_text.py

# 3. Engineer tabular features
python features.py

# 4. Prepare .pt dataset caches (combines all features)
python prepare_data.py

# 5. Train the fusion neural model (regression head and OOF storage)
python train_fusion.py --train_pt caches/train_data.pt --out_dir outputs/fusion

# 6. Train the LightGBM stacker
python stack_lgbm.py --train_pt caches/train_data.pt --test_pt caches/test_data.pt --out_dir outputs/stacker

# 7. Predict on the test set (fusion model)
python predict.py --out_csv outputs/final/test_out.csv

# 8. Blend neural and LightGBM predictions (final submission)
python blend_predict.py --cv_summary_dir outputs/fusion --stacker_dir outputs/stacker --test_csv dataset/test.csv --sample_out_csv dataset/sample_test_out.csv --out_csv outputs/final/test_out_blended.csv

```

---

## 📊 Results

| Model             | Approach               | CV SMAPE   | Notes                          |
| ----------------- | ---------------------- | ---------- | ------------------------------ |
| LightGBM Ensemble | Tabular + TF-IDF       | **43.0%**  | Baseline ensemble              |
| Multimodal Fusion | Text + Image + Tabular | **~42.5%** | Improved via gating + blending |

---

## 📈 Visualization & Evaluation

Generate and visualize model metrics:

```bash
python Model_Multimodal/utils_metrics.py
```

Feature importances and validation plots are saved under `outputs/calibrations/`.

---

## 🔁 Reproducibility

* Fixed random seeds for numpy, torch, and LightGBM
* Works on CPU-only systems (under 7 hours runtime)
* Tested on Python 3.10+

---
