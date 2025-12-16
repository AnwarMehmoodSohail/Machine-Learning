# Feature Selection vs PCA for Machine Learning Classification

This repository demonstrates a **clear and beginner-friendly comparison** between:

- ✅ **Baseline Machine Learning model**
- ✅ **Feature Selection using ANOVA (SelectKBest)**
- ✅ **Dimensionality Reduction using PCA**

The goal is to help understand:
- How **feature selection** differs from **PCA**
- Why **normalization** is important
- How to **avoid data leakage**
- How dimensionality reduction affects **classification accuracy**

---

## 📌 Problem Statement

High-dimensional datasets (e.g., sensor data, OBD-II / CAN-BUS data, biomedical signals) often:
- Contain noisy or redundant features
- Cause overfitting
- Increase training time

This project compares three approaches to address this issue using **Logistic Regression**.

---

## 📊 Models Implemented

### 1️⃣ Baseline Model (No Dimensionality Reduction)
- Uses **all original features**
- Standardized using `StandardScaler`
- Logistic Regression classifier

### 2️⃣ Feature Selection Model
- Uses **ANOVA F-test** (`SelectKBest + f_classif`)
- Selects top **K most discriminative features**
- Keeps original feature meaning (interpretable)

### 3️⃣ PCA Model
- Uses **Principal Component Analysis**
- Reduces features to **K principal components**
- Improves efficiency but loses interpretability

---

## 🧠 Key Concepts Covered

- Feature Selection vs Feature Extraction
- Normalization / Standardization
- `.fit_transform()` vs `.transform()`
- Avoiding **data leakage**
- Visual comparison of model performance

---

## 🗂️ Dataset Format

Replace the dataset with your own CSV file:

```text
your_high_dim_dataset.csv
