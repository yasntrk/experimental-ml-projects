# SEDS 537 — Machine Learning Take-Home Midterm

> **Student ID:** 323011020
> **Course:** SEDS 537 — Machine Learning
> **Instructor:** Prof. Dr. Aytuğ Onan
> **Institution:** Izmir Institute of Technology, Department of Software Engineering
> **Term:** Spring 2026

---

## 📋 Overview

This repository contains the complete solution for the SEDS 537 Machine Learning take-home midterm examination. The midterm covers **five core topics in machine learning**, each implemented as a separate, fully reproducible Jupyter notebook. A consolidated LaTeX report (and compiled PDF) summarizes the methodology, experiments, results, and discussion for all five questions.

| # | Topic | Dataset | Key Models |
|---|-------|---------|------------|
| Q1 | Regression | California Housing | Linear, Ridge (CV), Random Forest |
| Q2 | Classification (Imbalanced) | Credit Card Fraud | Logistic Regression, Random Forest, XGBoost |
| Q3 | Dimensionality Reduction | MNIST | PCA, t-SNE, k-NN |
| Q4 | Clustering | Wholesale Customers | K-Means, Agglomerative, DBSCAN |
| Q5 | Neural Networks | Fashion-MNIST | MLP, CNN (PyTorch) |

---

## 📁 Repository Structure

```
SEDS537_Midterm/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── SEDS537_Midterm_Report.tex             # Master LaTeX report (all 5 questions)
├── SEDS537_Midterm_Report.pdf             # Compiled PDF report (20 pages)
│
├── Q1_Regression.ipynb                    # Q1 — California Housing
├── Q2_Classification_Imbalance.ipynb      # Q2 — Credit Card Fraud
├── Q3_Dimensionality_Reduction.ipynb      # Q3 — MNIST (PCA + t-SNE)
├── Q4_Clustering.ipynb                    # Q4 — Wholesale Customers
├── Q5_Neural_Networks.ipynb               # Q5 — Fashion-MNIST (MLP vs CNN)
│
├── creditcard.csv                         # Dataset for Q2 (download separately)
├── data/                                  # Auto-downloaded datasets (Fashion-MNIST)
│
└── *.png                                  # Generated figures (18 total)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.9+**
- pip / conda
- (Optional) LaTeX distribution (TeX Live / MacTeX) — only required to recompile the PDF report

### 1. Clone the repository
```bash
git clone <repository-url>
cd SEDS537_Midterm
```

### 2. Create a virtual environment *(recommended)*
```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
imbalanced-learn
torch
torchvision
scipy
```

### 4. Download the Credit Card Fraud dataset (Q2 only)
Q2 requires the credit-card-fraud dataset, which is too large for GitHub.

Download `creditcard.csv` from Kaggle:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place it directly inside the `SEDS537_Midterm/` folder.

> All other datasets (California Housing, MNIST, Wholesale Customers, Fashion-MNIST) are loaded automatically by `scikit-learn` / `torchvision` / UCI repository.

---

## 🚀 How to Run

### Run the notebooks
Open each notebook in Jupyter / VS Code / Colab and execute all cells in order:

```bash
jupyter notebook
```

Or run them headlessly:
```bash
jupyter nbconvert --to notebook --execute Q1_Regression.ipynb --output Q1_Regression.ipynb
```

Each notebook will automatically save its figures as `.png` files into the project root. All 18 figures are required to compile the LaTeX report.

> **Tip:** All notebooks use `random_state=42` for reproducibility.

### Compile the LaTeX report
```bash
pdflatex SEDS537_Midterm_Report.tex
pdflatex SEDS537_Midterm_Report.tex     # second pass for table of contents
```

The output is `SEDS537_Midterm_Report.pdf` (~20 pages).

---

## 📚 Question-by-Question Summary

### Q1 — Regression: Predicting House Prices

- **Dataset:** California Housing (20,640 samples, 8 features)
- **Tasks:**
  - Exploratory Data Analysis (histograms, correlation matrix, scatter plots)
  - Train **Linear Regression**, **Ridge Regression** (with CV-tuned `alpha`), **Random Forest**
  - Compare with and without **standard scaling**
  - Add **polynomial features** (degree 2) and re-evaluate
  - Plot **residual plots** (fitted vs residuals)
- **Metrics:** RMSE, MAE, R²
- **Best result:** Random Forest — RMSE = 0.51, R² = 0.80

### Q2 — Classification: Handling Class Imbalance

- **Dataset:** Credit Card Fraud Detection (284,807 transactions, 0.17% fraud)
- **Tasks:**
  - Train **Logistic Regression**, **Random Forest**, **XGBoost**
  - Apply **SMOTE** (oversampling) and **Random Undersampling**
  - Compare metrics across resampling strategies
  - Plot **ROC curves** for the best model
  - Discuss the **Precision/Recall trade-off**
- **Metrics:** Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Best result:** Random Forest + SMOTE — F1 = 0.88, ROC-AUC = 0.978

### Q3 — Dimensionality Reduction & Visualisation

- **Dataset:** MNIST (handwritten digits 0–9)
- **Tasks:**
  - Apply **PCA** (2D and 50D); report explained variance for first 10 components
  - Apply **t-SNE** with perplexities 30 and 50
  - Visualize 2D embeddings, color-coded by digit label
  - Train **k-NN (k=5)** on 784D, PCA-50D, t-SNE-2D
  - Compare 5-fold CV accuracy
- **Discussion:** PCA vs t-SNE — strengths, weaknesses, downstream usability

### Q4 — Clustering: Unsupervised Learning

- **Dataset:** UCI Wholesale Customers (440 samples, 6 spending features)
- **Tasks:**
  - **K-Means** — choose optimal `k` via Elbow + Silhouette
  - **Agglomerative Clustering** (Ward linkage) + dendrogram
  - **DBSCAN** — tune `eps` via k-distance graph
  - Visualize clusters in 2D (PCA projection)
  - Interpret cluster profiles
- **Metrics:** Silhouette Score, Davies-Bouldin Index
- **Discussion:** DBSCAN parameter sensitivity, cluster shape comparison

### Q5 — Neural Networks: MLP vs CNN

- **Dataset:** Fashion-MNIST (10 clothing classes, 28×28 grayscale)
- **Models:**
  - **MLP** — 2 hidden layers (256, 128) + ReLU + Dropout 0.3
  - **CNN** — 2 conv layers (32, 64 filters) + MaxPool + Dense(128)
- **Training:** Adam optimizer, CrossEntropyLoss, **early stopping** (patience=4)
- **Metrics:** Accuracy, Macro-F1, Confusion Matrix
- **Visualization:** Training/validation curves, 5 misclassified examples per model
- **Result:** CNN ≈ 91% accuracy vs MLP ≈ 88% — CNN wins thanks to translation invariance & hierarchical feature learning

---

## 🔬 Reproducibility

All experiments are reproducible thanks to:
- **Fixed random seed:** `random_state=42` in all sklearn calls and `torch.manual_seed(42)` for PyTorch
- **Pinned dependencies:** see `requirements.txt`
- **Stratified splits:** for classification (Q2) to preserve class distribution
- **Single test-set evaluation:** the test set is used exactly once per model, only for final evaluation

---

## 📊 Headline Results

| Question | Best Model | Key Metric |
|----------|-----------|------------|
| Q1 — Regression | Random Forest | R² = 0.80 |
| Q2 — Classification | RF + SMOTE | F1 = 0.88, AUC = 0.978 |
| Q3 — k-NN on MNIST | k-NN @ t-SNE 2D | Acc ≈ 0.96 |
| Q4 — Clustering | K-Means (k=2) | Silhouette ≈ 0.50 |
| Q5 — Image Classification | CNN | Acc ≈ 0.91 |

---

## 📝 Report

The full report (`SEDS537_Midterm_Report.pdf`) includes:
- Cover page + automated table of contents
- Per-question: dataset description, methodology, results table, figures, and discussion
- Comparison tables and final conclusion

The LaTeX source (`SEDS537_Midterm_Report.tex`) is provided for editing and recompilation.

---

## 🧪 Tested Environments

| Component | Version |
|-----------|---------|
| Python | 3.9 / 3.10 / 3.11 |
| scikit-learn | ≥ 1.3 |
| PyTorch | ≥ 2.0 |
| XGBoost | ≥ 1.7 |
| imbalanced-learn | ≥ 0.11 |
| TeX Live | 2024 / 2025 |

---

## 📚 References

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Chawla et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR.
- van der Maaten & Hinton (2008). *Visualizing Data using t-SNE.* JMLR.
- LeCun et al. (1998). *Gradient-based Learning Applied to Document Recognition.* IEEE.
- Xiao et al. (2017). *Fashion-MNIST: A Novel Image Dataset for Benchmarking ML Algorithms.* arXiv:1708.07747.
- UCI Machine Learning Repository — Wholesale Customers Dataset.

---

## 📜 Academic Integrity

This work was completed individually as required by the course policy. All external sources are cited above. The code, experiments, and report are reproducible from the provided notebooks and `requirements.txt`.
