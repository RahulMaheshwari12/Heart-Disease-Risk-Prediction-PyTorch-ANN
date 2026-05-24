# Heart Disease Risk Prediction — PyTorch ANN with SQL Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)
![SQLite](https://img.shields.io/badge/SQLite-3-lightgrey)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9537-brightgreen)

---

## Project Overview

An end-to-end heart disease prediction system built on **630,000 patient records**.  
This project combines a **SQLite3 data pipeline**, **domain-driven feature engineering**, and a **PyTorch ANN** to predict heart disease risk — outperforming a Logistic Regression baseline with a ROC-AUC of **0.9537**.

> In medical diagnosis, missing a disease patient (false negative) is far more dangerous than a false alarm.  
> This model is optimised for **recall** using threshold tuning at 0.4 — achieving **91% recall** on the test set.

---

## Key Results

| Metric | Logistic Regression (Baseline) | PyTorch ANN |
|--------|-------------------------------|-------------|
| Accuracy | 88% | **89%** |
| ROC-AUC | 0.9515 | **0.9537** |
| Recall (Disease) | 0.88 | **0.91** |
| Threshold | 0.5 | **0.4 (tuned)** |

---

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** PyTorch
- **ML & Preprocessing:** Scikit-Learn (ColumnTransformer, PowerTransformer, StandardScaler)
- **Data Pipeline:** SQLite3 + ipython-sql (SQL magic)
- **Visualization:** Matplotlib, Seaborn
- **Data Handling:** Pandas, NumPy

---

## Dataset

- **Source:** [Heart Disease Dataset — Kaggle](https://www.kaggle.com/)
- **Size:** 630,000 patient records
- **Features:** 14 clinical features (Age, Sex, Chest Pain Type, Cholesterol, Max HR, ST Depression, etc.)
- **Target:** Heart Disease — Presence (1) / Absence (0)
- **Class Distribution:** 44.83% Presence / 55.17% Absence (moderate imbalance)

---

## Project Structure

```
Heart-Disease-Risk-Prediction-PyTorch-ANN/
│
├── heart_disease_prediction.ipynb   # Main notebook
├── heart_disease_dataset.xls        # Dataset
├── heart_disease_ann.pth            # Saved PyTorch model weights
├── preprocessor.pkl                 # Saved sklearn preprocessor
└── README.md                        # Project documentation
```

---

## Workflow

### 1. SQL Data Pipeline & EDA
- Loaded 630,000 records into **SQLite3** database
- Performed **16 SQL queries** using SQL magic for comprehensive EDA
- Key findings:
  - Exercise Angina patients: **80.63%** disease rate
  - Chest Pain Type 4 + Thallium 7: **90.32%** disease rate
  - 3 Vessels Fluro: **89.95%** disease rate
  - Males: **55.59%** vs Females: **17.88%** disease rate

### 2. Feature Engineering
Created 3 domain-driven features:
- `age_maxHR_ratio` = Age / Max HR — captures cardiac efficiency decline with age
- `chol_age_ratio` = Cholesterol / Age — cholesterol burden relative to age
- `maxhr_st_interaction` = Max HR × ST Depression — combined cardiac stress indicator

Dropped zero-correlation features: `BP` (-0.01) and `FBS over 120` (0.03)

### 3. Preprocessing
- `PowerTransformer` for skewed features: ST Depression, Cholesterol, chol_age_ratio
- `StandardScaler` for normal features: Age, Max HR, age_maxHR_ratio, maxhr_st_interaction
- `ColumnTransformer` pipeline ensures **no data leakage** during cross-validation
- `stratify=y` in train/test split maintains class distribution

### 4. Baseline Model
- Logistic Regression with `class_weight='balanced'`
- Achieved ROC-AUC: **0.9515** — strong baseline to beat

### 5. PyTorch ANN Architecture
```
Input (14 features)
    → Linear(14 → 64) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.2)
    → Linear(32 → 1) + Sigmoid
Output (probability 0-1)
```

Key design decisions:
- **BatchNorm** — stabilizes training, faster convergence
- **Dropout** — prevents overfitting
- **WeightedRandomSampler** — handles class imbalance during training
- **Early Stopping** (patience=5) — restores best model automatically

### 6. Evaluation & Threshold Tuning
```
Threshold | Precision | Recall
0.3       | 0.800     | 0.934
0.4       | 0.836     | 0.910  ← Selected
0.5       | 0.864     | 0.884
0.6       | 0.889     | 0.852
0.7       | 0.914     | 0.811
```
**Threshold 0.4 selected** — maximises recall to minimise missed diagnoses.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/RahulMaheshwari12/Heart-Disease-Risk-Prediction-PyTorch-ANN.git
cd Heart-Disease-Risk-Prediction-PyTorch-ANN
```

### 2. Install dependencies
```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch ipython-sql joblib
```

### 3. Run the notebook
```bash
jupyter notebook heart_disease_prediction.ipynb
```

---

## Author

**Rahul Maheshwari**  
B.Tech Computer Science & Engineering (IoT) — MLV Textile & Engineering College, RTU  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rahulmaheshwari73-blue)](https://linkedin.com/in/rahulmaheshwari73)
[![GitHub](https://img.shields.io/badge/GitHub-RahulMaheshwari12-black)](https://github.com/RahulMaheshwari12)
