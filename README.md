# 💳 Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using a simulated dataset of 1,000 customers and 800 merchants over a 2-year period (Jan 2019 – Dec 2020).

---

## 📌 Project Overview

Financial fraud costs banks billions annually. This project builds an end-to-end fraud detection pipeline — from raw transaction data to a deployable Streamlit dashboard — using techniques directly applicable to real-world risk and fraud analytics roles.

---

## 📂 Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection (Simulated)](https://www.kaggle.com/)
- **Generator:** Sparkov Data Generation Tool by Brandon Harris
- **Period:** January 1, 2019 – December 31, 2020
- **Size:** ~1.85M transactions across train and test sets
- **Customers:** 1,000 | **Merchants:** 800
- **Class Balance:** Highly imbalanced (~0.5% fraud)

| Column | Description |
|---|---|
| `trans_date_trans_time` | Transaction datetime |
| `merchant` | Merchant name |
| `category` | Merchant category |
| `amt` | Transaction amount |
| `city`, `state` | Cardholder location |
| `lat`, `long` | Cardholder coordinates |
| `merch_lat`, `merch_long` | Merchant coordinates |
| `city_pop` | City population |
| `job` | Cardholder's occupation |
| `dob` | Cardholder date of birth |
| `trans_num` | Unique transaction ID |
| `is_fraud` | Target (1 = Fraud, 0 = Legitimate) |

---

## 🗂️ Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
├── notebooks/
│   └── fraud_detection.ipynb       # Full analysis notebook
│
├── app/
│   └── app.py                      # Streamlit dashboard
│
├── models/
│   └── xgboost_fraud_model.pkl     # Saved model
│
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Class imbalance analysis
- Fraud patterns by category, hour, day, and geography
- Transaction amount distribution (fraud vs legitimate)

### 2. Feature Engineering
- **Age** derived from date of birth
- **Transaction hour, day, month** from datetime
- **Distance** between cardholder and merchant (Haversine formula)
- Label encoding for categorical variables

### 3. Handling Class Imbalance
- Applied **SMOTE** (Synthetic Minority Oversampling Technique) on training data

### 4. Models Trained
| Model | Purpose |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble benchmark |
| XGBoost | Primary model |

### 5. Explainability
- **SHAP values** used for global feature importance and individual prediction explanation

---

## 📊 Results

| Metric | XGBoost |
|---|---|
| AUC-ROC | ~0.98 |
| Precision (Fraud) | ~0.91 |
| Recall (Fraud) | ~0.87 |
| F1-Score (Fraud) | ~0.89 |

> Note: Results may vary slightly depending on SMOTE random state and hyperparameters.

---

## 🌐 Streamlit Dashboard

The dashboard includes:
- 📋 Transaction explorer with fraud filter
- 📈 Live AUC-ROC and Precision-Recall curves
- 🗺️ Geographic fraud heatmap
- 🔍 Single transaction fraud probability scorer
- 📊 SHAP feature importance chart

**To run locally:**
```bash
pip install -r requirements.txt
streamlit run app/app.py
```

---

## 🛠️ Tech Stack

| Library | Use |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn`, `plotly` | Visualizations |
| `scikit-learn` | Preprocessing & evaluation |
| `imbalanced-learn` | SMOTE oversampling |
| `xgboost` | Primary ML model |
| `shap` | Model explainability |
| `streamlit` | Web dashboard |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

---

## 📖 Key Learnings

- Fraud detection requires special handling of **severely imbalanced datasets**
- **Recall** is prioritized over Precision in fraud contexts — missing a fraud is more costly than a false alarm
- **Distance between customer and merchant** is a strong fraud signal
- **SHAP explainability** is critical for risk roles — models must be interpretable for regulators

---

## 👤 Author

**Your Name**
- 🎓 MSc Data Science & Analytics
- 💼 [LinkedIn](https://linkedin.com/in/yourprofile)
- 🐙 [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License.
