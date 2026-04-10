# 🛡️ SecureCred — Real-Time Credit Card Fraud Detection

> An end-to-end fraud detection platform built for the Indian banking ecosystem — combining ML, a live analytics dashboard, and a configurable rule engine.

---

## What is this?

SecureCred detects fraudulent credit card transactions in real time. A transaction comes in, gets evaluated by a business rule engine first, then scored by a machine learning model, and a risk decision is returned — with a plain-English explanation of *why* it was flagged.

It's not just a notebook. It has a working backend API, a live dashboard, and a full investigation workflow for fraud analysts.

---

## What makes it different?

### 1. Dual-Layer Defense — Rules + ML
Most fraud detection projects stop at the model. SecureCred runs two independent layers:

**Layer 1 — Rule Engine (instant, no ML needed)**
Fraud analysts can define policies like `amount > ₹50,000 → BLOCK` directly from the dashboard. These trigger *before* the model runs, with zero latency. No redeployment needed.

**Layer 2 — LightGBM Model (probabilistic scoring)**
When no rule fires, the transaction flows to a Optuna-tuned LightGBM model. Final decision is LOW / MEDIUM / HIGH risk, with explanations attached.

---

### 2. Bayesian Probability Calibration
This is the bit most teams miss. When a model is trained on balanced class weights, its raw `predict_proba()` scores are inflated — a 60% model score doesn't mean 60% real-world risk.

We apply Bayes' theorem to correct this back to the real-world fraud rate (~2%):

```
P(fraud | score) = (score × 0.02 / 0.50) /
                   (score × 0.02/0.50 + (1 - score) × 0.98/0.50)
```

This makes the dashboard scores actually meaningful, not just model-relative numbers.

---

### 3. No SMOTE — Principled Imbalance Handling
We didn't synthesize fake fraud samples. Instead:
- Random Forest → `class_weight='balanced'`
- XGBoost → `scale_pos_weight`
- LightGBM → `is_unbalance=True`

This avoids distribution artifacts from oversampling and makes the model more honest about real-world signal.

---

### 4. Full End-to-End Product
| What | How |
|---|---|
| Single transaction scoring | `POST /predict` with adjustable threshold slider |
| Bulk batch analysis | Upload CSV/JSON → scan thousands of transactions → export as CSV/JSON |
| Investigation queue | High/Medium priority queues with Freeze or Approve actions |
| Visual transaction graph | Node graph: Cardholder → Transaction → Merchant → Location → Status |
| Live geographic map | World map with pulsing red dots for fraud, green for legitimate |
| Policy control | Add/clear blocking rules from the UI, active immediately |

---

## ML Pipeline (tldr)

```
Dataset (severely imbalanced, ~2% fraud)
    ↓
Feature Engineering (9 derived features: z-scores, velocity flags, distance anomalies, etc.)
    ↓
Leakage removal (velocity columns dropped in training, kept for inference)
    ↓
3 models benchmarked: Random Forest, XGBoost, LightGBM
    ↓
Hyperparameter tuning via Optuna (TPE Sampler, 3-fold stratified CV)
    ↓
Best model selected by ROC-AUC → serialized to fraud_model.joblib
    ↓
Bayesian probability calibration applied at inference
    ↓
Adjustable decision threshold (default 0.75, tunable from dashboard)
```

Key features engineered: `amount_vs_avg_ratio`, `amount_zscore`, `international_high_amount`, `high_velocity`, `new_card_risk`, `extreme_distance`, `is_manual_entry`.

---

## Tech Stack

**ML:** Python · LightGBM · XGBoost · scikit-learn · Optuna · SHAP  
**Backend:** FastAPI · Uvicorn · Pydantic · joblib  
**Frontend:** React 19 · Vite · TailwindCSS · @xyflow/react · react-simple-maps

---

## Running Locally

```bash
# Backend
cd backend
python -m venv venv && venv\Scripts\activate   # Windows
pip install fastapi uvicorn pandas scikit-learn lightgbm xgboost optuna shap joblib python-multipart
python main.py
# → http://localhost:8000  |  Swagger docs → /docs

# Frontend (separate terminal)
cd frontend
npm install && npm run dev
# → http://localhost:5173
```

> Make sure the backend is running before opening the dashboard.

---

## Project Structure

```
securecred-main/
├── PerfectModel.ipynb       ← Full ML notebook (EDA → Training → SHAP)
├── model_clean.py           ← Clean training script
├── fraud_model.joblib       ← Serialized model (585 KB)
├── train_transactions.csv   ← Training data
├── backend/
│   └── main.py              ← FastAPI server (all API routes)
└── frontend/
    └── src/App.jsx          ← Dashboard (Live Monitor, Batch, Investigation, Rules)
```

---

> Built for the ML Hackathon · SecureCred Intelligence Protocol v4.0.1
