from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime

app = FastAPI(title="Real-Time Fraud Detection API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and preprocessing objects on startup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.joblib")

model_artifacts = None

@app.on_event("startup")
def load_model():
    global model_artifacts
    try:
        model_artifacts = joblib.load(MODEL_PATH)
        print("Model and preprocessors loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load model from {MODEL_PATH}. Error: {e}")

class TransactionInput(BaseModel):
    transaction_amount_inr: float
    is_international: int
    merchant_category: str
    country_code: str
    pos_entry_mode: str
    txn_hour: int
    txn_day: int
    txn_month: int
    txn_weekday: int
    is_weekend: int
    is_night_txn: int
    avg_txn_amount_30d: float = 500  # Some defs for derived feats
    credit_limit_inr: float = 50000
    std_txn_amount_30d: float = 100

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    risk_level: str
    explanation: list

# In-memory store
transactions = []
stats = {
    "total_transactions": 0,
    "fraud_count": 0,
    "fraud_rate": 0.0
}

def derive_features(tx: dict) -> dict:
    tx_copy = dict(tx)
    # Add derived features
    tx_copy['amount_vs_avg_ratio'] = tx_copy['transaction_amount_inr'] / (tx_copy['avg_txn_amount_30d'] + 1)
    tx_copy['amount_to_limit_ratio'] = tx_copy['transaction_amount_inr'] / (tx_copy['credit_limit_inr'] + 1)
    tx_copy['international_high_amount'] = 1 if (tx_copy['is_international'] == 1 and tx_copy['transaction_amount_inr'] > 1000) else 0
    tx_copy['amount_zscore'] = (tx_copy['transaction_amount_inr'] - tx_copy['avg_txn_amount_30d']) / (tx_copy['std_txn_amount_30d'] + 1)
    tx_copy['is_large_txn'] = 1 if tx_copy['amount_vs_avg_ratio'] > 3 else 0
    return tx_copy

def preprocess_input(tx_dict: dict):
    if not model_artifacts:
        raise Exception("Model not loaded.")
        
    df = pd.DataFrame([tx_dict])
    encoders = model_artifacts['encoders']
    scaler = model_artifacts['scaler']
    train_cols = model_artifacts['train_cols']

    for col in encoders:
        if col in df.columns:
            le = encoders[col]
            known_classes = set(le.classes_)
            # Handle unknown categories safely
            df[col] = df[col].astype(str).apply(lambda x: x if x in known_classes else le.classes_[0])
            df[col] = le.transform(df[col])
            
    # Align cols
    for col in train_cols:
        if col not in df:
            df[col] = 0
            
    df = df[train_cols]
    
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=train_cols)

def generate_explanation(tx_dict: dict, risk_level: str) -> list:
    reasons = []
    if risk_level == "LOW":
        reasons.append("Transaction appears normal based on historical patterns.")
        return reasons
        
    if tx_dict.get('is_international') == 1 and tx_dict.get('transaction_amount_inr', 0) > 1000:
        reasons.append("High amount on an international transaction.")
    if tx_dict.get('amount_vs_avg_ratio', 0) > 3:
        reasons.append("Transaction amount is significantly higher than 30-day average.")
    if tx_dict.get('is_night_txn') == 1:
        reasons.append("Transaction occurred during high-risk hours (night).")
    
    if not reasons:
        reasons.append("Anomaly detected by ML model based on complex feature interactions.")
        
    return reasons

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionInput, threshold: float = 0.5):
    try:
        tx_dict = tx.dict()
        tx_derived = derive_features(tx_dict)
        
        if model_artifacts:
            X_input = preprocess_input(tx_derived)
            model = model_artifacts['model']
            prob = float(model.predict_proba(X_input)[0, 1])
        else:
            # Fallback for testing frontend without model
            prob = tx.transaction_amount_inr / 10000.0 if tx.transaction_amount_inr < 10000 else 0.99
            
        is_fraud = prob >= threshold
        
        risk_level = "LOW"
        if prob > threshold:
            risk_level = "HIGH"
        elif prob > threshold * 0.7:
            risk_level = "MEDIUM"

        explanation = generate_explanation(tx_derived, risk_level)
        
        result = {
            "fraud_probability": round(prob, 4),
            "prediction": "FRAUD" if is_fraud else "LEGIT",
            "risk_level": risk_level,
            "explanation": explanation
        }
        
        # Save to DB
        txn_record = {
            "id": len(transactions) + 1,
            "amount": tx.transaction_amount_inr,
            "merchant": tx.merchant_category,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **result
        }
        transactions.insert(0, txn_record)
        if len(transactions) > 100:
            transactions.pop()
            
        stats["total_transactions"] += 1
        if is_fraud:
            stats["fraud_count"] += 1
        stats["fraud_rate"] = stats["fraud_count"] / max(stats["total_transactions"], 1)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    return stats

@app.get("/transactions")
def get_transactions():
    return transactions

@app.post("/simulate")
def simulate(threshold: float = 0.5):
    import random
    tx = TransactionInput(
        transaction_amount_inr=random.uniform(10, 5000),
        is_international=random.choice([0, 0, 0, 1]),
        merchant_category=random.choice(["retail", "online", "travel", "groceries"]),
        country_code="IN",
        pos_entry_mode="chip",
        txn_hour=random.randint(0, 23),
        txn_day=random.randint(1, 28),
        txn_month=random.randint(1, 12),
        txn_weekday=random.randint(0, 6),
        is_weekend=random.choice([0, 1]),
        is_night_txn=random.choice([0, 1])
    )
    return predict(tx, threshold)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
