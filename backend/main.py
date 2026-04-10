from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import io
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PROBABILITY CALIBRATION
# The model was trained with scale_pos_weight / is_unbalance=True to handle
# class imbalance during training. This causes raw predict_proba() scores to
# be significantly inflated relative to the real-world fraud base rate (~2%).
#
# We apply Bayes' theorem to correct the posterior probability:
#   P(fraud | score) = (score * real_prior / train_prior) /
#                      (score * real_prior / train_prior  +
#                       (1-score) * (1-real_prior) / (1-train_prior))
#
# REAL_FRAUD_PRIOR = 0.02  → only 2% of real-world transactions are fraud
# TRAIN_FRAUD_PRIOR = 0.50 → model trained on ~50/50 balanced-equivalent data
# ─────────────────────────────────────────────────────────────────────────────
REAL_FRAUD_PRIOR  = 0.02   # real-world prevalence
TRAIN_FRAUD_PRIOR = 0.50   # effective training prevalence (balanced weighting)

def calibrate_probability(raw_prob: float) -> float:
    """Bayesian prior correction: maps inflated model probabilities back to
    real-world scale so that ~2% of transactions are flagged as high risk."""
    eps = 1e-9  # avoid divide-by-zero
    raw_prob = max(eps, min(1 - eps, raw_prob))
    # Likelihood ratio of fraud vs legit at training prior
    lr_fraud  = raw_prob / TRAIN_FRAUD_PRIOR
    lr_legit  = (1 - raw_prob) / (1 - TRAIN_FRAUD_PRIOR)
    # Posterior using real-world prior
    numerator = lr_fraud * REAL_FRAUD_PRIOR
    calibrated = numerator / (numerator + lr_legit * (1 - REAL_FRAUD_PRIOR))
    return float(calibrated)

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
    avg_txn_amount_30d: float = 500
    credit_limit_inr: float = 50000
    std_txn_amount_30d: float = 100
    distance_from_home_km: float = 5.0
    card_age_days: int = 365
    velocity_last_1h: int = 1
    velocity_last_24h: int = 5

class PredictionResponse(BaseModel):
    fraud_probability: float
    prediction: str
    risk_level: str
    explanation: list
    investigation_summary: str
    status: str

class BulkPredictionResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    results: list

class RuleInput(BaseModel):
    field: str
    operator: str
    value: float
    action: str

class ActionUpdate(BaseModel):
    id: int
    status: str

# In-memory store
transactions = []
dynamic_rules = []
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
    
    # Banking specific derivations
    tx_copy['high_velocity'] = 1 if tx_copy.get('velocity_last_1h', 0) > 3 else 0
    tx_copy['new_card_risk'] = 1 if tx_copy.get('card_age_days', 365) < 30 else 0
    tx_copy['extreme_distance'] = 1 if tx_copy.get('distance_from_home_km', 0) > 500 else 0
    
    # Entry Mode Risk Weighting
    # Manual entry is typically 10x riskier for Card-Not-Present fraud
    tx_copy['is_manual_entry'] = 1 if tx_copy.get('pos_entry_mode') == 'manual' else 0
    tx_copy['is_chip_entry'] = 1 if tx_copy.get('pos_entry_mode') == 'chip' else 0
    
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
    if tx_dict.get('distance_from_home_km', 0) > 500:
        reasons.append(f"Extreme distance from home ({tx_dict.get('distance_from_home_km')} km).")
    if tx_dict.get('velocity_last_1h', 0) > 3:
        reasons.append(f"High transaction frequency detected ({tx_dict.get('velocity_last_1h')} in 1h).")
    if tx_dict.get('card_age_days', 365) < 30:
        reasons.append("New card with limited history.")
    
    if not reasons:
        reasons.append("Anomaly detected by ML model based on complex feature interactions.")
        
    return reasons

def evaluate_rules(tx_dict: dict) -> list:
    triggered_rules = []
    for r in dynamic_rules:
        fval = tx_dict.get(r['field'])
        if fval is not None:
            if r['operator'] == '>' and float(fval) > float(r['value']):
                triggered_rules.append(r)
            elif r['operator'] == '<' and float(fval) < float(r['value']):
                triggered_rules.append(r)
            elif r['operator'] == '==' and float(fval) == float(r['value']):
                triggered_rules.append(r)
    return triggered_rules

def simulate_llm_investigator(tx_dict: dict, risk_level: str, is_fraud: bool, triggered_rules: list) -> str:
    """Simulates an LLM summarizing the fraud parameters"""
    amt = tx_dict.get('transaction_amount_inr', 0)
    merchant = tx_dict.get('merchant_category', 'unknown').upper()
    is_intl = 'an international' if tx_dict.get('is_international') == 1 else 'a domestic'
    
    summary = f"Agent AI Analysis: Transaction of {amt:,.2f} INR at {merchant} ({is_intl} location). "
    
    if triggered_rules:
        summary += f"Automatically intercepted by Policy Rules ({len(triggered_rules)} rule(s) triggered). "
    
    if is_fraud:
        summary += "The ML engine signals highly anomalous behavior comparing current spending velocity against historical patterns. Recommend immediate card freeze and 2FA authentication."
    elif risk_level == "MEDIUM":
        summary += "Moderate risk indicators present. Consider sending a soft verification SMS."
    else:
        summary += "Behavior aligns with normal cardholder baseline. No action required."
        
    return summary

@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionInput, threshold: float = 0.75):
    try:
        tx_dict = tx.dict()
        tx_derived = derive_features(tx_dict)
        
        # Check dynamic rules first
        triggered_rules = evaluate_rules(tx_derived)
        rule_action = "BLOCK" if any(r['action'] == 'BLOCK' for r in triggered_rules) else None
        
        if rule_action == "BLOCK":
            prob = 0.999  # Cap at 99.9% even for rule-based blocks
            is_fraud = True
            risk_level = "CRITICAL (RULE)"
        else:
            if model_artifacts:
                X_input = preprocess_input(tx_derived)
                model = model_artifacts['model']
                raw_prob = float(model.predict_proba(X_input)[0, 1])
            else:
                # Fallback for testing frontend without model
                raw_prob = tx.transaction_amount_inr / 10000.0 if tx.transaction_amount_inr < 10000 else 0.99
            
            # ── CALIBRATION ──────────────────────────────────────────────────
            # The model was trained with class-imbalance compensation
            # (scale_pos_weight / is_unbalance), which inflates raw scores.
            # Correct back to the real-world ~2% fraud base rate.
            prob = calibrate_probability(raw_prob)
            
            # Cap at 99.9% — certainty of exactly 100% is never appropriate
            prob = min(prob, 0.999)

            # ── RISK TIERING ─────────────────────────────────────────────────
            # HIGH   : prob > threshold           → BLOCKED  (auto action)
            # MEDIUM : prob > threshold * 0.55    → MONITORED (human review)
            # LOW    : everything else            → APPROVED
            if prob > threshold:
                risk_level = "HIGH"
                is_fraud = True
            elif prob > threshold * 0.55:
                risk_level = "MEDIUM"
                is_fraud = False  # Medium risk is monitored, never auto-blocked
            else:
                risk_level = "LOW"
                is_fraud = False

        explanation = generate_explanation(tx_derived, risk_level)
        llm_summary = simulate_llm_investigator(tx_derived, risk_level, is_fraud, triggered_rules)
        
        result = {
            "fraud_probability": round(prob, 4),
            "prediction": "FRAUD" if is_fraud else "LEGIT",
            "risk_level": risk_level,
            "explanation": explanation,
            "investigation_summary": llm_summary,
            "status": "pending"
        }
        
        # Save to DB
        txn_record = {
            "id": len(transactions) + 1,
            "amount": tx.transaction_amount_inr,
            "merchant": tx.merchant_category,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "country_code": tx.country_code,
            **result
        }
        transactions.insert(0, txn_record)
        if len(transactions) > 5000:
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

@app.post("/action")
def update_action(update: ActionUpdate):
    for t in transactions:
        if t['id'] == update.id:
            t['status'] = update.status
            return {"success": True, "transaction": t}
    raise HTTPException(status_code=404, detail="Transaction not found")

@app.post("/rules")
def add_rule(rule: RuleInput):
    dynamic_rules.append(rule.model_dump() if hasattr(rule, 'model_dump') else rule.dict())
    return {"status": "Rule added", "rules": dynamic_rules}

@app.get("/rules")
def get_rules_endpoint():
    return dynamic_rules

@app.post("/rules/clear")
def clear_rules():
    dynamic_rules.clear()
    return {"status": "Rules cleared"}

@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(file: UploadFile = File(...), threshold: float = 0.75):
    content = await file.read()
    if file.filename.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(content))
    elif file.filename.endswith('.json'):
        df = pd.read_json(io.BytesIO(content))
    else:
        raise HTTPException(status_code=400, detail="Only CSV or JSON files allowed.")

    results = []
    fraud_count = 0
    
    # Process rows
    for _, row in df.iterrows():
        # Map row to TransactionInput format
        tx_data = row.to_dict()
        # Handle default falls for missing cols in CSV
        defaults = {
            "is_international": 0, "country_code": "IN", "pos_entry_mode": "chip",
            "txn_hour": 12, "txn_day": 1, "txn_month": 1, "txn_weekday": 1,
            "is_weekend": 0, "is_night_txn": 0, "distance_from_home_km": 5.0,
            "card_age_days": 365, "velocity_last_1h": 1, "velocity_last_24h": 5,
            "avg_txn_amount_30d": 500, "credit_limit_inr": 50000, "std_txn_amount_30d": 100
        }
        full_data = {**defaults, **tx_data}
        
        # We simulate the validation here
        try:
            tx_obj = TransactionInput(**full_data)
            pred = predict(tx_obj, threshold)
            results.append({
                "id": len(results) + 1,
                "amount": tx_obj.transaction_amount_inr,
                "merchant": tx_obj.merchant_category,
                "prediction": pred['prediction'],
                "risk_level": pred['risk_level']
            })
            if pred['prediction'] == "FRAUD":
                fraud_count += 1
        except Exception:
            continue
            
    return {
        "total_processed": len(results),
        "fraud_detected": fraud_count,
        "results": results
    }

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
