from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
import time
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.predict import FraudPredictor
from src.data.database import get_db, init_db, Transaction

# Initialize FastAPI app
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Production-ready fraud detection system using XGBoost",
    version="1.0.0"
)

# Load model at startup (once, not per request!)
print("🚀 Starting Fraud Detection API...")
predictor = FraudPredictor()
print("✅ Model loaded and ready!")

# Request schema
class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    Amount: float = Field(..., ge=0, description="Transaction amount in dollars")

    # All V columns from the Kaggle dataset (anonymized PCA features)
    V1: float = 0.0
    V2: float = 0.0
    V3: float = 0.0
    V4: float = 0.0
    V5: float = 0.0
    V6: float = 0.0
    V7: float = 0.0
    V8: float = 0.0
    V9: float = 0.0
    V10: float = 0.0
    V11: float = 0.0
    V12: float = 0.0
    V13: float = 0.0
    V14: float = 0.0
    V15: float = 0.0
    V16: float = 0.0
    V17: float = 0.0
    V18: float = 0.0
    V19: float = 0.0
    V20: float = 0.0
    V21: float = 0.0
    V22: float = 0.0
    V23: float = 0.0
    V24: float = 0.0
    V25: float = 0.0
    V26: float = 0.0
    V27: float = 0.0
    V28: float = 0.0

    # Engineered features
    hour_of_day: Optional[int] = Field(None, ge=0, le=23)
    amount_log: Optional[float] = None
    customer_avg_amount: Optional[float] = None
    customer_txn_count: Optional[int] = None
    high_amount_flag: Optional[int] = Field(None, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "TXN123456",
                "Amount": 250.75,
                "V1": -0.5,
                "V2": 0.8,
                "hour_of_day": 14,
                "customer_avg_amount": 120.0,
                "customer_txn_count": 15,
                "high_amount_flag": 0
            }
        }

# Response schema
class PredictionResponse(BaseModel):
    """Fraud prediction result."""
    transaction_id: str
    prediction: str
    fraud_probability: float
    risk_level: str
    model_version: str
    latency_ms: float

@app.get("/")
async def root():
    """API root - welcome message."""
    return {
        "message": "Real-Time Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_version": predictor.model_version
    }

@app.get("/model-info")
async def model_info():
    """Get model metadata."""
    return {
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "performance_metrics": predictor.metadata.get("performance", {}),
        "feature_count": len(predictor.feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest, db: Session = Depends(get_db)):
    """
    Make a fraud prediction on a transaction and store it in the database.

    Returns:
        - prediction: "FRAUD" or "LEGITIMATE"
        - fraud_probability: 0.0 to 1.0
        - risk_level: "LOW", "MEDIUM", "HIGH", or "CRITICAL"
        - latency_ms: prediction time in milliseconds
    """
    try:
        # Start latency timer
        start_time = time.time()

        # Convert request to dict
        transaction_data = transaction.model_dump(exclude={"transaction_id"})

        # Make prediction
        result = predictor.predict(transaction_data)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Store prediction in database
        db_transaction = Transaction(
            transaction_id=transaction.transaction_id,
            amount=transaction.Amount,
            prediction=result["prediction"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            model_version=result["model_version"],
            latency_ms=round(latency_ms, 2)
        )
        db.add(db_transaction)
        db.commit()

        # Build response
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            prediction=result["prediction"],
            fraud_probability=result["fraud_probability"],
            risk_level=result["risk_level"],
            model_version=result["model_version"],
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/transactions")
async def get_transactions(limit: int = 10, db: Session = Depends(get_db)):
    """
    Retrieve recent transactions from the database.

    Args:
        limit: Number of transactions to return (default 10, max 100)
    """
    if limit > 100:
        limit = 100

    transactions = db.query(Transaction).order_by(Transaction.created_at.desc()).limit(limit).all()

    return {
        "count": len(transactions),
        "transactions": [
            {
                "transaction_id": t.transaction_id,
                "amount": t.amount,
                "prediction": t.prediction,
                "fraud_probability": t.fraud_probability,
                "risk_level": t.risk_level,
                "latency_ms": t.latency_ms,
                "timestamp": t.timestamp.isoformat()
            }
            for t in transactions
        ]
    }

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database tables on API startup."""
    init_db()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
