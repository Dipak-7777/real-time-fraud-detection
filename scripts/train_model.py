import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
from datetime import datetime
import os

def train_and_save_model():
    print("="*60)
    print("  PRODUCTION MODEL TRAINING PIPELINE")
    print("="*60)

    print("\n1. Loading processed data...")
    df = pd.read_csv('data/processed/processed_creditcard.csv')

    y = df['Class']
    X = df.drop(columns=['Class', 'Time', 'customer_id', 'merchant_id'])

    # Store feature names (critical for production inference!)
    feature_names = X.columns.tolist()

    print("2. Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3. Training XGBoost with optimized parameters...")
    scale_weight = float(y_train.value_counts()[0] / y_train.value_counts()[1])

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("4. Evaluating model performance...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Use the optimized threshold from Phase 8
    OPTIMAL_THRESHOLD = 0.95
    y_pred = (y_pred_proba >= OPTIMAL_THRESHOLD).astype(int)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")

    print("\n5. Saving model artifacts...")
    os.makedirs('models', exist_ok=True)

    # Save the trained model
    model_path = 'models/fraud_model.joblib'
    joblib.dump(model, model_path)
    print(f"   ✓ Model saved: {model_path}")

    # Save metadata
    metadata = {
        "model_version": "1.0.0",
        "model_type": "XGBoost",
        "trained_on": datetime.now().isoformat(),
        "threshold": OPTIMAL_THRESHOLD,
        "feature_names": feature_names,
        "performance": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
    }

    metadata_path = 'models/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Metadata saved: {metadata_path}")

    print("\n" + "="*60)
    print("  MODEL PACKAGING COMPLETE! ✅")
    print("="*60)
    print(f"\nModel is ready for deployment!")
    print(f"Load it with: model = joblib.load('{model_path}')")

if __name__ == "__main__":
    train_and_save_model()
