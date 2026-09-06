import joblib
import json
import pandas as pd
import numpy as np

class FraudPredictor:
    """Production fraud prediction module."""

    def __init__(self, model_path='models/fraud_model.joblib', metadata_path='models/model_metadata.json'):
        """Load the trained model and metadata."""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.threshold = self.metadata['threshold']
        self.feature_names = self.metadata['feature_names']
        self.model_version = self.metadata['model_version']

        print(f"✓ Model loaded (version {self.model_version}, threshold={self.threshold})")

    def predict(self, transaction_data):
        """
        Make a fraud prediction on a single transaction.

        Args:
            transaction_data (dict): Transaction features as a dictionary

        Returns:
            dict: Prediction result with fraud probability, label, and risk level
        """
        # Convert to DataFrame with correct feature order
        df = pd.DataFrame([transaction_data])

        # Ensure features match training order
        df = df[self.feature_names]

        # Get fraud probability
        fraud_probability = self.model.predict_proba(df)[0, 1]

        # Apply threshold
        is_fraud = fraud_probability >= self.threshold

        # Determine risk level
        if fraud_probability >= 0.99:
            risk_level = "CRITICAL"
        elif fraud_probability >= 0.95:
            risk_level = "HIGH"
        elif fraud_probability >= 0.70:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "fraud_probability": float(fraud_probability),
            "prediction": "FRAUD" if is_fraud else "LEGITIMATE",
            "risk_level": risk_level,
            "model_version": self.model_version
        }


# Quick test when run directly
if __name__ == "__main__":
    print("Testing FraudPredictor...")

    predictor = FraudPredictor()

    # Create a fake test transaction (with all required features)
    test_transaction = {feature: 0.0 for feature in predictor.feature_names}
    test_transaction['Amount'] = 150.0
    test_transaction['amount_log'] = np.log1p(150.0)
    test_transaction['hour_of_day'] = 14

    result = predictor.predict(test_transaction)

    print("\nTest Prediction Result:")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Probability: {result['fraud_probability']:.4f}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Model Version: {result['model_version']}")
    print("\n✅ Predictor module works!")
