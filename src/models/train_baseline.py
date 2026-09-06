import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore') # Prevent clutter in terminal

def run_baseline():
    print("1. Loading processed data...")
    df = pd.read_csv('data/processed/processed_creditcard.csv')

    # 2. Separate Features (X) and Target (y)
    y = df['Class']

    # We drop 'Class' because it's the answer.
    # We drop 'Time' because we engineered 'hour_of_day'.
    # We drop IDs to prevent the model from memorizing specific customers (Data Leakage).
    X = df.drop(columns=['Class', 'Time', 'customer_id', 'merchant_id'])

    print("2. Splitting data (80% Train, 20% Test)...")
    # stratify=y ensures the 20% test set gets the exact same ratio of fraud as the 80% train set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("3. Scaling data for Logistic Regression...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("4. Training Logistic Regression Baseline...")
    # class_weight='balanced' tells the model to pay heavy attention to the rare fraud class
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print("5. Evaluating Model...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    # Calculate PR-AUC (Precision-Recall Area Under Curve - Gold standard for Fraud)
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    # Calculate False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)

    print("\n" + "="*40)
    print("   BASELINE MODEL RESULTS (LogReg)")
    print("="*40)
    print(f"Precision:           {precision:.4f}  <-- (% of flagged items that are actual fraud)")
    print(f"Recall:              {recall:.4f}  <-- (% of true fraud caught)")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}  <-- (Higher is better for imbalanced data)")
    print(f"False Positive Rate: {fpr:.4f}  <-- ({fp} legitimate users blocked!)")
    print("="*40)
    print("Conclusion: High Recall, but terrible Precision. We blocked too many innocent customers!")
    print("Next step: XGBoost.")

if __name__ == "__main__":
    run_baseline()
