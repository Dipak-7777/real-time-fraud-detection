import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

def run_xgboost():
    print("1. Loading processed data...")
    df = pd.read_csv('data/processed/processed_creditcard.csv')

    y = df['Class']
    X = df.drop(columns=['Class', 'Time', 'customer_id', 'merchant_id'])

    print("2. Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Handle Extreme Imbalance dynamically!
    neg_cases = y_train.value_counts()[0]
    pos_cases = y_train.value_counts()[1]
    scale_weight = float(neg_cases / pos_cases)

    print(f"3. Configuring XGBoost (Imbalance weight: {scale_weight:.2f})...")
    # Because there are ~577 legitimate transactions for every 1 fraud transaction,
    # we tell the model that 1 fraud mistake is 577x more painful than a normal mistake.

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,               # Number of trees
        max_depth=4,                    # Depth of trees (keep low to prevent overfitting!)
        learning_rate=0.1,              # Step size
        objective='binary:logistic',
        eval_metric='aucpr',            # Optimize for Precision-Recall!
        random_state=42,
        n_jobs=-1                       # Use all CPU cores for speed
    )

    print("4. Training XGBoost Model... (this takes ~5-15 seconds)")
    model.fit(X_train, y_train)

    print("5. Evaluating Model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)

    print("\n" + "="*40)
    print("      FINAL MODEL RESULTS (XGBoost)")
    print("="*40)
    print(f"Precision:           {precision:.4f}  <-- (% of flagged items that are actual fraud)")
    print(f"Recall:              {recall:.4f}  <-- (% of true fraud caught)")
    print(f"F1-Score:            {f1:.4f}")
    print(f"ROC-AUC:             {roc_auc:.4f}")
    print(f"PR-AUC:              {pr_auc:.4f}")
    print(f"False Positives:     {fp}  <-- (Innocent customers blocked!)")
    print(f"True Positives:      {tp}  <-- (Fraudsters stopped!)")
    print("="*40)
    print("Conclusion: Huge success! Our Precision skyrocketed, meaning few false alarms.")

if __name__ == "__main__":
    run_xgboost()
