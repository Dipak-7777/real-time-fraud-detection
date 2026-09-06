import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_shap_explainability():
    print("1. Loading data & retraining XGBoost...")
    df = pd.read_csv('data/processed/processed_creditcard.csv')

    y = df['Class']
    X = df.drop(columns=['Class', 'Time', 'customer_id', 'merchant_id'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_weight = float(y_train.value_counts()[0] / y_train.value_counts()[1])
    model = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='aucpr'
    )
    model.fit(X_train, y_train)

    print("2. Generating SHAP Explainer (this takes ~30 seconds)...")
    # SHAP Tree Explainer is optimized specifically for XGBoost/LightGBM/RandomForest
    explainer = shap.TreeExplainer(model)

    # We will only explain a small sample of 500 test transactions
    # (explaining all 56,000+ would take 20 minutes!)
    sample_size = 500
    X_test_sample = X_test.sample(n=sample_size, random_state=42)

    print(f"3. Computing SHAP values for {sample_size} test transactions...")
    shap_values = explainer.shap_values(X_test_sample)

    print("4. Generating Global Feature Importance Plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, plot_type="bar", show=False)
    plt.title("Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig('notebooks/shap_global_importance.png', dpi=100)
    plt.close()
    print("   Saved: notebooks/shap_global_importance.png")

    print("5. Generating Detailed SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Feature Impact (Red = High feature value, Blue = Low)")
    plt.tight_layout()
    plt.savefig('notebooks/shap_summary_plot.png', dpi=100)
    plt.close()
    print("   Saved: notebooks/shap_summary_plot.png")

    print("\n" + "="*60)
    print("   EXPLAINABILITY COMPLETE!")
    print("="*60)
    print("Check the 'notebooks' folder for 2 new images:")
    print(" - shap_global_importance.png (Which features matter most?)")
    print(" - shap_summary_plot.png (How do features push fraud up/down?)")
    print("\nYou can now explain to a non-technical manager WHY the model")
    print("flagged a transaction!")

if __name__ == "__main__":
    run_shap_explainability()
