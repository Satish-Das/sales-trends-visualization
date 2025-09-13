#!/usr/bin/env python3
"""
Customer Churn Prediction Model
This script builds a simple churn prediction model using RFM features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
import joblib
import os

warnings.filterwarnings('ignore')

def prepare_churn_data():
    """Prepare data for churn prediction"""

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION MODEL")
    print("=" * 60)

    print("\n[1/6] Loading data...")

    # Load RFM data
    rfm = pd.read_csv('data/rfm_table.csv')

    # Load main dataset for additional features
    df = pd.read_csv('data/superstore_clean.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    print(f"✓ Loaded {len(rfm):,} customers")

    # Define churn (customers who haven't ordered in last 90 days)
    # Since Recency is already days since last order, we can use it directly
    rfm['Churned'] = (rfm['Recency'] > 90).astype(int)

    # Calculate additional features
    print("\n[2/6] Engineering features...")

    # Customer-level aggregations
    customer_stats = df.groupby('Customer ID').agg({
        'Order ID': 'nunique',
        'Sales': ['sum', 'mean', 'std'],
        'Profit': ['sum', 'mean'],
        'Discount': 'mean',
        'Quantity': 'sum',
        'profit_margin': 'mean'
    }).reset_index()

    # Flatten column names
    customer_stats.columns = ['Customer ID', 'total_orders', 'total_sales', 'avg_sales',
                              'std_sales', 'total_profit', 'avg_profit', 'avg_discount',
                              'total_quantity', 'avg_margin']

    # Fill NaN values in std_sales with 0
    customer_stats['std_sales'] = customer_stats['std_sales'].fillna(0)

    # Merge with RFM data
    churn_data = rfm.merge(customer_stats, on='Customer ID', how='left')

    # Create derived features
    churn_data['avg_order_value'] = churn_data['Monetary'] / churn_data['Frequency']
    churn_data['order_frequency_rate'] = churn_data['Frequency'] / (churn_data['Recency'] + 1)
    churn_data['sales_volatility'] = churn_data['std_sales'] / (churn_data['avg_sales'] + 1)

    print(f"✓ Created {len(churn_data.columns) - 1} features")

    # Churn statistics
    churn_rate = churn_data['Churned'].mean() * 100
    print(f"\n[3/6] Churn Statistics:")
    print(f"  • Churned customers: {churn_data['Churned'].sum():,} ({churn_rate:.1f}%)")
    print(f"  • Active customers: {(1-churn_data['Churned']).sum():,} ({100-churn_rate:.1f}%)")

    return churn_data

def build_churn_model(churn_data):
    """Build and evaluate churn prediction model"""

    print("\n[4/6] Building prediction model...")

    # Select features for modeling
    feature_columns = ['Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
                       'total_orders', 'avg_sales', 'std_sales', 'avg_profit', 'avg_discount',
                       'total_quantity', 'avg_margin', 'avg_order_value', 'order_frequency_rate',
                       'sales_volatility']

    # Handle any remaining NaN values
    X = churn_data[feature_columns].fillna(0)
    y = churn_data['Churned']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"  • Training set: {len(X_train)} samples")
    print(f"  • Test set: {len(X_test)} samples")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models
    print("\n[5/6] Training models...")

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    print("✓ Models trained successfully")

    # Evaluate models
    print("\n[6/6] Model Evaluation:")
    print("\n" + "=" * 40)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 40)
    print(classification_report(y_test, lr_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, lr_pred_proba):.3f}")

    print("\n" + "=" * 40)
    print("RANDOM FOREST RESULTS")
    print("=" * 40)
    print(classification_report(y_test, rf_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, rf_pred_proba):.3f}")

    # Feature importance (Random Forest)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "=" * 40)
    print("TOP 10 FEATURE IMPORTANCE")
    print("=" * 40)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['feature']:20s}: {row['importance']:.4f}")

    # Save models
    os.makedirs('models', exist_ok=True)

    joblib.dump(lr_model, 'models/logistic_regression_churn.pkl')
    joblib.dump(rf_model, 'models/random_forest_churn.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    print("\n✓ Models saved to models/")

    # Save model performance metrics
    metrics = {
        'logistic_regression': {
            'roc_auc': float(roc_auc_score(y_test, lr_pred_proba)),
            'accuracy': float((lr_pred == y_test).mean()),
            'churn_precision': float(classification_report(y_test, lr_pred, output_dict=True)['1']['precision']),
            'churn_recall': float(classification_report(y_test, lr_pred, output_dict=True)['1']['recall'])
        },
        'random_forest': {
            'roc_auc': float(roc_auc_score(y_test, rf_pred_proba)),
            'accuracy': float((rf_pred == y_test).mean()),
            'churn_precision': float(classification_report(y_test, rf_pred, output_dict=True)['1']['precision']),
            'churn_recall': float(classification_report(y_test, rf_pred, output_dict=True)['1']['recall'])
        },
        'top_features': feature_importance.head(5).to_dict('records')
    }

    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    return rf_model, scaler, feature_columns, X_test, y_test, rf_pred_proba

def create_visualizations(rf_model, X_test, y_test, rf_pred_proba):
    """Create model performance visualizations"""

    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, rf_model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_pred_proba)
    auc_score = roc_auc_score(y_test, rf_pred_proba)
    axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[0, 1].set_xlim([0.0, 1.0])
    axes[0, 1].set_ylim([0.0, 1.05])
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend(loc="lower right")

    # 3. Prediction Distribution
    axes[1, 0].hist(rf_pred_proba[y_test == 0], bins=30, alpha=0.5, label='Not Churned', color='green')
    axes[1, 0].hist(rf_pred_proba[y_test == 1], bins=30, alpha=0.5, label='Churned', color='red')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Probability Distribution')
    axes[1, 0].legend()

    # 4. Feature Importance (Top 10)
    feature_importance = pd.DataFrame({
        'feature': rf_model.feature_names_in_,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)

    axes[1, 1].barh(range(len(feature_importance)), feature_importance['importance'])
    axes[1, 1].set_yticks(range(len(feature_importance)))
    axes[1, 1].set_yticklabels(feature_importance['feature'])
    axes[1, 1].set_xlabel('Importance')
    axes[1, 1].set_title('Top 10 Feature Importance')

    plt.tight_layout()
    plt.savefig('visuals/churn_model_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to visuals/churn_model_performance.png")

    plt.close()

def predict_at_risk_customers(churn_data, rf_model, scaler, feature_columns):
    """Identify at-risk customers for targeted interventions"""

    print("\n" + "=" * 60)
    print("IDENTIFYING AT-RISK CUSTOMERS")
    print("=" * 60)

    # Prepare features
    X_all = churn_data[feature_columns].fillna(0)

    # Make predictions
    churn_probabilities = rf_model.predict_proba(X_all)[:, 1]
    churn_data['Churn_Probability'] = churn_probabilities

    # Identify at-risk customers (probability > 0.5 but not yet churned)
    at_risk = churn_data[(churn_data['Churn_Probability'] > 0.5) & (churn_data['Churned'] == 0)]

    # Get top 20 at-risk customers with highest value
    top_at_risk = at_risk.nlargest(20, 'Monetary')[['Customer ID', 'Customer Name', 'Segment',
                                                     'Recency', 'Frequency', 'Monetary',
                                                     'Customer_Segment', 'Churn_Probability']]

    # Save at-risk customers list
    top_at_risk.to_csv('data/at_risk_customers.csv', index=False)

    print(f"\n✓ Identified {len(at_risk)} at-risk customers")
    print(f"✓ Saved top 20 high-value at-risk customers to data/at_risk_customers.csv")

    print("\nTop 5 At-Risk Customers (Highest Value):")
    print("=" * 80)
    for idx, row in top_at_risk.head(5).iterrows():
        print(f"• {row['Customer Name'][:30]:30s} | Segment: {row['Customer_Segment']:15s} | "
              f"Value: ${row['Monetary']:8.2f} | Risk: {row['Churn_Probability']:.1%}")

    return top_at_risk

def main():
    """Main execution function"""

    print("\n🔮 " * 20)
    print("CHURN PREDICTION MODEL DEVELOPMENT")
    print("🔮 " * 20)

    try:
        # Prepare data
        churn_data = prepare_churn_data()

        # Build model
        rf_model, scaler, feature_columns, X_test, y_test, rf_pred_proba = build_churn_model(churn_data)

        # Create visualizations
        create_visualizations(rf_model, X_test, y_test, rf_pred_proba)

        # Identify at-risk customers
        at_risk_customers = predict_at_risk_customers(churn_data, rf_model, scaler, feature_columns)

        print("\n" + "=" * 60)
        print("✅ CHURN PREDICTION MODEL COMPLETE!")
        print("=" * 60)

        print("\n📊 Model Performance Summary:")
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)

        print(f"  • ROC-AUC Score: {metrics['random_forest']['roc_auc']:.3f}")
        print(f"  • Accuracy: {metrics['random_forest']['accuracy']:.1%}")
        print(f"  • Churn Precision: {metrics['random_forest']['churn_precision']:.1%}")
        print(f"  • Churn Recall: {metrics['random_forest']['churn_recall']:.1%}")

        print("\n📁 Files Generated:")
        print("  • models/*.pkl - Trained models and scaler")
        print("  • models/model_metrics.json - Model performance metrics")
        print("  • data/at_risk_customers.csv - High-value at-risk customers")
        print("  • visuals/churn_model_performance.png - Model visualizations")

        print("\n💡 Business Recommendations:")
        print("  1. Focus retention efforts on identified at-risk customers")
        print("  2. Implement targeted campaigns based on churn probability")
        print("  3. Monitor recency as the top predictor of churn")
        print("  4. Develop incentives for customers with declining frequency")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()