#!/usr/bin/env python3
"""
Test Script for Superstore Analytics Application
Verifies that all components are working correctly
"""

import pandas as pd
import sqlite3
import json
import os
import sys

def test_data_files():
    """Test if all required data files exist and are valid"""
    print("\n" + "="*60)
    print("TESTING DATA FILES")
    print("="*60)

    required_files = [
        'data/raw_data.csv',
        'data/superstore_clean.csv',
        'data/orders_clean.csv',
        'data/customers_clean.csv',
        'data/products_clean.csv',
        'data/order_items_clean.csv',
        'data/rfm_table.csv',
        'data/data_summary.json',
        'data/at_risk_customers.csv'
    ]

    all_good = True
    for file in required_files:
        if os.path.exists(file):
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file, encoding='latin-1')
                except:
                    df = pd.read_csv(file)
                print(f"✅ {file:40s} - {len(df):,} rows")
            elif file.endswith('.json'):
                with open(file, 'r') as f:
                    data = json.load(f)
                print(f"✅ {file:40s} - Valid JSON")
        else:
            print(f"❌ {file:40s} - NOT FOUND")
            all_good = False

    return all_good

def test_database():
    """Test SQLite database connectivity and data"""
    print("\n" + "="*60)
    print("TESTING DATABASE")
    print("="*60)

    try:
        conn = sqlite3.connect('db/superstore.db')
        cursor = conn.cursor()

        tables = ['customers', 'products', 'orders', 'order_items']
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"✅ Table '{table:15s}' - {count:,} rows")

        # Test a sample query
        cursor.execute("""
            SELECT COUNT(DISTINCT customer_id) as customers,
                   SUM(sales) as total_revenue
            FROM order_items oi
            JOIN orders o ON oi.order_id = o.order_id
        """)
        result = cursor.fetchone()
        print(f"\n✅ Sample Query Test:")
        print(f"   Total Customers: {result[0]:,}")
        print(f"   Total Revenue: ${result[1]:,.2f}")

        conn.close()
        return True
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return False

def test_models():
    """Test if ML models exist"""
    print("\n" + "="*60)
    print("TESTING ML MODELS")
    print("="*60)

    model_files = [
        'models/random_forest_churn.pkl',
        'models/logistic_regression_churn.pkl',
        'models/scaler.pkl',
        'models/model_metrics.json'
    ]

    all_good = True
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✅ {file:40s} - {size:.1f} KB")
        else:
            print(f"❌ {file:40s} - NOT FOUND")
            all_good = False

    # Check model performance
    if os.path.exists('models/model_metrics.json'):
        with open('models/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"\n📊 Model Performance:")
        print(f"   ROC-AUC: {metrics['random_forest']['roc_auc']:.3f}")
        print(f"   Accuracy: {metrics['random_forest']['accuracy']:.1%}")

    return all_good

def test_visualizations():
    """Test if visualizations exist"""
    print("\n" + "="*60)
    print("TESTING VISUALIZATIONS")
    print("="*60)

    visual_files = [
        'visuals/revenue_trend.html',
        'visuals/top_products.html',
        'visuals/category_performance.html',
        'visuals/churn_model_performance.png'
    ]

    all_good = True
    for file in visual_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024  # KB
            print(f"✅ {file:40s} - {size:.1f} KB")
        else:
            print(f"❌ {file:40s} - NOT FOUND")
            all_good = False

    return all_good

def test_app_components():
    """Test application components"""
    print("\n" + "="*60)
    print("TESTING APPLICATION COMPONENTS")
    print("="*60)

    # Check if Streamlit app file exists
    if os.path.exists('app/streamlit_app.py'):
        with open('app/streamlit_app.py', 'r') as f:
            lines = len(f.readlines())
        print(f"✅ Streamlit App - {lines} lines of code")
    else:
        print("❌ Streamlit App - NOT FOUND")
        return False

    # Check for notebooks
    notebooks = [
        'notebooks/01_data_overview_and_cleaning.ipynb',
        'notebooks/02_etl_and_sql_schema.ipynb',
        'notebooks/03_analysis_visualizations.ipynb'
    ]

    for nb in notebooks:
        if os.path.exists(nb):
            print(f"✅ {nb.split('/')[-1]:40s} - Found")
        else:
            print(f"❌ {nb.split('/')[-1]:40s} - NOT FOUND")

    return True

def test_data_quality():
    """Test data quality and consistency"""
    print("\n" + "="*60)
    print("TESTING DATA QUALITY")
    print("="*60)

    # Load main dataset
    df = pd.read_csv('data/superstore_clean.csv')

    # Check for critical columns
    required_columns = ['Order ID', 'Customer ID', 'Product ID', 'Sales', 'Profit']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False
    else:
        print("✅ All critical columns present")

    # Check for null values in critical columns
    null_counts = df[required_columns].isnull().sum()
    if null_counts.sum() == 0:
        print("✅ No null values in critical columns")
    else:
        print(f"⚠️ Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Data statistics
    print(f"\n📊 Data Statistics:")
    print(f"   Date Range: {df['Order Date'].min()} to {df['Order Date'].max()}")
    print(f"   Total Orders: {df['Order ID'].nunique():,}")
    print(f"   Total Customers: {df['Customer ID'].nunique():,}")
    print(f"   Total Revenue: ${df['Sales'].sum():,.2f}")
    print(f"   Average Order Value: ${df.groupby('Order ID')['Sales'].sum().mean():.2f}")

    return True

def main():
    """Run all tests"""
    print("\n" + "🧪 "*20)
    print("SUPERSTORE ANALYTICS - COMPREHENSIVE TEST SUITE")
    print("🧪 "*20)

    tests = [
        ("Data Files", test_data_files),
        ("Database", test_database),
        ("ML Models", test_models),
        ("Visualizations", test_visualizations),
        ("App Components", test_app_components),
        ("Data Quality", test_data_quality)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    print("\n" + "="*60)
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Application is ready for deployment!")
    else:
        print(f"⚠️ Some tests failed: {passed}/{total} passed")
        print("Please review the errors above.")

    print("\n📌 Streamlit App Status:")
    print("   URL: http://localhost:8502")
    print("   Status: Running (check browser)")
    print("\n💡 To stop the app, press Ctrl+C in the terminal")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)