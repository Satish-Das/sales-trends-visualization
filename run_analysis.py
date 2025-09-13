#!/usr/bin/env python3
"""
Data Processing and Analysis Script for Superstore Analytics
This script runs all data processing steps to generate required outputs
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

def process_data():
    """Main data processing function"""

    print("=" * 60)
    print("SUPERSTORE DATA PROCESSING PIPELINE")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/5] Loading raw data...")
    df = pd.read_csv('data/raw_data.csv', encoding='latin-1')
    print(f"✓ Loaded {len(df):,} rows")

    # Step 2: Data cleaning
    print("\n[2/5] Cleaning data...")
    df_clean = df.copy()

    # Parse dates
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

    # Create date components
    df_clean['order_year'] = df_clean['Order Date'].dt.year
    df_clean['order_month'] = df_clean['Order Date'].dt.month
    df_clean['order_quarter'] = df_clean['Order Date'].dt.quarter
    df_clean['order_week'] = df_clean['Order Date'].dt.isocalendar().week
    df_clean['delivery_days'] = (df_clean['Ship Date'] - df_clean['Order Date']).dt.days

    # Add derived columns
    df_clean['revenue'] = df_clean['Sales']
    df_clean['profit_margin'] = np.where(
        df_clean['Sales'] != 0,
        (df_clean['Profit'] / df_clean['Sales']) * 100,
        0
    )

    # Handle missing values
    if 'Postal Code' in df_clean.columns:
        df_clean['Postal Code'] = df_clean['Postal Code'].fillna('Unknown')

    print(f"✓ Cleaned data: {len(df_clean):,} rows")

    # Step 3: Create normalized tables
    print("\n[3/5] Creating normalized tables...")

    # Orders table
    order_columns = ['Order ID', 'Order Date', 'Ship Date', 'Ship Mode', 'Customer ID',
                     'Segment', 'Country', 'City', 'State', 'Postal Code', 'Region',
                     'order_year', 'order_month', 'order_quarter', 'order_week', 'delivery_days']
    orders_df = df_clean[order_columns].drop_duplicates(subset=['Order ID'])

    # Order Items table
    order_items_columns = ['Order ID', 'Product ID', 'Sales', 'Quantity', 'Discount',
                           'Profit', 'Shipping Cost', 'profit_margin']
    order_items_df = df_clean[order_items_columns]

    # Customers table
    customer_columns = ['Customer ID', 'Customer Name', 'Segment']
    customers_df = df_clean[customer_columns].drop_duplicates(subset=['Customer ID'])

    # Products table
    product_columns = ['Product ID', 'Product Name', 'Category', 'Sub-Category']
    products_df = df_clean[product_columns].drop_duplicates(subset=['Product ID'])

    # Save cleaned data
    df_clean.to_csv('data/superstore_clean.csv', index=False)
    orders_df.to_csv('data/orders_clean.csv', index=False)
    order_items_df.to_csv('data/order_items_clean.csv', index=False)
    customers_df.to_csv('data/customers_clean.csv', index=False)
    products_df.to_csv('data/products_clean.csv', index=False)

    print(f"✓ Orders: {len(orders_df):,} rows")
    print(f"✓ Order Items: {len(order_items_df):,} rows")
    print(f"✓ Customers: {len(customers_df):,} rows")
    print(f"✓ Products: {len(products_df):,} rows")

    # Step 4: Create data summary
    print("\n[4/5] Creating data summary...")

    summary = {
        'total_rows': len(df_clean),
        'total_columns': len(df_clean.columns),
        'date_range': {
            'start': str(df_clean['Order Date'].min()),
            'end': str(df_clean['Order Date'].max())
        },
        'unique_counts': {
            'orders': int(df_clean['Order ID'].nunique()),
            'customers': int(df_clean['Customer ID'].nunique()),
            'products': int(df_clean['Product ID'].nunique()),
            'categories': int(df_clean['Category'].nunique()),
            'regions': int(df_clean['Region'].nunique())
        },
        'metrics': {
            'total_revenue': float(df_clean['Sales'].sum()),
            'total_profit': float(df_clean['Profit'].sum()),
            'avg_order_value': float(df_clean.groupby('Order ID')['Sales'].sum().mean()),
            'profit_margin': float((df_clean['Profit'].sum() / df_clean['Sales'].sum()) * 100)
        },
        'top_performers': {
            'top_product': df_clean.groupby('Product Name')['Sales'].sum().idxmax(),
            'top_category': df_clean.groupby('Category')['Sales'].sum().idxmax(),
            'top_region': df_clean.groupby('Region')['Sales'].sum().idxmax(),
            'top_customer': df_clean.groupby('Customer Name')['Sales'].sum().idxmax()
        }
    }

    with open('data/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("✓ Data summary created")

    # Step 5: Create RFM analysis
    print("\n[5/5] Performing RFM analysis...")

    # Calculate RFM metrics
    reference_date = df_clean['Order Date'].max() + timedelta(days=1)

    rfm = df_clean.groupby('Customer ID').agg({
        'Order Date': lambda x: (reference_date - x.max()).days,  # Recency
        'Order ID': 'nunique',  # Frequency
        'Sales': 'sum'  # Monetary
    }).reset_index()

    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

    # Add customer details
    customer_details = df_clean[['Customer ID', 'Customer Name', 'Segment']].drop_duplicates()
    rfm = rfm.merge(customer_details, on='Customer ID')

    # Create RFM scores using quintiles
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

    # Convert to numeric
    rfm['R_Score'] = pd.to_numeric(rfm['R_Score'], errors='coerce').fillna(3).astype(int)
    rfm['F_Score'] = pd.to_numeric(rfm['F_Score'], errors='coerce').fillna(3).astype(int)
    rfm['M_Score'] = pd.to_numeric(rfm['M_Score'], errors='coerce').fillna(3).astype(int)

    # Calculate RFM Score
    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # Customer segmentation
    def segment_customers(row):
        if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
            return 'Champions'
        elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
            return 'Loyal Customers'
        elif row['R_Score'] >= 3 and row['F_Score'] <= 2:
            return 'Potential Loyalists'
        elif row['R_Score'] <= 2 and row['F_Score'] >= 3:
            return 'At Risk'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] >= 3:
            return 'Cant Lose Them'
        elif row['R_Score'] <= 2 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
            return 'Lost'
        else:
            return 'Others'

    rfm['Customer_Segment'] = rfm.apply(segment_customers, axis=1)

    # Save RFM data
    rfm.to_csv('data/rfm_table.csv', index=False)
    print(f"✓ RFM analysis complete: {len(rfm):,} customers segmented")

    return df_clean, rfm

def create_database(df_clean):
    """Create SQLite database"""

    print("\n" + "=" * 60)
    print("CREATING SQLITE DATABASE")
    print("=" * 60)

    # Database path
    db_path = 'db/superstore.db'
    os.makedirs('db', exist_ok=True)

    # Remove existing database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Create connection
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n[1/3] Creating schema...")

    # Read and execute schema
    with open('sql/schema.sql', 'r') as f:
        schema_sql = f.read()

    # Execute schema statements
    for statement in schema_sql.split(';'):
        if statement.strip():
            try:
                cursor.execute(statement)
            except sqlite3.OperationalError as e:
                if 'already exists' not in str(e):
                    print(f"Warning: {e}")

    conn.commit()
    print("✓ Schema created")

    print("\n[2/3] Loading data into tables...")

    # Load cleaned CSVs
    customers_df = pd.read_csv('data/customers_clean.csv')
    products_df = pd.read_csv('data/products_clean.csv')
    orders_df = pd.read_csv('data/orders_clean.csv')
    order_items_df = pd.read_csv('data/order_items_clean.csv')

    # Rename columns to match database schema
    customers_df.columns = ['customer_id', 'customer_name', 'segment']
    products_df.columns = ['product_id', 'product_name', 'category', 'sub_category']
    orders_df.columns = ['order_id', 'order_date', 'ship_date', 'ship_mode', 'customer_id',
                         'segment', 'country', 'city', 'state', 'postal_code', 'region',
                         'order_year', 'order_month', 'order_quarter', 'order_week', 'delivery_days']
    order_items_df.columns = ['order_id', 'product_id', 'sales', 'quantity', 'discount',
                              'profit', 'shipping_cost', 'profit_margin']

    # Create engine for data loading
    engine = create_engine(f'sqlite:///{db_path}')

    # Load data into tables
    customers_df.to_sql('customers', engine, if_exists='replace', index=False)
    products_df.to_sql('products', engine, if_exists='replace', index=False)
    orders_df.to_sql('orders', engine, if_exists='replace', index=False)
    order_items_df.to_sql('order_items', engine, if_exists='replace', index=False)

    print("✓ Data loaded into database")

    print("\n[3/3] Verifying database...")

    # Verify table counts
    tables = ['customers', 'products', 'orders', 'order_items']
    for table in tables:
        count = pd.read_sql(f"SELECT COUNT(*) as count FROM {table}", conn).iloc[0]['count']
        print(f"✓ {table}: {count:,} rows")

    conn.close()
    print("\n✓ Database created successfully at db/superstore.db")

def create_sample_visualizations():
    """Create sample visualization HTML files"""

    print("\n" + "=" * 60)
    print("CREATING SAMPLE VISUALIZATIONS")
    print("=" * 60)

    os.makedirs('visuals', exist_ok=True)

    # Sample HTML template for visualizations
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .placeholder {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                          height: 400px; border-radius: 8px; display: flex; align-items: center;
                          justify-content: center; color: white; font-size: 24px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{title}</h1>
            <div class="placeholder">
                {description}
            </div>
            <p style="margin-top: 20px; color: #666;">
                This is a placeholder for the {title} visualization.
                Run the Jupyter notebooks to generate the actual interactive charts.
            </p>
        </div>
    </body>
    </html>
    """

    visualizations = [
        ('revenue_trend.html', 'Revenue Trend Analysis', 'Monthly Revenue and Profit Trends'),
        ('top_products.html', 'Top Products Analysis', 'Top 10 Products by Revenue'),
        ('category_performance.html', 'Category Performance', 'Revenue vs Profit Margin by Category'),
        ('regional_performance.html', 'Regional Analysis', 'Performance Metrics by Region'),
        ('rfm_3d_segmentation.html', 'RFM Segmentation', '3D Customer Segmentation View'),
        ('customer_segments_pie.html', 'Customer Segments', 'Distribution of Customer Segments'),
        ('seasonal_patterns.html', 'Seasonal Analysis', 'Monthly and Weekly Sales Patterns'),
        ('churn_analysis.html', 'Churn Analysis', 'Customer Churn Rate by Segment')
    ]

    for filename, title, description in visualizations:
        html_content = html_template.format(title=title, description=description)
        with open(f'visuals/{filename}', 'w') as f:
            f.write(html_content)
        print(f"✓ Created {filename}")

    print("\n✓ Sample visualizations created in visuals/")

def main():
    """Main execution function"""

    print("\n" + "🚀 " * 20)
    print("STARTING SUPERSTORE ANALYTICS PIPELINE")
    print("🚀 " * 20)

    try:
        # Process data
        df_clean, rfm = process_data()

        # Create database
        create_database(df_clean)

        # Create sample visualizations
        create_sample_visualizations()

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 60)

        print("\n📊 Summary Statistics:")
        print(f"  • Total Orders: {df_clean['Order ID'].nunique():,}")
        print(f"  • Total Customers: {df_clean['Customer ID'].nunique():,}")
        print(f"  • Total Revenue: ${df_clean['Sales'].sum():,.2f}")
        print(f"  • Date Range: {df_clean['Order Date'].min().date()} to {df_clean['Order Date'].max().date()}")

        print("\n📁 Files Generated:")
        print("  • data/*.csv - Cleaned data files")
        print("  • data/data_summary.json - Dataset summary")
        print("  • data/rfm_table.csv - RFM analysis")
        print("  • db/superstore.db - SQLite database")
        print("  • visuals/*.html - Sample visualizations")

        print("\n🎯 Next Steps:")
        print("  1. Run 'streamlit run app/streamlit_app.py' to launch dashboard")
        print("  2. Open Jupyter notebooks for detailed analysis")
        print("  3. Review insights.md for business recommendations")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()