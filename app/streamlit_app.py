import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import pickle
import joblib
import json
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Superstore Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2ca02c;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the data"""
    try:
        # Load main dataset
        df = pd.read_csv('./data/superstore_clean.csv')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df['Ship Date'] = pd.to_datetime(df['Ship Date'])

        # Load RFM data if available
        try:
            rfm_df = pd.read_csv('./data/rfm_table.csv')
        except:
            rfm_df = None

        return df, rfm_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def load_database_data():
    """Load data from SQLite database"""
    try:
        conn = sqlite3.connect('../db/superstore.db')

        # Load key tables
        customers = pd.read_sql("SELECT * FROM customers LIMIT 1000", conn)
        orders = pd.read_sql("SELECT * FROM orders LIMIT 1000", conn)
        order_items = pd.read_sql("SELECT * FROM order_items LIMIT 1000", conn)
        products = pd.read_sql("SELECT * FROM products LIMIT 1000", conn)

        conn.close()
        return customers, orders, order_items, products
    except Exception as e:
        st.warning(f"Could not load database: {e}")
        return None, None, None, None

def calculate_kpis(df):
    """Calculate key performance indicators"""
    total_revenue = df['Sales'].sum()
    total_profit = df['Profit'].sum()
    total_orders = df['Order ID'].nunique()
    total_customers = df['Customer ID'].nunique()
    avg_order_value = df.groupby('Order ID')['Sales'].sum().mean()
    profit_margin = (total_profit / total_revenue) * 100

    # YoY growth calculation
    current_year = df['Order Date'].dt.year.max()
    prev_year = current_year - 1

    current_year_revenue = df[df['Order Date'].dt.year == current_year]['Sales'].sum()
    prev_year_revenue = df[df['Order Date'].dt.year == prev_year]['Sales'].sum()
    yoy_growth = ((current_year_revenue - prev_year_revenue) / prev_year_revenue) * 100 if prev_year_revenue > 0 else 0

    # Churn rate calculation
    max_date = df['Order Date'].max()
    churn_threshold = max_date - timedelta(days=90)
    active_customers = df[df['Order Date'] > churn_threshold]['Customer ID'].nunique()
    churn_rate = ((total_customers - active_customers) / total_customers) * 100

    return {
        'total_revenue': total_revenue,
        'total_profit': total_profit,
        'total_orders': total_orders,
        'total_customers': total_customers,
        'avg_order_value': avg_order_value,
        'profit_margin': profit_margin,
        'yoy_growth': yoy_growth,
        'churn_rate': churn_rate,
        'active_customers': active_customers
    }

def create_revenue_trend(df, date_filter):
    """Create revenue trend chart"""
    filtered_df = df[df['Order Date'].between(date_filter[0], date_filter[1])]

    monthly_trend = filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M')).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique'
    }).reset_index()

    monthly_trend['Date'] = monthly_trend['Order Date'].dt.to_timestamp()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=monthly_trend['Date'], y=monthly_trend['Sales'],
                   mode='lines+markers', name='Revenue',
                   line=dict(color='#1f77b4', width=3)),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=monthly_trend['Date'], y=monthly_trend['Profit'],
                   mode='lines+markers', name='Profit',
                   line=dict(color='#2ca02c', width=2)),
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Revenue ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    fig.update_layout(title="Revenue and Profit Trends", height=400)

    return fig

def create_category_analysis(df, filters):
    """Create category performance analysis"""
    filtered_df = df.copy()

    if filters['region'] != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == filters['region']]

    if filters['segment'] != 'All':
        filtered_df = filtered_df[filtered_df['Segment'] == filters['segment']]

    category_perf = filtered_df.groupby('Category').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'Order ID': 'nunique'
    }).reset_index()

    category_perf['Profit_Margin'] = (category_perf['Profit'] / category_perf['Sales']) * 100

    fig = px.scatter(category_perf,
                     x='Sales',
                     y='Profit_Margin',
                     size='Order ID',
                     color='Category',
                     hover_data=['Quantity'],
                     title='Category Performance: Revenue vs Profit Margin')

    fig.update_layout(height=400)
    return fig

def create_geographic_analysis(df, filters):
    """Create geographic performance analysis"""
    filtered_df = df.copy()

    if filters['category'] != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == filters['category']]

    regional_perf = filtered_df.groupby('Region').agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Order ID': 'nunique',
        'Customer ID': 'nunique'
    }).reset_index()

    regional_perf['Profit_Margin'] = (regional_perf['Profit'] / regional_perf['Sales']) * 100
    regional_perf['AOV'] = regional_perf['Sales'] / regional_perf['Order ID']

    fig = px.bar(regional_perf,
                 x='Region',
                 y='Sales',
                 color='Profit_Margin',
                 title='Regional Performance',
                 color_continuous_scale='RdYlGn')

    fig.update_layout(height=400)
    return fig

def create_customer_segments_chart(rfm_df):
    """Create customer segmentation chart"""
    if rfm_df is None:
        return None

    segment_counts = rfm_df['Customer_Segment'].value_counts()

    fig = px.pie(values=segment_counts.values,
                 names=segment_counts.index,
                 title='Customer Segmentation Distribution')

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    return fig

@st.cache_resource
def load_prediction_models():
    """Load trained models for churn prediction"""
    import os
    try:
        # Get the directory where the app is running from
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), 'models')

        lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_churn.pkl'))
        rf_model = joblib.load(os.path.join(models_dir, 'random_forest_churn.pkl'))
        scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))

        with open(os.path.join(models_dir, 'model_metrics.json'), 'r') as f:
            metrics = json.load(f)

        return lr_model, rf_model, scaler, metrics
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error(f"Current working directory: {os.getcwd()}")
        st.error(f"Looking for models in: {models_dir if 'models_dir' in locals() else 'Path not set'}")
        return None, None, None, None

def prepare_prediction_features(customer_data):
    """Prepare features for prediction using the exact same features as training"""

    # Create feature DataFrame
    features_df = pd.DataFrame([customer_data])

    # Calculate derived features
    features_df['order_frequency_rate'] = features_df['Frequency'] / max(features_df['total_orders'].iloc[0], 1)
    features_df['avg_order_value'] = features_df['Monetary'] / max(features_df['Frequency'].iloc[0], 1)

    # Calculate RFM scores if not provided
    if 'R_Score' not in customer_data:
        if customer_data['Recency'] <= 30:
            features_df['R_Score'] = 5
        elif customer_data['Recency'] <= 60:
            features_df['R_Score'] = 4
        elif customer_data['Recency'] <= 120:
            features_df['R_Score'] = 3
        elif customer_data['Recency'] <= 240:
            features_df['R_Score'] = 2
        else:
            features_df['R_Score'] = 1

    if 'F_Score' not in customer_data:
        if customer_data['Frequency'] >= 10:
            features_df['F_Score'] = 5
        elif customer_data['Frequency'] >= 5:
            features_df['F_Score'] = 4
        elif customer_data['Frequency'] >= 3:
            features_df['F_Score'] = 3
        elif customer_data['Frequency'] >= 2:
            features_df['F_Score'] = 2
        else:
            features_df['F_Score'] = 1

    if 'M_Score' not in customer_data:
        if customer_data['Monetary'] >= 1000:
            features_df['M_Score'] = 5
        elif customer_data['Monetary'] >= 500:
            features_df['M_Score'] = 4
        elif customer_data['Monetary'] >= 200:
            features_df['M_Score'] = 3
        elif customer_data['Monetary'] >= 100:
            features_df['M_Score'] = 2
        else:
            features_df['M_Score'] = 1

    # Add the exact features that were used during training
    # These are estimates since we don't have access to detailed transaction history in the input
    features_df['avg_sales'] = features_df['avg_order_value']  # Approximate as avg_order_value
    features_df['std_sales'] = features_df['avg_order_value'] * 0.3  # Estimate std as 30% of avg
    features_df['avg_profit'] = features_df['avg_order_value'] * 0.2  # Estimate 20% profit margin
    features_df['avg_discount'] = features_df['avg_order_value'] * 0.1  # Estimate 10% discount
    features_df['avg_margin'] = 0.2  # Estimate 20% margin
    features_df['sales_volatility'] = features_df['std_sales'] / max(features_df['avg_sales'].iloc[0], 1)

    # Use the exact feature columns from training (in the same order)
    feature_columns = [
        'Recency', 'Frequency', 'Monetary', 'R_Score', 'F_Score', 'M_Score',
        'total_orders', 'avg_sales', 'std_sales', 'avg_profit', 'avg_discount',
        'total_quantity', 'avg_margin', 'avg_order_value', 'order_frequency_rate',
        'sales_volatility'
    ]

    return features_df[feature_columns]

def predict_churn(customer_data, models):
    """Predict churn probability for a customer"""
    lr_model, rf_model, scaler, metrics = models

    if any(model is None for model in models):
        return None

    # Prepare features
    features = prepare_prediction_features(customer_data)

    # Scale features
    features_scaled = scaler.transform(features)

    # Make predictions with both models
    rf_prob = rf_model.predict_proba(features_scaled)[0][1]
    lr_prob = lr_model.predict_proba(features_scaled)[0][1]

    # Get binary predictions
    rf_pred = rf_model.predict(features_scaled)[0]
    lr_pred = lr_model.predict(features_scaled)[0]

    return {
        'random_forest': {
            'probability': rf_prob,
            'prediction': bool(rf_pred),
            'accuracy': metrics['random_forest']['accuracy']
        },
        'logistic_regression': {
            'probability': lr_prob,
            'prediction': bool(lr_pred),
            'accuracy': metrics['logistic_regression']['accuracy']
        },
        'consensus': {
            'avg_probability': (rf_prob + lr_prob) / 2,
            'both_predict_churn': bool(rf_pred) and bool(lr_pred)
        }
    }

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability >= 0.7:
        return 'High Risk', '🔴'
    elif probability >= 0.4:
        return 'Medium Risk', '🟡'
    else:
        return 'Low Risk', '🟢'

def show_prediction_page():
    """Show the churn prediction page"""
    st.header("🔮 Customer Churn Prediction")

    # Load models
    models = load_prediction_models()

    if any(model is None for model in models):
        st.error("Could not load prediction models. Please check if model files exist.")
        return

    _, _, _, metrics = models

    # Show model performance
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌲 Random Forest Model")
        st.metric("Accuracy", f"{metrics['random_forest']['accuracy']:.1%}")
        st.metric("ROC AUC", f"{metrics['random_forest']['roc_auc']:.3f}")

    with col2:
        st.subheader("📈 Logistic Regression Model")
        st.metric("Accuracy", f"{metrics['logistic_regression']['accuracy']:.1%}")
        st.metric("ROC AUC", f"{metrics['logistic_regression']['roc_auc']:.3f}")

    st.divider()

    # Input form
    st.subheader("📝 Enter Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🕒 Recency & Frequency**")
        recency = st.number_input("Days since last order", min_value=0, value=30, step=1)
        frequency = st.number_input("Number of orders", min_value=1, value=3, step=1)
        total_orders = st.number_input("Total orders", min_value=1, value=frequency, step=1)

    with col2:
        st.markdown("**💰 Monetary Value**")
        monetary = st.number_input("Total amount spent ($)", min_value=0.0, value=200.0, step=10.0)
        avg_order_value = st.number_input("Average order value ($)", min_value=0.0, value=monetary/frequency, step=5.0)

    with col3:
        st.markdown("**📦 Order Details**")
        total_quantity = st.number_input("Total items purchased", min_value=1, value=frequency*2, step=1)

    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary"):

        # Prepare customer data
        customer_data = {
            'Recency': recency,
            'Frequency': frequency,
            'Monetary': monetary,
            'total_orders': total_orders,
            'avg_order_value': avg_order_value,
            'total_quantity': total_quantity
        }

        # Make prediction
        with st.spinner("Making prediction..."):
            results = predict_churn(customer_data, models)

        if results:
            st.divider()
            st.subheader("📊 Prediction Results")

            # Show consensus first
            consensus = results['consensus']
            risk_level, risk_icon = get_risk_level(consensus['avg_probability'])

            st.markdown(f"""
            <div class="insight-box">
                <h3>{risk_icon} Overall Risk Assessment</h3>
                <p><strong>Risk Level:</strong> {risk_level}</p>
                <p><strong>Average Churn Probability:</strong> {consensus['avg_probability']:.1%}</p>
                <p><strong>Model Consensus:</strong> {'Both models predict churn' if consensus['both_predict_churn'] else 'Models disagree or predict retention'}</p>
            </div>
            """, unsafe_allow_html=True)

            # Detailed results
            col1, col2 = st.columns(2)

            with col1:
                rf_result = results['random_forest']
                st.markdown("**🌲 Random Forest Prediction**")
                st.metric("Churn Probability", f"{rf_result['probability']:.1%}")
                st.metric("Prediction", "Will Churn" if rf_result['prediction'] else "Will Retain")

            with col2:
                lr_result = results['logistic_regression']
                st.markdown("**📈 Logistic Regression Prediction**")
                st.metric("Churn Probability", f"{lr_result['probability']:.1%}")
                st.metric("Prediction", "Will Churn" if lr_result['prediction'] else "Will Retain")

            # Recommendations
            st.subheader("💡 Recommendations")

            avg_prob = consensus['avg_probability']
            if avg_prob >= 0.7:
                st.error("""
                **🚨 HIGH RISK - Immediate Action Required!**
                - Offer significant discount or loyalty bonus
                - Personal outreach from customer success team
                - Expedited customer service support
                - Consider win-back campaign if they haven't ordered recently
                """)
            elif avg_prob >= 0.4:
                st.warning("""
                **⚠️ MEDIUM RISK - Monitor and Engage**
                - Send targeted retention email campaigns
                - Offer personalized product recommendations
                - Provide small incentives or discounts
                - Monitor ordering patterns closely
                """)
            else:
                st.success("""
                **✅ LOW RISK - Focus on Growth**
                - Continue regular engagement
                - Focus on upselling and cross-selling
                - Gather feedback for service improvement
                - Consider loyalty program enrollment
                """)

    # Feature importance
    st.divider()
    st.subheader("📈 Model Feature Importance")

    if 'top_features' in metrics:
        features_df = pd.DataFrame(metrics['top_features'])

        fig = px.bar(features_df,
                     x='importance',
                     y='feature',
                     orientation='h',
                     title='Top Features for Churn Prediction')
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">📊 Superstore Analytics Dashboard</h1>', unsafe_allow_html=True)

    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["📊 Analytics Dashboard", "🔮 Churn Prediction"]
    )

    if page == "🔮 Churn Prediction":
        show_prediction_page()
        return

    # Load data
    df, rfm_df = load_data()

    if df is None:
        st.error("Failed to load data. Please check if the data files exist.")
        return

    # Sidebar filters
    st.sidebar.header("🔧 Filters")

    # Date range filter
    min_date = df['Order Date'].min().date()
    max_date = df['Order Date'].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if len(date_range) == 2:
        date_filter = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))
    else:
        date_filter = (pd.to_datetime(min_date), pd.to_datetime(max_date))

    # Other filters
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    segments = ['All'] + sorted(df['Segment'].unique().tolist())
    categories = ['All'] + sorted(df['Category'].unique().tolist())

    filters = {
        'region': st.sidebar.selectbox("Region", regions),
        'segment': st.sidebar.selectbox("Customer Segment", segments),
        'category': st.sidebar.selectbox("Category", categories)
    }

    # Apply filters
    filtered_df = df[df['Order Date'].between(date_filter[0], date_filter[1])]

    if filters['region'] != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == filters['region']]
    if filters['segment'] != 'All':
        filtered_df = filtered_df[filtered_df['Segment'] == filters['segment']]
    if filters['category'] != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == filters['category']]

    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)

    # KPI Cards
    st.subheader("📈 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Total Revenue",
            value=f"${kpis['total_revenue']:,.0f}",
            delta=f"{kpis['yoy_growth']:.1f}% YoY"
        )

    with col2:
        st.metric(
            label="Total Profit",
            value=f"${kpis['total_profit']:,.0f}",
            delta=f"{kpis['profit_margin']:.1f}% margin"
        )

    with col3:
        st.metric(
            label="Total Orders",
            value=f"{kpis['total_orders']:,}",
            delta=f"${kpis['avg_order_value']:.2f} AOV"
        )

    with col4:
        st.metric(
            label="Active Customers",
            value=f"{kpis['active_customers']:,}",
            delta=f"{kpis['churn_rate']:.1f}% churn rate",
            delta_color="inverse"
        )

    # Charts Section
    st.subheader("📊 Analytics")

    # Revenue Trend
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_revenue_trend(filtered_df, date_filter), use_container_width=True)

    with col2:
        if rfm_df is not None:
            st.plotly_chart(create_customer_segments_chart(rfm_df), use_container_width=True)
        else:
            st.info("RFM analysis data not available. Run the analysis notebook first.")

    # Category and Geographic Analysis
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_category_analysis(filtered_df, filters), use_container_width=True)

    with col2:
        st.plotly_chart(create_geographic_analysis(filtered_df, filters), use_container_width=True)

    # Top Products Table
    st.subheader("🏆 Top Products")

    top_products = filtered_df.groupby(['Product Name', 'Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum'
    }).reset_index().sort_values('Sales', ascending=False).head(10)

    top_products['Profit Margin %'] = (top_products['Profit'] / top_products['Sales'] * 100).round(2)

    st.dataframe(
        top_products[['Product Name', 'Category', 'Sales', 'Profit', 'Profit Margin %', 'Quantity']],
        use_container_width=True
    )

    # Data Export
    st.subheader("💾 Data Export")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Export Filtered Data"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"superstore_filtered_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("Export Top Products"):
            csv = top_products.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"top_products_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col3:
        if rfm_df is not None and st.button("Export RFM Data"):
            csv = rfm_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    # Insights Section
    st.subheader("💡 Key Insights")

    insights = f"""
    <div class="insight-box">
    <h4>📊 Performance Summary</h4>
    <ul>
        <li><strong>Revenue:</strong> ${kpis['total_revenue']:,.0f} with {kpis['profit_margin']:.1f}% profit margin</li>
        <li><strong>Growth:</strong> {kpis['yoy_growth']:.1f}% year-over-year revenue growth</li>
        <li><strong>Customer Base:</strong> {kpis['total_customers']:,} total customers with {kpis['churn_rate']:.1f}% churn rate</li>
        <li><strong>Order Performance:</strong> {kpis['total_orders']:,} orders with ${kpis['avg_order_value']:.2f} average order value</li>
    </ul>
    </div>
    """

    st.markdown(insights, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        Built with Streamlit & Plotly |
        Data: Global Superstore Dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()