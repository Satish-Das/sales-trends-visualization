# 📊 Superstore Sales & Customer Analytics Dashboard

![Project Banner](/images/project_banner.png)

A comprehensive end-to-end data analytics project that transforms the Global Superstore dataset into actionable business intelligence. This project showcases advanced data processing, customer behavior analysis, predictive modeling, and interactive visualization capabilities through a production-ready dashboard.

## 🎯 Project Overview

This project delivers business value through data-driven insights:

- **Sales Performance Analysis**: Identify revenue drivers, seasonal patterns, and growth opportunities
- **Customer Segmentation**: RFM analysis to classify customers and target retention efforts
- **Predictive Churn Modeling**: ML-based prediction with 99.8% accuracy to identify at-risk customers
- **Interactive Dashboard**: Real-time business metrics with filtering and visualization
- **Strategic Recommendations**: Actionable insights with projected ROI for business growth
- **Data Engineering**: Comprehensive ETL pipeline, database design, and quality validation

## 🏗️ Project Structure

```
sales-trends-visualization/
├── app/                              # Interactive dashboard
│   └── streamlit_app.py              # Feature-rich Streamlit application
│
├── data/                             # Data files
│   ├── raw_data.csv                  # Original dataset
│   ├── superstore_clean.csv          # Cleaned master dataset
│   ├── customers_clean.csv           # Customer dimension table
│   ├── orders_clean.csv              # Orders dimension table 
│   ├── products_clean.csv            # Products dimension table
│   ├── order_items_clean.csv         # Order items fact table
│   ├── rfm_table.csv                 # RFM segmentation results
│   ├── at_risk_customers.csv         # High-value customers at risk of churning
│   ├── analysis_insights.txt         # Text-based analysis results
│   └── data_summary.json             # Dataset summary statistics
│
├── db/                               # Database files
│   └── superstore.db                 # SQLite database with normalized schema
│
├── models/                           # Machine learning models
│   └── model_metrics.json            # Performance metrics for ML models
│
├── notebooks/                        # Jupyter analysis notebooks
│   ├── 01_data_overview_and_cleaning.ipynb
│   ├── 02_etl_and_sql_schema.ipynb
│   └── 03_analysis_visualizations.ipynb
│
├── sql/                              # SQL files
│   ├── schema.sql                    # Database schema definitions
│   └── queries.sql                   # Analysis queries
│
├── visuals/                          # Generated visualizations
│   ├── revenue_trend.html            # Interactive revenue charts
│   ├── category_performance.html     # Category analysis 
│   ├── regional_performance.html     # Geographic analysis
│   ├── rfm_3d_segmentation.html      # 3D customer segmentation
│   ├── churn_model_performance.png   # ML model evaluation
│   ├── cohort_retention.png          # Retention analysis
│   └── more visualization files...
│
├── churn_prediction_model.py         # Customer churn prediction model
├── run_analysis.py                   # Data processing pipeline
├── test_app.py                       # Dashboard testing
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Container configuration
├── insights.md                       # Executive summary & recommendations
├── project_summary.md                # Implementation details
└── README.md                         # Project documentation
```

## 🚀 Getting Started

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Satish-Das/sales-trends-visualization
   cd sales-trends-visualization
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the data processing pipeline**
   ```bash
   python run_analysis.py
   ```

4. **Build the churn prediction model**
   ```bash
   python churn_prediction_model.py
   ```

5. **Launch the dashboard**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Access the dashboard at http://localhost:8501

### Option 2: Docker Deployment

1. **Build and run container**
   ```bash
   docker build -t sales-trends-visualization .
   docker run -p 8501:8501 sales-trends-visualization
   ```

2. **Access the dashboard**
   Navigate to http://localhost:8501

### Option 3: Explore Jupyter Notebooks

Run the notebooks in sequence to understand the full analysis process:
```bash
jupyter notebook notebooks/
```

## � Key Features

### 1. Comprehensive Data Pipeline
- **ETL Process**: Automated data cleaning, transformation, and loading
- **Data Quality**: Validation checks, anomaly detection, and error handling
- **Normalization**: Star schema database design with optimized query performance
- **Feature Engineering**: Derived metrics, time-based aggregations, and customer behavior indicators

### 2. Advanced Analytics
- **Customer Segmentation**: RFM (Recency, Frequency, Monetary) analysis with 7 segments
- **Cohort Analysis**: Monthly retention tracking and visualization
- **Predictive Modeling**: Machine learning-based churn prediction (99.8% accuracy)
- **Time Series Analysis**: Trend, seasonality, and cyclical pattern detection

### 3. Interactive Dashboard
- **Real-time Filtering**: Dynamic analysis by date, region, segment, and category
- **Interactive Visualizations**: Rich Plotly-powered charts with drill-down capabilities
- **KPI Monitoring**: Key business metrics with trend indicators
- **Data Export**: CSV export functionality for filtered results

### 4. Business Intelligence
- **Executive Summary**: Concise overview of key findings and recommendations
- **Strategic Insights**: Data-backed business opportunities with ROI projections
- **At-risk Customer Identification**: Targeted list of high-value customers for retention
- **Performance Comparison**: Regional, category, and temporal benchmarking

## � Key Insights & Business Impact

### Revenue & Profitability
- **Revenue Distribution**: West region leads ($3.2M) while Central region has highest margins (13.8%)
- **Category Performance**: Technology generates highest revenue ($4.7M), but Furniture shows margin improvement opportunities
- **Seasonal Patterns**: Q4 (Nov-Dec) represents 32% of annual revenue, highlighting seasonal opportunities

### Customer Behavior
- **Segment Distribution**: 11.2% Champions and 18.4% Loyal Customers generate 68% of total revenue
- **Retention Challenge**: Significant drop from 78% (Month 1) to 45% (Month 6) retention rate
- **Risk Analysis**: 23.1% of customers are classified as "At-Risk" based on RFM analysis

### Predictive Insights
- **Churn Prediction**: Model identifies high-value customers with 99.8% accuracy before they churn
- **Feature Importance**: Recency (38.3%) is the strongest predictor of customer churn
- **Growth Forecast**: 15.8% year-over-year growth with strategic initiatives

### ROI Projections (12-month implementation)
- **Revenue Opportunity**: 22-28% increase ($2.8M - $3.5M additional)
- **Retention Improvement**: Potential to increase Month 6 retention from 45% to 65%
- **Profit Enhancement**: 2-4 percentage point margin improvement possible

## � Technical Implementation

### Technologies Used
- **Python**: pandas, numpy, scikit-learn, plotly, seaborn, matplotlib
- **Database**: SQLite with optimized schema and indexes
- **Web Application**: Streamlit for interactive dashboard
- **Machine Learning**: Random Forest and Logistic Regression models
- **Containerization**: Docker for easy deployment
- **Version Control**: Git for collaborative development

### Machine Learning Pipeline
- **Feature Engineering**: 16 features derived from customer behavior
- **Model Selection**: Comparison of multiple algorithms with cross-validation
- **Hyperparameter Tuning**: Grid search for optimal model parameters
- **Evaluation**: ROC-AUC, precision, recall, and confusion matrix analysis
- **Deployment**: Pickled models for real-time prediction in dashboard

### Data Architecture
- **Normalized Schema**: Star schema with dimension and fact tables
- **ETL Pipeline**: Automated data processing with validation checks
- **Performance Optimization**: Indexed columns for common query patterns
- **Caching**: Strategic data caching for dashboard performance

## 📝 Strategic Recommendations

### 1. Implement Tiered Customer Retention Program
- VIP treatment for Champions and Loyal customers
- Targeted win-back campaigns for At-Risk customers
- Personalized recommendations based on purchase history

### 2. Optimize Regional Strategy
- Replicate Central region's margin success in other regions
- Address underperformance in specific geographic areas
- Localize marketing based on regional preferences

### 3. Enhance Product Portfolio
- Focus marketing on high-margin Technology products
- Address Furniture category margin challenges
- Develop strategic product bundles to increase order value

### 4. Leverage Seasonal Patterns
- Maximize Q4 peak season with inventory and marketing
- Develop off-season campaigns to smooth revenue curve
- Implement day-of-week optimizations for marketing spend

## 🧪 Testing & Quality Assurance

- **Data Validation**: Comprehensive checks for data integrity and consistency
- **Unit Tests**: Test coverage for key functions in `test_app.py`
- **Model Validation**: Cross-validation and holdout testing for ML models
- **Performance Testing**: Dashboard load time and query optimization

## 🔮 Future Enhancements

1. **Advanced Machine Learning Models**
   - Deep learning for demand forecasting
   - NLP for customer review analysis
   - Recommendation engine for product suggestions

2. **Real-time Analytics**
   - Streaming data pipeline for live updates
   - Alerting system for KPI threshold violations
   - Real-time churn risk monitoring

3. **Extended Functionality**
   - Mobile application for on-the-go analytics
   - API endpoints for system integration
   - PDF report generation for executive summaries

4. **Enhanced Visualizations**
   - Geographic mapping with drill-down capabilities
   - 3D visualization of customer segments
   - Advanced time-series forecasting charts

## 📞 Contact & Support

For questions, improvements, or collaboration:
- Create issues in the GitHub repository
- Fork the project for your own enhancements
- Submit pull requests for new features

---

## 📊 Dashboard Preview

The interactive dashboard provides:
- Real-time filtering by date, region, category, and customer segment
- KPI monitoring with trend indicators
- Interactive charts for revenue, category performance, and customer segments
- Churn prediction tool to assess customer risk
- Data export functionality for further analysis

## 📋 Project Requirements

- Python 3.8+
- 2GB RAM minimum (4GB recommended)
- All dependencies listed in `requirements.txt`

---

*This project demonstrates production-ready data analytics with actionable business insights, combining data science, engineering, and business intelligence in a comprehensive solution.*