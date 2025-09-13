-- ============================================
-- SUPERSTORE ANALYTICS SQL QUERIES
-- ============================================

-- 1. MONTHLY REVENUE & ORDERS TREND (Last 24 months)
-- ============================================
SELECT
    order_year,
    order_month,
    COUNT(DISTINCT o.order_id) as total_orders,
    SUM(oi.sales) as monthly_revenue,
    SUM(oi.profit) as monthly_profit,
    AVG(oi.sales) as avg_order_value,
    AVG(oi.profit_margin) as avg_profit_margin
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY order_year, order_month
ORDER BY order_year DESC, order_month DESC
LIMIT 24;

-- 2. YEAR-OVER-YEAR GROWTH
-- ============================================
WITH monthly_metrics AS (
    SELECT
        order_year,
        order_month,
        SUM(oi.sales) as revenue
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY order_year, order_month
),
yoy_comparison AS (
    SELECT
        curr.order_year,
        curr.order_month,
        curr.revenue as current_revenue,
        prev.revenue as previous_year_revenue,
        ((curr.revenue - prev.revenue) / prev.revenue) * 100 as yoy_growth_pct
    FROM monthly_metrics curr
    LEFT JOIN monthly_metrics prev
        ON curr.order_month = prev.order_month
        AND curr.order_year = prev.order_year + 1
)
SELECT * FROM yoy_comparison
WHERE previous_year_revenue IS NOT NULL
ORDER BY order_year DESC, order_month DESC;

-- 3. TOP 10 PRODUCTS BY REVENUE
-- ============================================
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    COUNT(DISTINCT oi.order_id) as orders_count,
    SUM(oi.quantity) as units_sold,
    SUM(oi.sales) as total_revenue,
    SUM(oi.profit) as total_profit,
    AVG(oi.profit_margin) as avg_margin
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category, p.sub_category
ORDER BY total_revenue DESC
LIMIT 10;

-- 4. TOP 10 PRODUCTS BY QUANTITY
-- ============================================
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.sub_category,
    SUM(oi.quantity) as units_sold,
    SUM(oi.sales) as total_revenue,
    AVG(oi.sales / NULLIF(oi.quantity, 0)) as avg_unit_price
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.product_id, p.product_name, p.category, p.sub_category
ORDER BY units_sold DESC
LIMIT 10;

-- 5. CATEGORY AND SUB-CATEGORY PERFORMANCE
-- ============================================
SELECT
    p.category,
    p.sub_category,
    COUNT(DISTINCT oi.order_id) as order_count,
    SUM(oi.quantity) as units_sold,
    SUM(oi.sales) as revenue,
    SUM(oi.profit) as profit,
    AVG(oi.profit_margin) as avg_margin,
    SUM(oi.profit) / NULLIF(SUM(oi.sales), 0) * 100 as profit_margin_pct
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
GROUP BY p.category, p.sub_category
ORDER BY revenue DESC;

-- 6. REGION & STATE PERFORMANCE
-- ============================================
SELECT
    o.region,
    o.state,
    COUNT(DISTINCT o.order_id) as order_count,
    COUNT(DISTINCT o.customer_id) as customer_count,
    SUM(oi.sales) as total_revenue,
    SUM(oi.profit) as total_profit,
    AVG(oi.profit_margin) as avg_margin,
    SUM(oi.sales) / COUNT(DISTINCT o.order_id) as avg_order_value
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
GROUP BY o.region, o.state
ORDER BY total_revenue DESC;

-- 7. CUSTOMER METRICS - AOV, REPEAT RATE
-- ============================================
WITH customer_metrics AS (
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(oi.sales) as total_spent,
        SUM(oi.profit) as total_profit,
        MIN(o.order_date) as first_order_date,
        MAX(o.order_date) as last_order_date,
        julianday(MAX(o.order_date)) - julianday(MIN(o.order_date)) as customer_lifetime_days
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id, c.customer_name, c.segment
)
SELECT
    AVG(total_spent / order_count) as avg_order_value,
    AVG(CASE WHEN order_count > 1 THEN 1 ELSE 0 END) * 100 as repeat_purchase_rate,
    AVG(order_count) as avg_orders_per_customer,
    AVG(total_spent) as avg_customer_lifetime_value,
    COUNT(CASE WHEN order_count = 1 THEN 1 END) as one_time_customers,
    COUNT(CASE WHEN order_count > 1 THEN 1 END) as repeat_customers
FROM customer_metrics;

-- 8. RFM ANALYSIS
-- ============================================
WITH rfm_calc AS (
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        julianday('2018-12-31') - julianday(MAX(o.order_date)) as recency_days,
        COUNT(DISTINCT o.order_id) as frequency,
        SUM(oi.sales) as monetary_value,
        MAX(o.order_date) as last_order_date
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id, c.customer_name, c.segment
),
rfm_scores AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY recency_days DESC) as recency_score,
        NTILE(5) OVER (ORDER BY frequency) as frequency_score,
        NTILE(5) OVER (ORDER BY monetary_value) as monetary_score
    FROM rfm_calc
)
SELECT
    customer_id,
    customer_name,
    segment,
    recency_days,
    frequency,
    monetary_value,
    recency_score,
    frequency_score,
    monetary_score,
    (recency_score + frequency_score + monetary_score) as rfm_total_score,
    CASE
        WHEN recency_score >= 4 AND frequency_score >= 4 AND monetary_score >= 4 THEN 'Champions'
        WHEN recency_score >= 3 AND frequency_score >= 3 AND monetary_score >= 3 THEN 'Loyal Customers'
        WHEN recency_score >= 3 AND frequency_score <= 2 THEN 'Potential Loyalists'
        WHEN recency_score <= 2 AND frequency_score >= 3 THEN 'At Risk'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score >= 3 THEN 'Cant Lose Them'
        WHEN recency_score <= 2 AND frequency_score <= 2 AND monetary_score <= 2 THEN 'Lost'
        ELSE 'Others'
    END as customer_segment
FROM rfm_scores
ORDER BY rfm_total_score DESC;

-- 9. COHORT ANALYSIS - MONTHLY RETENTION
-- ============================================
WITH cohort_data AS (
    SELECT
        c.customer_id,
        DATE(MIN(o.order_date), 'start of month') as cohort_month,
        DATE(o.order_date, 'start of month') as order_month
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    GROUP BY c.customer_id, DATE(o.order_date, 'start of month')
),
cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) as cohort_size
    FROM (
        SELECT
            customer_id,
            MIN(cohort_month) as cohort_month
        FROM cohort_data
        GROUP BY customer_id
    )
    GROUP BY cohort_month
),
retention_data AS (
    SELECT
        cd.cohort_month,
        cd.order_month,
        (julianday(cd.order_month) - julianday(cd.cohort_month)) / 30 as months_since_first,
        COUNT(DISTINCT cd.customer_id) as customers_retained
    FROM cohort_data cd
    GROUP BY cd.cohort_month, cd.order_month
)
SELECT
    rd.cohort_month,
    rd.months_since_first,
    rd.customers_retained,
    cs.cohort_size,
    (rd.customers_retained * 100.0 / cs.cohort_size) as retention_rate
FROM retention_data rd
JOIN cohort_size cs ON rd.cohort_month = cs.cohort_month
WHERE rd.months_since_first <= 12
ORDER BY rd.cohort_month, rd.months_since_first;

-- 10. CHURN ANALYSIS
-- ============================================
WITH customer_activity AS (
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        MAX(o.order_date) as last_order_date,
        julianday('2018-12-31') - julianday(MAX(o.order_date)) as days_since_last_order,
        COUNT(DISTINCT o.order_id) as total_orders,
        SUM(oi.sales) as total_revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id, c.customer_name, c.segment
)
SELECT
    segment,
    COUNT(CASE WHEN days_since_last_order > 90 THEN 1 END) as churned_customers,
    COUNT(*) as total_customers,
    (COUNT(CASE WHEN days_since_last_order > 90 THEN 1 END) * 100.0 / COUNT(*)) as churn_rate,
    AVG(CASE WHEN days_since_last_order > 90 THEN total_revenue END) as avg_churned_customer_value,
    AVG(CASE WHEN days_since_last_order <= 90 THEN total_revenue END) as avg_active_customer_value
FROM customer_activity
GROUP BY segment;

-- 11. CUSTOMER LIFETIME VALUE APPROXIMATION
-- ============================================
WITH customer_ltv AS (
    SELECT
        c.customer_id,
        c.customer_name,
        c.segment,
        COUNT(DISTINCT o.order_id) as purchase_frequency,
        SUM(oi.sales) as total_revenue,
        AVG(oi.sales) as avg_order_value,
        MIN(o.order_date) as first_purchase,
        MAX(o.order_date) as last_purchase,
        (julianday(MAX(o.order_date)) - julianday(MIN(o.order_date))) / 30 as customer_lifespan_months
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY c.customer_id, c.customer_name, c.segment
)
SELECT
    segment,
    AVG(total_revenue) as avg_customer_value,
    AVG(purchase_frequency) as avg_purchase_frequency,
    AVG(customer_lifespan_months) as avg_lifespan_months,
    AVG(total_revenue * (purchase_frequency / NULLIF(customer_lifespan_months, 0))) as estimated_ltv,
    COUNT(*) as customer_count
FROM customer_ltv
WHERE customer_lifespan_months > 0
GROUP BY segment
ORDER BY estimated_ltv DESC;

-- 12. PRODUCT AFFINITY ANALYSIS
-- ============================================
WITH product_pairs AS (
    SELECT
        oi1.product_id as product1,
        oi2.product_id as product2,
        COUNT(DISTINCT oi1.order_id) as co_occurrence_count
    FROM order_items oi1
    JOIN order_items oi2 ON oi1.order_id = oi2.order_id AND oi1.product_id < oi2.product_id
    GROUP BY oi1.product_id, oi2.product_id
    HAVING COUNT(DISTINCT oi1.order_id) > 10
)
SELECT
    p1.product_name as product1_name,
    p2.product_name as product2_name,
    pp.co_occurrence_count,
    p1.category as product1_category,
    p2.category as product2_category
FROM product_pairs pp
JOIN products p1 ON pp.product1 = p1.product_id
JOIN products p2 ON pp.product2 = p2.product_id
ORDER BY pp.co_occurrence_count DESC
LIMIT 20;

-- 13. SEASONAL TRENDS ANALYSIS
-- ============================================
SELECT
    order_month,
    AVG(sales_total) as avg_monthly_revenue,
    AVG(order_count) as avg_monthly_orders,
    AVG(profit_total) as avg_monthly_profit
FROM (
    SELECT
        order_year,
        order_month,
        SUM(oi.sales) as sales_total,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(oi.profit) as profit_total
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    GROUP BY order_year, order_month
) monthly_data
GROUP BY order_month
ORDER BY order_month;

-- 14. DISCOUNT IMPACT ANALYSIS
-- ============================================
SELECT
    CASE
        WHEN discount = 0 THEN 'No Discount'
        WHEN discount > 0 AND discount <= 0.1 THEN '1-10%'
        WHEN discount > 0.1 AND discount <= 0.2 THEN '11-20%'
        WHEN discount > 0.2 AND discount <= 0.3 THEN '21-30%'
        ELSE 'Over 30%'
    END as discount_range,
    COUNT(*) as order_count,
    AVG(sales) as avg_sale_amount,
    AVG(profit) as avg_profit,
    AVG(profit_margin) as avg_profit_margin,
    SUM(sales) as total_revenue,
    SUM(profit) as total_profit
FROM order_items
GROUP BY discount_range
ORDER BY MIN(discount);