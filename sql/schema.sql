
-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    customer_name TEXT NOT NULL,
    segment TEXT
);

-- Products table
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category TEXT,
    sub_category TEXT
);

-- Orders table
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    order_date DATE,
    ship_date DATE,
    ship_mode TEXT,
    customer_id TEXT,
    segment TEXT,
    country TEXT,
    city TEXT,
    state TEXT,
    postal_code TEXT,
    region TEXT,
    order_year INTEGER,
    order_month INTEGER,
    order_quarter INTEGER,
    order_week INTEGER,
    delivery_days INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

-- Order Items table
CREATE TABLE IF NOT EXISTS order_items (
    order_id TEXT,
    product_id TEXT,
    sales REAL,
    quantity INTEGER,
    discount REAL,
    profit REAL,
    shipping_cost REAL,
    profit_margin REAL,
    PRIMARY KEY (order_id, product_id),
    FOREIGN KEY (order_id) REFERENCES orders(order_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

-- Categories table
CREATE TABLE IF NOT EXISTS categories (
    category TEXT,
    sub_category TEXT,
    PRIMARY KEY (category, sub_category)
);

-- Regions table
CREATE TABLE IF NOT EXISTS regions (
    region TEXT,
    country TEXT,
    state TEXT,
    city TEXT
);
