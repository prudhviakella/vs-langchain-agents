"""NLQ SQL examples for the e-commerce KB feed.

    (question, paraphrases, sql, pattern, scenario, entities)

    pattern   aggregation · join · ranking · window · subquery · signal_threshold · ratio · temporal_compare
    scenario  revenue_analysis · product_performance · customer_analytics · payments_risk · fulfilment ·
              inventory · procurement · promotions · reviews · returns

SQL uses only constructs common to Spark SQL and DuckDB (date_trunc, EXTRACT,
CASE, window functions, CAST(bool AS INT)); every query is executed against the
cleaned data by build_kb_feed.py before it is written to the feed.
"""

E = []
def ex(question, paraphrases, sql, pattern, scenario, entities):
    E.append((question, paraphrases, sql.strip(), pattern, scenario, entities))

# ── revenue_analysis ─────────────────────────────────────────────────────
ex("What is total merchandise revenue by month?",
   ["monthly revenue", "sales per month", "revenue trend by month"],
   """
SELECT date_trunc('month', o.order_date) AS month,
       ROUND(SUM(o.total_amount), 2)      AS merchandise_revenue,
       COUNT(*)                           AS orders
FROM   ecom.orders o
GROUP  BY date_trunc('month', o.order_date)
ORDER  BY month""", "aggregation", "revenue_analysis", "orders; order_date; total_amount")

ex("What was revenue in 2024 compared with 2023?",
   ["year over year revenue", "2024 vs 2023 sales", "annual revenue comparison"],
   """
SELECT EXTRACT(YEAR FROM o.order_date)  AS year,
       ROUND(SUM(o.total_amount), 2)      AS merchandise_revenue,
       COUNT(*)                           AS orders,
       ROUND(SUM(o.total_amount) / COUNT(*), 2) AS avg_order_value
FROM   ecom.orders o
GROUP  BY EXTRACT(YEAR FROM o.order_date)
ORDER  BY year""", "temporal_compare", "revenue_analysis", "orders; order_date; total_amount")

ex("How much revenue did we actually keep after refunds and failed payments?",
   ["net revenue", "revenue kept", "collected revenue excluding refunds"],
   """
SELECT ROUND(SUM(CASE WHEN p.status = 'Completed' THEN p.amount ELSE 0 END), 2) AS kept_revenue,
       ROUND(SUM(CASE WHEN p.status = 'Refunded'  THEN p.amount ELSE 0 END), 2) AS refunded,
       ROUND(SUM(CASE WHEN p.status = 'Failed'    THEN p.amount ELSE 0 END), 2) AS never_charged
FROM   ecom.payment_transactions p""", "aggregation", "revenue_analysis", "payment_transactions; status; amount")

ex("What is the average order value by device type?",
   ["AOV by device", "average basket on mobile vs desktop", "order value per device"],
   """
SELECT o.device_type,
       COUNT(*)                                  AS orders,
       ROUND(SUM(o.total_amount) / COUNT(*), 2)  AS avg_order_value
FROM   ecom.orders o
GROUP  BY o.device_type
ORDER  BY avg_order_value DESC""", "aggregation", "revenue_analysis", "orders; device_type; total_amount")

ex("Which referral sources bring the most revenue?",
   ["revenue by traffic source", "best marketing channel by sales", "top referral sources"],
   """
SELECT o.referral_source,
       COUNT(*)                        AS orders,
       ROUND(SUM(o.total_amount), 2)   AS merchandise_revenue
FROM   ecom.orders o
GROUP  BY o.referral_source
ORDER  BY merchandise_revenue DESC""", "ranking", "revenue_analysis", "orders; referral_source; total_amount")

ex("Which states receive the most order revenue?",
   ["revenue by shipping state", "top destination states", "where do orders ship"],
   """
SELECT o.shipping_state,
       COUNT(*)                       AS orders,
       ROUND(SUM(o.total_amount), 2)  AS merchandise_revenue
FROM   ecom.orders o
GROUP  BY o.shipping_state
ORDER  BY merchandise_revenue DESC
LIMIT  10""", "ranking", "revenue_analysis", "orders; shipping_state; total_amount")

ex("What is total revenue including shipping and tax?",
   ["revenue with shipping and tax", "gross order value including tax", "total billed including shipping"],
   """
SELECT ROUND(SUM(o.total_amount), 2)                                  AS merchandise_revenue,
       ROUND(SUM(o.shipping_cost), 2)                                 AS shipping_charged,
       ROUND(SUM(o.tax_amount), 2)                                    AS tax,
       ROUND(SUM(o.total_amount + o.shipping_cost + o.tax_amount), 2) AS revenue_incl_shipping_and_tax
FROM   ecom.orders o""", "aggregation", "revenue_analysis", "orders; total_amount; shipping_cost; tax_amount")

ex("How did monthly revenue change from the previous month?",
   ["month over month revenue growth", "revenue change versus last month", "MoM sales growth"],
   """
SELECT month, merchandise_revenue,
       ROUND(merchandise_revenue - LAG(merchandise_revenue) OVER (ORDER BY month), 2) AS change_vs_prior_month,
       ROUND(100.0 * (merchandise_revenue - LAG(merchandise_revenue) OVER (ORDER BY month))
             / LAG(merchandise_revenue) OVER (ORDER BY month), 1)                     AS pct_change
FROM  (SELECT date_trunc('month', o.order_date) AS month, ROUND(SUM(o.total_amount), 2) AS merchandise_revenue
       FROM ecom.orders o GROUP BY date_trunc('month', o.order_date)) m
ORDER  BY month""", "window", "revenue_analysis", "orders; order_date; total_amount")

ex("How many gift orders were placed and what is their average value?",
   ["gift orders", "orders marked as gifts", "gift order value"],
   """
SELECT o.is_gift,
       COUNT(*)                                 AS orders,
       ROUND(SUM(o.total_amount) / COUNT(*), 2) AS avg_order_value
FROM   ecom.orders o
GROUP  BY o.is_gift""", "aggregation", "revenue_analysis", "orders; is_gift; total_amount")

# ── product_performance ──────────────────────────────────────────────────
ex("What are the top 10 best-selling products by units?",
   ["best sellers by quantity", "most sold products", "top products by units sold"],
   """
SELECT p.product_id, p.sku, p.product_name, p.brand,
       SUM(oi.quantity)              AS units_sold,
       ROUND(SUM(oi.total_price), 2) AS revenue
FROM   ecom.order_items oi
JOIN   ecom.products p ON p.product_id = oi.product_id
GROUP  BY p.product_id, p.sku, p.product_name, p.brand
ORDER  BY units_sold DESC
LIMIT  10""", "ranking", "product_performance", "order_items; products; quantity; total_price")

ex("What is revenue by top-level category?",
   ["sales by department", "revenue per main category", "category revenue rollup"],
   """
SELECT COALESCE(top2.category_name, top1.category_name, c.category_name) AS top_level_category,
       ROUND(SUM(oi.total_price), 2) AS revenue,
       SUM(oi.quantity)              AS units_sold
FROM   ecom.order_items oi
JOIN   ecom.products   p    ON p.product_id = oi.product_id
JOIN   ecom.categories c    ON c.category_id = p.category_id
LEFT JOIN ecom.categories top1 ON top1.category_id = c.parent_category_id
LEFT JOIN ecom.categories top2 ON top2.category_id = top1.parent_category_id
GROUP  BY COALESCE(top2.category_name, top1.category_name, c.category_name)
ORDER  BY revenue DESC""", "join", "product_performance", "order_items; products; categories; parent_category_id")

ex("Which brands generate the most revenue?",
   ["top brands by sales", "brand revenue ranking", "best performing brands"],
   """
SELECT p.brand,
       ROUND(SUM(oi.total_price), 2) AS revenue,
       SUM(oi.quantity)              AS units_sold,
       COUNT(DISTINCT p.product_id)  AS products
FROM   ecom.order_items oi
JOIN   ecom.products p ON p.product_id = oi.product_id
GROUP  BY p.brand
ORDER  BY revenue DESC
LIMIT  15""", "ranking", "product_performance", "order_items; products; brand")

ex("Which products have the highest gross margin percentage?",
   ["most profitable products", "highest margin items", "products with best margin"],
   """
SELECT p.product_id, p.sku, p.product_name, p.price, p.cost_price, p.margin_pct
FROM   ecom.products p
WHERE  p.is_active = TRUE
ORDER  BY p.margin_pct DESC
LIMIT  10""", "ranking", "product_performance", "products; margin_pct; price; cost_price")

ex("What gross profit did each category earn on its sales?",
   ["gross profit by category", "category profitability", "margin earned per category"],
   """
SELECT p.category,
       ROUND(SUM(oi.total_price), 2)                                     AS revenue,
       ROUND(SUM(oi.quantity * p.cost_price), 2)                         AS cost_of_goods,
       ROUND(SUM(oi.total_price) - SUM(oi.quantity * p.cost_price), 2)   AS gross_profit,
       ROUND(100.0 * (SUM(oi.total_price) - SUM(oi.quantity * p.cost_price)) / SUM(oi.total_price), 1) AS gross_margin_pct
FROM   ecom.order_items oi
JOIN   ecom.products p ON p.product_id = oi.product_id
GROUP  BY p.category
ORDER  BY gross_profit DESC""", "join", "product_performance", "order_items; products; cost_price; total_price")

ex("Which products sold fewer than 120 units in 2024?",
   ["slow selling products 2024", "products with low sales this year", "worst sellers in 2024"],
   """
SELECT p.product_id, p.sku, p.product_name, p.stock_quantity,
       COALESCE(s.units_sold, 0) AS units_sold_2024
FROM   ecom.products p
LEFT JOIN (SELECT oi.product_id, SUM(oi.quantity) AS units_sold
           FROM   ecom.order_items oi
           JOIN   ecom.orders o ON o.order_id = oi.order_id
           WHERE  o.order_date >= CAST('2024-01-01' AS TIMESTAMP)
             AND  o.order_date <  CAST('2025-01-01' AS TIMESTAMP)
           GROUP  BY oi.product_id) s ON s.product_id = p.product_id
WHERE  COALESCE(s.units_sold, 0) < 120
ORDER  BY units_sold_2024""", "subquery", "product_performance", "products; order_items; orders; quantity; order_date")

ex("What is the average selling price by category?",
   ["average price per category", "mean unit price by category", "category price levels"],
   """
SELECT p.category,
       ROUND(SUM(oi.total_price) / SUM(oi.quantity), 2) AS avg_selling_price,
       ROUND(AVG(p.price), 2)                           AS avg_list_price
FROM   ecom.order_items oi
JOIN   ecom.products p ON p.product_id = oi.product_id
GROUP  BY p.category
ORDER  BY avg_selling_price DESC""", "aggregation", "product_performance", "order_items; products; unit_price; price")

ex("Which product is the best seller in each top-level category?",
   ["top product per category", "best seller in every department", "leading product by category"],
   """
SELECT top_level_category, product_id, sku, product_name, units_sold
FROM  (SELECT COALESCE(top2.category_name, top1.category_name) AS top_level_category,
              p.product_id, p.sku, p.product_name, SUM(oi.quantity) AS units_sold,
              ROW_NUMBER() OVER (PARTITION BY COALESCE(top2.category_name, top1.category_name)
                                 ORDER BY SUM(oi.quantity) DESC) AS rn
       FROM   ecom.order_items oi
       JOIN   ecom.products   p    ON p.product_id = oi.product_id
       JOIN   ecom.categories c    ON c.category_id = p.category_id
       LEFT JOIN ecom.categories top1 ON top1.category_id = c.parent_category_id
       LEFT JOIN ecom.categories top2 ON top2.category_id = top1.parent_category_id
       GROUP  BY COALESCE(top2.category_name, top1.category_name), p.product_id, p.sku, p.product_name) ranked
WHERE  rn = 1
ORDER  BY units_sold DESC""", "window", "product_performance", "order_items; products; categories")

ex("How many featured products are there and how do they sell compared with others?",
   ["featured vs non-featured sales", "do featured products sell more", "featured product performance"],
   """
SELECT p.is_featured,
       COUNT(DISTINCT p.product_id)                                          AS products,
       ROUND(SUM(oi.total_price) / COUNT(DISTINCT p.product_id), 2)          AS revenue_per_product
FROM   ecom.products p
JOIN   ecom.order_items oi ON oi.product_id = p.product_id
GROUP  BY p.is_featured""", "aggregation", "product_performance", "products; is_featured; order_items")

# ── customer_analytics ───────────────────────────────────────────────────
ex("How many customers are in each segment and what do they spend?",
   ["customers by segment", "segment revenue", "spend per customer segment"],
   """
SELECT c.customer_segment,
       COUNT(*)                                AS customers,
       ROUND(SUM(c.lifetime_revenue), 2)       AS lifetime_revenue,
       ROUND(AVG(c.lifetime_revenue), 2)       AS avg_lifetime_revenue
FROM   ecom.customers c
GROUP  BY c.customer_segment
ORDER  BY lifetime_revenue DESC""", "aggregation", "customer_analytics", "customers; customer_segment; lifetime_revenue")

ex("Who are the top 10 customers by lifetime revenue?",
   ["best customers", "highest spending customers", "top customers by value"],
   """
SELECT c.customer_id, c.first_name, c.last_name, c.state, c.customer_segment,
       c.lifetime_orders, c.lifetime_revenue
FROM   ecom.customers c
ORDER  BY c.lifetime_revenue DESC
LIMIT  10""", "ranking", "customer_analytics", "customers; lifetime_revenue")

ex("Which acquisition channel brings the most valuable customers?",
   ["customer value by acquisition channel", "best signup channel", "revenue by acquisition source"],
   """
SELECT c.acquisition_channel,
       COUNT(*)                          AS customers,
       ROUND(AVG(c.lifetime_revenue), 2) AS avg_lifetime_revenue,
       ROUND(AVG(c.lifetime_orders), 2)  AS avg_orders
FROM   ecom.customers c
GROUP  BY c.acquisition_channel
ORDER  BY avg_lifetime_revenue DESC""", "aggregation", "customer_analytics", "customers; acquisition_channel; lifetime_revenue")

ex("What share of customers are repeat customers?",
   ["repeat purchase rate", "percentage of returning customers", "customers with more than one order"],
   """
SELECT SUM(CASE WHEN n_orders >= 2 THEN 1 ELSE 0 END)                         AS repeat_customers,
       COUNT(*)                                                              AS customers_with_orders,
       ROUND(100.0 * SUM(CASE WHEN n_orders >= 2 THEN 1 ELSE 0 END) / COUNT(*), 1) AS repeat_rate_pct
FROM  (SELECT o.customer_id, COUNT(*) AS n_orders FROM ecom.orders o GROUP BY o.customer_id) per_customer""",
   "ratio", "customer_analytics", "orders; customer_id")

ex("How many new customer accounts were created each month?",
   ["new signups per month", "customer acquisition trend", "accounts created by month"],
   """
SELECT date_trunc('month', c.created_at) AS month,
       COUNT(*)                          AS new_customers
FROM   ecom.customers c
GROUP  BY date_trunc('month', c.created_at)
ORDER  BY month""", "aggregation", "customer_analytics", "customers; created_at")

ex("Which home states have the most customers?",
   ["customers by state", "where do customers live", "customer count per state"],
   """
SELECT c.state,
       COUNT(*)                          AS customers,
       ROUND(SUM(c.lifetime_revenue), 2) AS lifetime_revenue
FROM   ecom.customers c
GROUP  BY c.state
ORDER  BY customers DESC
LIMIT  10""", "ranking", "customer_analytics", "customers; state")

ex("What is the age distribution of customers who placed orders in 2024?",
   ["customer age groups", "age bands of buyers", "how old are our customers"],
   """
SELECT CASE WHEN 2024 - c.birth_year < 25 THEN 'under 25'
            WHEN 2024 - c.birth_year < 35 THEN '25-34'
            WHEN 2024 - c.birth_year < 45 THEN '35-44'
            WHEN 2024 - c.birth_year < 55 THEN '45-54'
            ELSE '55+' END              AS age_band,
       COUNT(DISTINCT c.customer_id)     AS customers
FROM   ecom.customers c
JOIN   ecom.orders o ON o.customer_id = c.customer_id
WHERE  EXTRACT(YEAR FROM o.order_date) = 2024
GROUP  BY 1
ORDER  BY 1""", "aggregation", "customer_analytics", "customers; birth_year; orders")

ex("Do newsletter subscribers spend more than non-subscribers?",
   ["newsletter subscriber value", "email subscribers vs others", "spend by newsletter status"],
   """
SELECT c.is_newsletter_subscribed,
       COUNT(*)                          AS customers,
       ROUND(AVG(c.lifetime_revenue), 2) AS avg_lifetime_revenue,
       ROUND(AVG(c.lifetime_orders), 2)  AS avg_orders
FROM   ecom.customers c
GROUP  BY c.is_newsletter_subscribed""", "aggregation", "customer_analytics", "customers; is_newsletter_subscribed; lifetime_revenue")

ex("Which customers have not ordered since June 2024?",
   ["lapsed customers", "customers inactive since mid 2024", "customers with no recent orders"],
   """
SELECT c.customer_id, c.first_name, c.last_name, c.customer_segment, c.last_order_date, c.lifetime_revenue
FROM   ecom.customers c
WHERE  c.last_order_date < CAST('2024-06-01' AS TIMESTAMP)
ORDER  BY c.lifetime_revenue DESC
LIMIT  25""", "signal_threshold", "customer_analytics", "customers; last_order_date")

ex("What is each customer's default address?",
   ["customer default address", "primary address of customers", "main address per customer"],
   """
SELECT c.customer_id, c.first_name, c.last_name,
       a.address_type, a.address_line1, a.city, a.state, a.zip_code
FROM   ecom.customers c
JOIN   ecom.customer_addresses a ON a.customer_id = c.customer_id AND a.is_default = TRUE
ORDER  BY c.customer_id
LIMIT  25""", "join", "customer_analytics", "customers; customer_addresses; is_default")

ex("How many orders ship to a state other than the customer's home state?",
   ["orders shipped out of state", "orders to a different state than home", "gift or remote deliveries by state"],
   """
SELECT SUM(CASE WHEN o.shipping_state <> c.state THEN 1 ELSE 0 END)                         AS other_state,
       COUNT(*)                                                                            AS orders,
       ROUND(100.0 * SUM(CASE WHEN o.shipping_state <> c.state THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_other_state
FROM   ecom.orders o
JOIN   ecom.customers c ON c.customer_id = o.customer_id""", "ratio", "customer_analytics", "orders; customers; shipping_state; state")

# ── payments_risk ────────────────────────────────────────────────────────
ex("What share of payments failed, were refunded or completed?",
   ["payment status breakdown", "payment outcomes", "failed payment rate"],
   """
SELECT p.status,
       COUNT(*)                                            AS payments,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)  AS pct
FROM   ecom.payment_transactions p
GROUP  BY p.status
ORDER  BY payments DESC""", "ratio", "payments_risk", "payment_transactions; status")

ex("What is the chargeback rate by payment method?",
   ["disputes by payment method", "chargebacks per payment type", "which payment method has most chargebacks"],
   """
SELECT p.payment_method,
       COUNT(*)                                                                   AS payments,
       SUM(CAST(p.chargeback_raised AS INT))                                      AS chargebacks,
       ROUND(100.0 * SUM(CAST(p.chargeback_raised AS INT)) / COUNT(*), 2)         AS chargeback_rate_pct
FROM   ecom.payment_transactions p
GROUP  BY p.payment_method
ORDER  BY chargeback_rate_pct DESC""", "ratio", "payments_risk", "payment_transactions; chargeback_raised; payment_method")

ex("Which payments have a risk score above 60?",
   ["high risk payments", "risky transactions", "payments with high fraud score"],
   """
SELECT p.transaction_id, p.order_id, p.payment_method, p.amount, p.risk_score, p.ip_country, p.is_3d_secure, p.status
FROM   ecom.payment_transactions p
WHERE  p.risk_score > 60
ORDER  BY p.risk_score DESC""", "signal_threshold", "payments_risk", "payment_transactions; risk_score")

ex("Do payments without 3-D Secure carry more chargebacks?",
   ["3DS vs chargebacks", "chargeback rate with and without 3D secure", "does 3-D Secure reduce disputes"],
   """
SELECT p.is_3d_secure,
       COUNT(*)                                                           AS payments,
       ROUND(100.0 * SUM(CAST(p.chargeback_raised AS INT)) / COUNT(*), 2) AS chargeback_rate_pct,
       ROUND(AVG(p.risk_score), 1)                                        AS avg_risk_score
FROM   ecom.payment_transactions p
WHERE  p.payment_method IN ('Credit Card', 'Debit Card')
GROUP  BY p.is_3d_secure""", "ratio", "payments_risk", "payment_transactions; is_3d_secure; chargeback_raised")

ex("What is the average risk score by IP country?",
   ["risk by country", "payment risk per buyer country", "which countries are riskiest"],
   """
SELECT p.ip_country,
       COUNT(*)                     AS payments,
       ROUND(AVG(p.risk_score), 1)  AS avg_risk_score,
       SUM(CAST(p.chargeback_raised AS INT)) AS chargebacks
FROM   ecom.payment_transactions p
GROUP  BY p.ip_country
ORDER  BY avg_risk_score DESC""", "aggregation", "payments_risk", "payment_transactions; ip_country; risk_score")

ex("What did payment processing cost by provider?",
   ["processing fees by provider", "payment fees per gateway", "cost of payment processing"],
   """
SELECT p.payment_provider,
       COUNT(*)                                       AS payments,
       ROUND(SUM(p.processing_fee), 2)                AS total_fees,
       ROUND(100.0 * SUM(p.processing_fee) / SUM(p.amount), 2) AS fee_pct_of_amount
FROM   ecom.payment_transactions p
WHERE  p.status = 'Completed'
GROUP  BY p.payment_provider
ORDER  BY total_fees DESC""", "aggregation", "payments_risk", "payment_transactions; processing_fee; payment_provider")

ex("How much money was refunded for cancelled orders versus returned orders?",
   ["refunds by reason", "cancelled vs returned refunds", "refund amounts by order status"],
   """
SELECT o.order_status,
       COUNT(*)                       AS refunded_payments,
       ROUND(SUM(p.amount), 2)        AS refunded_amount
FROM   ecom.payment_transactions p
JOIN   ecom.orders o ON o.order_id = p.order_id
WHERE  p.status = 'Refunded'
GROUP  BY o.order_status""", "join", "payments_risk", "payment_transactions; orders; status; order_status")

# ── fulfilment ───────────────────────────────────────────────────────────
ex("What is the on-time delivery rate by carrier?",
   ["carrier on-time performance", "OTD by carrier", "which carrier delivers on time"],
   """
SELECT s.carrier,
       COUNT(*)                                          AS delivered_shipments,
       ROUND(100.0 * AVG(CAST(s.on_time AS INT)), 1)     AS on_time_rate_pct,
       ROUND(AVG(s.delivery_days_actual), 2)             AS avg_delivery_days
FROM   ecom.shipments s
WHERE  s.status = 'Delivered'
GROUP  BY s.carrier
ORDER  BY on_time_rate_pct DESC""", "ratio", "fulfilment", "shipments; carrier; on_time; delivery_days_actual")

ex("What is the average delivery time from each warehouse?",
   ["delivery days by warehouse", "warehouse shipping speed", "transit time per warehouse"],
   """
SELECT w.warehouse_name,
       COUNT(*)                               AS delivered_shipments,
       ROUND(AVG(s.delivery_days_actual), 2)  AS avg_delivery_days
FROM   ecom.shipments s
JOIN   ecom.warehouses w ON w.warehouse_id = s.warehouse_id
WHERE  s.status = 'Delivered'
GROUP  BY w.warehouse_name
ORDER  BY avg_delivery_days""", "join", "fulfilment", "shipments; warehouses; delivery_days_actual")

ex("How many shipments are still in transit by carrier?",
   ["shipments in transit", "undelivered shipments", "open shipments per carrier"],
   """
SELECT s.carrier,
       COUNT(*) AS in_transit
FROM   ecom.shipments s
WHERE  s.status = 'In Transit'
GROUP  BY s.carrier
ORDER  BY in_transit DESC""", "aggregation", "fulfilment", "shipments; status; carrier")

ex("What is the carrier's average cost per shipment and per pound?",
   ["shipping cost per carrier", "carrier cost per pound", "cheapest carrier"],
   """
SELECT s.carrier,
       COUNT(*)                                         AS shipments,
       ROUND(AVG(s.shipping_cost), 2)                   AS avg_carrier_cost,
       ROUND(SUM(s.shipping_cost) / SUM(s.weight_lbs), 3) AS cost_per_lb
FROM   ecom.shipments s
GROUP  BY s.carrier
ORDER  BY avg_carrier_cost""", "aggregation", "fulfilment", "shipments; shipping_cost; weight_lbs; carrier")

ex("How much did customers pay for shipping compared with what carriers cost?",
   ["shipping charged vs carrier cost", "shipping subsidy", "do we lose money on shipping"],
   """
SELECT ROUND(SUM(o.shipping_cost), 2)                    AS charged_to_customers,
       ROUND(SUM(s.shipping_cost), 2)                    AS carrier_cost,
       ROUND(SUM(s.shipping_cost) - SUM(o.shipping_cost), 2) AS shipping_subsidy
FROM   ecom.shipments s
JOIN   ecom.orders o ON o.order_id = s.order_id""", "join", "fulfilment", "orders; shipments; shipping_cost")

ex("Which shipping method do customers choose most and how fast is it?",
   ["shipping method mix", "express vs standard usage", "popular delivery options"],
   """
SELECT o.shipping_method,
       COUNT(*)                                                              AS orders,
       ROUND(AVG(CASE WHEN s.status = 'Delivered' THEN s.delivery_days_actual END), 2) AS avg_delivery_days
FROM   ecom.orders o
LEFT JOIN ecom.shipments s ON s.order_id = o.order_id
GROUP  BY o.shipping_method
ORDER  BY orders DESC""", "join", "fulfilment", "orders; shipments; shipping_method; delivery_days_actual")

ex("How many orders are cancelled or still processing, by month?",
   ["unshipped orders per month", "cancellations by month", "processing backlog"],
   """
SELECT date_trunc('month', o.order_date) AS month,
       SUM(CASE WHEN o.order_status = 'Cancelled'  THEN 1 ELSE 0 END) AS cancelled,
       SUM(CASE WHEN o.order_status = 'Processing' THEN 1 ELSE 0 END) AS processing
FROM   ecom.orders o
GROUP  BY date_trunc('month', o.order_date)
ORDER  BY month""", "aggregation", "fulfilment", "orders; order_status; order_date")

# ── returns ──────────────────────────────────────────────────────────────
ex("Which products have the highest return rate?",
   ["most returned products", "products with most returns", "highest return percentage"],
   """
SELECT p.product_id, p.sku, p.product_name, p.brand, p.return_rate_pct
FROM   ecom.products p
ORDER  BY p.return_rate_pct DESC
LIMIT  10""", "ranking", "returns", "products; return_rate_pct")

ex("What is the return rate by category?",
   ["returns by category", "category return percentage", "which categories get returned most"],
   """
SELECT p.category,
       COUNT(*)                                                                      AS shipped_lines,
       ROUND(100.0 * SUM(CASE WHEN o.order_status = 'Returned' THEN 1 ELSE 0 END) / COUNT(*), 1) AS return_rate_pct
FROM   ecom.order_items oi
JOIN   ecom.orders   o ON o.order_id = oi.order_id
JOIN   ecom.products p ON p.product_id = oi.product_id
WHERE  o.order_status IN ('Delivered', 'Shipped', 'Returned')
GROUP  BY p.category
ORDER  BY return_rate_pct DESC""", "ratio", "returns", "order_items; orders; products; order_status")

# ── inventory ────────────────────────────────────────────────────────────
ex("Which products are below their reorder point in any warehouse?",
   ["low stock items", "products needing reorder", "stock below reorder level"],
   """
SELECT w.warehouse_name, p.product_id, p.sku, p.product_name,
       i.quantity_available, i.reorder_point
FROM   ecom.inventory i
JOIN   ecom.products   p ON p.product_id = i.product_id
JOIN   ecom.warehouses w ON w.warehouse_id = i.warehouse_id
WHERE  i.quantity_available < i.reorder_point
ORDER  BY i.quantity_available""", "signal_threshold", "inventory", "inventory; products; warehouses; quantity_available; reorder_point")

ex("How much stock does each warehouse hold?",
   ["stock by warehouse", "inventory per warehouse", "units on hand by DC"],
   """
SELECT w.warehouse_name,
       COUNT(DISTINCT i.product_id)                       AS products,
       SUM(i.quantity_available)                          AS available_units,
       SUM(i.quantity_reserved)                           AS reserved_units,
       SUM(i.quantity_available + i.quantity_reserved)    AS on_hand_units
FROM   ecom.inventory i
JOIN   ecom.warehouses w ON w.warehouse_id = i.warehouse_id
GROUP  BY w.warehouse_name
ORDER  BY on_hand_units DESC""", "join", "inventory", "inventory; warehouses; quantity_available; quantity_reserved")

ex("What is the inventory value at cost by category?",
   ["stock value by category", "inventory valuation", "value of stock on hand"],
   """
SELECT p.category,
       SUM(i.quantity_available + i.quantity_reserved)                          AS on_hand_units,
       ROUND(SUM((i.quantity_available + i.quantity_reserved) * p.cost_price), 2) AS stock_value_at_cost
FROM   ecom.inventory i
JOIN   ecom.products p ON p.product_id = i.product_id
GROUP  BY p.category
ORDER  BY stock_value_at_cost DESC""", "join", "inventory", "inventory; products; cost_price")

ex("Which products have stock in only one warehouse?",
   ["single warehouse products", "products not spread across warehouses", "stock concentration risk"],
   """
SELECT p.product_id, p.sku, p.product_name, MAX(i.warehouse_id) AS warehouse_id, SUM(i.quantity_available) AS available
FROM   ecom.inventory i
JOIN   ecom.products p ON p.product_id = i.product_id
GROUP  BY p.product_id, p.sku, p.product_name
HAVING COUNT(*) = 1
ORDER  BY available DESC""", "aggregation", "inventory", "inventory; products; warehouse_id")

# ── procurement ──────────────────────────────────────────────────────────
ex("Who is the primary supplier of each product and at what cost?",
   ["primary supplier per product", "main vendor of each item", "product sourcing"],
   """
SELECT p.product_id, p.sku, p.product_name, s.supplier_name, s.country,
       ps.cost_price, ps.lead_time_days, ps.minimum_order_quantity
FROM   ecom.product_suppliers ps
JOIN   ecom.products  p ON p.product_id  = ps.product_id
JOIN   ecom.suppliers s ON s.supplier_id = ps.supplier_id
WHERE  ps.is_primary_supplier = TRUE
ORDER  BY p.product_id
LIMIT  25""", "join", "procurement", "product_suppliers; products; suppliers; is_primary_supplier")

ex("What is the average lead time by supplier country?",
   ["lead time by country", "how long suppliers take by country", "supplier country delivery time"],
   """
SELECT s.country,
       COUNT(DISTINCT s.supplier_id)        AS suppliers,
       ROUND(AVG(ps.lead_time_days), 1)     AS avg_lead_time_days
FROM   ecom.product_suppliers ps
JOIN   ecom.suppliers s ON s.supplier_id = ps.supplier_id
GROUP  BY s.country
ORDER  BY avg_lead_time_days DESC""", "join", "procurement", "product_suppliers; suppliers; lead_time_days; country")

ex("Which products have a cheaper backup supplier than their primary supplier?",
   ["cheaper alternative suppliers", "products where secondary supplier costs less", "sourcing savings"],
   """
SELECT p.product_id, p.sku, p.product_name,
       prim.cost_price                AS primary_cost,
       MIN(alt.cost_price)            AS cheapest_alternative_cost,
       ROUND(prim.cost_price - MIN(alt.cost_price), 2) AS saving_per_unit
FROM   ecom.product_suppliers prim
JOIN   ecom.product_suppliers alt ON alt.product_id = prim.product_id AND alt.is_primary_supplier = FALSE
JOIN   ecom.products p ON p.product_id = prim.product_id
WHERE  prim.is_primary_supplier = TRUE
GROUP  BY p.product_id, p.sku, p.product_name, prim.cost_price
HAVING MIN(alt.cost_price) < prim.cost_price
ORDER  BY saving_per_unit DESC""", "subquery", "procurement", "product_suppliers; products; cost_price; is_primary_supplier")

ex("Which suppliers are rated below 3 and how many products do they supply?",
   ["low rated suppliers", "poor supplier ratings", "suppliers to review"],
   """
SELECT s.supplier_id, s.supplier_name, s.country, s.rating,
       COUNT(ps.product_id) AS products_supplied,
       SUM(CAST(ps.is_primary_supplier AS INT)) AS primary_for
FROM   ecom.suppliers s
LEFT JOIN ecom.product_suppliers ps ON ps.supplier_id = s.supplier_id
WHERE  s.rating < 3
GROUP  BY s.supplier_id, s.supplier_name, s.country, s.rating
ORDER  BY s.rating""", "signal_threshold", "procurement", "suppliers; product_suppliers; rating")

# ── promotions ───────────────────────────────────────────────────────────
ex("Which coupon drives the most revenue?",
   ["coupon performance", "best coupon by sales", "revenue per coupon code"],
   """
SELECT o.coupon_code, pr.promotion_type, pr.discount_percent,
       COUNT(*)                          AS orders,
       ROUND(SUM(o.total_amount), 2)     AS merchandise_revenue,
       ROUND(SUM(o.discount_amount), 2)  AS discount_given
FROM   ecom.orders o
JOIN   ecom.promotions pr ON pr.promo_code = o.coupon_code
GROUP  BY o.coupon_code, pr.promotion_type, pr.discount_percent
ORDER  BY merchandise_revenue DESC""", "join", "promotions", "orders; promotions; coupon_code; promo_code; discount_amount")

ex("Do orders with a coupon have a higher average value than orders without?",
   ["coupon vs no coupon order value", "effect of coupons on basket size", "AOV with and without coupons"],
   """
SELECT CASE WHEN o.coupon_code IS NULL THEN 'no coupon' ELSE 'coupon' END AS coupon_use,
       COUNT(*)                                 AS orders,
       ROUND(SUM(o.total_amount) / COUNT(*), 2) AS avg_order_value
FROM   ecom.orders o
GROUP  BY 1""", "aggregation", "promotions", "orders; coupon_code; total_amount")

ex("Which promotions are currently active?",
   ["active promotions", "live offers", "current promotions"],
   """
SELECT pr.promotion_id, pr.promotion_name, pr.promotion_type, pr.promo_code,
       pr.discount_percent, pr.discount_amount, pr.start_date, pr.end_date
FROM   ecom.promotions pr
WHERE  pr.status = 'Active'
ORDER  BY pr.start_date""", "aggregation", "promotions", "promotions; status")

ex("How much discount did coupons give away in 2024?",
   ["total coupon discount 2024", "discount cost of coupons", "money given away in discounts"],
   """
SELECT o.coupon_code,
       COUNT(*)                          AS orders,
       ROUND(SUM(o.discount_amount), 2)  AS discount_given
FROM   ecom.orders o
WHERE  o.coupon_code IS NOT NULL
  AND  EXTRACT(YEAR FROM o.order_date) = 2024
GROUP  BY o.coupon_code
ORDER  BY discount_given DESC""", "aggregation", "promotions", "orders; coupon_code; discount_amount; order_date")

# ── reviews ──────────────────────────────────────────────────────────────
ex("Which products with at least 20 reviews have the lowest average rating?",
   ["worst rated products", "lowest rated items", "products customers dislike"],
   """
SELECT p.product_id, p.sku, p.product_name, p.brand,
       COUNT(*)                 AS reviews,
       ROUND(AVG(r.rating), 2)  AS avg_rating
FROM   ecom.reviews r
JOIN   ecom.products p ON p.product_id = r.product_id
GROUP  BY p.product_id, p.sku, p.product_name, p.brand
HAVING COUNT(*) >= 20
ORDER  BY avg_rating
LIMIT  10""", "signal_threshold", "reviews", "reviews; products; rating")

ex("What share of reviews are negative, by category?",
   ["negative reviews by category", "review sentiment per category", "categories with bad reviews"],
   """
SELECT p.category,
       COUNT(*)                                                                         AS reviews,
       ROUND(100.0 * SUM(CASE WHEN r.sentiment = 'Negative' THEN 1 ELSE 0 END) / COUNT(*), 1) AS negative_pct
FROM   ecom.reviews r
JOIN   ecom.products p ON p.product_id = r.product_id
GROUP  BY p.category
ORDER  BY negative_pct DESC""", "ratio", "reviews", "reviews; products; sentiment")

ex("Do reviews with images get more helpful votes?",
   ["photo reviews helpfulness", "image reviews vs text reviews", "helpful votes by image"],
   """
SELECT r.contains_image,
       COUNT(*)                        AS reviews,
       ROUND(AVG(r.helpful_votes), 1)  AS avg_helpful_votes,
       ROUND(AVG(r.rating), 2)         AS avg_rating
FROM   ecom.reviews r
GROUP  BY r.contains_image""", "aggregation", "reviews", "reviews; contains_image; helpful_votes")

ex("How often does the seller respond to negative reviews?",
   ["seller response rate", "replies to bad reviews", "seller engagement on negative reviews"],
   """
SELECT r.sentiment,
       COUNT(*)                                                    AS reviews,
       ROUND(100.0 * AVG(CAST(r.seller_responded AS INT)), 1)      AS response_rate_pct
FROM   ecom.reviews r
GROUP  BY r.sentiment
ORDER  BY response_rate_pct DESC""", "ratio", "reviews", "reviews; seller_responded; sentiment")
