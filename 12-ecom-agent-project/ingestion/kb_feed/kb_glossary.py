"""Glossary and capability card for the e-commerce KB feed.

    GLOSSARY   (term, category, meaning, synonyms, maps_to) — ONE row per term.
               category: metric · status · code · identifier · term
               maps_to: schema elements as  ecom.table.column, semicolon-separated
    CAPABILITY what the NLQ agent can and cannot answer, and the rules it follows
"""

T = "ecom."
GLOSSARY = [
    # ── metrics: how each number is computed ─────────────────────────────
    ("Revenue", "metric", "Merchandise sales: SUM(orders.total_amount) — shipping, tax and discount NOT included. Say 'merchandise revenue' in answers; add shipping/tax only when asked.", "sales; gross sales; GMV; turnover; order value", f"{T}orders.total_amount; {T}order_items.total_price"),
    ("Net Revenue", "metric", "Revenue actually kept: SUM(payment_transactions.amount) where status = 'Completed'. Excludes refunded (returned or cancelled) and failed payments.", "kept revenue; realised revenue; collected revenue", f"{T}payment_transactions.amount; {T}payment_transactions.status"),
    ("Average Order Value", "metric", "AOV = SUM(orders.total_amount) / COUNT(orders.order_id).", "AOV; average basket; average order size; avg order value", f"{T}orders.total_amount; {T}orders.order_id"),
    ("Units Sold", "metric", "SUM(order_items.quantity) — not COUNT(*) of lines.", "units; quantity sold; volume", f"{T}order_items.quantity"),
    ("Gross Margin", "metric", "Per product: margin_pct = (price - cost_price) / price x 100. Margin earned on sales: SUM(order_items.total_price) - SUM(order_items.quantity x products.cost_price).", "margin; profit margin; gross profit", f"{T}products.margin_pct; {T}products.cost_price; {T}order_items.total_price"),
    ("Return Rate", "metric", "Per product: products.return_rate_pct (percent of shipped lines whose order was Returned). Overall: Returned orders / shipped orders (Delivered + Shipped + Returned).", "returns %; return percentage; returns ratio", f"{T}products.return_rate_pct; {T}orders.order_status"),
    ("Refund Rate", "metric", "Refunded payments / all payments. Refunds come from returned AND cancelled orders.", "refund ratio; refunds %", f"{T}payment_transactions.status"),
    ("Payment Failure Rate", "metric", "Failed payments / all payments. Failed payments end as Cancelled orders.", "decline rate; failed payment rate", f"{T}payment_transactions.status"),
    ("Chargeback Rate", "metric", "Payments with chargeback_raised = TRUE / all payments.", "dispute rate; chargebacks %", f"{T}payment_transactions.chargeback_raised"),
    ("On-Time Delivery Rate", "metric", "AVG(CAST(shipments.on_time AS INT)) over status = 'Delivered'.", "OTD; on-time rate; delivered on time %", f"{T}shipments.on_time; {T}shipments.status"),
    ("Delivery Time", "metric", "AVG(shipments.delivery_days_actual) over status = 'Delivered' only — it is 0 when not delivered.", "transit time; days to deliver; delivery days", f"{T}shipments.delivery_days_actual"),
    ("Lifetime Value", "metric", "customers.lifetime_revenue = SUM of the customer's orders.total_amount (merchandise).", "LTV; CLV; customer lifetime value", f"{T}customers.lifetime_revenue"),
    ("Repeat Customer", "metric", "A customer with 2 or more orders (lifetime_orders >= 2).", "returning customer; repeat buyer", f"{T}customers.lifetime_orders; {T}orders.customer_id"),
    ("Stock on Hand", "metric", "inventory.quantity_available + inventory.quantity_reserved. Available alone is free-to-sell.", "on hand; total stock; physical stock", f"{T}inventory.quantity_available; {T}inventory.quantity_reserved"),
    ("Below Reorder Point", "metric", "inventory.quantity_available < inventory.reorder_point — needs replenishing.", "low stock; needs reorder; reorder needed", f"{T}inventory.quantity_available; {T}inventory.reorder_point"),
    ("Coupon Redemptions", "metric", "COUNT of orders with a given coupon_code (= promotions.times_used for coupons).", "coupon uses; promo redemptions", f"{T}orders.coupon_code; {T}promotions.times_used"),
    ("Age", "metric", "Reference year - customers.birth_year (only the year of birth is known).", "customer age; age group", f"{T}customers.birth_year"),
    # ── statuses ─────────────────────────────────────────────────────────
    ("Delivered", "status", "Order or shipment delivered to the customer.", "completed delivery; received", f"{T}orders.order_status; {T}shipments.status"),
    ("Shipped", "status", "Order status for an order in transit (its shipment status is 'In Transit').", "in transit; dispatched; on the way", f"{T}orders.order_status"),
    ("In Transit", "status", "Shipment status for a shipment not yet delivered; actual_delivery empty.", "shipping; on the way", f"{T}shipments.status"),
    ("Returned", "status", "Order sent back; its payment is Refunded and its shipment status is Returned.", "return; sent back", f"{T}orders.order_status; {T}shipments.status"),
    ("Processing", "status", "Order paid and being prepared; not shipped yet, so no shipment row.", "pending; being prepared; not yet shipped", f"{T}orders.order_status"),
    ("Cancelled", "status", "Order cancelled: its payment Failed or was Refunded. No shipment.", "canceled; void; abandoned", f"{T}orders.order_status"),
    ("Completed", "status", "Payment status: money taken and kept.", "paid; successful; captured", f"{T}payment_transactions.status"),
    ("Refunded", "status", "Payment status: money returned — for Returned AND Cancelled orders.", "refund; money back", f"{T}payment_transactions.status"),
    ("Failed", "status", "Payment status: never charged; the order is Cancelled.", "declined; unsuccessful payment", f"{T}payment_transactions.status"),
    ("Active Promotion", "status", "Promotion status Active as of 2024-12-31: started and not ended (coupons have no end date).", "live promotion; current offer", f"{T}promotions.status"),
    ("Expired Promotion", "status", "Promotion status Expired: ended before 2024-12-31.", "ended promotion; past offer", f"{T}promotions.status"),
    ("Scheduled Promotion", "status", "Promotion status Scheduled: starts after 2024-12-31.", "upcoming promotion; future offer", f"{T}promotions.status"),
    # ── codes ────────────────────────────────────────────────────────────
    ("Net 30", "code", "Supplier payment terms: pay the full invoice within 30 days.", "net thirty; 30-day terms", f"{T}suppliers.payment_terms"),
    ("2/10 Net 30", "code", "Supplier payment terms: 2% discount if paid within 10 days, otherwise full amount within 30.", "early payment discount; 2 10 net 30", f"{T}suppliers.payment_terms"),
    ("Net 15", "code", "Supplier payment terms: pay the full invoice within 15 days.", "net fifteen; 15-day terms", f"{T}suppliers.payment_terms"),
    ("Net 45", "code", "Supplier payment terms: pay the full invoice within 45 days.", "net forty-five; 45-day terms", f"{T}suppliers.payment_terms"),
    ("Net 60", "code", "Supplier payment terms: pay the full invoice within 60 days.", "net sixty; 60-day terms", f"{T}suppliers.payment_terms"),
    ("COD", "code", "Cash on delivery: the supplier is paid when goods arrive.", "cash on delivery; pay on delivery", f"{T}suppliers.payment_terms"),
    ("FREESHIP", "code", "Coupon giving free shipping: orders.shipping_cost = 0, no percent discount.", "free shipping coupon; free delivery", f"{T}orders.coupon_code; {T}promotions.promo_code"),
    ("Percentage Coupon", "code", "Coupons SAVE5, SAVE10, WELCOME10, SAVE15, SUMMER15, SAVE20, FLASH20, SAVE25, VIP30: the number is the percent off the merchandise subtotal.", "percent off; percentage discount", f"{T}orders.coupon_code; {T}orders.discount_amount; {T}promotions.discount_percent"),
    ("Store Pickup", "code", "Shipping method: the customer collects in store; no shipping charge. Its shipment row is the transfer to the store.", "click and collect; in-store pickup; BOPIS", f"{T}orders.shipping_method"),
    ("Express", "code", "Shipping method: faster paid delivery.", "expedited; fast shipping", f"{T}orders.shipping_method"),
    ("Overnight", "code", "Shipping method: next-day delivery.", "next day; overnight delivery", f"{T}orders.shipping_method"),
    ("Standard", "code", "Shipping method: regular delivery.", "regular shipping; ground", f"{T}orders.shipping_method"),
    ("VIP", "code", "Customer segment label VIP — 8,083 of 10,000 customers (81%). A label from the CRM, not a computed value tier: for top customers by value, rank lifetime_revenue instead.", "VIP customers; VIP segment", f"{T}customers.customer_segment"),
    ("Never Purchased", "code", "Customer segment of the 56 customers with no orders (lifetime_orders = 0, last_order_date empty).", "no orders; non-buyers; prospects", f"{T}customers.customer_segment; {T}customers.lifetime_orders"),
    ("Churned", "code", "Customer segment label Churned (118 customers) — assigned by the CRM.", "lost customers; churned customers", f"{T}customers.customer_segment"),
    ("At-Risk", "code", "Customer segment label At-Risk (148 customers) — assigned by the CRM.", "at risk; likely to churn", f"{T}customers.customer_segment"),
    ("Customer Segment", "term", "CRM label of a customer: VIP (8,083), Regular (1,363), New (232), At-Risk (148), Churned (118), Never Purchased (56). Labels, not computed tiers.", "tier; segment; customer group", f"{T}customers.customer_segment"),
    ("Acquisition Channel", "term", "How a customer first signed up (Organic Search, Paid Search, Social Media, …).", "signup channel; acquisition source", f"{T}customers.acquisition_channel"),
    ("Referral Source", "term", "Marketing source of an individual order (Google Organic, Facebook, Email, …) — per order, unlike acquisition channel.", "traffic source; order source; channel", f"{T}orders.referral_source"),
    ("Shipping Address", "term", "An order's own destination — not the customer's home address (they differ in 98% of orders).", "delivery address; ship-to", f"{T}orders.shipping_address; {T}orders.shipping_state"),
    ("Home State", "term", "The customer's own state, from customers.state.", "customer state; where customers live", f"{T}customers.state"),
    ("Default Address", "term", "The one saved address with is_default = TRUE.", "primary address; main address", f"{T}customer_addresses.is_default"),
    ("Primary Supplier", "term", "The one supplier per product with is_primary_supplier = TRUE; its cost is products.cost_price.", "main supplier; preferred vendor", f"{T}product_suppliers.is_primary_supplier"),
    ("Lead Time", "term", "Days from purchase order to delivery from a supplier.", "supplier lead time; replenishment time", f"{T}product_suppliers.lead_time_days"),
    ("Minimum Order Quantity", "term", "Fewest units a supplier accepts per purchase order.", "MOQ; minimum order", f"{T}product_suppliers.minimum_order_quantity"),
    ("3-D Secure", "term", "Card authentication step (TRUE in is_3d_secure); lowers fraud risk.", "3DS; strong customer authentication; SCA", f"{T}payment_transactions.is_3d_secure"),
    ("Chargeback", "term", "Cardholder dispute of a payment (chargeback_raised = TRUE).", "dispute; payment dispute", f"{T}payment_transactions.chargeback_raised"),
    ("Risk Score", "term", "Gateway fraud-risk score 0-100; higher is riskier.", "fraud score; risk level", f"{T}payment_transactions.risk_score"),
    ("Sentiment", "term", "Review tone derived from rating: 1-2 Negative, 3 Neutral, 4-5 Positive.", "review sentiment; tone", f"{T}reviews.sentiment; {T}reviews.rating"),
    ("Category Hierarchy", "term", "Categories at 3 levels; products at level 2 or 3; parent_category_id links up a level.", "category tree; department; sub-category", f"{T}categories.parent_category_id; {T}categories.category_level; {T}products.category_id"),
    ("Territory", "term", "Non-state US codes in state columns: DC, PR, GU, VI, AS, MP, FM, MH, PW.", "US territories; Puerto Rico; Guam", f"{T}customers.state; {T}orders.shipping_state; {T}customer_addresses.state"),
    # ── identifiers ──────────────────────────────────────────────────────
    ("SKU", "identifier", "Unique product code like NIK-RUN-1000 (brand-category-number).", "stock keeping unit; item code; product code", f"{T}products.sku"),
    ("Tracking Number", "identifier", "Carrier tracking number of a shipment.", "tracking ID; waybill", f"{T}shipments.tracking_number"),
    ("Authorization Code", "identifier", "Payment processor's approval reference.", "auth code; approval code", f"{T}payment_transactions.authorization_code"),
    ("Bin Location", "identifier", "Aisle-shelf-bin position of stock in a warehouse.", "bin; shelf location", f"{T}inventory.bin_location"),
    ("Order ID", "identifier", "Unique order number 3001-53000.", "order number; order #", f"{T}orders.order_id"),
    ("Customer ID", "identifier", "Unique customer number 1001-11000.", "customer number; account number", f"{T}customers.customer_id"),
]

CAPABILITY = [
    ("playbook", "WHAT THIS AGENT ANSWERS — a US online store, orders from 2023-01-01 to 2024-12-28: 50,000 orders from 10,000 customers; 120,156 order lines; one payment per order; 39,909 shipments from 5 warehouses by 5 carriers; 1,000 products in 46 categories (3-level hierarchy), 126 brands; stock per warehouse; 150 suppliers; 30,000 reviews; 10 coupons used in orders plus 100 catalogue promotions. Answers questions about sales and revenue, products and categories, customers and segments, payments, refunds and chargebacks, fulfilment and delivery, stock and suppliers, coupons, and reviews."),
    ("playbook", "WHAT IT CANNOT ANSWER — orders after 2024-12-28 (shipments run to 2024-12-30 and deliveries to 2025-01-09, nothing later); web traffic, sessions or conversion (there is no visit data); costs other than product cost, shipping and processing fees (no marketing spend, no salaries); exact dates of birth (only the year is kept); which promotion an order used other than through its coupon_code; who the 100 catalogue promotions were used by (no orders reference them)."),
    ("playbook", "RULES — Revenue = SUM(orders.total_amount), merchandise only; say so, and add shipping or tax only when asked. Money kept = payments with status Completed. Product, brand and category sales come from order_items joined to products. Units = SUM(quantity). Delivery time and on-time rate use Delivered shipments only. orders.shipping_cost is what the customer paid; shipments.shipping_cost is the carrier's cost. Customer location = customers.state; order destination = orders.shipping_state. Identify products by product_id or sku, never by name. Compare ZIPs and card digits as text. All timestamps are 'YYYY-MM-DD HH:MM:SS'."),
]
