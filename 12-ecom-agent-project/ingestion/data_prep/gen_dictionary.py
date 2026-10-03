"""Write DATA_DICTIONARY.md for the NLQ agent from the CLEANED data.

    clean CSVs ──► measured: type, nulls, distinct, allowed values, ranges
    TABLES / NOTES / RULES (below, written) ──► meaning, grain, keys, joins, query rules
                                         └──► DATA_DICTIONARY.md

Measured facts come from the data, so they cannot drift from it; meanings
are written once here.
"""
import sys
from pathlib import Path

import pandas as pd

TABLES = {
    "customers": ("One row per customer account.", "customer_id",
                  ["customer_addresses.customer_id", "orders.customer_id", "reviews.customer_id"]),
    "customer_addresses": ("One row per saved address of a customer (1-2 per customer).", "address_id",
                           ["customers.customer_id"]),
    "orders": ("One row per order. Header only — what was bought is in order_items.", "order_id",
               ["customers.customer_id", "order_items.order_id", "payment_transactions.order_id (1:1)",
                "shipments.order_id (0..1)", "promotions.promo_code = orders.coupon_code"]),
    "order_items": ("One row per product line in an order (1-5 lines per order).", "order_item_id",
                    ["orders.order_id", "products.product_id"]),
    "payment_transactions": ("One row per order's payment — exactly one per order.", "transaction_id",
                             ["orders.order_id"]),
    "shipments": ("One row per shipped order. Cancelled and Processing orders have none.", "shipment_id",
                  ["orders.order_id", "warehouses.warehouse_id"]),
    "products": ("One row per product (1,000).", "product_id",
                 ["categories.category_id", "order_items.product_id", "inventory.product_id",
                  "product_suppliers.product_id", "reviews.product_id"]),
    "categories": ("Category hierarchy, 3 levels. Products sit at level 2 or 3.", "category_id",
                   ["categories.parent_category_id -> categories.category_id", "products.category_id"]),
    "inventory": ("Stock of one product in one warehouse.", "inventory_id",
                  ["products.product_id", "warehouses.warehouse_id"]),
    "warehouses": ("The 5 distribution centres.", "warehouse_id", ["inventory.warehouse_id", "shipments.warehouse_id"]),
    "suppliers": ("One row per supplier (150).", "supplier_id", ["product_suppliers.supplier_id"]),
    "product_suppliers": ("Which supplier can supply which product, at what cost. Exactly one primary per product.",
                          "product_supplier_id", ["products.product_id", "suppliers.supplier_id"]),
    "promotions": ("Promotions. 100 catalogue promotions plus the 10 coupon codes used in orders.", "promotion_id",
                   ["orders.coupon_code = promotions.promo_code"]),
    "reviews": ("One row per product review. Every reviewer bought the product.", "review_id",
                ["products.product_id", "customers.customer_id"]),
}

NOTES = {
    ("orders", "total_amount"): "MERCHANDISE SUBTOTAL = SUM(order_items.total_price). Shipping, tax and discount are NOT included. This is the amount paid (= payment_transactions.amount).",
    ("orders", "discount_amount"): "Coupon discount: coupon rate x total_amount (SAVE10 = 10%). 0 without a coupon. Informational — not subtracted from total_amount.",
    ("orders", "shipping_cost"): "Shipping CHARGED TO THE CUSTOMER (mostly 0). 0 for FREESHIP and Store Pickup. Not included in total_amount. Not the carrier's cost — that is shipments.shipping_cost.",
    ("shipments", "shipping_cost"): "The CARRIER's cost for the shipment (4-25). Not what the customer paid — that is orders.shipping_cost.",
    ("shipments", "order_id"): "Store Pickup orders also have a shipment: the transfer of goods to the store.",
    ("products", "return_rate_pct"): "PERCENT (0-100) of the product's shipped order lines whose order was Returned.",
    ("orders", "tax_amount"): "Tax. Not included in total_amount.",
    ("orders", "coupon_code"): "Empty = no coupon. Joins promotions.promo_code.",
    ("orders", "shipping_address"): "This order's destination. NOT the customer's own address (98% of orders ship to another state than the customer's).",
    ("orders", "shipping_zip"): "5-digit text; keep leading zeros.",
    ("orders", "order_status"): "Delivered, Shipped (in transit), Returned (refunded), Processing (not shipped yet), Cancelled (failed or refunded payment).",
    ("payment_transactions", "status"): "Completed = money kept. Refunded = returned OR cancelled order. Failed = never charged.",
    ("payment_transactions", "amount"): "= orders.total_amount (merchandise subtotal).",
    ("payment_transactions", "card_last_four"): "4-digit text; empty for non-card methods.",
    ("shipments", "actual_delivery"): "Empty while In Transit or Returned.",
    ("shipments", "delivery_days_actual"): "Days from ship_date to actual_delivery; 0 when not delivered — exclude non-delivered shipments from averages.",
    ("shipments", "on_time"): "True only for delivered shipments that arrived by estimated_delivery.",
    ("customers", "created_at"): "Account creation. Never after the customer's first order or address.",
    ("customers", "birth_year"): "Year of birth only (source dates were all 1 January).",
    ("customers", "lifetime_orders"): "= COUNT(orders) for this customer. Prefer computing from orders.",
    ("customers", "lifetime_revenue"): "= SUM(orders.total_amount) for this customer (merchandise only).",
    ("customers", "zip_code"): "5-digit text; keep leading zeros.",
    ("customers", "state"): "US state, DC, or territory (PR, GU, VI, AS, MP, FM, MH, PW).",
    ("customer_addresses", "is_default"): "Exactly one True per customer.",
    ("products", "product_name"): "Adjective + category, e.g. 'Smart Running Shoes'. Not unique — identify products by product_id or sku.",
    ("products", "category"): "Category NAME (level 2 or 3). Prefer category_id for joins.",
    ("products", "stock_quantity"): "= SUM(inventory.quantity_available) across warehouses.",
    ("products", "cost_price"): "= the primary supplier's cost_price.",
    ("products", "margin_pct"): "(price - cost_price) / price x 100.",
    ("products", "avg_rating"): "Mean of reviews.rating for the product.",
    ("products", "review_count"): "= COUNT(reviews) for the product.",
    ("products", "tags"): "Comma-separated tags; use LIKE '%tag%'.",
    ("inventory", "quantity_available"): "Free-to-sell units. Reserved units are NOT included; on hand = available + reserved.",
    ("inventory", "quantity_reserved"): "Units held for open orders, in addition to available.",
    ("promotions", "promo_code"): "Unique. The 10 coupon promotions (promotion_id 12101-12110) match orders.coupon_code.",
    ("promotions", "times_used"): "Coupon promotions: COUNT of orders using the code. Catalogue promotions: the promotion system's own counter — no orders reference them.",
    ("promotions", "status"): "As of 2024-12-31: Expired if ended, Scheduled if not started, else Active. Coupons have no end date.",
    ("promotions", "end_date"): "Empty for the 10 coupons: no end date is known.",
    ("product_suppliers", "is_primary_supplier"): "Exactly one True per product.",
    ("reviews", "sentiment"): "Follows rating exactly: 1-2 Negative, 3 Neutral, 4-5 Positive.",
}

RULES = """## Rules for querying

1. **Revenue** means `SUM(orders.total_amount)` — merchandise only. Say so in the answer. Add shipping and tax only when asked ("including shipping").
2. **Money actually kept** = orders whose payment `status = 'Completed'`. Returned and cancelled orders were refunded; failed payments were never charged.
3. **Sales of a product or category** come from `order_items` (quantity, total_price), joined to `products`; not from orders.
4. **Stock**: `inventory.quantity_available` is free-to-sell; on hand = available + reserved. `products.stock_quantity` is the same total across warehouses.
5. **Shipping cost** means two things: `orders.shipping_cost` is what the customer was charged; `shipments.shipping_cost` is what the carrier cost. Ask which, or say which you used.
6. **Delivery time**: average `delivery_days_actual` over delivered shipments only (`status = 'Delivered'`); it is 0 when not delivered.
7. **Where customers are**: `customers.state` (home). **Where orders went**: `orders.shipping_state`. They differ for 98% of orders.
8. **A customer's default address**: `customer_addresses` with `is_default = True` (exactly one).
9. **Coupons and promotions**: `orders.coupon_code = promotions.promo_code`. Only the 10 coupon promotions have orders.
10. **Products** are identified by `product_id` or `sku`; names repeat (adjective + category).
11. **Category roll-ups** walk `categories.parent_category_id` (3 levels); products sit at level 2 or 3.
12. **Dates** are `YYYY-MM-DD HH:MM:SS` text; data covers 2023-01-01 to 2024-12-30.
13. **ZIP codes** and card digits are text: compare as strings, keep leading zeros.
"""


def enum_or_range(s: pd.Series) -> str:
    s = s.dropna()
    if s.empty:
        return "all empty"
    if pd.api.types.is_bool_dtype(s) or set(s.astype(str)) <= {"True", "False"}:
        return "True / False"
    if pd.api.types.is_numeric_dtype(s):
        return f"{s.min():g} to {s.max():g}"
    if s.nunique() <= 12:
        return " · ".join(f"`{v}`" for v in sorted(s.astype(str).unique()))
    v = s.astype(str)
    return f"e.g. `{v.iloc[0][:40]}`"


def main(folder: str, out: str) -> None:
    lines = ["# E-commerce dataset — data dictionary for the NLQ agent", "",
             "14 tables, US online store, orders from 2023-01-01 to 2024-12-28. Generated from the cleaned data: "
             "types, counts and values are measured, not typed by hand.", "",
             "```",
             "categories ◄── products ──► product_suppliers ──► suppliers",
             "                 │ │ ▲",
             "   inventory ◄───┘ │ └── reviews ◄── customers ──► customer_addresses",
             "       │           │                   │",
             "       ▼           └── order_items ◄── orders ──► payment_transactions (1:1)",
             "   warehouses ◄────────── shipments ◄──┘   │",
             "                                           └── coupon_code ──► promotions.promo_code",
             "```", "", RULES]
    for name, (what, key, joins) in TABLES.items():
        df = pd.read_csv(Path(folder) / f"{name}.csv", dtype={"zip_code": str, "shipping_zip": str, "card_last_four": str}, low_memory=False)
        lines += [f"## {name}", "", f"{what} **{len(df):,} rows.** Primary key: `{key}`.", "",
                  "Joins: " + "; ".join(f"`{j}`" for j in joins), "",
                  "| Column | Type | Empty | Values | Meaning |", "|---|---|---|---|---|"]
        for col in df.columns:
            s = df[col]
            typ = "text" if s.dtype == object or str(s.dtype) == "str" else str(s.dtype).replace("64", "")
            empty = f"{s.isna().mean():.0%}" if s.isna().any() else ""
            lines.append(f"| `{col}` | {typ} | {empty} | {enum_or_range(s)} | {NOTES.get((name, col), '')} |")
        lines.append("")
    Path(out).write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}: {len(TABLES)} tables, {sum(1 for l in lines if l.startswith('| `'))} columns")


if __name__ == "__main__":
    main(*sys.argv[1:3])
