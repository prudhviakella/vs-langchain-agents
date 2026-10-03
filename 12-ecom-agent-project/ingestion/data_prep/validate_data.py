"""Check every data rule the NLQ agent relies on. Independent of fix_data.py.

    cd ingestion
    python data_prep/validate_data.py data/clean     prints PASS/FAIL per rule, exits 1 on any FAIL

    links      every foreign key resolves (14 ID links + category_id + coupon -> promotion)
    time       no order or address predates its customer's account
    one truth  stock, cost, margin, totals, payments, coupons agree across tables
    formats    5-digit ZIP text, 4-digit card text, True/False flags, one timestamp format

Run it on the raw files to see what was wrong; on the clean files every rule passes.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

COUPON_PCT = {"SAVE5": 5, "SAVE10": 10, "WELCOME10": 10, "SAVE15": 15, "SUMMER15": 15,
              "SAVE20": 20, "FLASH20": 20, "SAVE25": 25, "VIP30": 30, "FREESHIP": 0}
AS_OF = pd.Timestamp("2024-12-31")


def main(folder: str) -> int:
    f = Path(folder)
    r = lambda n: pd.read_csv(f / f"{n}.csv", dtype={"zip_code": str, "shipping_zip": str, "card_last_four": str}, low_memory=False)
    cat, addr, cust, inv, items, orders, pay, ps, prod, promo, rev, ship, sup, wh = map(r, [
        "categories", "customer_addresses", "customers", "inventory", "order_items", "orders", "payment_transactions",
        "product_suppliers", "products", "promotions", "reviews", "shipments", "suppliers", "warehouses"])
    results = []

    def rule(name, ok, detail=""):
        results.append(bool(ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not ok else ""))

    print("LINKS")
    for child, col, parent, pcol in [(orders, "customer_id", cust, "customer_id"), (rev, "customer_id", cust, "customer_id"),
                                     (addr, "customer_id", cust, "customer_id"), (items, "order_id", orders, "order_id"),
                                     (items, "product_id", prod, "product_id"), (pay, "order_id", orders, "order_id"),
                                     (ship, "order_id", orders, "order_id"), (ship, "warehouse_id", wh, "warehouse_id"),
                                     (inv, "product_id", prod, "product_id"), (inv, "warehouse_id", wh, "warehouse_id"),
                                     (ps, "product_id", prod, "product_id"), (ps, "supplier_id", sup, "supplier_id"),
                                     (rev, "product_id", prod, "product_id"), (cat, "parent_category_id", cat, "category_id")]:
        missing = (~child[col].dropna().isin(parent[pcol])).sum()
        rule(f"{col} resolves ({len(child):,} rows)", missing == 0, f"{missing} orphans")
    rule("products.category_id resolves", "category_id" in prod and prod.category_id.isin(cat.category_id).all(), "column missing or orphans")
    used = set(orders.coupon_code.dropna())
    rule("every coupon used in orders is a promotion", used <= set(promo.promo_code), f"{len(used - set(promo.promo_code))} of {len(used)} codes unknown")

    print("TIME")
    created = cust.set_index("customer_id").created_at.pipe(pd.to_datetime)
    early = (pd.to_datetime(orders.order_date) < orders.customer_id.map(created)).sum()
    rule("no order before its customer's account", early == 0, f"{early:,} orders")
    early = (pd.to_datetime(addr.created_at) < addr.customer_id.map(created)).sum()
    rule("no address before its customer's account", early == 0, f"{early:,} addresses")

    print("ONE TRUTH")
    flag = addr.is_default.astype(str).isin(["True", "Yes"])
    defaults = flag.groupby(addr.customer_id).sum()
    rule("exactly one default address per customer", (defaults == 1).all(), f"{(defaults != 1).sum():,} customers")
    stock = prod.product_id.map(inv.groupby("product_id").quantity_available.sum())
    rule("stock_quantity = inventory available", (prod.stock_quantity == stock).all(), f"{(prod.stock_quantity != stock).sum()} products")
    primary = ps[ps.is_primary_supplier.astype(str).isin(["True", "Yes"])].set_index("product_id").cost_price
    rule("cost_price = primary supplier's cost", np.isclose(prod.cost_price, prod.product_id.map(primary), atol=0.005).all(),
         f"{(~np.isclose(prod.cost_price, prod.product_id.map(primary), atol=0.005)).sum()} products")
    rule("margin_pct = (price - cost) / price", np.isclose(prod.margin_pct, (prod.price - prod.cost_price) / prod.price * 100, atol=0.06).all())
    rule("product name ends with its category", (prod.product_name.str.split(n=1).str[1] == prod.category).all(),
         f"{(prod.product_name.str.split(n=1).str[1] != prod.category).sum()} products")
    subtotal = orders.order_id.map(items.groupby("order_id").total_price.sum())
    rule("order total = sum of its items", np.isclose(orders.total_amount, subtotal, atol=0.011).all())
    p = pay.merge(orders[["order_id", "total_amount", "order_status"]], on="order_id")
    rule("payment amount = order total", np.isclose(p.amount, p.total_amount, atol=0.011).all())
    bad = (p.order_status.eq("Cancelled") & p.status.eq("Completed")) | (p.order_status.eq("Returned") & ~p.status.eq("Refunded"))
    rule("payment status fits order status", not bad.any(), f"{bad.sum():,} payments")
    rate = orders.coupon_code.map(COUPON_PCT).fillna(0)
    rule("discount = coupon rate x subtotal", np.isclose(orders.discount_amount, (subtotal * rate / 100).round(2), atol=0.011).all(),
         f"{(~np.isclose(orders.discount_amount, (subtotal * rate / 100).round(2), atol=0.011)).sum():,} orders")
    rule("FREESHIP orders ship free", (orders.loc[orders.coupon_code.eq("FREESHIP"), "shipping_cost"] == 0).all())
    rule("Store Pickup orders carry no shipping charge", (orders.loc[orders.shipping_method.eq("Store Pickup"), "shipping_cost"] == 0).all(),
         f"{(orders.loc[orders.shipping_method.eq('Store Pickup'), 'shipping_cost'] != 0).sum()} orders")
    rule("no negative shipping charge", (orders.shipping_cost >= 0).all(), f"{(orders.shipping_cost < 0).sum()} orders")
    li = items.merge(orders[["order_id", "order_status"]], on="order_id")
    li = li[li.order_status.isin(["Delivered", "Shipped", "Returned"])]
    actual = (li.order_status.eq("Returned").groupby(li.product_id).mean() * 100).round(1)
    rule("return_rate_pct = actual returned share (percent)", np.isclose(prod.return_rate_pct, prod.product_id.map(actual), atol=0.051).all(),
         f"{(~np.isclose(prod.return_rate_pct, prod.product_id.map(actual), atol=0.051)).sum()} products")
    rule("promotion codes unique and real", promo.promo_code.is_unique and not promo.promo_code.str.contains("#").any())
    lim = promo.usage_limit.notna()
    rule("times_used within usage_limit", (promo.times_used[lim] <= promo.usage_limit[lim]).all(), f"{(promo.times_used[lim] > promo.usage_limit[lim]).sum()} promotions")
    expect = pd.Series("Active", index=promo.index).mask(pd.to_datetime(promo.end_date) < AS_OF, "Expired").mask(pd.to_datetime(promo.start_date) > AS_OF, "Scheduled")
    rule("promotion status matches its dates", (promo.status == expect).all(), f"{(promo.status != expect).sum()} promotions")

    print("FORMATS")
    for name, frame, col in [("customers", cust, "zip_code"), ("customer_addresses", addr, "zip_code"), ("orders", orders, "shipping_zip"), ("warehouses", wh, "zip_code")]:
        rule(f"{name}.{col} is 5-digit text", frame[col].str.fullmatch(r"\d{5}").all(), f"{(~frame[col].str.fullmatch(r'\d{5}')).sum():,} values")
    cards = pay.card_last_four.dropna()
    rule("card_last_four is 4-digit text", cards.str.fullmatch(r"\d{4}").all(), f"{(~cards.str.fullmatch(r'\d{4}')).sum():,} values")
    for name, frame, col in [("customer_addresses", addr, "is_default"), ("product_suppliers", ps, "is_primary_supplier"), ("reviews", rev, "verified_purchase")]:
        rule(f"{name}.{col} is True/False", set(frame[col].astype(str)) <= {"True", "False"}, f"values {sorted(set(frame[col].astype(str)))}")
    rule("customers carry birth_year, not a fake birth date", "birth_year" in cust and "date_of_birth" not in cust)
    for name, col in [("categories", "parent_category_id"), ("promotions", "usage_limit")]:
        raw = pd.read_csv(f / f"{name}.csv", dtype=str)[col].dropna()
        rule(f"{name}.{col} written as whole numbers", raw.str.fullmatch(r"\d+").all(), f"e.g. {raw.iloc[0]!r}")
    ts = re.compile(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")
    for name, frame, col in [("orders", orders, "order_date"), ("customers", cust, "created_at"), ("promotions", promo, "start_date")]:
        rule(f"{name}.{col} is 'YYYY-MM-DD HH:MM:SS'", frame[col].dropna().astype(str).str.fullmatch(ts).all())

    passed = sum(results)
    print(f"\n{passed} of {len(results)} rules pass")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
