"""Fix the e-commerce dataset so an NLQ agent can query it without contradictions.

    raw CSVs (15 files) ──► load ──► STEP 1..12 fixes ──► validate ──► clean CSVs (14 files)
                                                                  └──► changes/ (what changed, row by row)

    cd ingestion
    python data_prep/fix_data.py data/raw data/clean

WHY EACH FIX PICKS A "TRUTH"

When two tables disagree, one must win. The rule used throughout: the more
DETAILED or more PRIMARY record wins over a stored summary of it.

    stock            inventory (per warehouse)   beats   products.stock_quantity
    cost             primary supplier's cost      beats   products.cost_price
    account date     first real activity          beats   customers.created_at
    payment status   order status                 beats   payment status
    discount         the coupon code's own rate   beats   a random stored discount
    product name     the product's category       beats   a random adjective + noun

WHAT THIS DOES NOT DO

    - It does not change any amount a customer paid: orders.total_amount and
      payment_transactions.amount are untouched (they agree with each other
      and with order_items to the cent).
    - It does not invent order-level history: the 100 catalogue promotions
      keep their own times_used counter; no orders are attached to them.
    - It does not move shipping addresses to customers' addresses: an order's
      shipping address is its own destination (gifts, work, travel).
    - It does not touch reviews, shipments or order_items: they were consistent.
"""
import re
import sys
from pathlib import Path

import pandas as pd

AS_OF = pd.Timestamp("2024-12-31")          # the dataset's last day; statuses are judged at this date
FILES = ["categories", "customer_addresses", "customers", "inventory", "order_items", "orders",
         "payment_transactions", "product_suppliers", "products", "promotions", "reviews",
         "shipments", "suppliers", "warehouses"]

# The coupon codes used in orders, and what each one should do.
# "pct" = percent off the merchandise subtotal; "ship" = free shipping.
COUPONS = {"SAVE5": ("pct", 5), "SAVE10": ("pct", 10), "WELCOME10": ("pct", 10), "SAVE15": ("pct", 15),
           "SUMMER15": ("pct", 15), "SAVE20": ("pct", 20), "FLASH20": ("pct", 20), "SAVE25": ("pct", 25),
           "VIP30": ("pct", 30), "FREESHIP": ("ship", 0)}


def load(raw: Path) -> dict:
    # dtype=str for codes whose leading zeros must survive (ZIPs, card digits)
    text_cols = {"zip_code": str, "shipping_zip": str, "card_last_four": str}
    return {name: pd.read_csv(raw / f"{name}.csv", dtype=text_cols, low_memory=False) for name in FILES}


def fix(d: dict) -> tuple[dict, dict]:
    """Every fix, in order. Returns the fixed tables and a change log per fix."""
    log = {}
    orders, items, cust, addr = d["orders"], d["order_items"], d["customers"], d["customer_addresses"]
    for frame, cols in [(orders, ["order_date"]), (cust, ["created_at", "last_order_date"]), (addr, ["created_at"])]:
        for c in cols:
            frame[c] = pd.to_datetime(frame[c])

    # STEP 1 — account creation date: no order or address may predate its account.
    #          The account existed from its earliest real activity.
    first_order = orders.groupby("customer_id").order_date.min()
    first_addr = addr.groupby("customer_id").created_at.min()
    earliest = pd.concat([cust.set_index("customer_id").created_at, first_order, first_addr], axis=1).min(axis=1)
    new_created = cust.customer_id.map(earliest)
    changed = cust.created_at != new_created
    log["customers.created_at moved to first activity"] = cust.loc[changed, ["customer_id", "created_at"]].assign(new=new_created[changed])
    cust["created_at"] = new_created

    # STEP 2 — exactly one default address per customer, changing as little as possible:
    #          a customer with exactly one default keeps it; with two, the preferred of
    #          the two stays; with none, the preferred address becomes the default.
    #          Preference: an address used for Both, then Shipping, then the oldest.
    rank = addr.address_type.map({"Both": 0, "Shipping": 1, "Billing": 2})
    not_default = ~addr.is_default.eq("Yes")
    pick = (addr.assign(rank=rank, not_default=not_default)
                .sort_values(["customer_id", "not_default", "rank", "created_at"])
                .groupby("customer_id").address_id.first())
    new_default = addr.address_id.isin(pick)
    old_default = addr.is_default.eq("Yes")
    log["customer_addresses.is_default reassigned"] = addr.loc[new_default != old_default, ["address_id", "customer_id", "address_type", "is_default"]]
    addr["is_default"] = new_default

    # STEP 3 — product stock = the warehouses' total AVAILABLE stock.
    #          (quantity_available is free-to-sell stock; reserved is held separately.)
    prod, inv = d["products"], d["inventory"]
    stock = inv.groupby("product_id").quantity_available.sum()
    new_stock = prod.product_id.map(stock).fillna(0).astype(int)
    log["products.stock_quantity set from inventory"] = prod.loc[prod.stock_quantity != new_stock, ["product_id", "stock_quantity"]].assign(new=new_stock)
    prod["stock_quantity"] = new_stock

    # STEP 4 — product cost = the primary supplier's cost; margin recomputed from it.
    ps = d["product_suppliers"]
    primary = ps[ps.is_primary_supplier.eq("Yes")].set_index("product_id").cost_price
    new_cost = prod.product_id.map(primary)
    log["products.cost_price set from primary supplier"] = prod.loc[(prod.cost_price - new_cost).abs() > 0.005, ["product_id", "cost_price"]].assign(new=new_cost)
    prod["cost_price"] = new_cost.round(2)
    prod["margin_pct"] = ((prod.price - prod.cost_price) / prod.price * 100).round(1)

    # STEP 4b — product return rate = actual returns, as a PERCENT (like margin_pct).
    #           Share of the product's shipped order lines whose order was returned.
    #           The stored value was a fraction (0.25 = 25%) unrelated to the orders.
    lines = items.merge(orders[["order_id", "order_status"]], on="order_id")
    shipped = lines[lines.order_status.isin(["Delivered", "Shipped", "Returned"])]
    rate = shipped.order_status.eq("Returned").groupby(shipped.product_id).mean() * 100
    new_rate = prod.product_id.map(rate).fillna(0).round(1)
    log["products.return_rate_pct set from actual returns (percent)"] = prod[["product_id", "return_rate_pct"]].assign(new=new_rate)
    prod["return_rate_pct"] = new_rate

    # STEP 5 — product names and descriptions follow the category.
    #          New name: the original adjective + the category ("Smart Running Shoes").
    #          Descriptions are REGENERATED for every product from attributes known
    #          to be right — adjective, category, brand, the category's path:
    #            "Smart running shoes from Nike — Clothing & Apparel › Shoes & Footwear › Running Shoes."
    #          The originals were written to justify the random names ("makeup artist
    #          backpack", "Kids' Clothing … soft LED lighting for bedtime"); no rule
    #          reliably tells the coherent ones from the rest. They are kept, row by
    #          row, in changes/ — nothing is lost.
    cats = d["categories"].set_index("category_id")
    by_name = d["categories"].set_index("category_name").category_id

    def path(name: str) -> str:
        names, cid = [], by_name[name]
        while pd.notna(cid):
            names.append(cats.loc[int(cid), "category_name"])
            cid = cats.loc[int(cid), "parent_category_id"]
        return " › ".join(reversed(names))

    adjective = prod.product_name.str.split().str[0]
    new_name = adjective + " " + prod.category
    new_desc = [f"{a} {c.lower()} from {b} — {path(c)}." for a, c, b in zip(adjective, prod.category, prod.brand)]
    log["products renamed and described (old -> new)"] = pd.DataFrame({
        "product_id": prod.product_id, "old_name": prod.product_name, "new_name": new_name,
        "old_description": prod.description, "new_description": new_desc})
    prod["product_name"], prod["description"] = new_name, new_desc

    # STEP 6 — products carry category_id, not only the category's name.
    cat_id = d["categories"].set_index("category_name").category_id
    prod.insert(prod.columns.get_loc("category"), "category_id", prod.category.map(cat_id).astype(int))

    # STEP 7 — a cancelled order's completed payment was refunded.
    pay = d["payment_transactions"]
    status = pay.order_id.map(orders.set_index("order_id").order_status)
    wrong = pay.status.eq("Completed") & status.eq("Cancelled")
    log["payments of cancelled orders marked Refunded"] = pay.loc[wrong, ["transaction_id", "order_id", "status"]]
    pay.loc[wrong, "status"] = "Refunded"

    # STEP 8 — discounts follow the coupon code; FREESHIP makes shipping free.
    #          Orders without a coupon keep a 0 discount. total_amount is NOT
    #          changed: it is the merchandise subtotal that was paid.
    subtotal = orders.order_id.map(items.groupby("order_id").total_price.sum())
    kind = orders.coupon_code.map(lambda c: COUPONS[c][0] if c in COUPONS else None)
    rate = orders.coupon_code.map(lambda c: COUPONS[c][1] if c in COUPONS else 0)
    new_disc = (subtotal * rate / 100).round(2).where(kind.eq("pct"), 0.0)
    new_ship = orders.shipping_cost.where(~kind.eq("ship"), 0.0)
    log["orders.discount_amount / shipping_cost recomputed from coupon"] = orders.loc[
        (orders.discount_amount - new_disc).abs().gt(0.005) | (orders.shipping_cost != new_ship),
        ["order_id", "coupon_code", "discount_amount", "shipping_cost"]].assign(new_discount=new_disc, new_shipping=new_ship)
    orders["discount_amount"], orders["shipping_cost"] = new_disc, new_ship
    #          Store Pickup carries no shipping charge (some were negative, down to -0.98).
    pickup = orders.shipping_method.eq("Store Pickup") & orders.shipping_cost.ne(0)
    log["orders.shipping_cost of Store Pickup set to 0"] = orders.loc[pickup, ["order_id", "shipping_cost"]]
    orders.loc[pickup, "shipping_cost"] = 0.0

    # STEP 9 — promotions: the 10 real coupons become promotions, linked by
    #          promo_code = orders.coupon_code, with usage counted from orders.
    promo = d["promotions"]
    for c in ("start_date", "end_date"):
        promo[c] = pd.to_datetime(promo[c])
    used = orders.dropna(subset=["coupon_code"]).groupby("coupon_code").order_date.agg(["min", "max", "size"])
    rows = []
    for i, (code, (kind_, value)) in enumerate(sorted(COUPONS.items()), start=1):
        rows.append({"promotion_id": int(promo.promotion_id.max()) + i, "promotion_name": f"Coupon {code}",
                     "promotion_type": "Percentage" if kind_ == "pct" else "Free Shipping",
                     "discount_percent": float(value) if kind_ == "pct" else None, "discount_amount": None,
                     # start = first use; end unknown (last use is not an end date) -> left empty
                     "start_date": used.loc[code, "min"], "end_date": pd.NaT, "promo_code": code,
                     "min_purchase_amount": None, "max_discount": None, "status": None,
                     "usage_limit": None, "times_used": int(used.loc[code, "size"])})
    # STEP 10 — the catalogue promotions: real codes, limits respected, status by date.
    #           "ATCG##" -> "ATCG01" (the "##" placeholder becomes the promotion's number).
    promo["promo_code"] = [re.sub(r"#+$", f"{pid % 100:02d}", code) for code, pid in zip(promo.promo_code, promo.promotion_id)]
    over = promo.usage_limit.notna() & (promo.times_used > promo.usage_limit)
    log["promotions.times_used capped at usage_limit"] = promo.loc[over, ["promotion_id", "times_used", "usage_limit"]]
    promo.loc[over, "times_used"] = promo.loc[over, "usage_limit"].astype(int)
    promo = pd.concat([promo, pd.DataFrame(rows)], ignore_index=True)
    new_status = pd.Series("Active", index=promo.index).mask(promo.end_date < AS_OF, "Expired").mask(promo.start_date > AS_OF, "Scheduled")
    log["promotions.status recomputed as of 2024-12-31"] = promo.loc[promo.status.notna() & (promo.status != new_status), ["promotion_id", "status"]].assign(new=new_status)
    promo["status"] = new_status
    d["promotions"] = promo

    # STEP 11 — formats that lose information.
    #           ZIPs: 5-digit text ("7102" -> "07102"). Card digits: 4-digit text.
    for name, col in [("customers", "zip_code"), ("customer_addresses", "zip_code"), ("orders", "shipping_zip"), ("warehouses", "zip_code")]:
        d[name][col] = d[name][col].str.zfill(5)
    pay["card_last_four"] = pay.card_last_four.str.replace(r"\.0$", "", regex=True).str.zfill(4).where(pay.card_last_four.notna())
    #           Dates of birth are all 1 January: only the year is real.
    cust.insert(cust.columns.get_loc("date_of_birth"), "birth_year", pd.to_datetime(cust.date_of_birth).dt.year)
    cust.drop(columns="date_of_birth", inplace=True)
    #           Yes/No text flags -> True/False, like every other flag in the data.
    for name, col in [("product_suppliers", "is_primary_supplier"), ("reviews", "verified_purchase")]:
        d[name][col] = d[name][col].eq("Yes")

    # STEP 12a — IDs and counts with blanks stay integers. pandas stores a number
    #            column with blanks as float, so "39" would be written "39.0" —
    #            which an INTEGER column in Postgres rejects on load.
    for name, col in [("categories", "parent_category_id"), ("promotions", "usage_limit")]:
        d[name][col] = pd.to_numeric(d[name][col]).round().astype("Int64")

    # STEP 12 — write timestamps back in one format: "YYYY-MM-DD HH:MM:SS".
    for frame in d.values():
        for c in frame.columns:
            if pd.api.types.is_datetime64_any_dtype(frame[c]):
                frame[c] = frame[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(frame[c].notna())
    return d, log


def main(raw: str, out: str) -> None:
    raw, out = Path(raw), Path(out)
    (out / "changes").mkdir(parents=True, exist_ok=True)
    tables, log = fix(load(raw))
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    summary = []
    for i, (what, rows) in enumerate(log.items(), start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", what.lower()).strip("_")[:60]
        rows.to_csv(out / "changes" / f"{i:02d}_{slug}.csv", index=False)
        summary.append(f"{len(rows):>7,}  {what}")
    print("\n".join(summary))


if __name__ == "__main__":
    main(*sys.argv[1:3])
