-- ════════════════════════════════════════════════════════════════════════
-- E-commerce schema for the NLQ agent — RDS PostgreSQL (13+)
--
--   categories ◄── products ──► product_suppliers ──► suppliers
--                    │ │ ▲
--      inventory ◄───┘ │ └── reviews ◄── customers ──► customer_addresses
--          │           │                   │
--          ▼           └── order_items ◄── orders ──► payment_transactions (1:1)
--      warehouses ◄────────── shipments ◄──┘   │
--                                              └── coupon_code ──► promotions.promo_code
--
-- WHY THESE TYPES
--   money          NUMERIC(12,2)  exact cents; SUM never drifts, ROUND(x, 2) works
--                                 (Postgres has no ROUND(double precision, int))
--   ZIP, card      VARCHAR + CHECK keeps leading zeros; format enforced
--   timestamps     TIMESTAMP (no zone) — the source has no zone information
--
-- WHY FOREIGN KEYS ARE DEFERRABLE INITIALLY DEFERRED
--   load_postgres.py loads all 14 tables in ONE transaction. Deferred keys are
--   checked at COMMIT, so the load order inside it does not matter (categories
--   even references itself) — and a broken reference rolls the whole load back.
--
-- WHAT THIS DOES NOT DO
--   It does not drop anything. Every statement is CREATE … IF NOT EXISTS, so
--   running it twice is safe; changing a column type needs a migration.
-- ════════════════════════════════════════════════════════════════════════

CREATE SCHEMA IF NOT EXISTS ecom;

-- ── reference data ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ecom.categories (
    category_id         INTEGER      PRIMARY KEY,
    category_name       VARCHAR(60)  NOT NULL UNIQUE,
    parent_category_id  INTEGER,
    category_level      SMALLINT     NOT NULL CHECK (category_level BETWEEN 1 AND 3),
    description         TEXT         NOT NULL,
    created_at          TIMESTAMP    NOT NULL,
    CONSTRAINT fk_categories_parent FOREIGN KEY (parent_category_id)
        REFERENCES ecom.categories (category_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.warehouses (
    warehouse_id     INTEGER       PRIMARY KEY,
    warehouse_name   VARCHAR(60)   NOT NULL UNIQUE,
    address          TEXT          NOT NULL,
    city             VARCHAR(60)   NOT NULL,
    state            VARCHAR(2)    NOT NULL,
    zip_code         VARCHAR(5)    NOT NULL CHECK (zip_code ~ '^[0-9]{5}$'),
    country          VARCHAR(3)    NOT NULL,
    phone            VARCHAR(25)   NOT NULL,
    email            VARCHAR(80)   NOT NULL,
    capacity_sqft    INTEGER       NOT NULL CHECK (capacity_sqft > 0),
    manager_name     VARCHAR(60)   NOT NULL,
    operating_since  DATE          NOT NULL,
    lat              NUMERIC(9,6)  NOT NULL,
    lon              NUMERIC(9,6)  NOT NULL
);

CREATE TABLE IF NOT EXISTS ecom.suppliers (
    supplier_id      INTEGER       PRIMARY KEY,
    supplier_name    VARCHAR(80)   NOT NULL,
    contact_person   VARCHAR(60)   NOT NULL,
    email            VARCHAR(80)   NOT NULL,
    phone            VARCHAR(25)   NOT NULL,
    address          TEXT          NOT NULL,
    city             VARCHAR(60)   NOT NULL,
    state            VARCHAR(10)   NOT NULL,
    country          VARCHAR(40)   NOT NULL,
    payment_terms    VARCHAR(20)   NOT NULL,
    rating           NUMERIC(2,1)  NOT NULL CHECK (rating BETWEEN 1 AND 5),
    created_at       TIMESTAMP     NOT NULL
);

CREATE TABLE IF NOT EXISTS ecom.promotions (
    promotion_id         INTEGER        PRIMARY KEY,
    promotion_name       VARCHAR(80)    NOT NULL,
    promotion_type       VARCHAR(20)    NOT NULL
        CHECK (promotion_type IN ('Percentage', 'Fixed Amount', 'Buy One Get One', 'Free Shipping')),
    discount_percent     NUMERIC(5,1),
    discount_amount      NUMERIC(12,2),
    start_date           TIMESTAMP      NOT NULL,
    end_date             TIMESTAMP,                                      -- empty for coupons
    promo_code           VARCHAR(20)    NOT NULL UNIQUE,                 -- orders.coupon_code joins here
    min_purchase_amount  NUMERIC(12,2),
    max_discount         NUMERIC(12,2),
    status               VARCHAR(10)    NOT NULL CHECK (status IN ('Active', 'Expired', 'Scheduled')),
    usage_limit          INTEGER,
    times_used           INTEGER        NOT NULL CHECK (times_used >= 0)
);

-- ── customers ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ecom.customers (
    customer_id               INTEGER        PRIMARY KEY,
    first_name                VARCHAR(60)    NOT NULL,
    last_name                 VARCHAR(60)    NOT NULL,
    email                     VARCHAR(120)   NOT NULL UNIQUE,
    phone                     VARCHAR(25)    NOT NULL,
    address                   TEXT           NOT NULL,
    city                      VARCHAR(60)    NOT NULL,
    state                     VARCHAR(2)     NOT NULL,
    zip_code                  VARCHAR(5)     NOT NULL CHECK (zip_code ~ '^[0-9]{5}$'),
    country                   VARCHAR(3)     NOT NULL,
    created_at                TIMESTAMP      NOT NULL,
    lat                       NUMERIC(9,6)   NOT NULL,
    lon                       NUMERIC(9,6)   NOT NULL,
    gender                    VARCHAR(20)    NOT NULL,
    birth_year                SMALLINT       NOT NULL CHECK (birth_year BETWEEN 1900 AND 2025),
    customer_segment          VARCHAR(20)    NOT NULL,
    acquisition_channel       VARCHAR(30)    NOT NULL,
    loyalty_points            INTEGER        NOT NULL CHECK (loyalty_points >= 0),
    lifetime_orders           INTEGER        NOT NULL CHECK (lifetime_orders >= 0),
    lifetime_revenue          NUMERIC(12,2)  NOT NULL CHECK (lifetime_revenue >= 0),
    last_order_date           TIMESTAMP,                                 -- empty: no orders
    preferred_device          VARCHAR(10)    NOT NULL,
    is_newsletter_subscribed  BOOLEAN        NOT NULL
);

CREATE TABLE IF NOT EXISTS ecom.customer_addresses (
    address_id     INTEGER      PRIMARY KEY,
    customer_id    INTEGER      NOT NULL,
    address_type   VARCHAR(10)  NOT NULL CHECK (address_type IN ('Shipping', 'Billing', 'Both')),
    is_default     BOOLEAN      NOT NULL,
    address_line1  TEXT         NOT NULL,
    address_line2  TEXT,
    city           VARCHAR(60)  NOT NULL,
    state          VARCHAR(2)   NOT NULL,
    zip_code       VARCHAR(5)   NOT NULL CHECK (zip_code ~ '^[0-9]{5}$'),
    country        VARCHAR(3)   NOT NULL,
    phone          VARCHAR(25)  NOT NULL,
    created_at     TIMESTAMP    NOT NULL,
    CONSTRAINT fk_addresses_customer FOREIGN KEY (customer_id)
        REFERENCES ecom.customers (customer_id) DEFERRABLE INITIALLY DEFERRED
);
-- exactly one default address per customer
CREATE UNIQUE INDEX IF NOT EXISTS ux_addresses_one_default
    ON ecom.customer_addresses (customer_id) WHERE is_default;

-- ── catalogue ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ecom.products (
    product_id       INTEGER        PRIMARY KEY,
    product_name     VARCHAR(80)    NOT NULL,                            -- NOT unique
    category_id      INTEGER        NOT NULL,
    category         VARCHAR(60)    NOT NULL,
    brand            VARCHAR(60)    NOT NULL,
    price            NUMERIC(12,2)  NOT NULL CHECK (price > 0),
    stock_quantity   INTEGER        NOT NULL CHECK (stock_quantity >= 0),
    description      TEXT           NOT NULL,
    created_at       TIMESTAMP      NOT NULL,
    sku              VARCHAR(30)    NOT NULL UNIQUE,
    weight_kg        NUMERIC(8,2)   NOT NULL,
    cost_price       NUMERIC(12,2)  NOT NULL CHECK (cost_price >= 0),
    margin_pct       NUMERIC(5,1)   NOT NULL,
    avg_rating       NUMERIC(3,1)   NOT NULL,
    review_count     INTEGER        NOT NULL CHECK (review_count >= 0),
    return_rate_pct  NUMERIC(5,1)   NOT NULL CHECK (return_rate_pct BETWEEN 0 AND 100),
    is_featured      BOOLEAN        NOT NULL,
    is_active        BOOLEAN        NOT NULL,
    tags             TEXT           NOT NULL,
    CONSTRAINT fk_products_category FOREIGN KEY (category_id)
        REFERENCES ecom.categories (category_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.product_suppliers (
    product_supplier_id     INTEGER        PRIMARY KEY,
    product_id              INTEGER        NOT NULL,
    supplier_id             INTEGER        NOT NULL,
    cost_price              NUMERIC(12,2)  NOT NULL CHECK (cost_price >= 0),
    lead_time_days          INTEGER        NOT NULL CHECK (lead_time_days >= 0),
    minimum_order_quantity  INTEGER        NOT NULL CHECK (minimum_order_quantity > 0),
    is_primary_supplier     BOOLEAN        NOT NULL,
    last_order_date         TIMESTAMP      NOT NULL,
    CONSTRAINT uq_product_supplier UNIQUE (product_id, supplier_id),
    CONSTRAINT fk_ps_product  FOREIGN KEY (product_id)  REFERENCES ecom.products (product_id)   DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_ps_supplier FOREIGN KEY (supplier_id) REFERENCES ecom.suppliers (supplier_id) DEFERRABLE INITIALLY DEFERRED
);
-- exactly one primary supplier per product
CREATE UNIQUE INDEX IF NOT EXISTS ux_ps_one_primary
    ON ecom.product_suppliers (product_id) WHERE is_primary_supplier;

CREATE TABLE IF NOT EXISTS ecom.inventory (
    inventory_id        INTEGER      PRIMARY KEY,
    product_id          INTEGER      NOT NULL,
    warehouse_id        INTEGER      NOT NULL,
    quantity_available  INTEGER      NOT NULL CHECK (quantity_available >= 0),   -- free to sell
    quantity_reserved   INTEGER      NOT NULL CHECK (quantity_reserved >= 0),    -- in addition
    reorder_point       INTEGER      NOT NULL CHECK (reorder_point >= 0),
    last_restock_date   TIMESTAMP    NOT NULL,
    last_stock_check    TIMESTAMP    NOT NULL,
    bin_location        VARCHAR(20)  NOT NULL,
    CONSTRAINT uq_inventory_product_warehouse UNIQUE (product_id, warehouse_id),
    CONSTRAINT fk_inventory_product   FOREIGN KEY (product_id)   REFERENCES ecom.products (product_id)     DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_inventory_warehouse FOREIGN KEY (warehouse_id) REFERENCES ecom.warehouses (warehouse_id) DEFERRABLE INITIALLY DEFERRED
);

-- ── orders ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ecom.orders (
    order_id                 INTEGER        PRIMARY KEY,
    customer_id              INTEGER        NOT NULL,
    order_date               TIMESTAMP      NOT NULL,
    total_amount             NUMERIC(12,2)  NOT NULL CHECK (total_amount >= 0),  -- merchandise only
    payment_method           VARCHAR(20)    NOT NULL,
    order_status             VARCHAR(12)    NOT NULL
        CHECK (order_status IN ('Delivered', 'Shipped', 'Returned', 'Processing', 'Cancelled')),
    shipping_address         TEXT           NOT NULL,
    shipping_city            VARCHAR(60)    NOT NULL,
    shipping_state           VARCHAR(2)     NOT NULL,
    shipping_zip             VARCHAR(5)     NOT NULL CHECK (shipping_zip ~ '^[0-9]{5}$'),
    device_type              VARCHAR(10)    NOT NULL,
    shipping_method          VARCHAR(20)    NOT NULL,
    shipping_cost            NUMERIC(12,2)  NOT NULL CHECK (shipping_cost >= 0),
    tax_amount               NUMERIC(12,2)  NOT NULL CHECK (tax_amount >= 0),
    discount_amount          NUMERIC(12,2)  NOT NULL CHECK (discount_amount >= 0),
    coupon_code              VARCHAR(20),
    referral_source          VARCHAR(30)    NOT NULL,
    estimated_delivery_date  DATE           NOT NULL,
    is_gift                  BOOLEAN        NOT NULL,
    CONSTRAINT fk_orders_customer FOREIGN KEY (customer_id)
        REFERENCES ecom.customers (customer_id) DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_orders_coupon FOREIGN KEY (coupon_code)
        REFERENCES ecom.promotions (promo_code) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.order_items (
    order_item_id  INTEGER        PRIMARY KEY,
    order_id       INTEGER        NOT NULL,
    product_id     INTEGER        NOT NULL,
    quantity       INTEGER        NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(12,2)  NOT NULL CHECK (unit_price >= 0),
    total_price    NUMERIC(12,2)  NOT NULL CHECK (total_price >= 0),
    CONSTRAINT fk_items_order   FOREIGN KEY (order_id)   REFERENCES ecom.orders (order_id)     DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_items_product FOREIGN KEY (product_id) REFERENCES ecom.products (product_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.payment_transactions (
    transaction_id      INTEGER        PRIMARY KEY,
    order_id            INTEGER        NOT NULL UNIQUE,                  -- one payment per order
    payment_method      VARCHAR(20)    NOT NULL,
    payment_provider    VARCHAR(20)    NOT NULL,
    transaction_date    TIMESTAMP      NOT NULL,
    amount              NUMERIC(12,2)  NOT NULL CHECK (amount >= 0),
    currency            VARCHAR(3)     NOT NULL,
    status              VARCHAR(10)    NOT NULL CHECK (status IN ('Completed', 'Refunded', 'Failed')),
    card_last_four      VARCHAR(4)     CHECK (card_last_four ~ '^[0-9]{4}$'),
    authorization_code  VARCHAR(20)    NOT NULL,
    processing_fee      NUMERIC(12,2)  NOT NULL CHECK (processing_fee >= 0),
    risk_score          NUMERIC(4,1)   NOT NULL CHECK (risk_score BETWEEN 0 AND 100),
    ip_country          VARCHAR(2)     NOT NULL,
    is_3d_secure        BOOLEAN        NOT NULL,
    chargeback_raised   BOOLEAN        NOT NULL,
    CONSTRAINT fk_payments_order FOREIGN KEY (order_id)
        REFERENCES ecom.orders (order_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.shipments (
    shipment_id           INTEGER        PRIMARY KEY,
    order_id              INTEGER        NOT NULL UNIQUE,                -- one shipment per shipped order
    warehouse_id          INTEGER        NOT NULL,
    carrier               VARCHAR(20)    NOT NULL,
    tracking_number       VARCHAR(40)    NOT NULL UNIQUE,
    ship_date             TIMESTAMP      NOT NULL,
    estimated_delivery    TIMESTAMP      NOT NULL,
    actual_delivery       TIMESTAMP,                                     -- empty until delivered
    shipping_cost         NUMERIC(12,2)  NOT NULL CHECK (shipping_cost >= 0),   -- carrier's cost
    weight_lbs            NUMERIC(8,2)   NOT NULL CHECK (weight_lbs >= 0),
    status                VARCHAR(12)    NOT NULL CHECK (status IN ('Delivered', 'In Transit', 'Returned')),
    origin_lat            NUMERIC(9,6)   NOT NULL,
    origin_lon            NUMERIC(9,6)   NOT NULL,
    destination_lat       NUMERIC(9,6)   NOT NULL,
    destination_lon       NUMERIC(9,6)   NOT NULL,
    delivery_days_actual  INTEGER        NOT NULL CHECK (delivery_days_actual >= 0),
    on_time               BOOLEAN        NOT NULL,
    CONSTRAINT fk_shipments_order     FOREIGN KEY (order_id)     REFERENCES ecom.orders (order_id)         DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_shipments_warehouse FOREIGN KEY (warehouse_id) REFERENCES ecom.warehouses (warehouse_id) DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS ecom.reviews (
    review_id          INTEGER      PRIMARY KEY,
    product_id         INTEGER      NOT NULL,
    customer_id        INTEGER      NOT NULL,
    rating             SMALLINT     NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_title       TEXT         NOT NULL,
    review_text        TEXT         NOT NULL,
    review_date        TIMESTAMP    NOT NULL,
    verified_purchase  BOOLEAN      NOT NULL,
    helpful_votes      INTEGER      NOT NULL CHECK (helpful_votes >= 0),
    sentiment          VARCHAR(10)  NOT NULL CHECK (sentiment IN ('Positive', 'Neutral', 'Negative')),
    seller_responded   BOOLEAN      NOT NULL,
    contains_image     BOOLEAN      NOT NULL,
    CONSTRAINT fk_reviews_product  FOREIGN KEY (product_id)  REFERENCES ecom.products (product_id)   DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_reviews_customer FOREIGN KEY (customer_id) REFERENCES ecom.customers (customer_id) DEFERRABLE INITIALLY DEFERRED
);

-- ── indexes on join and filter columns (primary keys are indexed already) ──
CREATE INDEX IF NOT EXISTS ix_orders_customer        ON ecom.orders (customer_id);
CREATE INDEX IF NOT EXISTS ix_orders_date            ON ecom.orders (order_date);
CREATE INDEX IF NOT EXISTS ix_orders_status          ON ecom.orders (order_status);
CREATE INDEX IF NOT EXISTS ix_orders_coupon          ON ecom.orders (coupon_code) WHERE coupon_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_items_order            ON ecom.order_items (order_id);
CREATE INDEX IF NOT EXISTS ix_items_product          ON ecom.order_items (product_id);
CREATE INDEX IF NOT EXISTS ix_addresses_customer     ON ecom.customer_addresses (customer_id);
CREATE INDEX IF NOT EXISTS ix_products_category      ON ecom.products (category_id);
CREATE INDEX IF NOT EXISTS ix_categories_parent      ON ecom.categories (parent_category_id);
CREATE INDEX IF NOT EXISTS ix_inventory_warehouse    ON ecom.inventory (warehouse_id);
CREATE INDEX IF NOT EXISTS ix_ps_supplier            ON ecom.product_suppliers (supplier_id);
CREATE INDEX IF NOT EXISTS ix_shipments_warehouse    ON ecom.shipments (warehouse_id);
CREATE INDEX IF NOT EXISTS ix_reviews_product        ON ecom.reviews (product_id);
CREATE INDEX IF NOT EXISTS ix_reviews_customer       ON ecom.reviews (customer_id);
