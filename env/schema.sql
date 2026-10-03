-- Dokan shop schema. Timestamps are ISO 8601 strings with the +06:00 offset.

PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT NOT NULL UNIQUE,
    district    TEXT NOT NULL,
    is_dhaka    INTEGER NOT NULL CHECK (is_dhaka IN (0, 1)),
    tier        TEXT NOT NULL CHECK (tier IN ('regular', 'silver', 'gold'))
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    category   TEXT NOT NULL CHECK (
                   category IN ('clothing', 'electronics_accessories',
                                'home_goods', 'cosmetics')
               ),
    price_bdt  INTEGER NOT NULL CHECK (price_bdt > 0),
    stock      INTEGER NOT NULL CHECK (stock >= 0)
);

CREATE TABLE orders (
    order_id            TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers (customer_id),
    status              TEXT NOT NULL CHECK (
                            status IN ('pending', 'confirmed', 'shipped',
                                       'delivered', 'cancelled', 'returned')
                        ),
    payment_method      TEXT NOT NULL CHECK (
                            payment_method IN ('cod', 'bkash', 'nagad', 'card')
                        ),
    district            TEXT NOT NULL,
    address             TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    shipped_at          TEXT,
    delivered_at        TEXT,
    total_bdt           INTEGER NOT NULL CHECK (total_bdt >= 0),
    delivery_charge_bdt INTEGER NOT NULL CHECK (delivery_charge_bdt >= 0)
);

CREATE TABLE order_items (
    order_id       TEXT NOT NULL REFERENCES orders (order_id),
    product_id     TEXT NOT NULL REFERENCES products (product_id),
    quantity       INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_bdt INTEGER NOT NULL CHECK (unit_price_bdt > 0),
    PRIMARY KEY (order_id, product_id)
);

CREATE TABLE refunds (
    refund_id  TEXT PRIMARY KEY,
    order_id   TEXT NOT NULL REFERENCES orders (order_id),
    amount_bdt INTEGER NOT NULL,
    method     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE tickets (
    ticket_id  TEXT PRIMARY KEY,
    order_id   TEXT REFERENCES orders (order_id),
    department TEXT NOT NULL CHECK (
                   department IN ('logistics', 'billing',
                                  'product_quality', 'general')
               ),
    priority   TEXT NOT NULL CHECK (priority IN ('low', 'normal', 'high')),
    summary    TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE quotes (
    quote_id            TEXT PRIMARY KEY,
    customer_phone      TEXT NOT NULL,
    district            TEXT NOT NULL,
    subtotal_bdt        INTEGER NOT NULL,
    discount_percent    REAL NOT NULL,
    delivery_charge_bdt INTEGER NOT NULL,
    total_bdt           INTEGER NOT NULL,
    created_at          TEXT NOT NULL
);

CREATE TABLE quote_items (
    quote_id   TEXT NOT NULL REFERENCES quotes (quote_id),
    product_id TEXT NOT NULL REFERENCES products (product_id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    PRIMARY KEY (quote_id, product_id)
);

CREATE TABLE coupons (
    code          TEXT PRIMARY KEY,
    percent_off   REAL NOT NULL,
    valid_until   TEXT NOT NULL,
    min_order_bdt INTEGER NOT NULL
);

CREATE TABLE coupon_uses (
    order_id TEXT NOT NULL REFERENCES orders (order_id),
    code     TEXT NOT NULL REFERENCES coupons (code),
    PRIMARY KEY (order_id, code)
);
