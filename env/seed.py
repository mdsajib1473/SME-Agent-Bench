"""Deterministic builder for the Dokan seed database.

Running this module writes env/shop_seed.db. Every value derives from the fixed
seed and the fixed clock, so repeated builds are byte-comparable in content.
"""

import random
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEED = 42
DHAKA_TZ = timezone(timedelta(hours=6))
FIXED_CLOCK = datetime(2026, 10, 1, 10, 0, 0, tzinfo=DHAKA_TZ)

ENV_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = ENV_DIR / "schema.sql"
DEFAULT_DB_PATH = ENV_DIR / "shop_seed.db"

DHAKA_COUNT = 20

NAMES = [
    "Rafiqul Islam", "Nasrin Akter", "Mahmudul Hasan", "Shahana Begum",
    "Tanvir Ahmed", "Farhana Yasmin", "Jahidul Karim", "Ruma Khatun",
    "Imran Hossain", "Sabrina Sultana", "Kamrul Hasan", "Taslima Akter",
    "Shafiqur Rahman", "Nusrat Jahan", "Anwar Hossain", "Rehana Parvin",
    "Masud Rana", "Shirin Akter", "Habibur Rahman", "Momtaz Begum",
    "Saiful Islam", "Rokeya Khatun", "Delwar Hossain", "Afsana Khanam",
    "Ashraful Alam", "Jesmin Ara", "Mizanur Rahman", "Sultana Parvin",
    "Golam Mostafa", "Hasina Akter", "Nazrul Islam", "Shamima Nasrin",
    "Abdul Hakim", "Rabeya Khatun", "Monirul Islam", "Fatema Begum",
    "Zahirul Haque", "Salma Khatun", "Badrul Alam", "Marium Akter",
]

OUTSIDE_DISTRICTS = [
    "Chattogram", "Sylhet", "Rajshahi", "Khulna", "Barishal", "Rangpur",
    "Mymensingh", "Cumilla", "Bogura", "Jashore", "Dinajpur", "Narayanganj",
    "Gazipur", "Tangail", "Faridpur", "Kushtia", "Pabna", "Noakhali",
    "Feni", "Cox's Bazar",
]

DHAKA_AREAS = [
    "Dhanmondi", "Gulshan", "Mirpur", "Uttara", "Bashundhara", "Mohakhali",
    "Banani", "Shyamoli", "Badda", "Khilgaon",
]

PRODUCTS = [
    ("Cotton Panjabi", "clothing", 1450),
    ("Printed Saree", "clothing", 2300),
    ("Denim Jacket", "clothing", 2650),
    ("Kids Frock", "clothing", 890),
    ("Mens Formal Shirt", "clothing", 1250),
    ("Womens Kurti", "clothing", 1150),
    ("Winter Shawl", "clothing", 1750),
    ("Cotton Trouser", "clothing", 1320),
    ("USB-C Fast Charger", "electronics_accessories", 950),
    ("Bluetooth Earbuds", "electronics_accessories", 2150),
    ("Power Bank 10000mAh", "electronics_accessories", 1850),
    ("Laptop Sleeve 15 inch", "electronics_accessories", 1100),
    ("Wireless Mouse", "electronics_accessories", 780),
    ("HDMI Cable 2m", "electronics_accessories", 520),
    ("Phone Holder Stand", "electronics_accessories", 390),
    ("Memory Card 64GB", "electronics_accessories", 870),
    ("Ceramic Dinner Set", "home_goods", 3400),
    ("Non-stick Frying Pan", "home_goods", 1280),
    ("Cotton Bedsheet Double", "home_goods", 1650),
    ("Steel Water Bottle", "home_goods", 640),
    ("Storage Basket Set", "home_goods", 980),
    ("Wall Clock", "home_goods", 760),
    ("Table Lamp", "home_goods", 1420),
    ("Aloe Vera Face Wash", "cosmetics", 450),
    ("Herbal Hair Oil", "cosmetics", 380),
    ("Matte Lipstick", "cosmetics", 690),
    ("Sunscreen SPF 50", "cosmetics", 1150),
    ("Moisturizing Lotion", "cosmetics", 520),
    ("Kajal Eyeliner", "cosmetics", 310),
    ("Neem Face Pack", "cosmetics", 420),
]

TIERS = ["regular", "silver", "gold"]

REFUND_WINDOW_DAYS = 7

# Day 7 is skipped so that no delivered_at lands within 12 hours of the refund
# window boundary. Every delivered order is at least 24 hours clear of it.
DELIVERED_DAY_OFFSETS = [
    day for day in range(1, 21) if day != REFUND_WINDOW_DAYS
]

# Order counts per status. Delivered and returned orders carry delivered_at
# dates spread across the 20 days before the fixed clock, so the 7-day refund
# window contains some of them and excludes the rest.
STATUS_COUNTS = {
    "pending": 20,
    "confirmed": 20,
    "shipped": 20,
    "delivered": 40,
    "cancelled": 10,
    "returned": 10,
}

COUPONS = [
    ("EID300", 10.0, "2026-12-31", 3000),
    ("WELCOME5", 5.0, "2026-10-31", 1000),
    ("DOKAN10", 10.0, "2026-11-30", 2500),
    ("MONSOON15", 15.0, "2026-09-15", 1500),
]


def iso(moment):
    return moment.isoformat()


def delivery_charge(subtotal_bdt, is_dhaka):
    if subtotal_bdt >= 3000:
        return 0
    return 60 if is_dhaka else 120


def build(db_path=DEFAULT_DB_PATH):
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    connection = sqlite3.connect(db_path)
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

    districts = ["Dhaka"] * DHAKA_COUNT + OUTSIDE_DISTRICTS[: len(NAMES) - DHAKA_COUNT]
    rng.shuffle(districts)

    customers = []
    for index, name in enumerate(NAMES, start=1):
        district = districts[index - 1]
        is_dhaka = 1 if district == "Dhaka" else 0
        prefix = rng.choice(["013", "015", "016", "017", "018", "019"])
        phone = prefix + "".join(str(rng.randint(0, 9)) for _ in range(8))
        area = rng.choice(DHAKA_AREAS) if is_dhaka else f"{district} Sadar"
        customers.append(
            {
                "customer_id": f"CUS-{index:03d}",
                "name": name,
                "phone": phone,
                "district": district,
                "is_dhaka": is_dhaka,
                "tier": rng.choice(TIERS),
                "area": area,
            }
        )

    connection.executemany(
        "INSERT INTO customers (customer_id, name, phone, district, is_dhaka, tier)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                c["customer_id"], c["name"], c["phone"],
                c["district"], c["is_dhaka"], c["tier"],
            )
            for c in customers
        ],
    )

    products = []
    for index, (name, category, price) in enumerate(PRODUCTS, start=1):
        products.append(
            {
                "product_id": f"PRD-{index:03d}",
                "name": name,
                "category": category,
                "price_bdt": price,
                "stock": rng.randint(0, 120),
            }
        )

    connection.executemany(
        "INSERT INTO products (product_id, name, category, price_bdt, stock)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            (p["product_id"], p["name"], p["category"], p["price_bdt"], p["stock"])
            for p in products
        ],
    )

    statuses = []
    for status, count in STATUS_COUNTS.items():
        statuses.extend([status] * count)
    rng.shuffle(statuses)

    delivered_day_cycle = []
    for status in statuses:
        if status in ("delivered", "returned"):
            delivered_day_cycle.append(status)
    # Spread delivered_at evenly across the allowed days before the clock.
    delivered_offsets = [
        DELIVERED_DAY_OFFSETS[position % len(DELIVERED_DAY_OFFSETS)]
        for position in range(len(delivered_day_cycle))
    ]
    offset_iterator = iter(delivered_offsets)

    order_rows = []
    item_rows = []
    coupon_use_rows = []

    for index, status in enumerate(statuses):
        order_id = f"ORD-{1001 + index}"
        customer = customers[rng.randrange(len(customers))]

        item_count = rng.randint(1, 3)
        chosen = rng.sample(products, item_count)
        items = []
        subtotal = 0
        for product in chosen:
            quantity = rng.randint(1, 4)
            items.append((order_id, product["product_id"], quantity, product["price_bdt"]))
            subtotal += quantity * product["price_bdt"]

        charge = delivery_charge(subtotal, customer["is_dhaka"])

        shipped_at = None
        delivered_at = None
        if status in ("delivered", "returned"):
            days_ago = next(offset_iterator)
            delivered_moment = FIXED_CLOCK - timedelta(days=days_ago)
            shipped_moment = delivered_moment - timedelta(days=2)
            created_moment = shipped_moment - timedelta(days=1)
            delivered_at = iso(delivered_moment)
            shipped_at = iso(shipped_moment)
        elif status == "shipped":
            shipped_moment = FIXED_CLOCK - timedelta(days=rng.randint(1, 4))
            created_moment = shipped_moment - timedelta(days=1)
            shipped_at = iso(shipped_moment)
        elif status == "cancelled":
            created_moment = FIXED_CLOCK - timedelta(days=rng.randint(1, 15))
        else:
            created_moment = FIXED_CLOCK - timedelta(
                days=rng.randint(0, 5), hours=rng.randint(0, 23)
            )

        order_rows.append(
            (
                order_id, customer["customer_id"], status, rng.choice(
                    ["cod", "bkash", "nagad", "card"]
                ),
                customer["district"],
                f"House {rng.randint(1, 99)}, Road {rng.randint(1, 30)}, "
                f"{customer['area']}",
                iso(created_moment), shipped_at, delivered_at,
                subtotal + charge, charge,
            )
        )
        item_rows.extend(items)

    connection.executemany(
        "INSERT INTO orders (order_id, customer_id, status, payment_method,"
        " district, address, created_at, shipped_at, delivered_at, total_bdt,"
        " delivery_charge_bdt) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        order_rows,
    )
    connection.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price_bdt)"
        " VALUES (?, ?, ?, ?)",
        item_rows,
    )
    connection.executemany(
        "INSERT INTO coupons (code, percent_off, valid_until, min_order_bdt)"
        " VALUES (?, ?, ?, ?)",
        COUPONS,
    )
    connection.executemany(
        "INSERT INTO coupon_uses (order_id, code) VALUES (?, ?)", coupon_use_rows
    )

    connection.commit()
    connection.close()
    return db_path


if __name__ == "__main__":
    path = build()
    print(f"seed database written: {path}")
