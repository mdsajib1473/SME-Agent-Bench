"""Per-run shop database.

Each Shop copies the seed database to a fresh temporary file, so a task run can
mutate state freely without touching the seed or another run.
"""

import shutil
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

DHAKA_TZ = timezone(timedelta(hours=6))
FIXED_CLOCK = datetime(2026, 10, 1, 10, 0, 0, tzinfo=DHAKA_TZ)

ENV_DIR = Path(__file__).resolve().parent
SEED_DB_PATH = ENV_DIR / "shop_seed.db"

TABLES = [
    "customers",
    "products",
    "orders",
    "order_items",
    "refunds",
    "tickets",
    "quotes",
    "quote_items",
    "coupons",
    "coupon_uses",
]

# Column tuples used to order rows inside a snapshot, so two databases holding
# the same logical state compare equal regardless of insertion order.
SNAPSHOT_KEYS = {
    "customers": ("customer_id",),
    "products": ("product_id",),
    "orders": ("order_id",),
    "order_items": ("order_id", "product_id"),
    "refunds": ("refund_id",),
    "tickets": ("ticket_id",),
    "quotes": ("quote_id",),
    "quote_items": ("quote_id", "product_id"),
    "coupons": ("code",),
    "coupon_uses": ("order_id", "code"),
}


class Shop:
    def __init__(self, seed_db_path=SEED_DB_PATH):
        self.seed_db_path = Path(seed_db_path)
        if not self.seed_db_path.exists():
            raise FileNotFoundError(
                f"seed database missing: {self.seed_db_path}. Run env/seed.py first."
            )
        handle = tempfile.NamedTemporaryFile(
            prefix="dokan_run_", suffix=".db", delete=False
        )
        handle.close()
        self.db_path = Path(handle.name)
        shutil.copyfile(self.seed_db_path, self.db_path)
        self._connection = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self.clock = FIXED_CLOCK

    @property
    def connection(self):
        return self._connection

    def now(self):
        return self.clock

    def now_iso(self):
        return self.clock.isoformat()

    def snapshot(self):
        state = {}
        for table in TABLES:
            cursor = self._connection.execute(f"SELECT * FROM {table}")
            columns = [description[0] for description in cursor.description]
            rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
            keys = SNAPSHOT_KEYS[table]
            rows.sort(key=lambda row: tuple(str(row[key]) for key in keys))
            state[table] = rows
        return state

    def close(self):
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self.db_path.exists():
            self.db_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
