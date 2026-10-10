"""Write tasks/db_snapshot.csv: a plain English seed database snapshot per task."""

import csv
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_FILES = [ROOT / "tasks" / "tasks.jsonl"]
OUT = ROOT / "tasks" / "db_snapshot.csv"
CLOCK = datetime(2026, 10, 1, 10, 0, tzinfo=timezone(timedelta(hours=6)))

# Strong keywords always name the product; weak ones only when the product is
# already a line item on an order in the same task.
STRONG = {
    "PRD-001": ["panjabi"], "PRD-002": ["saree"], "PRD-003": ["denim", "jacket"],
    "PRD-004": ["frock"], "PRD-005": ["formal shirt", "shirt"], "PRD-006": ["kurti"],
    "PRD-007": ["shawl"], "PRD-008": ["trouser"], "PRD-009": ["charger"],
    "PRD-010": ["earbud"], "PRD-011": ["power bank"], "PRD-012": ["laptop sleeve"],
    "PRD-013": ["mouse"], "PRD-014": ["hdmi"], "PRD-015": ["phone holder"],
    "PRD-016": ["memory card"], "PRD-017": ["dinner set"], "PRD-018": ["frying pan"],
    "PRD-019": ["bedsheet"], "PRD-020": ["water bottle"], "PRD-021": ["storage basket"],
    "PRD-022": ["wall clock"], "PRD-023": ["lamp"], "PRD-024": ["face wash"],
    "PRD-025": ["hair oil"], "PRD-026": ["lipstick"], "PRD-027": ["sunscreen"],
    "PRD-028": ["lotion"], "PRD-029": ["kajal", "eyeliner"], "PRD-030": ["neem", "face pack"],
}
WEAK = {"PRD-020": ["bottle"]}


def date_words(iso):
    return datetime.fromisoformat(iso).strftime("%d %B %Y").lstrip("0")


def age_words(iso):
    hours = (CLOCK - datetime.fromisoformat(iso)).total_seconds() / 3600
    days = hours / 24
    text = f"{days:g}" if days == int(days) else f"{days:.2f}".rstrip("0")
    return f"{text} days before the fixed clock"


def pct(value):
    return f"{value:g}"


def main():
    con = sqlite3.connect(ROOT / "env" / "shop_seed.db")
    con.row_factory = sqlite3.Row
    products = {r["product_id"]: dict(r) for r in con.execute("SELECT * FROM products")}
    coupons = {r["code"]: dict(r) for r in con.execute("SELECT * FROM coupons")}
    used = {r["order_id"]: r["code"] for r in con.execute("SELECT * FROM coupon_uses")}

    rows = []
    for path in TASK_FILES:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            task = json.loads(line)
            gold = json.dumps(task["gold_actions"])
            text = " ".join([task["instruction"], task["notes"], gold])
            low = text.lower()

            order_ids = sorted(set(re.findall(r"ORD-\d+", text)))
            codes = [c for c in coupons if re.search(rf"\b{c}\b", text, re.I)]
            facts = []
            line_products = set()

            for oid in order_ids:
                o = con.execute(
                    "SELECT o.*, c.name, c.phone FROM orders o"
                    " JOIN customers c ON c.customer_id = o.customer_id"
                    " WHERE order_id = ?", (oid,)
                ).fetchone()
                if o is None:
                    facts.append(f"{oid}: not found in the seed database")
                    continue
                items = con.execute(
                    "SELECT oi.*, p.name FROM order_items oi JOIN products p"
                    " ON p.product_id = oi.product_id WHERE order_id = ?"
                    " ORDER BY oi.product_id", (oid,)
                ).fetchall()
                subtotal = sum(i["quantity"] * i["unit_price_bdt"] for i in items)
                discount = subtotal + o["delivery_charge_bdt"] - o["total_bdt"]
                facts.append(f"{oid} owner {o['name']}, phone {o['phone']}")
                facts.append(f"{oid} status {o['status']}")
                facts.append(f"{oid} payment method {o['payment_method']}")
                facts.append(f"{oid} district {o['district']}")
                if o["delivered_at"]:
                    facts.append(
                        f"{oid} delivered on {date_words(o['delivered_at'])}"
                        f" ({age_words(o['delivered_at'])})"
                    )
                if o["shipped_at"] and not o["delivered_at"]:
                    facts.append(
                        f"{oid} shipped on {date_words(o['shipped_at'])}"
                        f" ({age_words(o['shipped_at'])})"
                    )
                if not o["shipped_at"] and not o["delivered_at"]:
                    facts.append(f"{oid} not shipped")
                for i in items:
                    line_products.add(i["product_id"])
                    facts.append(
                        f"{oid} item {i['name']} ({i['product_id']}) x {i['quantity']}"
                        f" at {i['unit_price_bdt']}"
                    )
                facts.append(f"{oid} subtotal {subtotal}")
                coupon_note = f" (coupon {used[oid]})" if oid in used else ""
                facts.append(f"{oid} discount {discount}{coupon_note}")
                facts.append(f"{oid} delivery charge {o['delivery_charge_bdt']}")
                facts.append(f"{oid} total {o['total_bdt']}")

            for code in codes:
                c = coupons[code]
                facts.append(f"coupon {code} discount {pct(c['percent_off'])} percent")
                facts.append(f"coupon {code} minimum subtotal {c['min_order_bdt']}")
                facts.append(
                    f"coupon {code} expires {date_words(c['valid_until'] + 'T23:59:59+06:00')}"
                )

            named = set(re.findall(r"PRD-\d{3}", text))
            for pid, keys in STRONG.items():
                if any(re.search(rf"\b{k}", low) for k in keys) or (
                    pid == "PRD-009" and re.search(r"\busb", low)
                ):
                    named.add(pid)
            for pid, keys in WEAK.items():
                if pid in line_products and any(re.search(rf"\b{k}", low) for k in keys):
                    named.add(pid)
            for pid in sorted(named):
                p = products[pid]
                facts.append(f"product {p['name']} ({pid}) stock {p['stock']}")
                facts.append(f"product {p['name']} ({pid}) unit price {p['price_bdt']}")

            if not facts:
                facts.append("no order, coupon or product named")
            rows.append((task["task_id"], "; ".join(facts)))

    with OUT.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "db_snapshot"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
