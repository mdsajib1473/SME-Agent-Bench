"""Environment tests: seed determinism, tool behaviour, and policy permissiveness."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import seed, tools
from env.shop import FIXED_CLOCK, Shop

REFUND_WINDOW_DAYS = seed.REFUND_WINDOW_DAYS
DELAY_THRESHOLD_DAYS = seed.DELAY_THRESHOLD_DAYS


@pytest.fixture(scope="module")
def seed_db():
    path = seed.DEFAULT_DB_PATH
    if not path.exists():
        seed.build(path)
    return path


@pytest.fixture
def shop(seed_db):
    with Shop(seed_db) as instance:
        yield instance


@pytest.fixture
def registry(shop):
    return tools.build_registry(shop)


def _one(shop, query, params=()):
    row = shop.connection.execute(query, params).fetchone()
    return dict(row) if row is not None else None


def _order_with_status(shop, status):
    return _one(
        shop,
        "SELECT o.*, c.phone AS phone FROM orders o"
        " JOIN customers c ON c.customer_id = o.customer_id"
        " WHERE o.status = ? ORDER BY o.order_id LIMIT 1",
        (status,),
    )


def _delivered_order(shop, inside_window):
    cutoff = (FIXED_CLOCK - timedelta(days=REFUND_WINDOW_DAYS)).isoformat()
    comparison = ">=" if inside_window else "<"
    return _one(
        shop,
        "SELECT o.*, c.phone AS phone FROM orders o"
        " JOIN customers c ON c.customer_id = o.customer_id"
        f" WHERE o.status = 'delivered' AND o.delivered_at {comparison} ?"
        " ORDER BY o.order_id LIMIT 1",
        (cutoff,),
    )


def _single_item_order(shop, status):
    return _one(
        shop,
        "SELECT o.*, c.phone AS phone FROM orders o"
        " JOIN customers c ON c.customer_id = o.customer_id"
        " WHERE o.status = ? AND (SELECT COUNT(*) FROM order_items i"
        " WHERE i.order_id = o.order_id) = 1 ORDER BY o.order_id LIMIT 1",
        (status,),
    )


class TestSeedDeterminism:
    def test_two_builds_produce_identical_snapshots(self, tmp_path):
        first = seed.build(tmp_path / "first.db")
        second = seed.build(tmp_path / "second.db")
        with Shop(first) as shop_a, Shop(second) as shop_b:
            assert shop_a.snapshot() == shop_b.snapshot()

    def test_rebuild_matches_committed_seed(self, tmp_path, seed_db):
        rebuilt = seed.build(tmp_path / "rebuilt.db")
        with Shop(rebuilt) as shop_a, Shop(seed_db) as shop_b:
            assert shop_a.snapshot() == shop_b.snapshot()

    def test_seed_shape(self, shop):
        snapshot = shop.snapshot()
        assert len(snapshot["customers"]) == 40
        assert len(snapshot["products"]) == 30
        assert len(snapshot["orders"]) == 120
        assert len(snapshot["coupons"]) == 4
        assert sum(c["is_dhaka"] for c in snapshot["customers"]) == 20
        assert {o["status"] for o in snapshot["orders"]} == {
            "pending", "confirmed", "shipped", "delivered", "cancelled", "returned",
        }
        assert snapshot["refunds"] == []
        assert snapshot["tickets"] == []
        assert snapshot["quotes"] == []

    def test_phone_format(self, shop):
        for customer in shop.snapshot()["customers"]:
            assert tools.PHONE_PATTERN.match(customer["phone"])

    def test_refund_window_straddles_the_clock(self, shop):
        assert _delivered_order(shop, inside_window=True) is not None
        assert _delivered_order(shop, inside_window=False) is not None

    def test_no_delivery_near_the_refund_boundary(self, shop):
        boundary = FIXED_CLOCK - timedelta(days=REFUND_WINDOW_DAYS)
        for order in shop.snapshot()["orders"]:
            if not order["delivered_at"]:
                continue
            delivered = datetime.fromisoformat(order["delivered_at"])
            assert abs((delivered - boundary).total_seconds()) >= 12 * 3600, (
                f"{order['order_id']} sits within 12 hours of the refund boundary"
            )

    def test_shipped_orders_clear_the_delay_boundary(self, shop):
        boundary = FIXED_CLOCK - timedelta(days=DELAY_THRESHOLD_DAYS)
        ages = []
        for order in shop.snapshot()["orders"]:
            if order["status"] != "shipped":
                continue
            shipped = datetime.fromisoformat(order["shipped_at"])
            assert abs((shipped - boundary).total_seconds()) >= 24 * 3600, (
                f"{order['order_id']} sits within 24 hours of the delay boundary"
            )
            ages.append((FIXED_CLOCK - shipped).days)
        assert sum(1 for age in ages if age > DELAY_THRESHOLD_DAYS) >= 5
        assert {6, 7, 8, 9}.issubset(set(ages))
        assert max(ages) <= 9
        assert DELAY_THRESHOLD_DAYS not in ages

    def test_shipped_at_precedes_delivered_at(self, shop):
        for order in shop.snapshot()["orders"]:
            if order["shipped_at"] and order["delivered_at"]:
                assert order["shipped_at"] < order["delivered_at"], order["order_id"]

    def test_one_expired_coupon(self, shop):
        today = FIXED_CLOCK.date().isoformat()
        coupons = shop.snapshot()["coupons"]
        expired = [c for c in coupons if c["valid_until"] < today]
        assert len(expired) == 1


class TestRegistry:
    def test_registry_entry_count(self, registry):
        # The brief asked for 11 tools but named 12; all 12 named tools exist.
        assert len(registry) == 12
        assert len(tools.TOOL_SCHEMAS) == 12

    def test_schema_names_match_functions(self, registry):
        schema_names = {entry["function"]["name"] for entry in tools.TOOL_SCHEMAS}
        assert schema_names == set(registry)

    def test_schemas_are_well_formed(self):
        for entry in tools.TOOL_SCHEMAS:
            assert entry["type"] == "function"
            function = entry["function"]
            assert function["description"].strip()
            parameters = function["parameters"]
            assert parameters["type"] == "object"
            for name in parameters["required"]:
                assert name in parameters["properties"]
            for name, spec in parameters["properties"].items():
                assert spec["description"].strip(), name

    def test_call_tool_dispatch_and_unknown_name(self, shop):
        order = _order_with_status(shop, "pending")
        result = tools.call_tool(shop, "get_order", {"order_id": order["order_id"]})
        assert result["ok"]
        assert not tools.call_tool(shop, "no_such_tool", {})["ok"]
        assert not tools.call_tool(shop, "get_order", "{bad json")["ok"]


class TestToolsValidInput:
    def test_get_customer(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["get_customer"](phone=order["phone"])
        assert result["ok"]
        assert result["customer"]["phone"] == order["phone"]

    def test_get_order(self, shop, registry):
        order = _order_with_status(shop, "delivered")
        result = registry["get_order"](order_id=order["order_id"])
        assert result["ok"]
        assert result["order"]["customer_phone"] == order["phone"]
        assert result["order"]["items"]
        assert result["order"]["subtotal_bdt"] == (
            order["total_bdt"] - order["delivery_charge_bdt"]
        )

    def test_list_customer_orders(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["list_customer_orders"](phone=order["phone"])
        assert result["ok"]
        assert result["count"] >= 1
        assert order["order_id"] in [o["order_id"] for o in result["orders"]]

    def test_search_products(self, registry):
        result = registry["search_products"](query="cosmetics")
        assert result["ok"]
        assert result["count"] > 0
        assert all(p["category"] == "cosmetics" for p in result["products"])

    def test_check_stock(self, registry):
        result = registry["check_stock"](product_id="PRD-001")
        assert result["ok"]
        assert result["stock"] >= 0

    def test_cancel_order(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["cancel_order"](
            order_id=order["order_id"], reason="customer changed mind"
        )
        assert result["ok"]
        assert result["previous_status"] == "pending"
        assert _one(
            shop, "SELECT status FROM orders WHERE order_id = ?", (order["order_id"],)
        )["status"] == "cancelled"

    def test_update_order_address(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["update_order"](
            order_id=order["order_id"], field="address", value="House 9, Road 3, Mirpur"
        )
        assert result["ok"]
        assert result["current"] == "House 9, Road 3, Mirpur"

    def test_update_order_rejects_phone_field(self, shop, registry):
        order = _order_with_status(shop, "confirmed")
        result = registry["update_order"](
            order_id=order["order_id"], field="phone", value="01711111111"
        )
        assert not result["ok"]
        stored = _one(
            shop,
            "SELECT phone FROM customers WHERE customer_id = ?",
            (order["customer_id"],),
        )
        assert stored["phone"] == order["phone"]

    def test_update_order_quantity(self, shop, registry):
        order = _single_item_order(shop, "pending")
        result = registry["update_order"](
            order_id=order["order_id"], field="quantity", value=2
        )
        assert result["ok"]
        assert result["current"] == 2
        assert result["total_bdt"] == (
            result["subtotal_bdt"] + result["delivery_charge_bdt"]
        )

    def test_update_order_quantity_named_line(self, shop, registry):
        multi = _one(
            shop,
            "SELECT o.order_id FROM orders o WHERE (SELECT COUNT(*) FROM order_items i"
            " WHERE i.order_id = o.order_id) > 1 ORDER BY o.order_id LIMIT 1",
        )
        items = shop.connection.execute(
            "SELECT product_id FROM order_items WHERE order_id = ? ORDER BY product_id",
            (multi["order_id"],),
        ).fetchall()
        product_id = items[0]["product_id"]
        result = registry["update_order"](
            order_id=multi["order_id"], field="quantity", value=f"{product_id}:5"
        )
        assert result["ok"]
        assert result["product_id"] == product_id
        assert result["current"] == 5

    def test_issue_refund(self, shop, registry):
        order = _delivered_order(shop, inside_window=True)
        result = registry["issue_refund"](
            order_id=order["order_id"],
            amount_bdt=500,
            method="bkash",
            reason="partial refund for late delivery",
        )
        assert result["ok"]
        assert result["refund_id"] == "REF-0001"
        assert result["created_at"] == FIXED_CLOCK.isoformat()

    def test_refund_ids_are_sequential(self, shop, registry):
        order = _delivered_order(shop, inside_window=True)
        first = registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="bkash", reason="one"
        )
        second = registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="bkash", reason="two"
        )
        assert [first["refund_id"], second["refund_id"]] == ["REF-0001", "REF-0002"]

    def test_create_quote(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["create_quote"](
            customer_phone=order["phone"],
            district="Dhaka",
            items=[
                {"product_id": "PRD-029", "quantity": 20},
                {"product_id": "PRD-030", "quantity": 10},
            ],
            discount_percent=5,
        )
        assert result["ok"]
        assert result["quote_id"] == "QUO-0001"
        assert result["total_units"] == 30
        expected_subtotal = 20 * 310 + 10 * 420
        assert result["subtotal_bdt"] == expected_subtotal
        assert result["discount_amount_bdt"] == round(expected_subtotal * 0.05)
        assert result["total_bdt"] == (
            expected_subtotal - result["discount_amount_bdt"]
            + result["delivery_charge_bdt"]
        )

    def test_create_quote_delivery_charge_rules(self, shop, registry):
        order = _order_with_status(shop, "pending")
        cheap_dhaka = registry["create_quote"](
            customer_phone=order["phone"],
            district="Dhaka",
            items=[{"product_id": "PRD-029", "quantity": 1}],
            discount_percent=0,
        )
        assert cheap_dhaka["delivery_charge_bdt"] == 60
        cheap_outside = registry["create_quote"](
            customer_phone=order["phone"],
            district="Khulna",
            items=[{"product_id": "PRD-029", "quantity": 1}],
            discount_percent=0,
        )
        assert cheap_outside["delivery_charge_bdt"] == 120
        large = registry["create_quote"](
            customer_phone=order["phone"],
            district="Khulna",
            items=[{"product_id": "PRD-029", "quantity": 40}],
            discount_percent=5,
        )
        assert large["subtotal_bdt"] >= 3000
        assert large["delivery_charge_bdt"] == 0

    def test_route_ticket(self, shop, registry):
        order = _order_with_status(shop, "shipped")
        result = registry["route_ticket"](
            order_id=order["order_id"],
            department="logistics",
            priority="high",
            summary="parcel delayed 6 days",
        )
        assert result["ok"]
        assert result["ticket_id"] == "TKT-0001"

    def test_route_ticket_without_order(self, registry):
        result = registry["route_ticket"](
            department="general", priority="normal", summary="asks about opening hours"
        )
        assert result["ok"]
        assert result["order_id"] is None

    def test_lookup_policy(self, registry):
        result = registry["lookup_policy"](topic="refunds")
        assert result["ok"]
        assert "7 days" in result["section"]
        assert registry["lookup_policy"](topic="cancel")["topic"] == "Cancellation"
        assert registry["lookup_policy"](topic="address")["topic"] == "Address Change"
        assert registry["lookup_policy"](topic="delivery")["ok"]
        assert registry["lookup_policy"](topic="complaint routing")["ok"]

    def test_apply_coupon(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["apply_coupon"](order_id=order["order_id"], code="welcome5")
        assert result["ok"]
        assert result["code"] == "WELCOME5"
        assert result["total_bdt"] == (
            result["subtotal_bdt"] - result["discount_amount_bdt"]
            + order["delivery_charge_bdt"]
        )


class TestToolsInvalidInput:
    def test_get_customer_bad_phone(self, registry):
        assert not registry["get_customer"](phone="12345")["ok"]

    def test_get_customer_unknown_phone(self, registry):
        assert not registry["get_customer"](phone="01999999999")["ok"]

    def test_get_order_bad_id(self, registry):
        assert not registry["get_order"](order_id="1001")["ok"]

    def test_get_order_unknown_id(self, registry):
        assert not registry["get_order"](order_id="ORD-9999")["ok"]

    def test_list_customer_orders_bad_phone(self, registry):
        assert not registry["list_customer_orders"](phone="")["ok"]

    def test_search_products_empty_query(self, registry):
        assert not registry["search_products"](query="   ")["ok"]

    def test_check_stock_bad_id(self, registry):
        assert not registry["check_stock"](product_id="PRODUCT-1")["ok"]
        assert not registry["check_stock"](product_id="PRD-999")["ok"]

    def test_cancel_order_bad_reason(self, shop, registry):
        order = _order_with_status(shop, "pending")
        assert not registry["cancel_order"](order_id=order["order_id"], reason="")["ok"]

    def test_update_order_bad_field(self, shop, registry):
        order = _order_with_status(shop, "pending")
        assert not registry["update_order"](
            order_id=order["order_id"], field="district", value="Dhaka"
        )["ok"]

    def test_update_order_bad_quantity_value(self, shop, registry):
        order = _single_item_order(shop, "pending")
        assert not registry["update_order"](
            order_id=order["order_id"], field="quantity", value="many"
        )["ok"]
        assert not registry["update_order"](
            order_id=order["order_id"], field="quantity", value=0
        )["ok"]

    def test_update_order_quantity_needs_a_named_line(self, shop, registry):
        multi = _one(
            shop,
            "SELECT o.order_id FROM orders o WHERE (SELECT COUNT(*) FROM order_items i"
            " WHERE i.order_id = o.order_id) > 1 ORDER BY o.order_id LIMIT 1",
        )
        assert not registry["update_order"](
            order_id=multi["order_id"], field="quantity", value=3
        )["ok"]

    def test_issue_refund_bad_arguments(self, shop, registry):
        order = _delivered_order(shop, inside_window=True)
        assert not registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=-5, method="bkash", reason="x"
        )["ok"]
        assert not registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="cheque", reason="x"
        )["ok"]
        assert not registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="bkash", reason=" "
        )["ok"]

    def test_create_quote_bad_arguments(self, shop, registry):
        order = _order_with_status(shop, "pending")
        assert not registry["create_quote"](
            customer_phone="01", district="Dhaka",
            items=[{"product_id": "PRD-001", "quantity": 1}], discount_percent=5,
        )["ok"]
        assert not registry["create_quote"](
            customer_phone=order["phone"], district="Dhaka", items=[],
            discount_percent=5,
        )["ok"]
        assert not registry["create_quote"](
            customer_phone=order["phone"], district="Dhaka",
            items=[{"product_id": "PRD-999", "quantity": 1}], discount_percent=5,
        )["ok"]
        assert not registry["create_quote"](
            customer_phone=order["phone"], district="Dhaka",
            items=[{"product_id": "PRD-001", "quantity": 0}], discount_percent=5,
        )["ok"]
        assert not registry["create_quote"](
            customer_phone=order["phone"], district="Dhaka",
            items=[{"product_id": "PRD-001", "quantity": 1}], discount_percent=120,
        )["ok"]

    def test_route_ticket_bad_arguments(self, registry):
        assert not registry["route_ticket"](
            department="warehouse", priority="normal", summary="x"
        )["ok"]
        assert not registry["route_ticket"](
            department="general", priority="urgent", summary="x"
        )["ok"]
        assert not registry["route_ticket"](
            department="general", priority="normal", summary=""
        )["ok"]
        assert not registry["route_ticket"](
            department="general", priority="normal", summary="x", order_id="ORD-9999"
        )["ok"]

    def test_lookup_policy_unknown_topic(self, registry):
        result = registry["lookup_policy"](topic="warranty transfer")
        assert not result["ok"]
        assert result["available_topics"]

    def test_apply_coupon_bad_arguments(self, shop, registry):
        order = _order_with_status(shop, "pending")
        assert not registry["apply_coupon"](
            order_id=order["order_id"], code="NOPE123"
        )["ok"]
        assert not registry["apply_coupon"](order_id="ORD-9999", code="WELCOME5")["ok"]
        registry["apply_coupon"](order_id=order["order_id"], code="WELCOME5")
        assert not registry["apply_coupon"](
            order_id=order["order_id"], code="DOKAN10"
        )["ok"]


class TestToolsDoNotEnforcePolicy:
    """Policy breaches must succeed at the tool layer; the agent is the gate."""

    def test_refund_outside_window_succeeds(self, shop, registry):
        order = _delivered_order(shop, inside_window=False)
        result = registry["issue_refund"](
            order_id=order["order_id"],
            amount_bdt=200,
            method="bkash",
            reason="late damage claim",
        )
        assert result["ok"]
        assert result["order_delivered_at"] < (
            FIXED_CLOCK - timedelta(days=REFUND_WINDOW_DAYS)
        ).isoformat()

    def test_refund_above_order_total_succeeds(self, shop, registry):
        order = _delivered_order(shop, inside_window=True)
        result = registry["issue_refund"](
            order_id=order["order_id"],
            amount_bdt=order["total_bdt"] * 10,
            method="bkash",
            reason="overpayment",
        )
        assert result["ok"]
        assert result["amount_bdt"] > result["order_total_bdt"]

    def test_refund_to_mismatched_method_succeeds(self, shop, registry):
        order = _one(
            shop,
            "SELECT * FROM orders WHERE status = 'delivered' AND payment_method = 'card'"
            " ORDER BY order_id LIMIT 1",
        )
        result = registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="nagad", reason="x"
        )
        assert result["ok"]

    def test_refund_on_pending_order_succeeds(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["issue_refund"](
            order_id=order["order_id"], amount_bdt=100, method="bkash", reason="x"
        )
        assert result["ok"]

    def test_cancelling_a_shipped_order_succeeds(self, shop, registry):
        order = _order_with_status(shop, "shipped")
        result = registry["cancel_order"](
            order_id=order["order_id"], reason="customer called"
        )
        assert result["ok"]
        assert result["previous_status"] == "shipped"

    def test_cancelling_a_delivered_order_succeeds(self, shop, registry):
        order = _order_with_status(shop, "delivered")
        assert registry["cancel_order"](
            order_id=order["order_id"], reason="customer called"
        )["ok"]

    def test_address_change_after_shipping_succeeds(self, shop, registry):
        order = _order_with_status(shop, "shipped")
        result = registry["update_order"](
            order_id=order["order_id"], field="address", value="New address, Uttara"
        )
        assert result["ok"]
        assert result["order_status"] == "shipped"

    def test_quantity_change_on_delivered_order_succeeds(self, shop, registry):
        order = _single_item_order(shop, "delivered")
        result = registry["update_order"](
            order_id=order["order_id"], field="quantity", value=3
        )
        assert result["ok"]
        assert result["order_status"] == "delivered"

    def test_quantity_above_stock_succeeds(self, shop, registry):
        order = _single_item_order(shop, "pending")
        product_id = _one(
            shop,
            "SELECT product_id FROM order_items WHERE order_id = ?",
            (order["order_id"],),
        )["product_id"]
        stock = _one(
            shop, "SELECT stock FROM products WHERE product_id = ?", (product_id,)
        )["stock"]
        result = registry["update_order"](
            order_id=order["order_id"], field="quantity", value=stock + 50
        )
        assert result["ok"]
        assert result["current"] > stock

    def test_discount_above_ten_percent_succeeds(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["create_quote"](
            customer_phone=order["phone"],
            district="Dhaka",
            items=[{"product_id": "PRD-029", "quantity": 2}],
            discount_percent=40,
        )
        assert result["ok"]
        assert result["discount_percent"] == 40.0

    def test_expired_coupon_succeeds(self, shop, registry):
        order = _order_with_status(shop, "pending")
        result = registry["apply_coupon"](order_id=order["order_id"], code="MONSOON15")
        assert result["ok"]
        assert result["valid_until"] < FIXED_CLOCK.date().isoformat()

    def test_coupon_below_minimum_on_shipped_order_succeeds(self, shop, registry):
        order = _one(
            shop,
            "SELECT * FROM orders WHERE status = 'shipped'"
            " AND total_bdt - delivery_charge_bdt < 3000 ORDER BY order_id LIMIT 1",
        )
        result = registry["apply_coupon"](order_id=order["order_id"], code="EID300")
        assert result["ok"]
        assert result["subtotal_bdt"] < result["min_order_bdt"]
        assert result["order_status"] == "shipped"

    def test_wrong_department_and_priority_succeed(self, shop, registry):
        order = _order_with_status(shop, "delivered")
        result = registry["route_ticket"](
            order_id=order["order_id"],
            department="billing",
            priority="low",
            summary="item arrived damaged",
        )
        assert result["ok"]

    def test_tools_never_check_identity(self, shop, registry):
        """No tool takes a phone number to verify against the order."""
        order = _order_with_status(shop, "pending")
        assert registry["cancel_order"](
            order_id=order["order_id"], reason="caller gave no phone number"
        )["ok"]
