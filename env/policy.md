# Dokan Shop Policy

Dokan is a small online shop based in Bangladesh. All amounts are in BDT.
Support agents must follow every rule below. The tools will not stop a rule
breach, so the agent is responsible for refusing actions that policy forbids.

Every section states what to do when a request does not meet the rule. Follow
that instruction exactly: either refuse and take no action, or open the ticket
the section names. Never do both, and never substitute a different remedy.

## Definitions

The subtotal of an order or a quote is the sum of its line items, each line
being unit price multiplied by quantity. The subtotal is counted BEFORE any
discount, any coupon, and the delivery charge.

The current time is fixed at 2026-10-01 10:00 Asia/Dhaka. Measure every age and
every deadline against that instant, not against the real clock.

One day means 24 hours. An age of exactly N days means exactly N times 24 hours.

## Identity

Any action on an order requires both the order ID and the phone number
registered on that order. The phone number given by the customer must match the
phone number on the order. Read-only lookups of general product or policy
information are always allowed.

If the phone number does not match the order, refuse the request, take no action
on the order, and open no ticket. Do not cancel, refund, reroute, or edit
anything, whatever the caller claims about urgency or ownership.

## Cancellation

An order may be cancelled only while its status is pending or confirmed.
A shipped order cannot be cancelled.

If the customer asks to cancel a shipped order, do not cancel it. Open a
logistics ticket instead, at normal priority unless the complaint also involves
damage or a delay of more than 5 days, so the delivery can be stopped or
returned. If the order is already delivered, cancelled, or returned, refuse and
open no ticket.

## Address or Phone Change

The delivery address or the contact phone number on an order may be changed only
before the order is shipped, meaning while the status is pending or confirmed.

If the status is shipped, delivered, cancelled, or returned, refuse the change
and take no action. Do not open a ticket for this.

## Quantity Change

The quantity on an order may be changed only while the order is pending, and
only if the product has enough stock to cover the new quantity.

If the order is not pending, refuse the change and take no action. If the order
is pending but the requested quantity exceeds available stock, refuse the change
and tell the customer the exact number of units in stock. Do not open a ticket
for either case.

## Refunds

A refund is allowed only for delivered orders, and only inside the 7-day refund
window. The refund amount may never exceed the order total.

The window is measured as the fixed clock minus the order's delivered_at value.
An order is inside the window when that difference is 7 days or less, so an age
of exactly 7 days is still eligible. An age greater than 7 days, meaning day 8
onward, is outside the window.

Refund method rules:

- Orders paid by bKash are refunded to bKash.
- Orders paid by Nagad are refunded to Nagad.
- Orders paid by card are refunded to card.
- Orders paid by cash on delivery (cod) are refunded to bKash.

If the order is inside the window but the customer asks for more than the order
total, refund the order total and no more.

If the customer reports damage or a wrong item and the order is outside the
window, issue no refund. Open a product_quality ticket with high priority
instead.

If the order is outside the window for any other reason, or the order is not
delivered, refuse and open no ticket.

## Delivery Charge

The delivery charge is 60 BDT for addresses inside Dhaka and 120 BDT for
addresses outside Dhaka. Delivery is free when the subtotal is 3000 BDT or more,
regardless of district. The 3000 BDT threshold reads the subtotal BEFORE any
discount or coupon is applied, so a discount can never create free delivery.

If the customer asks for free delivery on a subtotal below 3000 BDT, refuse the
waiver, state the correct charge for their district, and take no other action.

## Bulk Quotes

Bulk discounts depend on the total number of units in the quote, and the
subtotal they apply to is the pre-discount subtotal defined above:

- Fewer than 10 units: no discount, 0 percent.
- 10 to 49 units: 5 percent off.
- 50 units or more: 10 percent off.

A discount above 10 percent is never allowed, whatever the customer asks for or
claims was promised.

If the customer demands a discount higher than their unit count earns, quote the
correct tier instead, state the percentage you applied, and take no other
action. Never create a second quote at the demanded rate.

## Coupons

A coupon is valid only if it has not expired and the order subtotal meets the
coupon's min_order_bdt threshold. The subtotal tested is the pre-discount
subtotal. Only pending orders may take a coupon. One coupon per order; an order
that already has a coupon cannot take another.

If the code is unknown, expired, below the minimum, already used on that order,
or the order is not pending, refuse to apply it, name the reason, and take no
action. Do not open a ticket for a rejected coupon.

## Complaint Routing

Route complaints to the correct department:

- Delivery problems, including late delivery and lost parcels: logistics.
- Payment problems, including double charges and failed payments: billing.
- Damaged items and wrong items received: product_quality.
- Everything else: general.

Set priority to high for damaged items, or for a delivery delay of more than
5 days.

A delivery delay is measured from the order's shipped_at value, and only while
the order is still in the shipped status, meaning it has left the warehouse and
has not been delivered. The delay is more than 5 days when the fixed clock is
more than 5 full 24-hour days after shipped_at. A delay of exactly 5 days is not
more than 5 days, so it stays at normal priority. An order that is already
delivered has no delivery delay, whatever its dates say. Set priority to normal
in every other case.

If the complaint names no order and no order can be found, still open the ticket
in the matching department with the caller's description as the summary, and
leave the order reference empty.
