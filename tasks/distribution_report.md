# Task distribution report

Merged set: 80 tasks, 10 approved in `tasks.jsonl` and 70 awaiting review in `tasks_draft.jsonl`.

## Category by difficulty

| Category | Easy | Medium | Hard | Total |
| --- | --- | --- | --- | --- |
| refund | 6 | 6 | 4 | 16 |
| order_change | 6 | 6 | 4 | 16 |
| complaint_routing | 6 | 6 | 4 | 16 |
| quotation | 6 | 6 | 4 | 16 |
| inquiry | 6 | 6 | 4 | 16 |
| **total** | 30 | 30 | 20 | 80 |

## Category by language

| Category | en | banglish | banglish share |
| --- | --- | --- | --- |
| refund | 12 | 4 | 25% |
| order_change | 12 | 4 | 25% |
| complaint_routing | 12 | 4 | 25% |
| quotation | 12 | 4 | 25% |
| inquiry | 12 | 4 | 25% |
| **total** | 60 | 20 | 25% |

## Trap status

| Category | Traps | Non-traps | Trap share | Hard traps (of 4 hard) |
| --- | --- | --- | --- | --- |
| refund | 4 | 12 | 25% | 3 |
| order_change | 3 | 13 | 19% | 3 |
| complaint_routing | 3 | 13 | 19% | 3 |
| quotation | 3 | 13 | 19% | 3 |
| inquiry | 3 | 13 | 19% | 3 |
| **total** | 16 | 64 | 20% | 15 |

## Multi-step tasks

Tasks with 3 or more write actions in a fixed order: 7. Of the 20 hard tasks, 7 need 3 or more write actions (35%).

- `CMP-H-04` (complaint_routing): route_ticket, issue_refund, cancel_order
- `ORD-H-02` (order_change): apply_coupon, update_order, cancel_order
- `ORD-H-04` (order_change): update_order, cancel_order, apply_coupon
- `QUO-H-02` (quotation): create_quote, cancel_order, route_ticket
- `QUO-H-04` (quotation): create_quote, cancel_order, route_ticket
- `REF-H-03` (refund): issue_refund, route_ticket, cancel_order
- `REF-H-04` (refund): issue_refund, cancel_order, route_ticket

## Trap types covered

| Trap type | Tasks |
| --- | --- |
| identity mismatch | `CMP-H-01`, `CMP-H-03`, `INQ-H-01` |
| outside refund window | `REF-H-01`, `CMP-H-02` |
| order not delivered | `REF-H-02` |
| refund above order total | `REF-H-03` |
| wrong refund method | `REF-M-06` |
| shipped order cancellation | `ORD-H-01` |
| change after shipping | `ORD-H-03` |
| insufficient stock | `ORD-H-02` |
| discount above the band | `QUO-H-01`, `QUO-H-02`, `QUO-H-03` |
| expired coupon | `INQ-H-02` |
| coupon below minimum | `INQ-H-03` |

## Order id reuse

55 distinct order ids appear across the instructions. Reuse counts: 1 task(s): 43 ids, 2 task(s): 12 ids.

No order id appears in more than 3 tasks.

## Validator result

`python eval/validate_tasks.py tasks/tasks.jsonl tasks/tasks_draft.jsonl` reports 0 schema or replay problems, a 100 percent oracle success rate, and an 8 percent no-op success rate that matches the share of do-nothing traps exactly.
