# Benchmark card: sme-agent-bench

Numbers come from env/, tasks/ and config.yaml. Task numbers describe the final reviewed
80 task set in tasks/tasks.jsonl.

## Environment

* Dokan, a simulated online shop in Bangladesh, in SQLite, rebuilt from a seeded
  generator (seed 42) and copied fresh for every run. Clock fixed at 2026-10-01 10:00
  Asia/Dhaka; all amounts in BDT.
* Seed data: 40 customers, 30 products, 120 orders (20 pending, 20 confirmed, 20 shipped,
  40 delivered, 10 cancelled, 10 returned), 249 order lines, 4 coupons.
* 12 tools: 5 lookups (get_customer, get_order, list_customer_orders, search_products,
  check_stock), lookup_policy, and 6 writes (cancel_order, update_order, issue_refund,
  create_quote, route_ticket, apply_coupon). Tools check only that inputs are well formed
  and IDs exist; policy is the agent's job.

## Policy summary [env/policy.md]

* Identity: any action on an order needs the order ID and the phone number registered on
  it; on a mismatch, refuse and open no ticket.
* Cancellation only while pending or confirmed; a shipped order gets a logistics ticket
  instead. Address change only before shipping; quantity change only while pending and
  within stock; the phone number on an order is never changed.
* Refunds only for delivered orders within 7 days, never above the order total (the
  amount paid, including delivery), to the payment method (cash on delivery refunds go to bKash); damage outside the window gets a
  high priority product_quality ticket instead.
* Delivery charge 60 BDT in Dhaka, 120 BDT outside, free from a 3000 BDT pre discount subtotal.
* Bulk quotes: 0 percent below 10 units, 5 percent for 10 to 49, 10 percent from 50; never above 10.
* Coupons: valid, unexpired, minimum met, pending orders only, one per order.
* Complaints routed to logistics, billing, product_quality or general; high priority for
  damage or a shipping delay of more than 5 days.

## Task distribution

* 80 single turn tasks, 16 per category: refund, order_change, complaint_routing,
  quotation, inquiry.
* Difficulty: 30 easy, 30 medium, 20 hard (6, 6 and 4 per category).
* Language: 60 English, 20 Banglish (4 per category; 10 easy, 5 medium, 5 hard).
* Traps: 16 (refund 4, every other category 3; 15 of them hard). 7 traps need no write
  action: in 6 the agent only refuses, in 1 it must also name the coupon minimum. The
  other 9 need a different action than the one requested (a ticket, a lower discount, a
  capped refund). Trap types: identity mismatch, outside the refund
  window, order not delivered, refund above the total, wrong refund method, cancelling a
  shipped order, change after shipping, insufficient stock, discount above the band,
  expired coupon, coupon below the minimum.
* Gold write actions per task: none for 20 tasks, one for 52, two for 1, three for 7.
* The pilots used an earlier 10 task file (7 English, 3 Banglish, 4 traps) whose tasks
  are part of the final set (REF-E-01 was reworded during review).

## Scoring [eval/scorer.py]

success = state_match and output_match and not policy_violation and not empty_reply.

* state_match: the gold actions are replayed on a fresh shop and the whole database is
  compared with the agent's final state (free text columns ignored).
* output_match: every required value appears in the reply (case and thousands separators
  ignored); 51 tasks have at least one required value.
* policy_violation: a forbidden write that succeeded, optionally only under an argument
  condition; a call that returned an error is not a violation.
* empty_reply: an empty reply fails the task, even when doing nothing was correct.

## Architectures [agents/]

All three share one LLM client, one system prompt with the full policy, the 12 tools and a
budget of 20 LLM calls per task (config.yaml: temperature 0.3, at most 1024 output tokens
per call, seeds 0, 1, 2, request timeout 180 s).

* ReAct: one agent with all tools, at most 15 iterations.
* Plan and Execute: a planner without tools (at most 2 attempts, plans of up to 6 steps
  requested), an executor with all tools (at most 4 calls per step), at most 1 replan, and
  a responder without tools (1 call reserved).
* Supervisor: a supervisor without shop tools delegating to four specialists (OrderAgent,
  RefundAgent, SalesAgent, SupportAgent) with tool subsets; at most 6 delegations, 4 calls
  per specialist, 15 supervisor turns, 1 call reserved for the supervisor.

## Known limitations

* The shop is simulated: a small synthetic database, fixed clock, no real payments or users.
* Tasks are single turn: the agent cannot ask the customer a question.
* Only small local models (Qwen2.5 7B and 14B, Q4_K_M quantized) on one consumer GPU.
* Banglish messages were written by the authors, not collected from real customers.
* Identity checks are scored only through trap tasks: no tool checks ownership, and an
  agent that skips the phone check loses points only in the identity trap tasks, where
  the phone does not match.
* Tools are policy permissive by design, so the benchmark measures whether the agent
  enforces policy, not whether it can work around tool guards.
* Energy is GPU board energy only (no CPU or system), measured with NVML with about 10
  percent uncertainty (lag corrected sampled power against the energy counter; see
  docs/method_notes.md).
