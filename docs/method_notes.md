# Method notes

Design decisions of the benchmark harness, frozen after pilot3 (2026-10-09).
The file or setting each value comes from is given in brackets.

## Models and serving

* Ollama 0.35.1, OpenAI compatible endpoint `http://localhost:11434/v1` [config.yaml].
* Two model tags built from Modelfiles [models/]:
  * `qwen2.5-7b-8k` from `qwen2.5:7b` (Ollama IDs 30edf3b19f51 for the tag, 845dbda0ea48 for the base).
  * `qwen2.5-14b-8k` from `qwen2.5:14b` (Ollama IDs 3b962add16b4 for the tag, 7cdf5a0187d5 for the base).
* Context size 8192 tokens for both (`PARAMETER num_ctx 8192`).
* GPU: NVIDIA GeForce RTX 3060, 12 GB, driver 616.92. GPU placement is recorded per run; both models ran 100% on GPU in pilot3.
* Python 3.11.9 on Windows 11.

## Sampling, seeds and limits per call [config.yaml]

* Temperature 0.3 for every call in every role.
* Seeds 0, 1, 2. Run i of a task uses seed i, sent with every request. The main run uses all three seeds.
* At most 1024 output tokens per call (`max_tokens_per_call`). Without a cap a runaway generation runs to the request timeout, and Ollama keeps generating after the client gives up, which slows the next run.
* Request timeout 180 s. Client retries are disabled, so every request the model sees is counted.

## Shared call budget

* 20 LLM calls per task (`max_llm_calls_per_task`), shared by all roles of an architecture and enforced in the one LLM client every architecture uses [agents/llm.py]. Failed requests count against it.
* When the budget is spent, the next call is refused, the run ends with stop reason `budget_exceeded` and an empty final reply, and the task scores as failed.
* Plan and Execute holds back 1 call for its responder and Supervisor holds back 1 call for the supervisor, so they stop delegating or executing early instead of losing the final reply.
* An endpoint error or timeout ends the run with stop reason `llm_error` and an empty reply.

## Architectures

All three use the same client, the same shared system prompt, the same 12 tools (counted over all roles), and the same budget. Each architecture only appends its own role text after the shared prompt.

ReAct [agents/react.py]

* One agent with all 12 tools in one tool calling loop.
* At most 15 LLM calls (iterations). The run ends at the first reply without a tool call. Reaching 15 ends the run with stop reason `max_iterations` and an empty reply.

Plan and Execute [agents/plan_execute.py]

* Planner, no tools: writes a JSON plan of goals in plain words, with no tool names, arguments or IDs. The prompt asks for at most 6 steps; the code does not cut longer plans. Up to 2 attempts when the JSON is invalid; after 2 invalid plans a single fallback step is used ("Handle the customer's request as the policy requires.").
* Executor: each step is a fresh conversation with all 12 tools and at most 4 LLM calls. It sees the customer request, the plan and the earlier step results as plain sentences.
* A step fails when the executor gives no report within its calls, when the report contains the uppercase word FAILED or starts with "failed" in any case, or when its last tool call returned an error.
* At most 1 replan per task. Once the replan is used, a further failed step ends execution and goes to the responder.
* Responder, no tools: writes the reply from the step results, as plain text, with the customer's message placed last. It has no language instruction of its own and relies on the shared rule.

Supervisor [agents/supervisor.py]

* The supervisor has no shop tools, only 4 delegation tools, one per specialist, each taking one instruction string.
* Specialists and their tools:
  * OrderAgent: get_customer, get_order, list_customer_orders, cancel_order, update_order, apply_coupon.
  * RefundAgent: get_order, issue_refund, lookup_policy.
  * SalesAgent: search_products, check_stock, create_quote.
  * SupportAgent: get_order, route_ticket, lookup_policy.
* A specialist sees only the instruction, runs at most 4 LLM calls, and returns a text report.
* At most 6 delegations and at most 15 supervisor turns per task. A delegation is refused when it would leave no call for the supervisor.
* The supervisor is told to pass the customer's request and every identifier word for word, never to paste policy text into an instruction, and never to tell a specialist which tool to call.

## Shared system prompt [agents/prompts.py]

Identical for every role of every architecture, in this order:

1. Role: customer service assistant for Dokan, a small online shop in Bangladesh.
2. Fixed time 2026-10-01 10:00 Asia/Dhaka (UTC+06:00), to be used for every age and deadline instead of the real clock.
3. Language rule: reply in plain English to English, in Banglish to Banglish (Bangla in Latin letters), only in Latin letters, never in Bangla script, Chinese or any other script.
4. The full policy [env/policy.md] inside `<policy>` tags.
5. Conversation rules: a single message conversation, so never ask the customer to confirm or for more information; never guess an order ID or product ID; only state actions actually performed; reply in the customer's language style.

`prompt_sha256` covers exactly this shared text. Role text and per role messages are covered by `harness_sha256` (see Hashes).

## Environment and tools [env/]

* SQLite shop built from a seeded generator; every run starts from a fresh copy of the seed database. The shop clock is fixed at 2026-10-01 10:00 Asia/Dhaka.
* 12 tools. They check only that inputs are well formed and that referenced IDs exist. They never enforce policy: a refund outside the window, on an undelivered order, above the order total or to the wrong method, a cancellation of a shipped order, and any bulk discount all succeed. Policy compliance is what is measured.
* Business state checks that return an error (state, not policy): a second refund on the same order, cancelling an order that is already cancelled, and a second coupon on the same order.
* Ownership is not checked by any tool. Matching the customer's phone number to the order is a policy rule (Identity) that the agent must apply; identity trap tasks forbid every write tool, so acting on a mismatched phone is a violation.

## Tool call handling and counters [agents/llm.py]

* Native tool calls are used. If a response has none, JSON tool calls written in the message text are parsed and executed (`text_tool_calls`).
* A call with an unknown tool name or invalid arguments is not executed; the model receives an error and `malformed_tool_calls` is incremented.
* `dropped_tool_calls` counts responses with output tokens but no content and no tool call (finish reason stop or length). Ollama discards a generated tool call whose name matches no supplied tool, together with its text. Such a response is not retried; the agent treats it as an empty reply.

## Scoring [eval/scorer.py]

success = state_match and output_match and not policy_violation and not empty_reply.

* state_match: the gold write actions are replayed on a fresh shop and the whole database is compared with the agent's final state. Free text columns (refund reason, ticket summary) are ignored.
* output_match: every required output string appears in the reply, ignoring case and thousands separators.
* policy_violation: a forbidden write tool call that succeeded, optionally only when an argument condition holds. A call that returned an error is not a violation.
* empty_reply: an empty or whitespace reply fails the task, even when doing nothing was correct, because every task needs a reply to the customer.
* wrong_script, recorded per run [eval/script_check.py]: true when the reply to an English or Banglish task contains any non Latin letter or non ASCII digit. It is reported separately and does not affect success.
* Task file validation [eval/validate_tasks.py]: every gold action replays without error, the oracle (gold actions plus required outputs) passes 100%, the same oracle with an empty reply passes 0%, and a no action agent passes only the do nothing traps.

## Model reset before every run [agents/llm.py, run.py]

* Before every run the model is unloaded (keep_alive 0, waiting until Ollama reports it gone) and reloaded with a fixed warm up request ("hi", 1 output token, temperature 0, seed 0).
* Reason: Ollama reuses cached prompt prefixes across requests, and a different cache state changes the output even with the same seed. The reset makes every run start from the same state, independent of run order.
* The reset is outside the timed and energy measured block; its duration is recorded as `reset_s`. The model is also reset after an LLM timeout. Before switching models all other models are unloaded.
* Run order: model (outer), architecture, task, seed. Each run appends one line to `runs.jsonl` (flushed and synced) and writes its full trace.

## Time and energy [run.py, telemetry/energy.py]

* `wall_time_s`: duration of the agent run only, excluding reset and scoring. `llm_latency_s`: sum of request latencies. `run_total_s`: everything for the run, reset and scoring included; used for time projections.
* Energy is GPU board energy from NVML on GPU 0 only, not CPU or whole system energy.
* Sampled energy (`energy_wh`): board power read every 100 ms on a background thread during the agent run and integrated with the trapezoid rule.
* Idle power (`idle_power_w` in meta.json): mean sampled board power over 30 s, measured once per run directory after a 5 s settle, with the first model loaded and no request running.
* Counter energy (`energy_counter_wh`): the change in the NVML cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption) over the same block. Stored as null when the driver does not expose the counter.
* Net energy: the energy minus idle_power_w times the measured run duration, so it is the energy above idle. Recorded for both sources: `net_energy_counter_wh` from the counter and `net_energy_wh` from the samples.
* The primary energy figure is `net_energy_counter_wh`. The sampled `net_energy_wh` is the cross check. In pilot3 the counter read above the sampled value by a nearly constant amount per run (median 81 J, range 66 to 94 J), not correlated with run duration (r = 0.07 in magnitude); as a ratio this is about 16% for short 7B ReAct runs and 3% for long 14B runs.
* Runs recorded before the counter fields were renamed (pilot3 and earlier) store the counter as `counter_energy_wh`; the summary derives their net counter value with wall_time_s as the duration and marks it as derived.
* `peak_vram_mib` is recorded per run.

## Hashes

Hashes are SHA256 over the decoded UTF8 text, so CRLF and LF checkouts agree. Each is stored in meta.json and in every runs.jsonl line. The runner refuses to resume a run directory if any hash or any fairness setting in config.yaml (call budget, token cap, temperature, seeds) changed.

* policy_sha256: `108c9136b218c6b16ced3ae58f5f5fb4ff053baa4ca5b5889041b98f3f62aade` (env/policy.md).
* prompt_sha256: `6c5bdb1a99ca4ee6d0b8ac63194c33d495f2e400ca940d8505d5a64c9efbeb29` (the shared system prompt only).
* harness_sha256: `a1f94412682cb3ad393f81964c861f2d4f82db08c6fc46b8433a1cffee87678e` [agents/harness.py]. One hash over the shared prompt, every role prompt and fixed per role instruction of the three architectures (planner, executor, responder, supervisor, specialists), the specialist and shop tool schemas, the source of eval/scorer.py, the call budget, token cap and temperature from config.yaml, and the iteration, step, delegation and replan limits. Seeds and the model list are not part of it.
* tasks_sha256: to be filled in once all 80 tasks are merged into tasks/tasks.jsonl. The 10 task pilot file is `74fb7fde4d69428465e5566d30f332331c8e800fcfd0fc590ceb8e3ffc2a792d`.

## Change after pilot3

The Plan and Execute responder message lost its own English or Banglish instruction after pilot3, so every architecture now relies only on the shared language rule. pilot3 Plan and Execute runs still had that instruction. The change is outside the shared prompt, so prompt_sha256 did not change. harness_sha256 was introduced afterwards and is not recorded for pilot3.
