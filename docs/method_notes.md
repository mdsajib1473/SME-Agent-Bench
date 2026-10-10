# Method notes

Design decisions of the benchmark harness, frozen after pilot3 (2026-10-09).
The file or setting each value comes from is given in brackets.

## Models and serving

* Ollama 0.40.2, OpenAI compatible endpoint `http://localhost:11434/v1` [config.yaml].
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
* Settle wait [config.yaml `settle_seconds`, 5 s]: after the reset and its warm up request the runner waits 5 s before the timed and energy measured block. The wait is outside that block and is recorded as `settle_seconds` in meta.json and in every runs.jsonl line. `tail_w_before_run` stores the mean sampled board power over the last 2 s of the wait as a diagnostic; it reads above idle (about 20 to 30 W) because the reported power trails the true power (see Time and energy).
* Reason for the settle wait: after the warm up the GPU stays at about 36 to 40 W for about 3 s. Without a wait, a 10 s block with no request measured about 80 J above idle (79 ± 9 J, 6 blocks), all of it the reset aftermath. With a 5 s wait the same block measured 2.0 ± 0.9 J; 3 s still left 13.5 ± 6.4 J. 5 s is the shortest wait tested (0, 3, 5, 8, 12, 15 s, 8 repeats each) that brings the aftermath under 10 J [results/calibration/energy_settle_*].
* Run order: model (outer), architecture, task, seed. Each run appends one line to `runs.jsonl` (flushed and synced) and writes its full trace.

## Time and energy [run.py, telemetry/energy.py]

* `wall_time_s`: duration of the agent run only, excluding reset, settle wait and scoring. `llm_latency_s`: sum of request latencies. `run_total_s`: everything for the run, reset, settle wait, meter tail and scoring included; used for time projections.
* Energy is GPU board energy from NVML on GPU 0 only, not CPU or whole system energy.
* Reported power lag: on this GPU (Ampere) nvmlDeviceGetPowerUsage returns board power averaged over 1 s (nvidia smi documents power.draw this way for Ampere and newer), and the value refreshes only about every 0.5 s (median 537 ms). Cross correlation against power derived from the energy counter gave a lag of 630 to 850 ms on three 300 token generations and 610 to 710 ms on three real 7B ReAct runs [results/calibration/power_lag_*]. `power_lag_s` in config.yaml is 0.65 s.
* Sampled energy (`energy_wh`): board power read every 100 ms on a background thread. The sample timestamps are shifted back by `power_lag_s`, sampling continues for 1 s after the block ends so the shifted series covers the whole block, and the shifted series is integrated over the block only with the trapezoid rule (cut at the block edges by linear interpolation). `energy_sampled_raw_wh` keeps the unshifted integral over the block, as recorded before the correction. The 1 s tail is outside `wall_time_s`.
* Counter energy (`energy_counter_wh`): the change in the NVML cumulative energy counter (nvmlDeviceGetTotalEnergyConsumption) over the same block, read at the block end. Stored as null when the driver does not expose the counter.
* Idle power: mean sampled board power over 10 s, with the model resident and no request running, after a 5 s settle. It is measured at the start of every session (including a resume), whenever the model or the architecture changes, and every 25 runs. Each run uses the most recent measurement and stores it as `idle_w_used`. Every measurement is listed in meta.json under `idle_measurements` (time, model, architecture, reason, run number in the session, watts); `idle_power_w` in meta.json is the first one.
* Reason for the idle schedule: the idle reading is not stable over a session. In the settle test, measured idle was 11.04 W at the start, the counter then read about 6.2 W at idle for several minutes and returned to about 11.1 W, while the reported power moved by about 2 W. Across sessions idle measured 11.0 to 13.0 W. For a 5 s run, a baseline 2 to 5 W off is 10 to 25 J.
* Net energy: the energy minus `idle_w_used` times the measured block duration, so it is the energy above idle. Recorded for both sources: `net_energy_wh` from the lag corrected samples and `net_energy_counter_wh` from the counter.
* The primary energy figure is `net_energy_wh` (sampled, lag corrected). `net_energy_counter_wh` is the cross check. Reason: the counter is less stable between sessions than the sampled power. The gross counter energy of the same 7B ReAct run on `REF-E-01` was 492 to 506 J in one session and 457 to 459 J in another, about 9 percent apart, while the raw sampled energy of the same runs was 419 to 427 J and 418 to 424 J. The counter idle reading also drifted by about 5 W within a session (above). The uncorrected sampled figure misses the energy of the last lag interval of a run and so reads low by a roughly constant amount per run; the lag correction removes that.
* Pilot3 and earlier used the counter as the primary figure. There the counter read above the uncorrected sampled value by a nearly constant amount per run (median 81 J, range 66 to 94 J), not correlated with run duration (r = 0.07 in magnitude); that gap is mostly the power lag. Runs recorded before the correction have no `energy_sampled_raw_wh`, their `energy_wh` is uncorrected, and the summary says so.
* Runs recorded before the counter fields were renamed (pilot3 and earlier) store the counter as `counter_energy_wh`; the summary derives their net counter value with wall_time_s as the duration and marks it as derived.
* Validation of the correction (2026-10-09): ReAct 7B on `REF-E-01`, seed 0, five separate runner sessions in the normal way (reset, warm up, 5 s settle, measured block). Mean `net_energy_wh` 353.5 J, mean `net_energy_counter_wh` 380.6 J, a difference of 7.1 percent (limit 10 percent); coefficient of variation of `net_energy_wh` 0.88 percent (limit 3 percent). Accepted. Raw sampled gross energy 337 to 347 J, lag corrected gross energy 417 to 425 J, counter gross energy 445 to 451 J [results/energy_validation_1 to 5].
* `peak_vram_mib` is recorded per run.

## Hashes

Hashes are SHA256 over the decoded UTF8 text, so CRLF and LF checkouts agree. Each is stored in meta.json and in every runs.jsonl line. The runner refuses to resume a run directory if any hash, any fairness setting in config.yaml (call budget, token cap, temperature, seeds) or any energy measurement setting (`settle_seconds`, `power_lag_s`) changed. The energy measurement settings are not part of harness_sha256.

* policy_sha256: `afb5e8f6b705211fd126a10d66752257f313a8b203f69f2efab1d26076e8ba5c` (env/policy.md).
* prompt_sha256: `1e21e82ffaad58cdbb6e6fd197609f8813f7ea316b4be14fa78d66b03a32d86e` (the shared system prompt only).
* harness_sha256: `70cf0f7a774f072822a684519a4b42eef9487ce349aa780d0c40ab34e61023a5` [agents/harness.py]. One hash over the shared prompt, every role prompt and fixed per role instruction of the three architectures (planner, executor, responder, supervisor, specialists), the specialist and shop tool schemas, the source of eval/scorer.py, the call budget, token cap and temperature from config.yaml, and the iteration, step, delegation and replan limits. Seeds and the model list are not part of it.
* tasks_sha256: `9d33b0aaf9af811de5430c14f31646844d6b7fb57867d9c90dc5ff473354cdb8` (tasks/tasks.jsonl, all 80 tasks). The 10 task pilot file was `74fb7fde4d69428465e5566d30f332331c8e800fcfd0fc590ceb8e3ffc2a792d`.

## Change after pilot3

The Plan and Execute responder message lost its own English or Banglish instruction after pilot3, so every architecture now relies only on the shared language rule. pilot3 Plan and Execute runs still had that instruction. The change is outside the shared prompt, so prompt_sha256 did not change. harness_sha256 was introduced afterwards and is not recorded for pilot3.

## Change at the task merge (2026-10-11)

The 80 reviewed tasks were merged into tasks/tasks.jsonl before the main run, with one authorized policy edit. env/policy.md gained two additions: the refunds section now states that the refundable amount is the full amount the customer paid, including the delivery charge, and the address change section now states that phone numbers on an order cannot be changed through this channel (refuse and take no action). policy_sha256, prompt_sha256 and harness_sha256 changed as a result; the values above are the new ones. Pilot runs used the earlier policy and the 10 task file and are not reported.
