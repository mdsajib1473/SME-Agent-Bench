# sme-agent-bench

A benchmark of LLM agent architectures for customer service in a small online shop.
Three architectures (ReAct, Plan and Execute, Supervisor) run on the same local
models, with the same tools, the same written policy and the same call budget, and
are compared on task success, policy violations, time and GPU energy.

Repository link for review: anonymous link (to be added).

## What the benchmark is

* **Environment.** A simulated Bangladeshi online shop called Dokan, stored in SQLite
  [env/]. The seed database holds 40 customers, 30 products, 120 orders and 4 coupons,
  generated with a fixed seed. The shop clock is fixed at 2026-10-01 10:00 Asia/Dhaka.
* **Tools.** 12 tools: get_customer, get_order, list_customer_orders, search_products,
  check_stock, cancel_order, update_order, issue_refund, create_quote, route_ticket,
  lookup_policy, apply_coupon [env/tools.py]. The tools only check that inputs are well
  formed and that IDs exist. They never enforce the policy, so a policy breach succeeds
  and is counted.
* **Policy.** A written shop policy in ten sections (definitions, identity, cancellation,
  address change, quantity change, refunds, delivery charge, bulk quotes, coupons,
  complaint routing) [env/policy.md]. The full text is part of every system prompt.
* **Tasks.** 80 single turn customer messages in English or Banglish (Bangla written in
  Latin letters), in five categories, three difficulty levels, with trap tasks where the
  correct behaviour is to refuse or to do something other than what the customer asks
  [tasks/]. All 80 tasks were reviewed by two authors and are in tasks/tasks.jsonl.
  tasks/tasks_draft.jsonl keeps the 70 drafts as they were before review.
* **Scoring.** By the final state of the shop database, not by the conversation: the gold
  actions are replayed on a fresh shop and the whole database is compared with the
  agent's final state. A task succeeds when the state matches, every required value
  appears in the reply, no forbidden action succeeded, and the reply is not empty
  [eval/scorer.py].

See docs/benchmark_card.md for a one page summary and docs/method_notes.md for every
design decision.

## Hardware and software used

* CPU AMD Ryzen 7 7700, GPU NVIDIA GeForce RTX 3060 12 GB (driver 616.92), Windows 11.
* Python 3.11.9.
* Ollama 0.40.2 for the main run. The runner records the Ollama version of every
  run in results/<tag>/meta.json; check it before comparing runs.
* Models, both Q4_K_M quantized as distributed by Ollama:
  * `qwen2.5-7b-8k`, built from `qwen2.5:7b` (Ollama IDs 30edf3b19f51 for the tag, 845dbda0ea48 for the base)
  * `qwen2.5-14b-8k`, built from `qwen2.5:14b` (Ollama IDs 3b962add16b4 for the tag, 7cdf5a0187d5 for the base)

The 8k context tags are created from the Modelfiles in models/, which contain only the
base model and the context size:

    FROM qwen2.5:7b
    PARAMETER num_ctx 8192

Create them with:

    ollama pull qwen2.5:7b
    ollama pull qwen2.5:14b
    ollama create qwen2.5-7b-8k -f models/qwen2.5-7b-8k.Modelfile
    ollama create qwen2.5-14b-8k -f models/qwen2.5-14b-8k.Modelfile

If `ollama` is not on PATH on Windows, use `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

## Installation

Python 3.11 is required. In PowerShell:

    py -3.11 -m venv .venv
    .venv\Scripts\python.exe -m pip install -r requirements.txt

On Linux or macOS use `python3.11 -m venv .venv` and `.venv/bin/python`. Always call
Python through the virtual environment. The versions used for the paper were openai
3.24.0, nvidia-ml-py 13.615.71, PyYAML 6.0.3, pandas 3.0.6, numpy 2.4.6, scipy 1.17.1,
statsmodels 0.15.0, matplotlib 3.11.2, pytest 9.1.1 and tqdm 4.70.1.

Energy is read from the GPU through NVML (nvidia-ml-py). Without an NVIDIA GPU the runs
still work and the energy fields stay empty.

## Reproducing everything

One command runs the task validator, the unit tests, the full experiment and the
analysis, and stops with a message if Ollama or a model is missing:

    powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1

or on Linux or macOS:

    bash scripts/reproduce_all.sh

The steps one by one (PowerShell; on Linux use `.venv/bin/python`):

* Validate the task file: `.venv\Scripts\python.exe eval\validate_tasks.py`
* Unit tests (no GPU, no Ollama needed): `.venv\Scripts\python.exe -m pytest -q`
* Full experiment, 2 models x 3 architectures x 80 tasks x 3 seeds = 1,440 runs:
  `.venv\Scripts\python.exe run.py --tag main --runs 3 --resume`
  (resumable: rerun the same command after an interruption)
* All tables, figures and sheets: `.venv\Scripts\python.exe -m analysis.run_all --tag main`
* Run overview and time projection: `.venv\Scripts\python.exe scripts\summarize_run.py --tag main`

Each table and figure of the paper comes from the analysis command above and is written
to results/main/analysis/:

* Main results table: table_main_results.tex (data in main_results.csv)
* McNemar tests: table_mcnemar.tex (data in mcnemar.csv)
* Cost against accuracy: fig_cost_accuracy.pdf and .png
* Success by category: fig_category.pdf and .png
* Success by language: fig_language.pdf and .png
* Breakdowns by category, difficulty, language and trap status: breakdowns.csv

The energy calibration behind the measurement settings (settle wait and power lag
correction, see docs/method_notes.md) is reproduced with
`.venv\Scripts\python.exe scripts\calibrate_energy.py --settle` and
`.venv\Scripts\python.exe scripts\measure_power_lag.py --real`; both need the GPU and
write to results/calibration/.

## Expected runtime

* Full experiment: about 15 GPU hours on the hardware above (model reset before every run,
  a 5 s settle wait and periodic idle measurements included). The 14B model takes about
  two thirds of it.
* Analysis: a few minutes on a CPU.
* Validator and unit tests: under a minute.

## Output folders

* results/<tag>/runs.jsonl: one line per run with success, scores, counters, tokens, wall
  time, energy and hashes.
* results/<tag>/meta.json: configuration, hashes, Ollama version, GPU, model placement and
  every idle power measurement.
* results/<tag>/traces/: one JSON file per run with every message, LLM call, tool call
  and result, the score details and the final reply.
* results/<tag>/analysis/: tables (CSV and LaTeX), figures (PDF and PNG), the automatic
  failure signals (failures_auto.csv), the failure labeling sheets with their guide, and
  summary.md, a plain language summary with the limits of the run.
* results/calibration/: raw data of the energy calibration.

The results folder is not part of the repository.

## How the tasks were created

The tasks were drafted with LLM assistance and then reviewed by the authors. Each draft
states the customer message, the gold write actions, the values the reply must contain,
the forbidden actions and a note that explains the expected outcome from the policy and
the seed data. Every draft is checked automatically by eval/validate_tasks.py: the gold
actions must replay without error, an oracle that performs exactly the gold actions must
pass every task, the same oracle with an empty reply must fail every task, and an agent
that does nothing may pass only the traps whose correct behaviour is to do nothing.

The review record is kept in tasks/:

* review_sheet.csv is the blank review sheet: every task with its message, notes, gold
  actions, required outputs and forbidden actions, plus empty review columns.
* review_log.csv is the record of the review: one row per task with a reviewer code (R1
  or R2, never a name), the approval decision and the reviewer comment.
* review_pending.md lists rows still unresolved after the merge (none) and how each row
  marked not approved was resolved.
* db_snapshot.csv gives the reviewer a plain English view of the seed database rows each
  task depends on (built by scripts/task_generation/build_snapshot.py).
* review_replacements.csv holds replacement drafts for tasks withdrawn during review.
* distribution_report.md summarises the distribution of the merged set.

Approved tasks are in tasks/tasks.jsonl. Banglish messages were written by the authors.

## License

See LICENSE.
