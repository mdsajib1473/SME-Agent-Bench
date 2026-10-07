# sme-agent-bench

Project rules. These apply to every session in this repository.

## Rules

- Never run git commands (no add, commit, push, branch).
- No emojis and no em dashes anywhere: code, comments, markdown, logs, or output. Plain punctuation only.
- Minimal comments; only where logic is not obvious.
- Log only meaningful state changes, not every step.
- Target OS is Windows 11. Use pathlib for paths. Commands must work in PowerShell.
- All randomness must be seeded. The shop environment uses a fixed clock of 2026-10-01 10:00 (Asia/Dhaka); never call datetime.now() inside the environment.
- Tools are policy-permissive: they validate only that inputs are well-formed and IDs exist. They never enforce business policy. Enforcing policy is the agent's job, and violations are what we measure.
- Every architecture must use the same LLM client, the same tools, the same policy text, and the same call budget. Fairness between arms is the top priority.

## Environment notes

This machine has several Python installations (3.10, 3.12, and a 32-bit 3.14 as the
py launcher default). Use only Python 3.11, through the project virtual environment.

Create the virtual environment with:

    py -3.11 -m venv .venv

Always call tools through `.venv\Scripts\python.exe`, never bare `python`.

If `ollama` is not on PATH in the current session, use
`%LOCALAPPDATA%\Programs\Ollama\ollama.exe`.

## Known limitations

- get_order computes the subtotal as total minus delivery, so after a coupon is
  applied a later quantity change starts from the discounted amount. No current
  task combines the two. Do not fix it.

## Layout

    env/        shop database, seed data, policy, tools
    tasks/      task files
    eval/       scorer, task validator
    agents/     LLM client, prompts, three architectures
    telemetry/  GPU energy sampling
    analysis/   statistics, figures, tables
    results/    run outputs (gitignored)
    scripts/    operational scripts
    models/     Modelfile variants
    tests/      tests
