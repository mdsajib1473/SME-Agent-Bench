"""Smoke run of every architecture on the same few tasks.

Checks that Ollama is reachable and the model exists, resets the model, measures
idle GPU power, then runs and scores each architecture on the same tasks with
the same seed. Before every run the model is unloaded and reloaded (outside the
timed and measured block) so each run starts from the same prompt-cache state.
Prints per-run results, a comparison table, the trace of the
first task for each architecture, and peak VRAM. Full results go to
results/smoke_architectures.json.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\smoke_react.py
    .venv\\Scripts\\python.exe scripts\\smoke_react.py --archs react supervisor --tasks REF-E-01
"""

import argparse
import json
import sys
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import LLMClient, load_config, native_base, ollama_get, reset_model_state
from agents.registry import ARCHITECTURES, build_agent
from env.shop import Shop
from eval import scorer
from telemetry.energy import EnergyMeter, measure_idle_power

MODEL = "qwen2.5-7b-8k"
DEFAULT_TASKS = ("REF-E-01", "ORD-H-01", "CMP-M-01")
RESULTS_PATH = ROOT / "results" / "smoke_architectures.json"
TRACE_TEXT_LIMIT = 300


def check_ollama(native_url, model):
    """Return an error message, or None when the server and model are ready."""
    try:
        tags = ollama_get(f"{native_url}/api/tags", timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return (
            f"Ollama is not reachable at {native_url} ({error}). Start it with"
            " 'ollama serve' or launch the Ollama app, then retry."
        )
    names = {
        str(entry.get(key, "")).removesuffix(":latest")
        for entry in tags.get("models", [])
        for key in ("name", "model")
    }
    if model not in names:
        return (
            f"model {model} is not installed in Ollama. Create it with"
            f" 'ollama create {model} -f models\\{model}.Modelfile'."
        )
    return None


def resident_vram_gib(native_url, model):
    try:
        running = ollama_get(f"{native_url}/api/ps", timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    for entry in running.get("models", []):
        if str(entry.get("name", "")).removesuffix(":latest") == model:
            size = entry.get("size_vram")
            return size / 1024**3 if size else None
    return None


def short(value, limit=TRACE_TEXT_LIMIT):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\r", "").replace("\n", " | ")
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def print_trace(result):
    meta = result.metadata
    print(f"--- trace: {meta['architecture']} on {meta['task_id']} ---")
    for event in result.trace:
        stamp = f"[{event['t']:6.2f}s]"
        agent = event.get("agent") or ""
        kind = event["type"]
        if kind == "message":
            content = event["content"]
            if event["role"] == "system":
                content = f"({len(content)} chars, policy sha256 {meta['policy_sha256'][:12]})"
            print(f"{stamp} {agent} {event['role'].upper()}: {short(content)}")
        elif kind == "llm_call":
            print(
                f"{stamp} {agent} LLM call {event['call_index']}: {event['latency_s']:.2f}s,"
                f" {event['prompt_tokens']}+{event['completion_tokens']} tokens"
            )
            if event["content"]:
                print(f"           says: {short(event['content'])}")
            for call in event["tool_calls"]:
                flag = f" MALFORMED ({call['error']})" if call["error"] else ""
                print(f"           -> {call['source']} {call['name']}{short(call['raw_arguments'])}{flag}")
        elif kind == "tool_call":
            print(f"{stamp} {agent} TOOL {event['name']}: {short(event['result'])}")
        elif kind == "plan":
            status = "valid" if event["valid"] else f"INVALID ({event['error']})"
            print(f"{stamp} {event['kind'].upper()} attempt {event['attempt']} {status}: {event['steps']}")
        elif kind == "step":
            print(f"{stamp} STEP {event['index']} {event['status'].upper()} ({event['llm_calls']} calls): {short(event['report'])}")
        elif kind == "delegation":
            if event["refused"]:
                print(f"{stamp} DELEGATION REFUSED to {event['specialist']}: {event['refused']}")
            else:
                print(f"{stamp} DELEGATION {event['index']} to {event['specialist']} ({event['llm_calls']} calls)")
                print(f"           instruction: {short(event['instruction'])}")
                print(f"           report: {short(event['report'])}")
        else:
            print(f"{stamp} {kind.upper()}: {short({k: v for k, v in event.items() if k not in ('t', 'type')})}")
    print(f"FINAL REPLY: {short(result.final_reply, 600)}")
    print()


def run_one(agent, task, llm, seed, idle_w, reset):
    """Resets happen outside the timed and energy-measured block."""
    resets = {"reset_s": None, "post_timeout_reset_s": None}
    if reset:
        resets["reset_s"] = reset_model_state(llm.model, llm.config)
    with Shop() as shop:
        with EnergyMeter() as meter:
            result = agent.run(task, shop, llm, seed)
        score = scorer.score(task, shop.snapshot(), result.final_reply, result.tool_calls)
    if result.metadata["llm_timeout"]:
        resets["post_timeout_reset_s"] = reset_model_state(llm.model, llm.config)
    energy = {
        "energy_wh": meter.energy_wh,
        "net_energy_wh": meter.net_energy_wh(idle_w),
        "counter_energy_wh": meter.counter_energy_wh,
        "mean_power_w": meter.mean_power_w,
        "peak_vram_mib": meter.peak_vram_mib,
        "samples": len(meter.samples),
    }
    return result, score, energy, resets


def print_run(task, result, score, energy, resets):
    meta = result.metadata
    print(
        f"{meta['architecture']} / {task['task_id']} ({task['category']}, {task['difficulty']},"
        f" trap={task['is_trap']}): success={score['success']} stop={meta['stop_reason']}"
        f" reset={fmt(resets['reset_s'], '.2f')}s"
    )
    if resets["post_timeout_reset_s"] is not None:
        print(f"  LLM timeout; reset after it took {resets['post_timeout_reset_s']:.2f}s")
    if score["diff"]:
        print(f"  diff: {score['diff']}")
    if score["missing_outputs"]:
        print(f"  missing outputs: {score['missing_outputs']}")
    if score["violation_details"]:
        print(f"  violations: {score['violation_details']}")
    extras = {k: meta[k] for k in ("plan_valid", "planner_calls", "replans", "steps_run",
                                   "steps_failed", "delegations", "delegations_refused") if k in meta}
    if extras:
        print(f"  {' '.join(f'{k}={v}' for k, v in extras.items())}")


COLUMNS = (
    ("arch", 12), ("task", 9), ("ok", 3), ("state", 5), ("out", 3), ("viol", 4), ("stop", 15),
    ("llm", 3), ("tools", 5), ("malf", 4), ("text", 4), ("prompt", 6), ("compl", 5),
    ("lat_s", 6), ("wall_s", 6), ("net_Wh", 6),
)


def yes(value):
    return "Y" if value else "-"


def print_table(rows):
    header = " ".join(name.ljust(width) for name, width in COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        meta, score, energy = row["metadata"], row["score"], row["energy"]
        cells = (
            meta["architecture"], meta["task_id"], yes(score["success"]), yes(score["state_match"]),
            yes(score["output_match"]), "Y" if score["policy_violation"] else "-", meta["stop_reason"],
            meta["llm_calls"], meta["tool_calls"], meta["malformed_tool_calls"], meta["text_tool_calls"],
            meta["prompt_tokens"], meta["completion_tokens"], f"{meta['llm_latency_s']:.1f}",
            f"{meta['wall_time_s']:.1f}", fmt(energy["net_energy_wh"], ".3f"),
        )
        print(" ".join(str(cell).ljust(width) for cell, (_, width) in zip(cells, COLUMNS)))
    print()

    print("per architecture totals")
    print("arch         success  llm  tools  malf  text  prompt_tok  compl_tok  lat_s  net_Wh")
    for arch in dict.fromkeys(row["metadata"]["architecture"] for row in rows):
        mine = [row for row in rows if row["metadata"]["architecture"] == arch]
        total = lambda key: sum(row["metadata"][key] for row in mine)
        energies = [row["energy"]["net_energy_wh"] for row in mine]
        net = None if any(e is None for e in energies) else sum(energies)
        print(
            f"{arch:<12} {sum(row['score']['success'] for row in mine)}/{len(mine):<6}"
            f" {total('llm_calls'):<4} {total('tool_calls'):<6} {total('malformed_tool_calls'):<5}"
            f" {total('text_tool_calls'):<5} {total('prompt_tokens'):<11} {total('completion_tokens'):<10}"
            f" {total('llm_latency_s'):<6.1f} {fmt(net, '.3f')}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--archs", nargs="+", default=list(ARCHITECTURES), choices=list(ARCHITECTURES))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--idle-seconds", type=float, default=30)
    parser.add_argument("--no-reset", action="store_true",
                        help="skip the model reset before each run (for comparison only)")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    config = load_config()
    native_url = native_base(config["ollama_base_url"])
    seed = config["seeds"][0]

    problem = check_ollama(native_url, args.model)
    if problem:
        print(f"STOP: {problem}")
        return 1

    tasks_by_id = {task["task_id"]: task for task in scorer.load_tasks()}
    unknown = [task_id for task_id in args.tasks if task_id not in tasks_by_id]
    if unknown:
        print(f"STOP: unknown task ids {unknown}; available: {sorted(tasks_by_id)}")
        return 1

    print(f"model {args.model}, seed {seed}, budget {config['max_llm_calls_per_task']} calls per task")
    reset_s = reset_model_state(args.model, config)
    print(f"initial model reset: {reset_s:.2f} s")
    print(f"measuring idle GPU power for {args.idle_seconds:.0f} s with the model resident...")
    idle_w = measure_idle_power(seconds=args.idle_seconds)
    print(f"idle power: {fmt(idle_w, '.1f')} W")
    print()

    llm = LLMClient(args.model, config=config)
    rows = []
    first_results = []
    peak_vram_mib = None
    for arch in args.archs:
        agent = build_agent(arch)
        for index, task_id in enumerate(args.tasks):
            task = tasks_by_id[task_id]
            result, score, energy, resets = run_one(
                agent, task, llm, seed, idle_w, reset=not args.no_reset
            )
            print_run(task, result, score, energy, resets)
            if energy["peak_vram_mib"] is not None:
                peak_vram_mib = max(peak_vram_mib or 0, energy["peak_vram_mib"])
            if index == 0:
                first_results.append(result)
            rows.append({"metadata": result.metadata, "score": score, "energy": energy,
                         "resets": resets, "final_reply": result.final_reply,
                         "trace": result.trace})
    print()
    reset_times = [row["resets"]["reset_s"] for row in rows if row["resets"]["reset_s"] is not None]
    if reset_times:
        print(
            f"model reset per run: mean {sum(reset_times) / len(reset_times):.2f} s,"
            f" min {min(reset_times):.2f} s, max {max(reset_times):.2f} s over {len(reset_times)} resets"
        )
        print()

    for result in first_results:
        print_trace(result)

    print_table(rows)
    hashes = {row["metadata"]["policy_sha256"] for row in rows}
    print(f"policy sha256 identical across runs: {len(hashes) == 1} ({next(iter(hashes))[:16]})")
    print(f"peak VRAM used (whole device): {fmt(peak_vram_mib, '.0f')} MiB")
    model_vram = resident_vram_gib(native_url, args.model)
    print(f"model resident VRAM (ollama ps): {fmt(model_vram, '.2f')} GiB")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"model": args.model, "seed": seed, "idle_power_w": idle_w,
             "peak_vram_mib": peak_vram_mib, "runs": rows},
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"full results: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
