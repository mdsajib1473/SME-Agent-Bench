"""Experiment runner.

Loops model (outer), architecture, task, seed. Before each model it unloads any
other model and warms the target up. Before every run it resets the model
(unload, reload, fixed warm-up) outside the timed and energy-measured block,
and resets again after an LLM timeout. Each run appends one line to
results/<tag>/runs.jsonl (flushed and synced) and writes its full trace to
results/<tag>/traces/. --resume skips combinations already recorded.

Usage (PowerShell):
    .venv\\Scripts\\python.exe run.py --tag main
    .venv\\Scripts\\python.exe run.py --tag pilot --limit 10 --runs 1
    .venv\\Scripts\\python.exe run.py --tag main --resume
"""

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import (
    CONFIG_PATH,
    LLMClient,
    check_ollama,
    load_config,
    model_residency,
    native_base,
    ollama_get,
    reset_model_state,
    running_models,
    unload_model,
    warm_up_model,
)
from agents.prompts import load_policy, policy_sha256
from agents.registry import ARCHITECTURES, build_agent
from env.shop import Shop
from eval import scorer
from telemetry.energy import EnergyMeter, gpu_info, measure_idle_power

RESULTS_DIR = ROOT / "results"
DEFAULT_TASKS_PATH = ROOT / "tasks" / "tasks.jsonl"
DEFAULT_ARCHS = ("react", "plan_execute", "supervisor")
IDLE_SECONDS = 30
IDLE_SETTLE_S = 5
# A resumed run must not mix results produced under different fairness settings.
FAIRNESS_KEYS = ("max_llm_calls_per_task", "max_tokens_per_call", "temperature", "seeds")


def text_sha256(path):
    # Hashes decoded text, so CRLF and LF checkouts agree, as for the policy.
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def run_key(model, arch, task_id, seed):
    return (model, arch, task_id, int(seed))


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(text))


def load_done(runs_path):
    """Keys already recorded. A line cut off by a crash is dropped from the file."""
    if not runs_path.exists():
        return set()
    lines = runs_path.read_text(encoding="utf-8").splitlines()
    good, done = [], set()
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        good.append(line)
        done.add(run_key(row["model"], row["arch"], row["task_id"], row["seed"]))
    if len(good) != len([line for line in lines if line.strip()]):
        backup = runs_path.with_name(runs_path.name + ".bak")
        backup.write_text("\n".join(lines) + "\n", encoding="utf-8")
        runs_path.write_text("".join(f"{line}\n" for line in good), encoding="utf-8")
        tqdm.write(f"dropped {len(lines) - len(good)} unreadable line(s); original kept as {backup.name}")
    return done


def append_line(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def prepare_model(model, config):
    """Unload every other model, then load and warm up this one. Returns seconds."""
    native_url = native_base(config["ollama_base_url"])
    timeout = config["request_timeout_s"]
    started = time.perf_counter()
    for entry in running_models(native_url, timeout):
        name = entry.get("name") or entry.get("model")
        if str(name).removesuffix(":latest") != model.removesuffix(":latest"):
            unload_model(native_url, name, timeout)
    warm_up_model(native_url, model, timeout)
    return time.perf_counter() - started


def ollama_version(config):
    try:
        return ollama_get(f"{native_base(config['ollama_base_url'])}/api/version", 10).get("version")
    except OSError:
        return None


def run_once(agent, arch, task, seed, llm, config, idle_w, hashes, traces_dir):
    started = time.perf_counter()
    reset_s = reset_model_state(llm.model, config)
    residency = model_residency(llm.model, config)

    meter = EnergyMeter()
    result, score, error, error_trace, partial_trace = None, None, None, None, []
    with Shop() as shop:
        run_started = time.perf_counter()
        try:
            with meter:
                result = agent.run(task, shop, llm, seed)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            error_trace = traceback.format_exc()
            partial_trace = getattr(exc, "agent_trace", [])
        run_wall = time.perf_counter() - run_started
        if result is not None:
            try:
                score = scorer.score(task, shop.snapshot(), result.final_reply, result.tool_calls)
            except Exception as exc:
                error = f"scoring failed: {type(exc).__name__}: {exc}"
                error_trace = traceback.format_exc()

    meta = result.metadata if result is not None else {}
    totals = meta if result is not None else llm.snapshot_totals()
    post_timeout_reset_s = None
    if meta.get("llm_timeout"):
        post_timeout_reset_s = reset_model_state(llm.model, config)

    stop_reason = meta.get("stop_reason", "exception")
    trace_name = f"{safe_name(llm.model)}__{arch}__{safe_name(task['task_id'])}__s{seed}.json"
    row = {
        "task_id": task["task_id"],
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "language": task.get("language"),
        "is_trap": task.get("is_trap"),
        "model": llm.model,
        "arch": arch,
        "seed": seed,
        "success": bool(score and score["success"]),
        "state_match": score["state_match"] if score else None,
        "output_match": score["output_match"] if score else None,
        "policy_violation": score["policy_violation"] if score else None,
        "llm_calls": totals["llm_calls"],
        "tool_calls": totals["tool_calls"],
        "malformed_tool_calls": totals["malformed_tool_calls"],
        "text_tool_calls": totals["text_tool_calls"],
        "prompt_tokens": totals["prompt_tokens"],
        "completion_tokens": totals["completion_tokens"],
        "wall_time_s": meta.get("wall_time_s", round(run_wall, 4)),
        "llm_latency_s": round(totals["llm_latency_s"], 4),
        "energy_wh": meter.energy_wh,
        "net_energy_wh": meter.net_energy_wh(idle_w),
        "budget_exceeded": stop_reason == "budget_exceeded",
        "llm_timeout": bool(meta.get("llm_timeout")),
        "stop_reason": stop_reason,
        "delegations": meta.get("delegations") if arch == "supervisor" else None,
        "replans": meta.get("replans") if arch == "plan_execute" else None,
        "policy_sha256": meta.get("policy_sha256", hashes["policy_sha256"]),
        "tasks_sha256": hashes["tasks_sha256"],
        "error": error or meta.get("error"),
        "reset_s": round(reset_s, 3),
        "post_timeout_reset_s": post_timeout_reset_s,
        "run_total_s": None,
        "counter_energy_wh": meter.counter_energy_wh,
        "peak_vram_mib": meter.peak_vram_mib,
        "gpu_fraction": residency["gpu_fraction"] if residency else None,
        "trace_file": f"traces/{trace_name}",
    }
    write_json(
        traces_dir / trace_name,
        {
            "row": row,
            "metadata": meta,
            "score": score,
            "final_reply": result.final_reply if result is not None else None,
            "error_traceback": error_trace,
            "residency": residency,
            "trace": result.trace if result is not None else partial_trace,
        },
    )
    row["run_total_s"] = round(time.perf_counter() - started, 3)
    return row


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument("--models", nargs="+", default=list(config["models"]))
    parser.add_argument("--archs", nargs="+", default=list(DEFAULT_ARCHS), choices=list(ARCHITECTURES))
    parser.add_argument("--runs", type=int, default=3, help="seeds per task, taken from config seeds")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS_PATH)
    parser.add_argument("--limit", type=int, default=None, help="first N tasks only, for pilots")
    parser.add_argument("--tag", required=True, help="run name; output goes to results/<tag>/")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    seeds = list(config["seeds"])[: args.runs]
    if args.runs > len(config["seeds"]):
        print(f"STOP: --runs {args.runs} but config lists only {len(config['seeds'])} seeds")
        return 1
    tasks = scorer.load_tasks(args.tasks)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    hashes = {
        "policy_sha256": policy_sha256(load_policy()),
        "tasks_sha256": text_sha256(args.tasks),
    }

    out_dir = RESULTS_DIR / safe_name(args.tag)
    runs_path = out_dir / "runs.jsonl"
    meta_path = out_dir / "meta.json"
    traces_dir = out_dir / "traces"

    meta = None
    if meta_path.exists() or runs_path.exists():
        if not args.resume:
            print(f"STOP: results/{out_dir.name} already exists. Use --resume to continue it or pick a new --tag.")
            return 1
        if not meta_path.exists():
            print(f"STOP: results/{out_dir.name}/runs.jsonl exists but meta.json is missing; cannot verify hashes.")
            return 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, label in (("policy_sha256", "env/policy.md"), ("tasks_sha256", str(args.tasks))):
            if meta.get(key) != hashes[key]:
                print(
                    f"STOP: {label} changed since this run started ({key} was {meta.get(key)},"
                    f" now {hashes[key]}). Results would mix two versions; start a new --tag instead."
                )
                return 1
        changed = [k for k in FAIRNESS_KEYS if meta["config"].get(k) != config.get(k)]
        if changed:
            print(f"STOP: config.yaml changed since this run started: {', '.join(changed)}. Start a new --tag.")
            return 1

    problem = check_ollama(args.models, config)
    if problem:
        print(f"STOP: {problem}")
        return 1

    done = load_done(runs_path) if args.resume else set()
    plan = [
        (model, arch, task, seed)
        for model in args.models
        for arch in args.archs
        for task in tasks
        for seed in seeds
        if run_key(model, arch, task["task_id"], seed) not in done
    ]
    if not plan:
        print(f"nothing to do: all {len(done)} runs already recorded in results/{out_dir.name}")
        return 0

    traces_dir.mkdir(parents=True, exist_ok=True)
    if meta is None:
        first_model = plan[0][0]
        load_s = prepare_model(first_model, config)
        time.sleep(IDLE_SETTLE_S)
        print(f"measuring idle GPU power for {IDLE_SECONDS} s with {first_model} resident...")
        idle_w = measure_idle_power(seconds=IDLE_SECONDS)
        meta = {
            "tag": args.tag,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "config": config,
            "config_yaml": CONFIG_PATH.read_text(encoding="utf-8"),
            "ollama_version": ollama_version(config),
            "gpu": gpu_info(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "idle_power_w": idle_w,
            "idle_power_state": f"{first_model} resident, no request running",
            "policy_sha256": hashes["policy_sha256"],
            "tasks_sha256": hashes["tasks_sha256"],
            "tasks_path": str(args.tasks),
            "task_ids": [task["task_id"] for task in tasks],
            "seeds": seeds,
            "models": {first_model: {"first_load_s": round(load_s, 3)}},
            "resumes": [],
        }
        write_json(meta_path, meta)
        print(f"idle power {idle_w if idle_w is None else round(idle_w, 2)} W; meta written to results/{out_dir.name}/meta.json")
    else:
        meta["resumes"].append({"at_utc": datetime.now(timezone.utc).isoformat(), "remaining": len(plan)})
        write_json(meta_path, meta)
        print(f"resuming results/{out_dir.name}: {len(done)} done, {len(plan)} to go")
    idle_w = meta["idle_power_w"]

    agents = {arch: build_agent(arch) for arch in args.archs}
    progress = tqdm(total=len(plan), unit="run", dynamic_ncols=True, smoothing=0.05)
    successes = 0
    current_model, llm = None, None
    for model, arch, task, seed in plan:
        if model != current_model:
            load_s = prepare_model(model, config)
            residency = model_residency(model, config)
            info = meta["models"].setdefault(model, {})
            info.update({"load_s": round(load_s, 3), "residency": residency})
            write_json(meta_path, meta)
            tqdm.write(f"loaded {model} in {load_s:.1f} s ({residency['processor'] if residency else 'placement unknown'})")
            if residency and residency["gpu_fraction"] is not None and residency["gpu_fraction"] < 1:
                tqdm.write(f"WARNING: {model} is partly on CPU ({residency['processor']})")
            llm = LLMClient(model, config=config)
            current_model = model

        progress.set_postfix_str(f"{model} {arch} {task['task_id']} s{seed}", refresh=False)
        try:
            row = run_once(agents[arch], arch, task, seed, llm, config, idle_w, hashes, traces_dir)
        except (OSError, RuntimeError) as error:
            # Agent failures are caught inside run_once; anything here is Ollama
            # or disk trouble, so stop rather than record a run that never happened.
            progress.close()
            print(f"STOP: infrastructure failure before {model} {arch} {task['task_id']} s{seed}: {error}")
            print(f"Fix it, then continue with: run.py --tag {args.tag} --resume")
            return 1
        append_line(runs_path, row)
        successes += row["success"]
        if row["error"] and row["stop_reason"] == "exception":
            tqdm.write(f"run failed with an exception: {model} {arch} {task['task_id']} s{seed}: {row['error']}")
        if row["llm_timeout"]:
            tqdm.write(f"LLM timeout: {model} {arch} {task['task_id']} s{seed}; model reset")
        progress.update(1)
    progress.close()
    print(f"done: {len(plan)} runs this session, {successes} successful; results in results/{out_dir.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
