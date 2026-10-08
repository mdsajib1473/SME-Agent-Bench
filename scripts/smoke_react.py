"""Smoke run of the ReAct agent on a few tasks.

Checks that Ollama is reachable and the model exists, warms the model, measures
idle GPU power, then runs and scores the tasks and prints per-task results, the
trace of the first task, and peak VRAM. Full results go to results/smoke_react.json.

Usage (PowerShell):
    .venv\\Scripts\\python.exe scripts\\smoke_react.py
    .venv\\Scripts\\python.exe scripts\\smoke_react.py --tasks REF-E-01 ORD-H-01 CMP-M-01
"""

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm import LLMClient, load_config
from agents.react import ReActAgent
from env.shop import Shop
from eval import scorer
from telemetry.energy import EnergyMeter, measure_idle_power

MODEL = "qwen2.5-7b-8k"
DEFAULT_TASKS = ("REF-E-01", "ORD-H-01", "CMP-M-01")
RESULTS_PATH = ROOT / "results" / "smoke_react.json"
TRACE_TEXT_LIMIT = 400


def native_base(base_url):
    return base_url.rstrip("/").removesuffix("/v1")


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_ollama(native_url, model):
    """Return an error message, or None when the server and model are ready."""
    try:
        tags = get_json(f"{native_url}/api/tags", timeout=10)
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


def warm_up(native_url, model, timeout):
    post_json(
        f"{native_url}/api/generate",
        {"model": model, "prompt": "hi", "stream": False, "options": {"num_predict": 1, "seed": 0}},
        timeout,
    )


def resident_vram_gib(native_url, model):
    try:
        running = get_json(f"{native_url}/api/ps", timeout=10)
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    for entry in running.get("models", []):
        if str(entry.get("name", "")).removesuffix(":latest") == model:
            size = entry.get("size_vram")
            return size / 1024**3 if size else None
    return None


def short(value, limit=TRACE_TEXT_LIMIT):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\r", "")
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def fmt(value, spec):
    return "n/a" if value is None else format(value, spec)


def print_trace(result):
    print(f"--- trace of {result.metadata['task_id']} ---")
    for event in result.trace:
        stamp = f"[{event['t']:7.2f}s]"
        if event["type"] == "message":
            content = event["content"]
            if event["role"] == "system":
                content = f"({len(content)} chars, policy sha256 {result.metadata['policy_sha256'][:12]})"
            print(f"{stamp} {event['role'].upper()}: {short(content)}")
        elif event["type"] == "llm_call":
            print(
                f"{stamp} LLM call {event['call_index']}: {event['latency_s']:.2f}s,"
                f" {event['prompt_tokens']} prompt + {event['completion_tokens']} completion tokens,"
                f" finish={event['finish_reason']}"
            )
            if event["content"]:
                print(f"           ASSISTANT: {short(event['content'])}")
            for call in event["tool_calls"]:
                flag = f" MALFORMED ({call['error']})" if call["error"] else ""
                print(f"           -> {call['source']} call {call['name']}{short(call['raw_arguments'])}{flag}")
        elif event["type"] == "tool_call":
            print(f"{stamp} TOOL {event['name']} ({event['latency_s'] * 1000:.1f} ms): {short(event['result'])}")
    print(f"FINAL REPLY: {result.final_reply}")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--idle-seconds", type=float, default=30)
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
    warm_up(native_url, args.model, config["request_timeout_s"])
    print(f"measuring idle GPU power for {args.idle_seconds:.0f} s with the model resident...")
    idle_w = measure_idle_power(seconds=args.idle_seconds)
    print(f"idle power: {fmt(idle_w, '.1f')} W")
    print()

    llm = LLMClient(args.model, config=config)
    agent = ReActAgent()
    rows = []
    results = []
    peak_vram_mib = None
    for task_id in args.tasks:
        task = tasks_by_id[task_id]
        with Shop() as shop:
            with EnergyMeter() as meter:
                result = agent.run(task, shop, llm, seed)
            score = scorer.score(task, shop.snapshot(), result.final_reply, result.tool_calls)
        if meter.peak_vram_mib is not None:
            peak_vram_mib = max(peak_vram_mib or 0, meter.peak_vram_mib)
        meta = result.metadata
        energy = {
            "energy_wh": meter.energy_wh,
            "net_energy_wh": meter.net_energy_wh(idle_w),
            "counter_energy_wh": meter.counter_energy_wh,
            "mean_power_w": meter.mean_power_w,
            "peak_vram_mib": meter.peak_vram_mib,
            "samples": len(meter.samples),
        }
        results.append((result, score))
        rows.append({"metadata": meta, "score": score, "energy": energy,
                     "final_reply": result.final_reply, "trace": result.trace})

        print(f"{task_id} ({task['category']}, {task['difficulty']}, {task['language']}, trap={task['is_trap']})")
        print(
            f"  success={score['success']} state_match={score['state_match']}"
            f" output_match={score['output_match']} violation={score['policy_violation']}"
            f" stop={meta['stop_reason']}"
        )
        if score["diff"]:
            print(f"  diff: {score['diff']}")
        if score["missing_outputs"]:
            print(f"  missing outputs: {score['missing_outputs']}")
        if score["violation_details"]:
            print(f"  violations: {score['violation_details']}")
        print(
            f"  llm_calls={meta['llm_calls']} tool_calls={meta['tool_calls']}"
            f" malformed={meta['malformed_tool_calls']} text_calls={meta['text_tool_calls']}"
        )
        print(
            f"  tokens: {meta['prompt_tokens']} prompt + {meta['completion_tokens']} completion;"
            f" llm latency {meta['llm_latency_s']:.2f} s; wall {meta['wall_time_s']:.2f} s"
        )
        print(
            f"  energy: {fmt(energy['energy_wh'], '.4f')} Wh total,"
            f" {fmt(energy['net_energy_wh'], '.4f')} Wh above idle,"
            f" NVML counter {fmt(energy['counter_energy_wh'], '.4f')} Wh,"
            f" mean {fmt(energy['mean_power_w'], '.1f')} W over {energy['samples']} samples"
        )
        print()

    print_trace(results[0][0])

    successes = sum(1 for _, score in results if score["success"])
    print(f"success: {successes}/{len(results)}")
    print(f"peak VRAM used (whole device): {fmt(peak_vram_mib, '.0f')} MiB")
    model_vram = resident_vram_gib(native_url, args.model)
    print(f"model resident VRAM (ollama ps): {fmt(model_vram, '.2f')} GiB")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(
            {"model": args.model, "seed": seed, "idle_power_w": idle_w,
             "peak_vram_mib": peak_vram_mib, "tasks": rows},
            ensure_ascii=False, indent=2, default=str,
        ),
        encoding="utf-8",
    )
    print(f"full results: {RESULTS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
