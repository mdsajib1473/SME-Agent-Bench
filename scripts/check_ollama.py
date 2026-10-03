"""Smoke check for the configured Ollama models.

For each model: force a load, send one chat request carrying a single dummy tool
definition, verify a well-formed tool call comes back, and report load time,
generation throughput, and GPU memory in use.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
SEED = 0

DUMMY_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock",
        "description": "Look up the stock level of a single product by its id.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "Product id, for example P-1001.",
                }
            },
            "required": ["product_id"],
        },
    },
}

PROMPT = "Check the stock level of product P-1001. Use the available tool."


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def native_base(base_url):
    # config holds the OpenAI-compatible path; the native API sits one level up.
    return base_url.rstrip("/").removesuffix("/v1")


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def unload(native_url, model, timeout):
    """Evict the model so the next load time is a true cold start."""
    try:
        post_json(
            f"{native_url}/api/generate",
            {"model": model, "prompt": "", "keep_alive": 0, "stream": False},
            timeout,
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        pass


def measure_load(native_url, model, timeout):
    """Warm the model with a one-token generation; returns load seconds.

    An empty prompt reports load_duration 0, so a single token is generated to
    force a real load and leave the model resident for the timed chat request.
    """
    result = post_json(
        f"{native_url}/api/generate",
        {
            "model": model,
            "prompt": "hi",
            "stream": False,
            "options": {"num_predict": 1, "seed": SEED},
        },
        timeout,
    )
    return result.get("load_duration", 0) / 1e9


def resident_vram_bytes(native_url, model, timeout):
    """Ollama reports names tagged, so compare without the tag."""
    wanted = model.split(":")[0]
    try:
        running = get_json(f"{native_url}/api/ps", timeout)
    except urllib.error.URLError:
        return None
    for entry in running.get("models", []):
        names = {
            str(entry.get(key, "")).split(":")[0] for key in ("name", "model")
        }
        if wanted in names:
            return entry.get("size_vram")
    return None


def gpu_memory():
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        devices = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            devices.append(
                {
                    "name": name,
                    "used_mib": info.used / 1024**2,
                    "total_mib": info.total / 1024**2,
                }
            )
        return devices
    finally:
        pynvml.nvmlShutdown()


def check_model(client, native_url, model, temperature, timeout):
    report = {"model": model, "ok": False}

    unload(native_url, model, timeout)
    try:
        report["load_s"] = measure_load(native_url, model, timeout)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        report["error"] = f"load failed: {error}"
        return report

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            tools=[DUMMY_TOOL],
            temperature=temperature,
            extra_body={"seed": SEED},
        )
    except Exception as error:
        report["error"] = f"chat request failed: {error}"
        return report
    elapsed = time.perf_counter() - started

    report["elapsed_s"] = elapsed
    usage = getattr(response, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
    report["completion_tokens"] = completion_tokens
    report["tokens_per_s"] = (
        completion_tokens / elapsed if completion_tokens and elapsed > 0 else None
    )

    message = response.choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        report["error"] = "no tool call returned"
        report["content"] = (message.content or "")[:200]
        return report

    call = tool_calls[0]
    report["tool_name"] = call.function.name
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, json.JSONDecodeError) as error:
        report["error"] = f"tool arguments are not valid JSON: {error}"
        report["raw_arguments"] = call.function.arguments
        return report

    report["tool_arguments"] = arguments
    if call.function.name != DUMMY_TOOL["function"]["name"]:
        report["error"] = f"unexpected tool name: {call.function.name}"
        return report
    if "product_id" not in arguments:
        report["error"] = "tool call is missing the required product_id argument"
        return report

    report["vram_bytes"] = resident_vram_bytes(native_url, model, timeout)
    report["ok"] = True
    return report


def print_report(report):
    print(f"model: {report['model']}")
    if "load_s" in report:
        print(f"  load time:        {report['load_s']:.2f} s")
    if report.get("elapsed_s") is not None:
        print(f"  request wall:     {report['elapsed_s']:.2f} s")
    if report.get("tokens_per_s") is not None:
        print(
            f"  throughput:       {report['tokens_per_s']:.1f} tok/s "
            f"({report['completion_tokens']} completion tokens)"
        )
    if report.get("tool_name"):
        print(f"  tool call:        {report['tool_name']}{report.get('tool_arguments', '')}")
    if report.get("vram_bytes"):
        print(f"  model vram:       {report['vram_bytes'] / 1024**3:.2f} GiB resident")
    devices = gpu_memory()
    if devices:
        for device in devices:
            print(
                f"  gpu {device['name']}: {device['used_mib']:.0f} / "
                f"{device['total_mib']:.0f} MiB used"
            )
    else:
        print("  gpu: pynvml unavailable")
    if report["ok"]:
        print("  result:           PASS")
    else:
        print(f"  result:           FAIL ({report.get('error', 'unknown')})")
        if report.get("content"):
            print(f"  content:          {report['content']}")
    print()


def main():
    config = load_config()
    base_url = config["ollama_base_url"]
    native_url = native_base(base_url)
    timeout = config["request_timeout_s"]
    client = OpenAI(base_url=base_url, api_key="ollama", timeout=timeout)

    reports = []
    for model in config["models"]:
        report = check_model(
            client, native_url, model, config["temperature"], timeout
        )
        print_report(report)
        reports.append(report)

    failures = [r["model"] for r in reports if not r["ok"]]
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("all models passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
