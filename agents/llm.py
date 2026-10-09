"""Shared LLM client used by every architecture.

One client wraps the OpenAI-compatible Ollama endpoint, enforces the per-task
call budget, turns whatever the model emits into tool calls (native or written
as JSON in the message text), executes them against the shop, and keeps the
per-task totals. Routing every architecture through this one class is what
keeps the arms comparable.

Counter meanings:
    llm_calls             requests sent to the model, including failed ones
    tool_calls            shop tool calls executed (native and text)
    malformed_tool_calls  tool calls rejected before execution
    text_tool_calls       tool calls recovered from message text, executed or not
    dropped_tool_calls    responses with output tokens but no content and no tool
                          call: the serving layer discarded what the model wrote
"""

import json
import re
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import tools as shop_tools

CONFIG_PATH = ROOT / "config.yaml"

TOTAL_KEYS = (
    "llm_calls",
    "tool_calls",
    "malformed_tool_calls",
    "text_tool_calls",
    "dropped_tool_calls",
    "prompt_tokens",
    "completion_tokens",
    "llm_latency_s",
)


class BudgetExceeded(Exception):
    """The per-task LLM call budget is spent."""


class LLMError(Exception):
    """The model endpoint failed: connection, timeout, or HTTP error."""

    def __init__(self, message, timeout=False):
        super().__init__(message)
        self.timeout = timeout


def load_config(path=CONFIG_PATH):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


RESET_WARMUP = {"prompt": "hi", "options": {"num_predict": 1, "temperature": 0, "seed": 0}}
UNLOAD_WAIT_S = 60
UNLOAD_POLL_S = 0.1


def native_base(base_url):
    """Config holds the OpenAI-compatible path; the native API sits one level up."""
    return base_url.rstrip("/").removesuffix("/v1")


def ollama_get(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def ollama_post(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _bare(name):
    return str(name or "").removesuffix(":latest")


def running_models(native_url, timeout):
    return ollama_get(f"{native_url}/api/ps", timeout).get("models", []) or []


def _running_entry(native_url, model, timeout):
    for entry in running_models(native_url, timeout):
        if _bare(model) in (_bare(entry.get("name")), _bare(entry.get("model"))):
            return entry
    return None


def _model_loaded(native_url, model, timeout):
    return _running_entry(native_url, model, timeout) is not None


def unload_model(native_url, model, timeout):
    ollama_post(f"{native_url}/api/generate", {"model": model, "keep_alive": 0}, timeout)
    deadline = time.perf_counter() + UNLOAD_WAIT_S
    while _model_loaded(native_url, model, timeout):
        if time.perf_counter() > deadline:
            raise RuntimeError(f"{model} still loaded {UNLOAD_WAIT_S} s after unload request")
        time.sleep(UNLOAD_POLL_S)


def warm_up_model(native_url, model, timeout):
    """Load the model (if needed) with one fixed, seeded, one-token request."""
    ollama_post(
        f"{native_url}/api/generate",
        {"model": model, "stream": False, **RESET_WARMUP},
        timeout,
    )


def reset_model_state(model, config=None):
    """Unload the model from Ollama, then load it with one fixed warm-up request.

    Unloading drops the prompt cache, so every run starts from the same cache
    state; Ollama reuses cached prefixes across requests, and a different
    cached prefix changes the output even with the same seed. Returns seconds.
    """
    config = config if config is not None else load_config()
    native_url = native_base(config["ollama_base_url"])
    timeout = config["request_timeout_s"]
    started = time.perf_counter()
    unload_model(native_url, model, timeout)
    warm_up_model(native_url, model, timeout)
    return time.perf_counter() - started


def model_residency(model, config):
    """Where Ollama placed the loaded model: bytes total, bytes in VRAM, GPU share."""
    native_url = native_base(config["ollama_base_url"])
    entry = _running_entry(native_url, model, config["request_timeout_s"])
    if entry is None:
        return None
    size = entry.get("size") or 0
    vram = entry.get("size_vram") or 0
    fraction = vram / size if size else None
    if fraction is None:
        processor = "unknown"
    elif fraction >= 1:
        processor = "100% GPU"
    else:
        processor = f"{round((1 - fraction) * 100)}%/{round(fraction * 100)}% CPU/GPU"
    return {
        "size_bytes": size,
        "size_vram_bytes": vram,
        "gpu_fraction": fraction,
        "processor": processor,
        "context_length": entry.get("context_length"),
    }


def check_ollama(models, config):
    """Return an error message, or None when the server is up and every model exists."""
    native_url = native_base(config["ollama_base_url"])
    try:
        tags = ollama_get(f"{native_url}/api/tags", timeout=10)
    except OSError as error:
        return (
            f"Ollama is not reachable at {native_url} ({error}). Start it with"
            " 'ollama serve' or launch the Ollama app, then retry."
        )
    installed = {
        _bare(entry.get(key)) for entry in tags.get("models", []) for key in ("name", "model")
    }
    missing = [model for model in models if _bare(model) not in installed]
    if missing:
        return "models not installed in Ollama: " + ", ".join(
            f"{model} (create with 'ollama create {model} -f models\\{model}.Modelfile')"
            for model in missing
        )
    return None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None
    raw_arguments: str
    source: str
    error: str | None = None

    @property
    def malformed(self):
        return self.error is not None


@dataclass
class ChatResult:
    message: dict
    content: str
    tool_calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    finish_reason: str | None = None
    call_index: int = 0
    dropped: bool = False


def _parse_arguments(raw):
    """Return (arguments_dict, error)."""
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    if not isinstance(raw, str):
        return None, f"tool arguments must be a JSON object, got {type(raw).__name__}"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        return None, f"tool arguments are not valid JSON: {error}"
    if not isinstance(value, dict):
        return None, f"tool arguments must be a JSON object, got {type(value).__name__}"
    return value, None


def _raw_text(value):
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _as_call_payload(payload):
    """Return (name, raw_arguments) if payload looks like a tool call, else None."""
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("function"), dict):
        payload = payload["function"]
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    for key in ("arguments", "parameters"):
        if key in payload:
            return name, payload[key]
    return None


def find_text_tool_calls(content):
    """Find tool calls written as JSON objects inside message text.

    Returns (calls, residual_text) where calls is a list of (name, raw_arguments)
    and residual_text is the content with those JSON objects and any tool_call
    tags or empty code fences removed.
    """
    if not content or "{" not in content:
        return [], content or ""
    decoder = json.JSONDecoder()
    calls = []
    spans = []
    index = 0
    while True:
        start = content.find("{", index)
        if start < 0:
            break
        try:
            payload, end = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        parsed = _as_call_payload(payload)
        if parsed is not None:
            calls.append(parsed)
            spans.append((start, end))
        index = end
    if not calls:
        return [], content

    pieces = []
    cursor = 0
    for start, end in spans:
        pieces.append(content[cursor:start])
        cursor = end
    pieces.append(content[cursor:])
    residual = "".join(pieces)
    residual = re.sub(r"</?tool_call>", "", residual)
    residual = re.sub(r"```(?:json)?\s*```", "", residual)
    return calls, residual.strip()


class LLMClient:
    def __init__(self, model, config=None, client=None):
        self.config = config if config is not None else load_config()
        self.model = model
        self.temperature = self.config["temperature"]
        self.max_calls = int(self.config["max_llm_calls_per_task"])
        # A runaway generation would otherwise run to the request timeout, and
        # Ollama keeps generating after the client gives up, skewing the next run.
        self.max_tokens = self.config.get("max_tokens_per_call")
        if client is None:
            from openai import OpenAI

            # Retries are disabled so every request the model sees is counted.
            client = OpenAI(
                base_url=self.config["ollama_base_url"],
                api_key="ollama",
                timeout=self.config["request_timeout_s"],
                max_retries=0,
            )
        self.client = client
        self.reset()

    def reset(self):
        """Start a new task: zero every counter and restore the full budget."""
        self.totals = {key: 0 for key in TOTAL_KEYS}
        self.totals["llm_latency_s"] = 0.0

    @property
    def calls_remaining(self):
        return self.max_calls - self.totals["llm_calls"]

    def snapshot_totals(self):
        return dict(self.totals)

    def chat(self, messages, tools=None, *, seed):
        if self.totals["llm_calls"] >= self.max_calls:
            raise BudgetExceeded(
                f"budget of {self.max_calls} LLM calls per task is spent"
            )
        self.totals["llm_calls"] += 1
        call_index = self.totals["llm_calls"]

        request = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "extra_body": {"seed": seed},
        }
        if self.max_tokens:
            request["max_tokens"] = int(self.max_tokens)
        if tools:
            request["tools"] = tools

        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(**request)
        except Exception as error:
            self.totals["llm_latency_s"] += time.perf_counter() - started
            if _is_openai_error(error):
                raise LLMError(
                    f"{type(error).__name__}: {error}", timeout=_is_timeout(error)
                ) from error
            raise
        latency = time.perf_counter() - started
        self.totals["llm_latency_s"] += latency

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        self.totals["prompt_tokens"] += prompt_tokens
        self.totals["completion_tokens"] += completion_tokens

        choice = response.choices[0]
        raw_message = choice.message
        content = raw_message.content or ""
        known = {entry["function"]["name"] for entry in tools or []}

        tool_calls = []
        for position, native in enumerate(getattr(raw_message, "tool_calls", None) or []):
            tool_calls.append(
                self._make_call(
                    call_id=getattr(native, "id", None),
                    name=native.function.name,
                    raw=native.function.arguments,
                    source="native",
                    known=known,
                    call_index=call_index,
                    position=position,
                )
            )

        history_content = content
        if not tool_calls and tools:
            text_calls, residual = find_text_tool_calls(content)
            if text_calls:
                history_content = residual
                for position, (name, raw) in enumerate(text_calls):
                    tool_calls.append(
                        self._make_call(
                            call_id=None,
                            name=name,
                            raw=raw,
                            source="text",
                            known=known,
                            call_index=call_index,
                            position=position,
                        )
                    )

        message = {"role": "assistant", "content": history_content}
        if tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # Invalid JSON echoed back would make the endpoint reject the
                        # next request, so malformed calls go back with empty arguments.
                        "arguments": json.dumps(call.arguments, ensure_ascii=False)
                        if call.arguments is not None
                        else "{}",
                    },
                }
                for call in tool_calls
            ]

        finish_reason = getattr(choice, "finish_reason", None)
        # Ollama discards a generated tool call whose name matches no supplied
        # tool, together with its text, so the model's output never reaches us.
        dropped = (
            finish_reason in ("stop", "length")
            and not content.strip()
            and not tool_calls
            and completion_tokens > 0
        )
        if dropped:
            self.totals["dropped_tool_calls"] += 1

        return ChatResult(
            message=message,
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency,
            finish_reason=finish_reason,
            call_index=call_index,
            dropped=dropped,
        )

    def _make_call(self, call_id, name, raw, source, known, call_index, position):
        call_id = call_id or f"call_{call_index}_{position}"
        name = name if isinstance(name, str) else ""
        arguments, error = _parse_arguments(raw)
        if name not in known:
            error = f"unknown tool {name!r}; available tools: {', '.join(sorted(known))}"
            arguments = None
        return ToolCall(
            id=call_id,
            name=name,
            arguments=arguments,
            raw_arguments=_raw_text(raw),
            source=source,
            error=error,
        )

    def check_tool_call(self, call):
        """Count one parsed call; return an error result if malformed, else None.

        Calls handled outside the shop (the supervisor's delegate) go through
        here so text and malformed counts stay comparable across architectures.
        """
        if call.source == "text":
            self.totals["text_tool_calls"] += 1
        if call.malformed:
            self.totals["malformed_tool_calls"] += 1
            return {"ok": False, "error": f"malformed tool call: {call.error}"}
        return None

    def execute_tool_call(self, shop, call):
        """Run one parsed call against the shop, or return its parse error."""
        error = self.check_tool_call(call)
        if error is not None:
            return error
        self.totals["tool_calls"] += 1
        return shop_tools.call_tool(shop, call.name, call.arguments)

    @staticmethod
    def tool_message(call, result):
        return {
            "role": "tool",
            "tool_call_id": call.id,
            "name": call.name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        }


def _is_openai_error(error):
    try:
        import openai
    except ImportError:
        return False
    return isinstance(error, openai.OpenAIError)


def _is_timeout(error):
    try:
        import openai
    except ImportError:
        return False
    return isinstance(error, openai.APITimeoutError)
