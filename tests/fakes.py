"""Fake OpenAI-style endpoint so agent tests run without Ollama."""

import json
from types import SimpleNamespace


def native_call(name, arguments, call_id="call_x"):
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def response(content="", tool_calls=None, prompt_tokens=100, completion_tokens=10):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        ),
    )


class FakeClient:
    """Answers from a scripted list, or from handler(request) when given a callable.

    Every request is recorded with a copy of its message list, because agents
    keep appending to the list they sent.
    """

    def __init__(self, responses):
        self.handler = responses if callable(responses) else None
        self.responses = [] if self.handler else list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        request = dict(request, messages=list(request["messages"]))
        self.requests.append(request)
        if self.handler is not None:
            return self.handler(request)
        if not self.responses:
            raise AssertionError("fake client ran out of scripted responses")
        return self.responses.pop(0)


def system_text(request):
    return request["messages"][0]["content"]


def tool_names(request):
    return [entry["function"]["name"] for entry in request.get("tools") or []]
