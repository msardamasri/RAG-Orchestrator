"""Generators: the language model that writes the answer, rewrites follow-ups and condenses memory."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence

import httpx

from ragcore.interfaces import Message

THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OpenAICompatGenerator:
    """Any server speaking the OpenAI chat API: a local mlx-lm / llama.cpp / Ollama server, or OpenAI itself.

    ``extra_body`` is merged into every request, e.g. ``{"chat_template_kwargs": {"enable_thinking": false}}``
    to turn off a reasoning model's thinking.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8742/v1", model: str = "", api_key_env: str | None = None,
                 extra_body: dict | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"openai_compat:{model or base_url}"
        self.extra_body = extra_body or {}
        headers = {}
        if api_key_env:
            api_key = os.getenv(api_key_env)
            if not api_key:
                raise RuntimeError(f"{api_key_env} is not set")
            headers["Authorization"] = f"Bearer {api_key}"
        self.client = httpx.Client(timeout=timeout, headers=headers)

    def available(self) -> bool:
        try:
            return self.client.get(f"{self.base_url}/models", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def complete(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.1) -> str:
        body = {"messages": list(messages), "max_tokens": max_tokens, "temperature": temperature, **self.extra_body}
        if self.model:
            body["model"] = self.model
        response = self.client.post(f"{self.base_url}/chat/completions", json=body)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"].get("content") or ""
        return THINK.sub("", content).strip()


class FakeGenerator:
    """A scripted generator for tests: ``responder(messages) -> str``.

    The default responder is a stand-in that makes the pipeline runnable with no model: asked to answer from
    sources it quotes the first source and cites it; asked anything else it echoes the last user message.
    """

    name = "fake"

    def __init__(self, responder: Callable[[Sequence[Message]], str] | None = None):
        self.responder = responder or self._default
        self.calls: list[list[Message]] = []

    @staticmethod
    def _default(messages: Sequence[Message]) -> str:
        last = messages[-1]["content"]
        match = re.search(r"^\[1\] (?:\([^)]*\) )?(.+)$", last, re.MULTILINE)
        if match:
            return f"{match.group(1).strip()[:200]} [1]"
        return last.strip().splitlines()[-1] if last.strip() else ""

    def complete(self, messages: Sequence[Message], max_tokens: int = 512, temperature: float = 0.1) -> str:
        self.calls.append(list(messages))
        return self.responder(messages)
