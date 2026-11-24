"""
The small model interface used by model-based scorers (claim checking now, LLM-as-judge later).

A model is anything with `complete(prompt) -> str`. Scorers only need text in and text out, so
this stays independent of the agent adapters: the model that grades can be a different (often
cheaper, or stronger) model than the one under test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TextModel(Protocol):
    def complete(self, prompt: str) -> str: ...


class FunctionModel:
    """Wrap a plain `prompt -> str` function, e.g. a fake for tests or a call into your own stack."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self.fn = fn

    def complete(self, prompt: str) -> str:
        return self.fn(prompt)


class OpenAIChatModel:
    """Any OpenAI-compatible chat completions client (sync), called with one user message."""

    def __init__(self, client: Any, model: str, temperature: float = 0.0, system: str | None = None) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.system = system

    def complete(self, prompt: str) -> str:
        messages = [{"role": "system", "content": self.system}] if self.system else []
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.temperature
        )
        return resp.choices[0].message.content or ""
