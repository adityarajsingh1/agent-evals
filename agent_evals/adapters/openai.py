"""
A tool-calling agent on the OpenAI chat completions API, with every turn recorded in the trace.

The client is duck-typed: anything with `chat.completions.create(...)` that returns
OpenAI-shaped responses works, sync or async. That covers `openai.OpenAI`, `openai.AsyncOpenAI`,
Azure, and the many servers that copy the API (vLLM, Ollama, LiteLLM, llmops). It also means the
tests use a small fake client and need no API key.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent_evals.models import AgentResult, Step, TestCase, ToolCall, Trace


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    # JSON schema for the arguments, as the API expects it
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class OpenAIToolsAgent:
    def __init__(
        self,
        client: Any,
        model: str,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
        max_turns: int = 10,
        prices: dict[str, tuple[float, float]] | None = None,
        name: str | None = None,
    ) -> None:
        """`prices` maps a model name to (input, output) USD per million tokens, for cost tracking."""
        self.client = client
        self.model = model
        self.tools = {t.name: t for t in tools or []}
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.prices = prices or {}
        self.name = name or f"openai-tools:{model}"

    async def _complete(self, **kwargs: Any) -> Any:
        create = self.client.chat.completions.create
        if inspect.iscoroutinefunction(create):
            return await create(**kwargs)
        return await asyncio.to_thread(create, **kwargs)

    def _cost(self, model: str, prompt: int, completion: int) -> float:
        price_in, price_out = self.prices.get(model, (0.0, 0.0))
        return (prompt * price_in + completion * price_out) / 1_000_000

    async def run(self, case: TestCase) -> AgentResult:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": case.input})
        trace = Trace()
        kwargs: dict[str, Any] = {"model": self.model}
        if self.tools:
            kwargs["tools"] = [t.spec() for t in self.tools.values()]

        for _ in range(self.max_turns):
            start = time.perf_counter()
            resp = await self._complete(messages=messages, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            msg = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            prompt, completion = (usage.prompt_tokens, usage.completion_tokens) if usage else (0, 0)
            model = getattr(resp, "model", None) or self.model
            trace.steps.append(
                Step(
                    type="llm",
                    content=msg.content,
                    model=model,
                    input_tokens=prompt,
                    output_tokens=completion,
                    cost_usd=self._cost(model, prompt, completion),
                    duration_ms=elapsed,
                )
            )
            calls = getattr(msg, "tool_calls", None) or []
            if not calls:
                return AgentResult(output=msg.content or "", trace=trace)

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.function.name, "arguments": c.function.arguments},
                        }
                        for c in calls
                    ],
                }
            )
            for c in calls:
                content = await self._call_tool(c.function.name, c.function.arguments, trace)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": content})

        return AgentResult(output=None, trace=trace, error=f"no final answer after {self.max_turns} turns")

    async def _call_tool(self, name: str, raw_args: str, trace: Trace) -> str:
        """Run one tool call and record it. Errors go back to the model as text, like most agents do."""
        start = time.perf_counter()
        args: dict[str, Any] = {}
        result: Any = None
        error: str | None = None
        try:
            args = json.loads(raw_args or "{}")
            tool = self.tools.get(name)
            if tool is None:
                raise KeyError(f"unknown tool {name!r}")
            result = await tool.fn(**args) if inspect.iscoroutinefunction(tool.fn) else tool.fn(**args)
        except Exception as e:  # the model sees the error and can try again
            error = f"{type(e).__name__}: {e}"
        trace.steps.append(
            Step(
                type="tool",
                tool_call=ToolCall(name=name, arguments=args, result=result, error=error),
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        )
        if error:
            return f"error: {error}"
        return result if isinstance(result, str) else json.dumps(result, default=str)
