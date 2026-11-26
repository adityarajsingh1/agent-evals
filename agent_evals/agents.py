"""
The interface every agent under test implements, and the simplest adapter: a plain function.

An agent is anything with a `name` and an async `run(case)` that returns an `AgentResult`.
Adapters for specific frameworks live in `agent_evals.adapters`, one file each, so supporting a
new framework means adding one file that implements this protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from agent_evals.models import AgentResult, Step, TestCase, ToolCall, Trace


@runtime_checkable
class Agent(Protocol):
    name: str

    async def run(self, case: TestCase) -> AgentResult: ...


class Recorder:
    """Lets a plain-function agent report what it did, so trajectory scorers have something to check.

    Pass a function that takes two arguments to `FunctionAgent` and it gets one of these::

        def my_agent(question, rec):
            hits = search(question)
            rec.tool("search", {"q": question}, result=hits)
            return answer(question, hits)
    """

    def __init__(self) -> None:
        self.trace = Trace()

    def tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        call = ToolCall(name=name, arguments=arguments or {}, result=result, error=error)
        self.trace.steps.append(Step(type="tool", tool_call=call, duration_ms=duration_ms))

    def llm(
        self,
        content: str | None = None,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: float = 0.0,
    ) -> None:
        self.trace.steps.append(
            Step(
                type="llm",
                content=content,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                duration_ms=duration_ms,
            )
        )


AgentFn = Callable[..., "str | AgentResult | Awaitable[str | AgentResult]"]


class FunctionAgent:
    """Wraps `fn(input)` or `fn(input, recorder)`, sync or async, returning a string or an AgentResult.

    Sync functions run in a worker thread so the runner can still run cases in parallel.
    """

    def __init__(self, fn: AgentFn, name: str | None = None) -> None:
        self.fn = fn
        self.name: str = name or str(getattr(fn, "__name__", "agent"))
        self._wants_recorder = len(inspect.signature(fn).parameters) >= 2

    async def run(self, case: TestCase) -> AgentResult:
        rec = Recorder()
        args: tuple[Any, ...] = (case.input, rec) if self._wants_recorder else (case.input,)
        start = time.perf_counter()
        if inspect.iscoroutinefunction(self.fn):
            out = await self.fn(*args)
        else:
            out = await asyncio.to_thread(self.fn, *args)
            if inspect.isawaitable(out):
                out = await out
        if isinstance(out, AgentResult):
            return out
        if not isinstance(out, str):
            raise TypeError(f"{self.name} returned {type(out).__name__}, expected str or AgentResult")
        if not rec.trace.steps:
            # nothing was recorded; keep at least the wall time so cost/latency checks see something
            rec.llm(content=out, duration_ms=(time.perf_counter() - start) * 1000)
        return AgentResult(output=out, trace=rec.trace)
