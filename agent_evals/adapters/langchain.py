"""
Run any LangChain runnable (an AgentExecutor, a LangGraph graph, a plain chain) as an agent under test.

Nothing about the runnable has to change: a callback handler is attached for the call and records
each chat model call (text, token usage) and each tool call (name, arguments, result or error).
Needs `pip install agent-evals[langchain]`.
"""

from __future__ import annotations

import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from langchain_core.runnables import Runnable

from agent_evals.models import AgentResult, Step, TestCase, ToolCall, Trace


class TraceHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.trace = Trace()
        self._started: dict[UUID, float] = {}
        self._tool_inputs: dict[UUID, tuple[str, dict[str, Any]]] = {}

    def _elapsed(self, run_id: UUID) -> float:
        start = self._started.pop(run_id, None)
        return (time.perf_counter() - start) * 1000 if start is not None else 0.0

    async def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = time.perf_counter()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        gen: Any = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        model = (getattr(msg, "response_metadata", None) or {}).get("model_name")
        self.trace.steps.append(
            Step(
                type="llm",
                content=getattr(gen, "text", None) or None,
                model=model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                duration_ms=self._elapsed(run_id),
            )
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = time.perf_counter()
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        self._tool_inputs[run_id] = (name, inputs if inputs is not None else {"input": input_str})

    def _tool_step(self, run_id: UUID, result: Any = None, error: str | None = None) -> None:
        name, args = self._tool_inputs.pop(run_id, ("tool", {}))
        result = getattr(result, "content", result)  # ToolMessage -> its content
        call = ToolCall(name=name, arguments=args, result=result, error=error)
        self.trace.steps.append(Step(type="tool", tool_call=call, duration_ms=self._elapsed(run_id)))

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_step(run_id, result=output)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._tool_step(run_id, error=f"{type(error).__name__}: {error}")


class LangChainAgent:
    def __init__(
        self,
        runnable: Runnable[Any, Any],
        name: str | None = None,
        input_key: str | None = "input",
        output_key: str = "output",
    ) -> None:
        """`input_key=None` passes the case input as a bare string instead of `{input_key: input}`."""
        self.runnable = runnable
        self.name = name or runnable.get_name()
        self.input_key = input_key
        self.output_key = output_key

    async def run(self, case: TestCase) -> AgentResult:
        handler = TraceHandler()
        payload: Any = case.input if self.input_key is None else {self.input_key: case.input}
        out = await self.runnable.ainvoke(payload, config={"callbacks": [handler]})
        return AgentResult(output=self._text(out), trace=handler.trace)

    def _text(self, out: Any) -> str:
        if isinstance(out, BaseMessage):
            return out.text if isinstance(out.text, str) else str(out.content)
        if isinstance(out, dict):
            if self.output_key in out:
                return self._text(out[self.output_key])
            if out.get("messages"):  # LangGraph state: the last message is the answer
                return self._text(out["messages"][-1])
        return out if isinstance(out, str) else str(out)
