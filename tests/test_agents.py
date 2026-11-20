import asyncio

import pytest

from agent_evals.agents import Agent, FunctionAgent
from agent_evals.models import AgentResult, TestCase

CASE = TestCase(id="c1", input="What's 2 + 2?")


async def test_plain_sync_function():
    agent = FunctionAgent(lambda q: f"you asked: {q}", name="echo")
    assert isinstance(agent, Agent)
    result = await agent.run(CASE)
    assert result.output == "you asked: What's 2 + 2?"
    # nothing recorded, so the call itself becomes one step with its wall time
    assert len(result.trace.steps) == 1


async def test_async_function_with_recorder():
    async def calculator_agent(question, rec):
        rec.llm(content=None, model="fake", input_tokens=20, output_tokens=5, cost_usd=0.001)
        rec.tool("add", {"a": 2, "b": 2}, result=4)
        await asyncio.sleep(0)
        return "4"

    result = await FunctionAgent(calculator_agent).run(CASE)
    assert result.output == "4"
    assert [c.name for c in result.trace.tool_calls] == ["add"]
    assert result.trace.tool_calls[0].arguments == {"a": 2, "b": 2}
    assert result.trace.cost_usd == pytest.approx(0.001)


async def test_function_can_return_agent_result():
    agent = FunctionAgent(lambda q: AgentResult(output="4", error=None))
    assert (await agent.run(CASE)).output == "4"


async def test_wrong_return_type_is_an_error():
    with pytest.raises(TypeError, match="expected str or AgentResult"):
        await FunctionAgent(lambda q: 4, name="bad").run(CASE)
