"""The LangChain adapter records model and tool calls from inside an unchanged runnable."""

import pytest

pytest.importorskip("langchain_core")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.runnables import RunnableConfig, RunnableLambda  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from agent_evals.adapters.langchain import LangChainAgent  # noqa: E402
from agent_evals.models import TestCase  # noqa: E402


@tool
def lookup(city: str) -> str:
    """Look up the weather for a city."""
    return f"{city}: 12°C"


@tool
def broken(city: str) -> str:
    """A tool that always fails."""
    raise RuntimeError("weather service down")


def make_chain(tool_to_call):
    model = GenericFakeChatModel(messages=iter([AIMessage(content="It's 12°C in Berlin.")]))

    async def agent(inputs: dict, config: RunnableConfig) -> dict:
        # a hand-rolled agent: one tool call, then the model answers
        try:
            observation = await tool_to_call.ainvoke({"city": "Berlin"}, config)
        except RuntimeError:
            observation = "no data"
        answer = await model.ainvoke(f"{inputs['input']}\n{observation}", config)
        return {"output": answer}

    return RunnableLambda(agent, name="weather-chain")


async def test_records_tool_and_model_calls():
    result = await LangChainAgent(make_chain(lookup)).run(TestCase(id="1", input="Weather in Berlin?"))
    assert result.output == "It's 12°C in Berlin."
    assert [s.type for s in result.trace.steps] == ["tool", "llm"]
    call = result.trace.tool_calls[0]
    assert call.name == "lookup" and call.arguments == {"city": "Berlin"} and call.result == "Berlin: 12°C"
    assert result.trace.steps[1].content == "It's 12°C in Berlin."


async def test_records_tool_errors():
    result = await LangChainAgent(make_chain(broken)).run(TestCase(id="1", input="Weather?"))
    assert result.trace.tool_calls[0].error == "RuntimeError: weather service down"


async def test_name_and_bare_input():
    chain = RunnableLambda(lambda text: text.upper(), name="shout")
    agent = LangChainAgent(chain, input_key=None)
    assert agent.name == "shout"
    assert (await agent.run(TestCase(id="1", input="hi"))).output == "HI"
