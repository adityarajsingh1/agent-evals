"""The OpenAI adapter against a fake client that replays scripted responses."""

import json
from types import SimpleNamespace as NS

import pytest

from agent_evals.adapters.openai import OpenAIToolsAgent, Tool
from agent_evals.models import TestCase


def reply(content=None, tool_calls=None, prompt=100, completion=20):
    return NS(
        model="gpt-test",
        usage=NS(prompt_tokens=prompt, completion_tokens=completion),
        choices=[NS(message=NS(content=content, tool_calls=tool_calls))],
    )


def call(id, name, args):
    return NS(id=id, function=NS(name=name, arguments=json.dumps(args)))


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def weather(city: str) -> dict:
    return {"city": city, "temp_c": 12}


WEATHER = Tool(
    "get_weather",
    "Current weather for a city",
    weather,
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
CASE = TestCase(id="w1", input="Weather in Berlin?")


async def test_tool_loop_is_traced():
    client = FakeClient(
        [
            reply(tool_calls=[call("t1", "get_weather", {"city": "Berlin"})]),
            reply("It's 12°C in Berlin.", prompt=150, completion=10),
        ]
    )
    agent = OpenAIToolsAgent(
        client, "gpt-test", [WEATHER], system_prompt="Be brief.", prices={"gpt-test": (1.0, 4.0)}
    )
    result = await agent.run(CASE)

    assert result.output == "It's 12°C in Berlin."
    assert [s.type for s in result.trace.steps] == ["llm", "tool", "llm"]
    tool_call = result.trace.tool_calls[0]
    assert tool_call.arguments == {"city": "Berlin"} and tool_call.result == {"city": "Berlin", "temp_c": 12}
    assert result.trace.tokens == 280
    assert result.trace.cost_usd == pytest.approx((250 * 1.0 + 30 * 4.0) / 1e6)

    # the second request carries the tool result back to the model
    second = client.requests[1]["messages"]
    assert second[0] == {"role": "system", "content": "Be brief."}
    assert second[-1]["role"] == "tool" and json.loads(second[-1]["content"])["temp_c"] == 12
    assert client.requests[0]["tools"][0]["function"]["name"] == "get_weather"


async def test_tool_errors_go_back_to_the_model():
    client = FakeClient(
        [
            reply(tool_calls=[call("t1", "get_wether", {"city": "Berlin"})]),
            reply("Sorry, I couldn't check the weather."),
        ]
    )
    result = await OpenAIToolsAgent(client, "gpt-test", [WEATHER]).run(CASE)
    assert "unknown tool 'get_wether'" in (result.trace.tool_calls[0].error or "")
    assert client.requests[1]["messages"][-1]["content"].startswith("error: KeyError")
    assert result.error is None


async def test_gives_up_after_max_turns():
    looping = [reply(tool_calls=[call(f"t{i}", "get_weather", {"city": "Berlin"})]) for i in range(3)]
    result = await OpenAIToolsAgent(FakeClient(looping), "gpt-test", [WEATHER], max_turns=3).run(CASE)
    assert result.output is None
    assert result.error == "no final answer after 3 turns"
    assert len(result.trace.tool_calls) == 3


async def test_async_client():
    class AsyncClient(FakeClient):
        async def create(self, **kwargs):
            return FakeClient.create(self, **kwargs)

        def __init__(self, responses):
            super().__init__(responses)
            self.chat = NS(completions=NS(create=self.create))

    result = await OpenAIToolsAgent(AsyncClient([reply("hi")]), "gpt-test").run(CASE)
    assert result.output == "hi"
