"""An agent whose behaviour comes from a prompt, for trying `agent-evals improve`:

    agent-evals improve examples/city_agent.py:make_agent examples/cities.jsonl \
        --prompts examples/prompts --init "Answer with the name of one city."

`make_agent` gets a `PromptVersion` and returns the agent, so `improve` can build one agent per
prompt version and compare them.

With OPENAI_API_KEY set it calls a real model. Without it, it uses `toy_model`, a stand-in that
behaves like a small model with one blind spot: it reads every comparison as "which is warmer"
unless its prompt has shown it what "colder" means. That's enough to watch the loop work
offline; the numbers it produces say nothing about any real model.
"""

import os

from agent_evals.agents import FunctionAgent
from agent_evals.learn import PromptVersion
from agent_evals.llm import FunctionModel, OpenAIChatModel, TextModel

TEMPS = {"Berlin": 12, "Lisbon": 19, "Oslo": 2, "Madrid": 21, "Helsinki": -1, "Rome": 16}


def toy_model(prompt: str) -> str:
    seen, _, question = prompt.rpartition("Q: ")
    cities = [c for c in TEMPS if c in question]
    if not cities:
        return "I don't know."
    colder = "colder" in question and "colder" in seen
    return (min if colder else max)(cities, key=TEMPS.__getitem__)


def model() -> TextModel:
    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        return OpenAIChatModel(OpenAI(), model=os.environ.get("CITY_AGENT_MODEL", "gpt-4o-mini"))
    return FunctionModel(toy_model)


def make_agent(version: PromptVersion) -> FunctionAgent:
    m = model()
    temps = ", ".join(f"{c} {t}°C" for c, t in TEMPS.items())
    # a real model needs the temperatures; the toy one already knows them
    if not isinstance(m, FunctionModel):
        version = version.model_copy(update={"system": f"{version.system}\n\nToday: {temps}."})

    def answer(question: str) -> str:
        return m.complete(version.render(question)).strip()

    return FunctionAgent(answer, name="city-agent")
