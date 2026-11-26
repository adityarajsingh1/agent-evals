"""A small stand-in agent for trying the CLI:

    agent-evals run examples/weather_agent.py:agent examples/weather.jsonl

It "calls" a fake weather tool for each city it recognises and answers from the results, so the
example dataset has something real to score. Swap in your own agent with the same signature,
or any LangChain / OpenAI-style agent through the adapters.
"""

from agent_evals.agents import Recorder

WEATHER = {"Berlin": 12, "Lisbon": 19, "Oslo": 2}


def agent(question: str, rec: Recorder) -> str:
    cities = [c for c in WEATHER if c.lower() in question.lower()]
    if not cities:
        return "Overcast means the sky is completely covered by cloud."
    temps = {}
    for city in cities:
        temps[city] = WEATHER[city]
        rec.tool("get_weather", {"city": city}, result={"city": city, "temp_c": WEATHER[city]})
    if len(temps) == 1:
        city, t = next(iter(temps.items()))
        return f"It's {t}°C in {city}."
    warmest = max(temps, key=lambda c: temps[c])
    return f"{warmest} is warmer: " + ", ".join(f"{c} {t}°C" for c, t in temps.items()) + "."
