# agent-evals

Regression tests for AI agents.

Agents break quietly. Someone tweaks a prompt or switches the model, and the agent starts calling the wrong tool, passing the wrong arguments, or answering confidently from nothing. Unit tests don't catch it because the code didn't change, and the first people to notice are usually users.

I'm building agent-evals to be something like pytest for agents. You write test cases (an input plus what a good answer and a good trajectory look like), run your agent over them in CI, and get a failing build when accuracy, tool use, groundedness or cost gets worse than the last good run. Later on, real production failures get turned into new test cases, so the suite grows from what actually went wrong.

## Status

Early. So far there's the package and its data models (`agent_evals/models.py`):

- a **test case** is an input plus whatever should be checked: the expected answer, the tool calls a good trajectory should make, the sources the answer must stick to, and step and cost limits
- a **trace** records what the agent did: each model call and tool call, with tokens, cost and timing
- a **run** collects the scores for every case, along with the git commit of the agent under test, so a regression can be traced to a change

Agents plug in through a small interface (`agent_evals/agents.py`): anything with a `name` and an async `run(case)` that returns the answer and its trace. There are adapters for:

- **plain functions**, sync or async. A function that takes a second argument gets a recorder for logging its tool calls
- **LangChain** runnables (agents, LangGraph graphs, chains), traced through a callback handler without changing the runnable (`pip install agent-evals[langchain]`)
- **OpenAI-style tool calling**, with any client that speaks the chat completions API, including local servers. Token counts and cost come from the API's usage numbers

Test cases live in JSONL files, one case per line (see [examples/weather.jsonl](examples/weather.jsonl)). A typo or a duplicate id fails with the file and line number. The runner runs cases concurrently, each with a timeout. A case that crashes or times out is recorded as failed, and the rest of the run carries on:

```python
from agent_evals.agents import FunctionAgent
from agent_evals.dataset import load_dataset
from agent_evals.runner import run_eval_sync


def weather_agent(question, rec):
    # a stand-in for a real agent: calls its tool for Berlin and answers
    rec.tool("get_weather", {"city": "Berlin"}, result={"temp_c": 12})
    return "It's 12°C in Berlin."


run = run_eval_sync(FunctionAgent(weather_agent), load_dataset("examples/weather.jsonl"), concurrency=4)
for case in run.cases:
    print(case.case_id, [c.name for c in case.result.trace.tool_calls], case.result.output)
```

Each `Run` records the agent, the dataset, the git commit and every trace. Scoring hooks in through `scorers=[...]`: any function that takes the case and the result and returns a `Score`. The built-in scorers are the next step.

The rest is coming in this order:

1. scorers for answers, tool use, hallucinations, plus LLM-as-judge
2. a results store, a CLI with `compare` for CI, and a dashboard
3. turning production failures into test cases

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # ruff and a few file checks on every commit
pytest
```

CI runs ruff, mypy (strict) and the tests on Python 3.10 to 3.13 for every push and pull request.

## License

MIT
