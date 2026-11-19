# agent-evals

Regression tests for AI agents.

Agents break quietly. Someone tweaks a prompt or switches the model, and the agent starts calling the wrong tool, passing the wrong arguments, or answering confidently from nothing. Unit tests don't catch it because the code didn't change, and the first people to notice are usually users.

I'm building agent-evals to be something like pytest for agents. You write test cases (an input plus what a good answer and a good trajectory look like), run your agent over them in CI, and get a failing build when accuracy, tool use, groundedness or cost gets worse than the last good run. Later on, real production failures get turned into new test cases, so the suite grows from what actually went wrong.

## Status

Early. So far there's the package and its data models (`agent_evals/models.py`):

- a **test case** is an input plus whatever should be checked: the expected answer, the tool calls a good trajectory should make, the sources the answer must stick to, and step and cost limits
- a **trace** records what the agent did: each model call and tool call, with tokens, cost and timing
- a **run** collects the scores for every case, along with the git commit of the agent under test, so a regression can be traced to a change

The rest is coming in this order:

1. adapters for plain functions, LangChain and OpenAI tool-calling agents, and a parallel test runner
2. scorers for answers, tool use, hallucinations, plus LLM-as-judge
3. a results store, a CLI with `compare` for CI, and a dashboard
4. turning production failures into test cases

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
