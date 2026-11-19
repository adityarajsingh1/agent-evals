# agent-evals

Regression tests for AI agents.

Agents break quietly. Someone tweaks a prompt or switches the model, and the agent starts calling the wrong tool, passing the wrong arguments, or answering confidently from nothing. Unit tests don't catch it because the code didn't change, and the first people to notice are usually users.

I'm building agent-evals to be something like pytest for agents. You write test cases (an input plus what a good answer and a good trajectory look like), run your agent over them in CI, and get a failing build when accuracy, tool use, groundedness or cost gets worse than the last good run. Later on, real production failures get turned into new test cases, so the suite grows from what actually went wrong.

## Status

Early. Right now this is the package skeleton. The pieces are coming in this order:

1. data models for test cases, runs and traces
2. adapters for plain functions, LangChain and OpenAI tool-calling agents, and a parallel test runner
3. scorers for answers, tool use, hallucinations, plus LLM-as-judge
4. a results store, a CLI with `compare` for CI, and a dashboard
5. turning production failures into test cases

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
