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

Each `Run` records the agent, the dataset, the git commit and every trace.

### Scorers

Scoring hooks in through `scorers=[...]`. A scorer is anything that takes the case and the result and returns a `Score` (0 to 1, pass or fail, and a reason). Scorers that need something a case doesn't have, like an expected answer, skip that case. That way one list of scorers works for a whole dataset. The built-in ones, by name:

| Name | Checks | Options |
|---|---|---|
| `exact` | the answer equals `expected_output`, after normalising case, punctuation and articles | `normalized` |
| `contains` | `expected_output` appears in the answer as whole words | |
| `numeric` | every number in `expected_output` appears in the answer ("1,234.50 EUR" matches 1234.5) | `rel_tol`, `abs_tol` |
| `regex` | the answer matches a pattern, given as an option or per case in `metadata.regex` | `pattern`, `ignore_case` |
| `json_schema` | the answer contains JSON valid against a schema, given as an option or in `metadata.json_schema` (`pip install agent-evals[schema]`) | `schema` |
| `tools` | the agent called the tools in `expected_tools` (by name), in order if `tools_in_order` is set. `"expected_tools": []` means no tool should be called | `allow_extra` |
| `tool_args` | the same, and each call has the expected arguments. Only the listed arguments are checked, strings are compared case-insensitively and `"3"` matches `3`. A failure shows the closest call the agent actually made | `allow_extra`, `rel_tol`, `case_sensitive` |
| `groundedness` | the share of the answer's claims that `sources` (and, by default, the agent's own tool results) back up. Fails below `threshold` and lists the unsupported claims | `threshold`, `min_coverage`, `include_tool_results`, `checker` |
| `judge` | a model's rubric scores for the answer, see below | `model`, `rubric`, `threshold`, `samples`, `include_tools` |
| `max_steps` | no more than `max_steps` model and tool calls | `limit` (default for cases without one) |
| `max_cost` | no more than `max_cost_usd`, from the trace | `limit` |
| `max_latency` | total step time under `metadata.max_latency_ms` | `limit` |

```python
from agent_evals.scorers import build_scorers, default_scorers

scorers = build_scorers(["contains", {"name": "numeric", "rel_tol": 0.01}, "tool_args"])
run = run_eval_sync(agent, cases, scorers)

# or the defaults: contains, tool_args, max_steps and max_cost
run = run_eval_sync(agent, cases, default_scorers())
```

A failing trajectory says what went wrong. For example, `missing get_weather(city='Oslo') (closest call: get_weather(city='Lisbon'))`.

### Hallucinations

`groundedness` splits the answer into claims (sentences and list items, minus filler like "Let me know if you need anything else" and questions) and checks each one against the sources. The default checker is lexical and needs no model. A claim counts as supported when a source passage (a sentence, or two neighbouring sentences) contains most of its content words and **every** number in it. For a refund-policy source that says "Refunds are available within 30 days of purchase":

```
Refunds are available within 30 days of purchase.   supported
Refunds take 5 business days.                        unsupported (number not in sources: 5)
Refunds are available within 60 days of purchase.    unsupported (number not in sources: 60)
```

That catches the hallucinations that hurt most in agent answers: invented figures, dates, names and products. It's cheap enough to run on every commit. It doesn't understand paraphrase or contradictions phrased with the same words. For those, pass a model and claims are checked by it instead:

```python
import openai

from agent_evals.llm import OpenAIChatModel
from agent_evals.scorers import get_scorer

judge = OpenAIChatModel(openai.OpenAI(), "gpt-4o-mini")
groundedness = get_scorer("groundedness", model=judge, extract_with_model=True)
```

The model labels each claim supported, contradicted or not found, and it has to quote the passage it relied on. If the quote isn't really in that source, the claim counts as unsupported, so the checker can't make up support of its own. `extract_with_model=True` also has the model split answers into one-fact claims. "Refunds take 30 days and include shipping" is one sentence but two facts. A model is anything with `complete(prompt) -> str` (`FunctionModel` wraps a plain function), and `checker=` takes any object with a `check(claim, sources)` method.

### LLM-as-judge

Some things only a model can grade: whether an answer actually helps, or is right when it's worded differently from the reference. `judge` asks a model to score the answer on each criterion of a rubric, from 1 to 5, and passes when the weighted score (rescaled to 0 to 1) reaches `threshold`, 0.75 by default:

```python
correctness = get_scorer("judge", model=judge, rubric="correctness")
custom = get_scorer("judge", model=judge, threshold=0.8, rubric={
    "name": "support-tone",
    "criteria": [
        {"name": "empathetic", "description": "Acknowledges the customer's problem.", "weight": 2},
        {"name": "on_policy", "description": "Doesn't promise anything the refund policy doesn't allow."},
    ],
})
```

The built-in rubrics are `correctness` (uses `expected_output` as a reference when there is one), `helpfulness` and `safety`. The judge sees the question, the reference, the answer and the tool calls. If its reply can't be read, the case fails with the reason instead of passing silently. `samples=3` asks three times and averages, and the reason shows the spread, a quick way to see how noisy a judge is.

A new scorer is a subclass of `BaseScorer` with a `name` and a `score()` method, registered with `@register`. A scorer in another package is registered through the `agent_evals.scorers` entry point group, so it shows up by name without any change here.

The rest is coming in this order:

1. a check of how well the judge agrees with human labels
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
