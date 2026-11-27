import json
import shutil
from pathlib import Path

import pytest

from agent_evals.cli import load_agent, main, parse_scorer
from agent_evals.store import ResultStore

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    shutil.copytree(EXAMPLES, tmp_path / "examples")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def run(*args):
    return main(["--db", "results.sqlite", *args])


def test_parse_scorer():
    assert parse_scorer("contains") == "contains"
    assert parse_scorer("numeric:rel_tol=0.01,abs_tol=0") == {
        "name": "numeric",
        "rel_tol": 0.01,
        "abs_tol": 0,
    }
    assert parse_scorer("regex:pattern=^ok") == {"name": "regex", "pattern": "^ok"}


def test_load_agent_from_file_and_module(workdir):
    agent = load_agent("examples/weather_agent.py:agent")
    assert agent.name == "weather_agent:agent"
    assert load_agent("agent_evals.agents:FunctionAgent").name  # any callable works


def test_run_saves_and_passes(workdir, capsys):
    assert run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "--json", "out.json") == 0
    out = capsys.readouterr()
    assert "3/3 passed (100.0%)" in out.out
    assert out.err.strip() == "..."
    with ResultStore("results.sqlite") as store:
        [summary] = store.list_runs()
    assert summary.agent == "weather_agent:agent" and summary.pass_rate == 1.0
    assert json.loads(Path("out.json").read_text())["id"] == summary.id


def test_fail_under_and_failure_reasons(workdir, capsys):
    code = run(
        "run",
        "examples/weather_agent.py:agent",
        "examples/weather.jsonl",
        "-s",
        "exact",
        "--fail-under",
        "0.9",
        "-q",
        "--no-save",
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "FAIL compare: exact: expected 'Lisbon'" in out
    assert "below --fail-under 90.0%" in out
    assert not Path("results.sqlite").exists()


def test_tags(workdir, capsys):
    assert (
        run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-t", "definition", "-q") == 0
    )
    assert "1/1 passed" in capsys.readouterr().out


@pytest.mark.parametrize(
    "args, message",
    [
        (["run", "missing.py:agent", "examples/weather.jsonl"], "no such file: missing.py"),
        (["run", "examples/weather_agent.py:nope", "examples/weather.jsonl"], "has no attribute 'nope'"),
        (["run", "examples/weather_agent.py:agent", "missing.jsonl"], "No such file"),
        (
            ["run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-s", "nope"],
            "unknown scorer",
        ),
        (
            ["run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-t", "x"],
            "no cases tagged x",
        ),
    ],
)
def test_errors_exit_2(workdir, capsys, args, message):
    assert run(*args) == 2
    assert message in capsys.readouterr().err


def test_report_latest_text_markdown_json(workdir, capsys):
    run(
        "run",
        "examples/weather_agent.py:agent",
        "examples/weather.jsonl",
        "-s",
        "contains",
        "-s",
        "exact",
        "-q",
    )
    capsys.readouterr()

    assert run("report") == 0
    text = capsys.readouterr().out
    assert text.startswith("weather_agent:agent on examples/weather.jsonl: 0/3 passed")
    assert "FAIL compare: exact: expected 'Lisbon'" in text

    assert run("report", "latest", "-f", "markdown", "--all") == 0
    md = capsys.readouterr().out
    assert md.startswith("### ❌ weather_agent:agent: 0/3 passed (0.0%)")
    assert "| contains | 1.000 | 1 |" in md and "| exact | 0.000 | 1 |" in md
    assert "| `berlin-now` | **fail** | no scorer applied |" in md

    assert run("report", "latest", "-f", "json") == 0
    assert len(json.loads(capsys.readouterr().out)["cases"]) == 3


def test_report_by_prefix_previous_and_json_file(workdir, capsys):
    run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q", "--json", "first.json")
    run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q", "-s", "exact")
    capsys.readouterr()
    first_id = json.loads(Path("first.json").read_text())["id"]

    assert run("report", "previous") == 0
    assert f"run {first_id}" in capsys.readouterr().out
    assert run("report", first_id[:6]) == 0
    assert "3/3 passed" in capsys.readouterr().out
    assert run("report", "first.json") == 0
    assert "3/3 passed" in capsys.readouterr().out


def test_report_errors(workdir, capsys):
    assert run("report") == 2
    assert "no latest run" in capsys.readouterr().err
    assert run("report", "abc123") == 2


BROKEN_AGENT = """
def agent(question, rec):
    if "Lisbon" in question:
        return "Probably Lisbon."  # forgot to check the weather
    if "Berlin" in question:
        rec.tool("get_weather", {"city": "Berlin"}, result={"temp_c": 12})
        return "It's 12°C in Berlin."
    return "Overcast means the sky is completely covered by cloud."
"""


def test_compare_exits_1_on_regression(workdir, capsys):
    Path("examples/broken_agent.py").write_text(BROKEN_AGENT)
    assert run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q") == 0
    assert run("run", "examples/broken_agent.py:agent", "examples/weather.jsonl", "-q") == 0
    capsys.readouterr()

    assert run("compare") == 1
    out = capsys.readouterr().out
    assert "REGRESSED compare: tool_args: missing get_weather(city='Lisbon'), get_weather(city='Oslo')" in out
    assert out.rstrip().endswith("FAIL: 1 case(s) that passed before now fail")

    assert run("compare", "--max-regressions", "1", "-f", "markdown") == 0
    assert capsys.readouterr().out.startswith("### ✅ No regressions")

    assert run("compare", "latest", "previous") == 0  # the other way round it's a fix
    assert "FIXED compare" in capsys.readouterr().out


def test_compare_needs_two_runs(workdir, capsys):
    run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q")
    capsys.readouterr()
    assert run("compare") == 2
    assert "no previous run" in capsys.readouterr().err
    assert run("compare", "latest", "latest") == 2


def test_agent_with_a_broken_import(workdir, capsys):
    Path("examples/bad_agent.py").write_text("import not_a_real_package\n\ndef agent(q):\n    return q\n")
    assert run("run", "examples/bad_agent.py:agent", "examples/weather.jsonl") == 2
    assert (
        "importing examples/bad_agent.py failed: No module named 'not_a_real_package'"
        in capsys.readouterr().err
    )


def test_name_groups_versions_of_an_agent(workdir, capsys):
    Path("examples/broken_agent.py").write_text(BROKEN_AGENT)
    run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q", "-n", "weather-bot")
    run("run", "examples/broken_agent.py:agent", "examples/weather.jsonl", "-q", "-n", "weather-bot")
    capsys.readouterr()
    assert run("compare", "--agent", "weather-bot") == 1
    with ResultStore("results.sqlite") as store:
        assert store.groups() == [("weather-bot", "examples/weather.jsonl", 2)]


def test_feedback_adds_ready_cases_and_drafts(workdir, capsys):
    assert run("feedback", "examples/feedback.jsonl", "--into", "examples/weather.jsonl", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "5 feedback record(s), 3 new input(s)" in out
    assert "1 skipped" in out  # the Berlin question is already in the dataset
    assert "would add 2 to examples/weather.jsonl" in out
    assert "x2  Will it rain in Lisbon tomorrow?" in out
    assert len(Path("examples/weather.jsonl").read_text().splitlines()) == 4

    assert run("feedback", "examples/feedback.jsonl", "--into", "examples/weather.jsonl") == 0
    out = capsys.readouterr().out
    assert "Fill in expected_output" in out
    assert len(Path("examples/weather.drafts.jsonl").read_text().splitlines()) == 1

    # importing the same log again adds nothing: everything is in the dataset or the drafts
    assert run("feedback", "examples/feedback.jsonl", "--into", "examples/weather.jsonl") == 0
    assert "4 skipped" in capsys.readouterr().out
    assert len(Path("examples/weather.jsonl").read_text().splitlines()) == 6

    # the new cases run, and the toy agent fails the Munich one: it never looks up the weather
    assert run("run", "examples/weather_agent.py:agent", "examples/weather.jsonl", "-q") == 0
    out = capsys.readouterr().out
    assert "4/5 passed" in out


def test_feedback_missing_log(workdir, capsys):
    assert run("feedback", "nope.jsonl", "--into", "examples/weather.jsonl") == 2
    assert "no such file" in capsys.readouterr().err
