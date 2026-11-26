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
