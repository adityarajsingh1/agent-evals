"""
The `agent-evals` command.

    agent-evals run examples/weather_agent.py:agent examples/weather.jsonl
    agent-evals report latest
    agent-evals compare <base-run> <new-run>

Runs are saved to the results store (`--db`, default `.agent-evals/results.sqlite`), so `report`
and `compare` can refer to them by id, id prefix, or `latest` / `previous`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agent_evals import __version__
from agent_evals.agents import Agent, FunctionAgent
from agent_evals.compare import compare_runs
from agent_evals.dataset import DatasetError, load_dataset
from agent_evals.models import CaseResult, Run
from agent_evals.report import (
    render_comparison_markdown,
    render_comparison_text,
    render_markdown,
    render_text,
)
from agent_evals.runner import run_eval_sync
from agent_evals.scorers import DEFAULT_SCORERS, build_scorers
from agent_evals.store import ResultStore

DEFAULT_DB = ".agent-evals/results.sqlite"


class CLIError(Exception):
    pass


# --- loading agents and scorers ----------------------------------------------------------------


def load_object(spec: str) -> Any:
    """`package.module:attr` or `path/to/file.py:attr`."""
    target, _, attr = spec.partition(":")
    if not attr:
        raise CLIError(f"expected module:attribute, got {spec!r}")
    if target.endswith(".py"):
        path = Path(target)
        if not path.exists():
            raise CLIError(f"no such file: {target}")
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        if module_spec is None or module_spec.loader is None:
            raise CLIError(f"can't import {target}")
        # the file's folder goes on the path so it can import its neighbours, as `python file.py` would
        sys.path.insert(0, str(path.parent.resolve()))
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[path.stem] = module
        try:
            module_spec.loader.exec_module(module)
        except ImportError as e:
            raise CLIError(f"importing {target} failed: {e}") from e
    else:
        sys.path.insert(0, str(Path.cwd()))
        try:
            module = importlib.import_module(target)
        except ImportError as e:
            raise CLIError(f"can't import {target}: {e}") from e
    try:
        return getattr(module, attr)
    except AttributeError:
        raise CLIError(f"{target} has no attribute {attr!r}") from None


def load_agent(spec: str) -> Agent:
    obj = load_object(spec)
    if isinstance(obj, Agent):
        return obj
    if callable(obj):
        # name it after where it came from, so runs of different agents don't all say "agent"
        target, _, attr = spec.partition(":")
        return FunctionAgent(obj, name=f"{Path(target).stem if target.endswith('.py') else target}:{attr}")
    raise CLIError(f"{spec} is neither an agent nor a function")


def parse_scorer(text: str) -> str | dict[str, Any]:
    """`numeric:rel_tol=0.01,abs_tol=0` -> {"name": "numeric", "rel_tol": 0.01, "abs_tol": 0}."""
    name, _, opts = text.partition(":")
    if not opts:
        return name
    spec: dict[str, Any] = {"name": name}
    for pair in opts.split(","):
        key, eq, value = pair.partition("=")
        if not eq:
            raise CLIError(f"bad scorer option {pair!r} in {text!r} (expected key=value)")
        try:
            spec[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            spec[key.strip()] = value
    return spec


# --- output ------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x:.1%}"


def resolve_run(store: ResultStore, ref: str, agent: str | None = None, dataset: str | None = None) -> Run:
    """A run id (or prefix), `latest`, `previous`, or a JSON file written by `run --json`."""
    if ref.endswith(".json") and Path(ref).exists():
        return Run.model_validate_json(Path(ref).read_text(encoding="utf-8"))
    if ref in {"latest", "previous"}:
        runs = store.list_runs(agent=agent, dataset=dataset, limit=2)
        index = 0 if ref == "latest" else 1
        if len(runs) <= index:
            raise CLIError(f"no {ref} run in {store.path}")
        return store.load_run(runs[index].id)
    return store.load_run(ref)


# --- commands ----------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    agent = load_agent(args.agent)
    if args.name:
        agent.name = args.name
    try:
        cases = load_dataset(args.dataset)
    except (DatasetError, OSError) as e:
        raise CLIError(str(e)) from e
    if args.tag:
        cases = [c for c in cases if set(args.tag) & set(c.tags)]
        if not cases:
            raise CLIError(f"no cases tagged {', '.join(args.tag)}")
    scorers = build_scorers([parse_scorer(s) for s in args.scorer] if args.scorer else list(DEFAULT_SCORERS))

    def progress(c: CaseResult) -> None:
        print("." if c.passed else "F", end="", file=sys.stderr, flush=True)

    run = run_eval_sync(
        agent,
        cases,
        scorers,
        dataset=str(args.dataset),
        concurrency=args.concurrency,
        timeout_s=args.timeout,
        on_result=None if args.quiet else progress,
        config={"scorers": [repr(s) for s in scorers]},
    )
    if not args.quiet:
        print(file=sys.stderr)
    if not args.no_save:
        with ResultStore(args.db) as store:
            store.save(run)
    if args.json:
        Path(args.json).write_text(run.model_dump_json(indent=2), encoding="utf-8")
    print("\n" + render_text(run))
    if args.fail_under is not None and run.pass_rate < args.fail_under:
        print(f"\npass rate {_pct(run.pass_rate)} is below --fail-under {_pct(args.fail_under)}")
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with ResultStore(args.db) as store:
        run = resolve_run(store, args.run, args.agent, args.dataset)
    if args.format == "json":
        print(run.model_dump_json(indent=2))
    elif args.format == "markdown":
        print(render_markdown(run, show_all=args.all), end="")
    else:
        print(render_text(run, show_all=args.all, limit=10**9 if args.all else 20))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    with ResultStore(args.db) as store:
        head = resolve_run(store, args.head, args.agent, args.dataset)
        base = resolve_run(store, args.base, args.agent, args.dataset)
    if base.id == head.id:
        raise CLIError(f"both sides are run {base.id}; nothing to compare")
    cmp = compare_runs(base, head)
    problems = cmp.gate(args.max_regressions, args.max_score_drop, args.max_cost_increase)
    if args.format == "markdown":
        print(render_comparison_markdown(cmp, problems), end="")
    else:
        print(render_comparison_text(cmp, problems))
    return 1 if problems else 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError as e:
        raise CLIError("the dashboard needs `pip install agent-evals[dashboard]`") from e
    import os
    import subprocess

    app = Path(__file__).parent / "dashboard" / "app.py"
    env = {**os.environ, "AGENT_EVALS_DB": str(Path(args.db).resolve())}
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(args.port)]
    if args.headless:
        cmd += ["--server.headless", "true"]
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-evals", description="Regression tests for AI agents.")
    parser.add_argument("--version", action="version", version=f"agent-evals {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"results store (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an agent over a dataset and score it")
    run.add_argument("agent", help="module:attr or file.py:attr; an Agent, or a function taking the input")
    run.add_argument("dataset", type=Path, help="JSONL file of test cases")
    run.add_argument("-n", "--name", help="agent name to record (runs are grouped and compared by name)")
    run.add_argument(
        "-s",
        "--scorer",
        action="append",
        metavar="NAME[:opt=value,...]",
        help=f"repeatable; default: {', '.join(DEFAULT_SCORERS)}",
    )
    run.add_argument("-t", "--tag", action="append", help="only run cases with this tag (repeatable)")
    run.add_argument("-j", "--concurrency", type=int, default=8)
    run.add_argument("--timeout", type=float, default=60.0, help="seconds per case")
    run.add_argument("--fail-under", type=float, metavar="RATE", help="exit 1 if the pass rate is below this")
    run.add_argument("--json", metavar="PATH", help="also write the full run as JSON")
    run.add_argument("--no-save", action="store_true", help="don't save the run to the store")
    run.add_argument("-q", "--quiet", action="store_true", help="no progress dots")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="show a saved run")
    report.add_argument(
        "run", nargs="?", default="latest", help="run id or prefix, latest, previous, or a .json file"
    )
    report.add_argument("--agent", help="with latest/previous: only runs of this agent")
    report.add_argument("--dataset", help="with latest/previous: only runs on this dataset")
    report.add_argument("-f", "--format", choices=["text", "markdown", "json"], default="text")
    report.add_argument("-a", "--all", action="store_true", help="list passing cases too")
    report.set_defaults(func=cmd_report)

    compare = sub.add_parser("compare", help="compare two runs and exit 1 on regressions")
    compare.add_argument(
        "base", nargs="?", default="previous", help="the run to compare against (default previous)"
    )
    compare.add_argument("head", nargs="?", default="latest", help="the new run (default latest)")
    compare.add_argument("--agent", help="with latest/previous: only runs of this agent")
    compare.add_argument("--dataset", help="with latest/previous: only runs on this dataset")
    compare.add_argument(
        "--max-regressions", type=int, default=0, help="newly failing cases allowed (default 0)"
    )
    compare.add_argument(
        "--max-score-drop", type=float, metavar="X", help="fail if any scorer's mean drops by more than this"
    )
    compare.add_argument(
        "--max-cost-increase",
        type=float,
        metavar="FRACTION",
        help="fail if the total cost grows by more than this, e.g. 0.2 for 20%%",
    )
    compare.add_argument("-f", "--format", choices=["text", "markdown"], default="text")
    compare.set_defaults(func=cmd_compare)

    dash = sub.add_parser("dashboard", help="open the Streamlit dashboard over the results store")
    dash.add_argument("--port", type=int, default=8501)
    dash.add_argument("--headless", action="store_true", help="don't open a browser")
    dash.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (CLIError, KeyError, ValueError) as e:
        print(f"agent-evals: error: {e}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
