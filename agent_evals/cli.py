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
from agent_evals.dataset import DatasetError, load_dataset
from agent_evals.models import CaseResult, Run
from agent_evals.runner import run_eval_sync
from agent_evals.scorers import DEFAULT_SCORERS, BaseScorer, build_scorers
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
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[path.stem] = module
        module_spec.loader.exec_module(module)
    else:
        sys.path.insert(0, str(Path.cwd()))
        module = importlib.import_module(target)
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


def print_summary(run: Run, scorers: Sequence[BaseScorer]) -> None:
    failed = [c for c in run.cases if not c.passed]
    print(
        f"\n{run.agent} on {run.dataset}: {len(run.cases) - len(failed)}/{len(run.cases)} passed "
        f"({_pct(run.pass_rate)}), cost ${run.cost_usd:.4f}, run {run.id}"
    )
    for s in scorers:
        mean = run.mean_score(s.name)
        if mean is not None:
            n = sum(1 for c in run.cases if c.score(s.name) is not None)
            print(f"  {s.name:<14} {mean:.3f}  ({n} cases)")
    for c in failed[:10]:
        print(f"  FAIL {c.case_id}: {failure_reason(c)}")
    if len(failed) > 10:
        print(f"  ... and {len(failed) - 10} more failing cases")


def failure_reason(case: CaseResult) -> str:
    if case.result.error:
        return case.result.error
    if not case.scores:
        return "no scorer applied"
    return "; ".join(f"{s.name}: {s.reason or 'failed'}" for s in case.scores if not s.passed)


# --- commands ----------------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    agent = load_agent(args.agent)
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
    print_summary(run, scorers)
    if args.fail_under is not None and run.pass_rate < args.fail_under:
        print(f"\npass rate {_pct(run.pass_rate)} is below --fail-under {_pct(args.fail_under)}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-evals", description="Regression tests for AI agents.")
    parser.add_argument("--version", action="version", version=f"agent-evals {__version__}")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"results store (default {DEFAULT_DB})")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an agent over a dataset and score it")
    run.add_argument("agent", help="module:attr or file.py:attr; an Agent, or a function taking the input")
    run.add_argument("dataset", type=Path, help="JSONL file of test cases")
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
