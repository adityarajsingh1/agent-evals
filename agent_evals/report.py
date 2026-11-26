"""Rendering runs (and comparisons, later) as text for a terminal or Markdown for PRs.

Markdown output is meant for `$GITHUB_STEP_SUMMARY` or a PR comment, so it sticks to tables and
short lines, and keeps long reasons truncated.
"""

from __future__ import annotations

from agent_evals.models import CaseResult, Run


def score_names(run: Run) -> list[str]:
    seen: dict[str, None] = {}
    for c in run.cases:
        for s in c.scores:
            seen.setdefault(s.name, None)
    return list(seen)


def failure_reason(case: CaseResult) -> str:
    if case.result.error:
        return case.result.error
    if not case.scores:
        return "no scorer applied"
    return "; ".join(f"{s.name}: {s.reason or 'failed'}" for s in case.scores if not s.passed)


def _short(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _md_cell(text: str) -> str:
    return _short(text).replace("|", "\\|")


def run_header(run: Run) -> str:
    passed = sum(c.passed for c in run.cases)
    commit = f", commit {run.git_commit[:10]}" if run.git_commit else ""
    return (
        f"{run.agent} on {run.dataset}: {passed}/{len(run.cases)} passed ({run.pass_rate:.1%}), "
        f"cost ${run.cost_usd:.4f}{commit}, run {run.id}"
    )


def render_text(run: Run, show_all: bool = False, limit: int = 20) -> str:
    lines = [run_header(run)]
    for name in score_names(run):
        mean = run.mean_score(name)
        n = sum(1 for c in run.cases if c.score(name) is not None)
        lines.append(f"  {name:<14} {mean:.3f}  ({n} cases)")
    shown = run.cases if show_all else [c for c in run.cases if not c.passed]
    for c in shown[:limit]:
        status = "PASS" if c.passed else "FAIL"
        detail = "" if c.passed else f": {_short(failure_reason(c))}"
        lines.append(f"  {status} {c.case_id}{detail}")
    if len(shown) > limit:
        lines.append(f"  ... and {len(shown) - limit} more")
    return "\n".join(lines)


def render_markdown(run: Run, show_all: bool = False, limit: int = 50) -> str:
    passed = sum(c.passed for c in run.cases)
    icon = "✅" if passed == len(run.cases) else "❌"
    lines = [
        f"### {icon} {run.agent}: {passed}/{len(run.cases)} passed ({run.pass_rate:.1%})",
        "",
        f"Dataset `{run.dataset}`, run `{run.id}`"
        + (f", commit `{run.git_commit[:10]}`" if run.git_commit else "")
        + f", cost ${run.cost_usd:.4f}",
        "",
    ]
    names = score_names(run)
    if names:
        lines += ["| Scorer | Mean | Cases |", "|---|---|---|"]
        for name in names:
            n = sum(1 for c in run.cases if c.score(name) is not None)
            lines.append(f"| {name} | {run.mean_score(name):.3f} | {n} |")
        lines.append("")
    shown = run.cases if show_all else [c for c in run.cases if not c.passed]
    if shown:
        lines += ["| Case | Result | Why |", "|---|---|---|"]
        for c in shown[:limit]:
            why = "" if c.passed else _md_cell(failure_reason(c))
            lines.append(f"| `{c.case_id}` | {'pass' if c.passed else '**fail**'} | {why} |")
        if len(shown) > limit:
            lines.append(f"\n…and {len(shown) - limit} more.")
    return "\n".join(lines) + "\n"
