import pytest

from agent_evals.compare import compare_runs
from agent_evals.models import AgentResult, CaseResult, Run, Score, Step, Trace
from agent_evals.report import render_comparison_markdown, render_comparison_text


def make_run(rid, results, cost=0.0, commit=None):
    cases = []
    for case_id, ok, g in results:
        trace = Trace(steps=[Step(type="llm", cost_usd=cost / len(results))])
        cases.append(
            CaseResult(
                case_id=case_id,
                result=AgentResult(output="x", trace=trace),
                scores=[
                    Score(
                        name="tool_args",
                        value=float(ok),
                        passed=ok,
                        reason="" if ok else "missing search(q='x')",
                    ),
                    Score(name="groundedness", value=g, passed=True),
                ],
            )
        )
    return Run(id=rid, agent="bot", dataset="d.jsonl", cases=cases, git_commit=commit)


BASE = make_run(
    "base",
    [("a", True, 1.0), ("b", True, 1.0), ("c", False, 0.8), ("d", True, 0.9)],
    cost=0.10,
    commit="1111111111aa",
)
HEAD = make_run(
    "head",
    [("a", True, 1.0), ("b", False, 0.6), ("c", True, 0.8), ("e", True, 1.0)],
    cost=0.15,
    commit="2222222222bb",
)


def test_case_level_changes():
    cmp = compare_runs(BASE, HEAD)
    assert [c.case_id for c in cmp.regressions] == ["b"]
    assert cmp.regressions[0].reason == "tool_args: missing search(q='x')"
    assert [c.case_id for c in cmp.fixes] == ["c"]
    assert cmp.added == ["e"] and cmp.removed == ["d"]
    # same pass rate on both sides, which is why the gate looks at cases
    assert cmp.pass_rate_delta == 0.0
    assert cmp.score_delta("groundedness") == pytest.approx(0.85 - 0.925)
    assert cmp.cost_delta == pytest.approx(0.05)


def test_gate():
    cmp = compare_runs(BASE, HEAD)
    assert cmp.gate() == ["1 case(s) that passed before now fail"]
    assert cmp.gate(max_regressions=1) == []
    assert cmp.gate(max_regressions=1, max_score_drop=0.05) == [
        "mean groundedness dropped by 0.075 (allowed 0.05)"
    ]
    assert cmp.gate(max_regressions=1, max_cost_increase=0.2) == ["cost went up 50% (allowed 20%)"]
    assert compare_runs(BASE, BASE).gate(max_score_drop=0.0) == []


def test_text_and_markdown():
    cmp = compare_runs(BASE, HEAD)
    problems = cmp.gate()
    text = render_comparison_text(cmp, problems)
    assert text.splitlines()[0] == "base -> head: pass rate 75.0% -> 75.0% (+0.0%), cost $0.1000 -> $0.1500"
    assert "  groundedness   0.925 -> 0.850  -0.075" in text
    assert "  REGRESSED b: tool_args: missing search(q='x')" in text
    assert "  FIXED c" in text
    assert text.endswith("FAIL: 1 case(s) that passed before now fail")

    md = render_comparison_markdown(cmp, problems)
    assert md.startswith("### ❌ Regressions: bot")
    assert "Base run `base` (`1111111111`), new run `head` (`2222222222`)." in md
    assert "| `b` | **regressed** | tool_args: missing search(q='x') |" in md
    assert "| groundedness | 0.925 | 0.850 | -0.075 |" in md
    assert md.rstrip().endswith("Other cases: 1 new, 1 removed.")
    assert render_comparison_markdown(compare_runs(BASE, BASE), []).startswith("### ✅ No regressions")
