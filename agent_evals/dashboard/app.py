"""
The agent-evals dashboard. Start it with `agent-evals dashboard` (or `streamlit run` on this file
with `AGENT_EVALS_DB` set).

Three views over the results store:
- Trends: pass rate, hallucination rate, scorer means and cost over time for one agent + dataset
- Run: the failing cases of one run, each with its full trace
- Case: how a single case did across runs and commits
"""

from __future__ import annotations

import os

import streamlit as st

from agent_evals.dashboard.data import case_rows, trace_rows, trend, trend_table
from agent_evals.report import failure_reason
from agent_evals.store import ResultStore

DB = os.environ.get("AGENT_EVALS_DB", ".agent-evals/results.sqlite")

st.set_page_config(page_title="agent-evals", layout="wide")
store = ResultStore(DB)
groups = store.groups()

st.sidebar.title("agent-evals")
if not groups:
    st.info(f"No runs in `{DB}` yet. Run `agent-evals run <agent> <dataset>` first.")
    st.stop()

labels = [f"{agent} · {dataset} ({n} runs)" for agent, dataset, n in groups]
choice = st.sidebar.selectbox("Agent and dataset", range(len(groups)), format_func=lambda i: labels[i]) or 0
agent, dataset, _ = groups[int(choice)]
view = st.sidebar.radio("View", ["Trends", "Run", "Case"])
st.sidebar.caption(f"Store: `{os.path.basename(DB)}`")

points = trend(store, agent, dataset)
rows = trend_table(points)
latest = points[-1]

if view == "Trends":
    st.header(f"{agent} on {dataset}")
    prev = points[-2] if len(points) > 1 else None
    cols = st.columns(4)
    cols[0].metric(
        "Pass rate", f"{latest.pass_rate:.1%}", f"{latest.pass_rate - prev.pass_rate:+.1%}" if prev else None
    )
    if latest.hallucination_rate is not None:
        delta = None
        if prev and prev.hallucination_rate is not None:
            delta = f"{latest.hallucination_rate - prev.hallucination_rate:+.1%}"
        # fewer hallucinations is better, so a drop shows green
        cols[1].metric("Hallucination rate", f"{latest.hallucination_rate:.1%}", delta, delta_color="inverse")
    cols[2].metric("Cost of last run", f"${latest.cost_usd:.4f}")
    cols[3].metric("Runs", len(points))

    st.subheader("Over time")
    # runs are numbered rather than plotted by time: several runs a day would bunch up otherwise
    x = list(range(1, len(rows) + 1))
    chart = {"run": x, "pass rate": [r["pass_rate"] for r in rows]}
    if any(r["hallucination_rate"] is not None for r in rows):
        chart["hallucination rate"] = [r["hallucination_rate"] for r in rows]
    for key in rows[-1]:
        if key.startswith("score:"):
            chart[key.removeprefix("score:")] = [r[key] for r in rows]
    st.line_chart(chart, x="run")
    if any(r["cost_usd"] > 0 for r in rows):
        st.subheader("Cost per run (USD)")
        st.bar_chart({"run": x, "cost": [r["cost_usd"] for r in rows]}, x="run")
    st.subheader("Runs")
    numbered = [{"#": i, **r} for i, r in zip(x, rows, strict=True)]
    st.dataframe(list(reversed(numbered)), hide_index=True, use_container_width=True)

elif view == "Run":
    when = {p.run_id: f"{p.started_at:%Y-%m-%d %H:%M}" for p in points}
    run_id = st.sidebar.selectbox("Run", list(reversed(when)), format_func=lambda r: f"{r} ({when[r]})")
    run = store.load_run(run_id or latest.run_id)
    failing = [c for c in run.cases if not c.passed]
    st.header(f"Run {run.id}")
    st.caption(
        f"{run.started_at:%Y-%m-%d %H:%M} UTC · commit `{run.git_commit or 'unknown'}` · "
        f"{len(run.cases) - len(failing)}/{len(run.cases)} passed · ${run.cost_usd:.4f}"
    )
    only_failing = st.toggle("Only failing cases", value=True)
    if only_failing and not failing:
        st.success("Every case passed in this run.")
    else:
        st.dataframe(case_rows(run, only_failing=only_failing), hide_index=True, use_container_width=True)
    for c in failing if only_failing else run.cases:
        with st.expander(f"{'❌' if not c.passed else '✅'} {c.case_id}"):
            if not c.passed:
                st.error(failure_reason(c))
            st.markdown("**Answer**")
            st.text(c.result.output or "(no answer)")
            st.markdown("**Scores**")
            st.dataframe(
                [
                    {"scorer": s.name, "value": round(s.value, 3), "passed": s.passed, "reason": s.reason}
                    for s in c.scores
                ],
                hide_index=True,
                use_container_width=True,
            )
            st.markdown("**Trace**")
            st.dataframe(trace_rows(c), hide_index=True, use_container_width=True)

else:
    run = store.load_run(latest.run_id)
    case_id = st.sidebar.selectbox("Case", [c.case_id for c in run.cases]) or run.cases[0].case_id
    history = store.case_history(case_id, agent, dataset, limit=50)
    st.header(f"Case {case_id}")
    passes = sum(h.passed for h in history)
    st.caption(f"Passed {passes} of the last {len(history)} runs")
    st.dataframe(
        [
            {
                "run": h.run_id,
                "when": h.started_at,
                "commit": (h.git_commit or "")[:10],
                "passed": h.passed,
                "error": h.error or "",
            }
            for h in history
        ],
        hide_index=True,
        use_container_width=True,
    )
    if history and 0 < passes < len(history):
        flips = sum(a.passed != b.passed for a, b in zip(history, history[1:], strict=False))
        if flips > 1:
            st.warning(
                f"This case changed result {flips} times. If the agent didn't change, it may be flaky."
            )
