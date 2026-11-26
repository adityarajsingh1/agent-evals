"""
Keeping run history in SQLite, so runs can be compared over time and across commits.

Each run is stored three ways: a row in `runs` with the summary (pass rate, cost, git commit),
a row per case in `cases`, and a row per score in `scores`. The summary columns make history
queries cheap, and the full run (with traces) is kept as JSON so `load_run` gives back exactly
what was saved.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent_evals.models import Run

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    dataset TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    git_commit TEXT,
    n_cases INTEGER NOT NULL,
    pass_rate REAL NOT NULL,
    cost_usd REAL NOT NULL,
    data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_agent_dataset ON runs (agent, dataset, started_at);

CREATE TABLE IF NOT EXISTS cases (
    run_id TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    passed INTEGER NOT NULL,
    error TEXT,
    output TEXT,
    steps INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    PRIMARY KEY (run_id, case_id)
);

CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    case_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    passed INTEGER NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id, name)
);
CREATE INDEX IF NOT EXISTS scores_name ON scores (name);
"""


@dataclass
class RunSummary:
    id: str
    agent: str
    dataset: str
    started_at: datetime
    git_commit: str | None
    n_cases: int
    pass_rate: float
    cost_usd: float


@dataclass
class CaseHistoryEntry:
    run_id: str
    started_at: datetime
    git_commit: str | None
    passed: bool
    error: str | None


class ResultStore:
    def __init__(self, path: str | Path = ".agent-evals/results.sqlite") -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn:
            yield self._conn

    def save(self, run: Run) -> None:
        """Store a run. Saving the same run id again replaces it."""
        with self._tx() as db:
            db.execute("DELETE FROM runs WHERE id = ?", (run.id,))
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run.id,
                    run.agent,
                    run.dataset,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.git_commit,
                    len(run.cases),
                    run.pass_rate,
                    run.cost_usd,
                    run.model_dump_json(),
                ),
            )
            db.executemany(
                "INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run.id,
                        c.case_id,
                        c.passed,
                        c.result.error,
                        c.result.output,
                        len(c.result.trace.steps),
                        c.result.trace.cost_usd,
                    )
                    for c in run.cases
                ],
            )
            db.executemany(
                "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (run.id, c.case_id, s.name, s.value, s.passed, s.reason)
                    for c in run.cases
                    for s in c.scores
                ],
            )

    def load_run(self, run_id: str) -> Run:
        row = self._conn.execute("SELECT data FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            # let people paste a prefix, like a short git hash
            rows = self._conn.execute("SELECT data FROM runs WHERE id LIKE ?", (run_id + "%",)).fetchall()
            if len(rows) != 1:
                raise KeyError(f"no run {run_id!r}" if not rows else f"{run_id!r} matches {len(rows)} runs")
            row = rows[0]
        return Run.model_validate_json(row["data"])

    def list_runs(
        self, agent: str | None = None, dataset: str | None = None, limit: int = 50
    ) -> list[RunSummary]:
        """Newest first."""
        where, args = [], []
        if agent is not None:
            where.append("agent = ?")
            args.append(agent)
        if dataset is not None:
            where.append("dataset = ?")
            args.append(dataset)
        sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "")
        sql += " ORDER BY started_at DESC LIMIT ?"
        rows = self._conn.execute(sql, (*args, limit)).fetchall()
        return [
            RunSummary(
                r["id"],
                r["agent"],
                r["dataset"],
                datetime.fromisoformat(r["started_at"]),
                r["git_commit"],
                r["n_cases"],
                r["pass_rate"],
                r["cost_usd"],
            )
            for r in rows
        ]

    def latest_run(self, agent: str, dataset: str, before: str | None = None) -> Run | None:
        """The most recent run of `agent` on `dataset`, optionally the one before run id `before`."""
        sql = "SELECT id FROM runs WHERE agent = ? AND dataset = ?"
        args: list[str] = [agent, dataset]
        if before is not None:
            sql += " AND started_at < (SELECT started_at FROM runs WHERE id = ?)"
            args.append(before)
        row = self._conn.execute(sql + " ORDER BY started_at DESC LIMIT 1", args).fetchone()
        return self.load_run(row["id"]) if row else None

    def score_history(
        self, name: str, agent: str, dataset: str, limit: int = 50
    ) -> list[tuple[datetime, float]]:
        """Mean value of one scorer per run, oldest first, e.g. groundedness over the last 50 runs."""
        rows = self._conn.execute(
            """SELECT r.started_at, AVG(s.value) AS mean FROM scores s JOIN runs r ON r.id = s.run_id
               WHERE s.name = ? AND r.agent = ? AND r.dataset = ?
               GROUP BY r.id ORDER BY r.started_at DESC LIMIT ?""",
            (name, agent, dataset, limit),
        ).fetchall()
        return [(datetime.fromisoformat(r["started_at"]), r["mean"]) for r in reversed(rows)]

    def case_history(self, case_id: str, agent: str, dataset: str, limit: int = 20) -> list[CaseHistoryEntry]:
        """How one case did across runs, newest first. Useful for telling a flaky case from a regression."""
        rows = self._conn.execute(
            """SELECT r.id, r.started_at, r.git_commit, c.passed, c.error
               FROM cases c JOIN runs r ON r.id = c.run_id
               WHERE c.case_id = ? AND r.agent = ? AND r.dataset = ?
               ORDER BY r.started_at DESC LIMIT ?""",
            (case_id, agent, dataset, limit),
        ).fetchall()
        return [
            CaseHistoryEntry(
                r["id"],
                datetime.fromisoformat(r["started_at"]),
                r["git_commit"],
                bool(r["passed"]),
                r["error"],
            )
            for r in rows
        ]

    def groups(self) -> list[tuple[str, str, int]]:
        """Every (agent, dataset) pair with its number of runs, most recently run first."""
        rows = self._conn.execute(
            "SELECT agent, dataset, COUNT(*) AS n, MAX(started_at) AS last FROM runs "
            "GROUP BY agent, dataset ORDER BY last DESC"
        ).fetchall()
        return [(r["agent"], r["dataset"], r["n"]) for r in rows]

    def delete_run(self, run_id: str) -> None:
        with self._tx() as db:
            db.execute("DELETE FROM runs WHERE id = ?", (run_id,))
