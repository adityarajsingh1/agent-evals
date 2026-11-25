"""Scorers: turn a test case and an agent's result into a `Score`."""

# importing the modules registers the built-in scorers
from agent_evals.scorers import answer, grounding, judge, limits, structure, trajectory  # noqa: F401
from agent_evals.scorers.base import BaseScorer, available_scorers, build_scorers, get_scorer, register

DEFAULT_SCORERS = ["contains", "tool_args", "max_steps", "max_cost"]


def default_scorers() -> list[BaseScorer]:
    """A sensible starting set. Each one only runs on cases that have what it checks."""
    return build_scorers(list(DEFAULT_SCORERS))


__all__ = [
    "DEFAULT_SCORERS",
    "BaseScorer",
    "available_scorers",
    "build_scorers",
    "default_scorers",
    "get_scorer",
    "register",
]
