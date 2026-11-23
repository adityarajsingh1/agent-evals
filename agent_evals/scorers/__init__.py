"""Scorers: turn a test case and an agent's result into a `Score`."""

# importing the modules registers the built-in scorers
from agent_evals.scorers import answer, structure, trajectory  # noqa: F401
from agent_evals.scorers.base import BaseScorer, available_scorers, build_scorers, get_scorer, register

__all__ = ["BaseScorer", "available_scorers", "build_scorers", "get_scorer", "register"]
