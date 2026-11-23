"""The base class for scorers and the registry that finds them by name.

A scorer looks at one test case and what the agent produced, and returns a `Score`. Built-in
scorers register themselves with `@register`. Scorers in other packages are picked up through
the `agent_evals.scorers` entry point group, so a plugin is one file plus one line of packaging:

    [project.entry-points."agent_evals.scorers"]
    sql_equivalent = "my_package.scorers:SqlEquivalent"
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Any, ClassVar, TypeVar

from agent_evals.models import AgentResult, Score, TestCase


class BaseScorer(ABC):
    name: ClassVar[str]

    def __call__(self, case: TestCase, result: AgentResult) -> Score:
        return self.score(case, result)

    @abstractmethod
    def score(self, case: TestCase, result: AgentResult) -> Score: ...

    def applies_to(self, case: TestCase) -> bool:
        """Whether this case has what the scorer needs (e.g. an expected output).

        The runner skips scorers that don't apply, so one scorer list can serve a dataset where
        some cases check the answer and others only check tool use.
        """
        return True

    def result(self, value: float, passed: bool, reason: str = "") -> Score:
        return Score(name=self.name, value=max(0.0, min(1.0, value)), passed=passed, reason=reason)

    def __repr__(self) -> str:
        opts = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({opts})"


S = TypeVar("S", bound=type[BaseScorer])
_REGISTRY: dict[str, type[BaseScorer]] = {}
_plugins_loaded = False


def register(cls: S) -> S:
    """Class decorator that makes a scorer available by its `name`."""
    name = getattr(cls, "name", None)
    if not name:
        raise TypeError(f"{cls.__name__} needs a `name` to be registered")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ValueError(f"scorer name {name!r} is already used by {existing.__module__}.{existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def _load_plugins() -> None:
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="agent_evals.scorers"):
        cls = ep.load()
        if not (isinstance(cls, type) and issubclass(cls, BaseScorer)):
            raise TypeError(f"entry point {ep.name!r} ({ep.value}) is not a BaseScorer subclass")
        if ep.name not in _REGISTRY:
            _REGISTRY[ep.name] = cls


def available_scorers() -> list[str]:
    _load_plugins()
    return sorted(_REGISTRY)


def get_scorer(name: str, **options: Any) -> BaseScorer:
    """Build a registered scorer by name, e.g. `get_scorer("numeric", rel_tol=0.01)`."""
    _load_plugins()
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown scorer {name!r}; available: {', '.join(available_scorers()) or 'none'}"
        ) from None
    return cls(**options)


def build_scorers(specs: list[str | dict[str, Any]]) -> list[BaseScorer]:
    """Build scorers from config: a name, or a dict with `name` plus options.

    `["exact", {"name": "numeric", "rel_tol": 0.01}]`
    """
    scorers = []
    for spec in specs:
        if isinstance(spec, str):
            scorers.append(get_scorer(spec))
        else:
            opts = dict(spec)
            name = opts.pop("name", None)
            if not isinstance(name, str):
                raise ValueError(f"scorer spec needs a 'name': {spec!r}")
            scorers.append(get_scorer(name, **opts))
    return scorers
