"""Plugin registry — how the swappable edge plugs into the stable core.

STABLE CORE. Adapters register themselves by (kind, name); the pipeline resolves
them by name from config. This is the mechanism that makes adopting a future
model a one-line change (Invariant I3).

    from reweave.core.registry import register

    @register("regenerator", "llama3-local")
    class Llama3Regenerator(Regenerator):
        ...

Then select `regenerator="llama3-local"` in config. The core never imports the
adapter directly.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple, Type, TypeVar

KIND_SCRUBBER = "scrubber"
KIND_EXTRACTOR = "extractor"
KIND_REGENERATOR = "regenerator"
KIND_SCORER = "scorer"
KIND_GUARD = "guard"
KIND_FACTCHECKER = "factchecker"

_VALID_KINDS = frozenset({
    KIND_SCRUBBER, KIND_EXTRACTOR, KIND_REGENERATOR, KIND_SCORER, KIND_GUARD,
    KIND_FACTCHECKER,
})

_REGISTRY: Dict[Tuple[str, str], Type] = {}

T = TypeVar("T")


def register(kind: str, name: str) -> Callable[[Type[T]], Type[T]]:
    """Class decorator: register an adapter under (kind, name)."""
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown adapter kind {kind!r}; valid: {sorted(_VALID_KINDS)}")

    def deco(cls: Type[T]) -> Type[T]:
        key = (kind, name)
        if key in _REGISTRY:
            raise ValueError(f"{kind}:{name} already registered by {_REGISTRY[key]!r}")
        _REGISTRY[key] = cls
        return cls

    return deco


def resolve(kind: str, name: str) -> Type:
    """Look up a registered adapter class by (kind, name)."""
    try:
        return _REGISTRY[(kind, name)]
    except KeyError:
        available = sorted(n for (k, n) in _REGISTRY if k == kind)
        raise LookupError(
            f"no {kind} named {name!r}. registered {kind}s: {available or '(none)'}"
        ) from None


def available(kind: str | None = None) -> list[Tuple[str, str]]:
    """List registered adapters, optionally filtered by kind."""
    return sorted(k for k in _REGISTRY if kind is None or k[0] == kind)
