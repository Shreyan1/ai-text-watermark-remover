"""Reweave — rebuild text from meaning into a genuine human voice.

Removing any embedded statistical watermark is a side effect of regeneration, not
a targeted operation. See ARCHITECTURE.md for the contract and the invariants.

Public surface is intentionally small: the core contracts and the Pipeline. Edge
adapters are reached by name through the registry, never imported directly by
callers of the core.
"""

from __future__ import annotations

from .core.pipeline import Pipeline, PipelineConfig
from .core.types import (
    Document,
    HumanSignature,
    Meaning,
    Point,
    TransformResult,
    Verdict,
    VoiceProfile,
)

__version__ = "0.0.1"

__all__ = [
    "Pipeline",
    "PipelineConfig",
    "Document",
    "Meaning",
    "Point",
    "HumanSignature",
    "TransformResult",
    "Verdict",
    "VoiceProfile",
    "__version__",
]
