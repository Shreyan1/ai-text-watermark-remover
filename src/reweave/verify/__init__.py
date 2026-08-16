"""Fact preservation, the gate's second, truth-value test.

The semantic guard asks "same topic?"; this asks "same claims?". Embeddings
cannot answer the second (measured: negation pairs score 0.776–0.959), so it
gets its own contract and two complementary implementations:

    ConstraintChecker  rules, numbers, names, negation (NegEx), antonyms.
                       Deterministic and free; blind to reworded reversal.
    NLIChecker         entailment, catches reworded reversal.
                       Costs a model call per claim; bounded by its backend.
    CompositeChecker   both. Rules first; NLI can only add findings.
"""

from .composite import CompositeChecker
from .constraint_checker import ConstraintChecker
from .constraints import extract_constraints
from .nli import NLIBackend, NLIScores
from .nli_checker import NLIChecker
from .ollama_nli import OllamaNLIBackend

__all__ = [
    "CompositeChecker",
    "ConstraintChecker",
    "NLIBackend",
    "NLIChecker",
    "NLIScores",
    "OllamaNLIBackend",
    "extract_constraints",
]
