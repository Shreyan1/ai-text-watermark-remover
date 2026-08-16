"""Stage ② — extractor (document → meaning)."""

from .outline_extractor import LocalLLMExtractor, SentenceFallbackExtractor
from .ollama_extractor import OllamaExtractor

__all__ = ["SentenceFallbackExtractor", "LocalLLMExtractor", "OllamaExtractor"]
