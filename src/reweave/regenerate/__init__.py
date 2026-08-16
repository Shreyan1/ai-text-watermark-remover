"""Stage ③ — regenerator (meaning → fresh, un-watermarked prose)."""

from .local_regenerator import EchoStubRegenerator, LocalLLMRegenerator
from .ollama_regenerator import OllamaRegenerator

__all__ = ["EchoStubRegenerator", "LocalLLMRegenerator", "OllamaRegenerator"]
