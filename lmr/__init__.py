"""
Local Model Router (LMR) - A full-featured local LLM server
============================================================
A drop-in, full-featured local LLM server using llama.cpp (llama-server)
as the inference backend. Provides full API compatibility with both
the native local model API and the OpenAI API.
"""

from lmr.config import LMR_VERSION, LMR_NAME

__version__ = LMR_VERSION
__all__ = ["LMR_VERSION", "LMR_NAME"]
