#!/usr/bin/env python3
"""
Local Model Router (LMR) - A full-featured local LLM server
============================================================
Author: g023 (https://github.com/g023)
License: MIT

A drop-in, full-featured local LLM server using llama.cpp (llama-server)
as the inference backend. Provides full API compatibility with both
the native local model API and the OpenAI API.

Installation:
    pip install fastapi uvicorn pydantic requests huggingface_hub aiohttp

Usage:
    python lmr.py                           # Default: http://localhost:11434
    python lmr.py --port 8080               # Custom port
    python lmr.py --models-json models.json # Custom model config
    python lmr.py --llama-server /path/to/llama-server  # Custom binary

Models Configuration (models.json):
    {
        "llama3.2": "/path/to/llama-3.2-8b.Q5_K_M.gguf",
        "deepseek-coder": "hf:deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GGUF:deepseek-coder-v2-lite-instruct.Q4_K_M.gguf",
        "qwen2.5": {"path": "/path/to/qwen2.5.gguf", "num_gpu": 99, "ctx_size": 8192}
    }

requirements.txt:
    fastapi>=0.100.0
    uvicorn>=0.23.0
    pydantic>=2.0.0
    requests>=2.28.0
    huggingface_hub>=0.20.0
    aiohttp>=3.9.0

This file is a thin entry point. The actual implementation lives in the lmr/ package.
Run: python lmr.py [options]
Or:  python -m lmr [options]
"""

from lmr.cli import main

if __name__ == "__main__":
    main()
def parse_keep_alive(value: Any) -> float:
    """Parse keep_alive value to seconds. Supports: int (seconds), string like '5m', '1h', '-1' (forever)."""
