"""Utility functions for LMR."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from lmr.config import (
    DEFAULT_CTX_SIZE,
    DEFAULT_KEEP_ALIVE,
    LLAMA_SERVER_PORT_END,
    LLAMA_SERVER_PORT_START,
    logger,
)


def parse_keep_alive(value: Any) -> float:
    """Parse keep_alive value to seconds. Supports: int (seconds), string like '5m', '1h', '-1' (forever)."""
    if value is None:
        return DEFAULT_KEEP_ALIVE
    if isinstance(value, (int, float)):
        if value == -1:
            return -1
        if value == 0:
            return 0
        return float(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "-1":
            return -1
        if value == "0":
            return 0
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(s|m|h|ms)?$", value)
        if match:
            num = float(match.group(1))
            unit = match.group(2) or "s"
            if unit == "ms":
                return num / 1000
            elif unit == "s":
                return num
            elif unit == "m":
                return num * 60
            elif unit == "h":
                return num * 3600
        try:
            return float(value)
        except ValueError:
            return DEFAULT_KEEP_ALIVE
    return DEFAULT_KEEP_ALIVE


def find_free_port(start: int = LLAMA_SERVER_PORT_START, end: int = LLAMA_SERVER_PORT_END) -> int:
    """Find an available port in the given range."""
    for port in range(start, end):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free ports available in range {start}-{end}")


def get_timestamp() -> str:
    """Get current timestamp in ISO format with timezone."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def ns_to_duration(ns: float) -> int:
    """Convert seconds to nanoseconds for API responses."""
    return int(ns * 1e9)


def detect_gpu_memory() -> int:
    """Detect available GPU memory in MB. Returns 0 if no GPU detected."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            memories = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return max(memories) if memories else 0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if "free" in line.lower() or line.strip().isdigit():
                    parts = line.split(",")
                    for p in parts:
                        try:
                            return int(p.strip()) // (1024 * 1024)
                        except ValueError:
                            continue
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return 0


def estimate_context_size(model_size_bytes: int, gpu_memory_mb: int) -> int:
    """Estimate optimal context size based on model size and GPU memory."""
    if gpu_memory_mb <= 0:
        return DEFAULT_CTX_SIZE
    model_size_mb = model_size_bytes / (1024 * 1024)
    available_for_ctx = gpu_memory_mb - (model_size_mb * 1.2) - 1024
    if available_for_ctx <= 0:
        return DEFAULT_CTX_SIZE
    model_ratio = max(1.0, model_size_mb / 4096)
    mb_per_1k_ctx = 8.0 * (model_ratio ** 0.5)
    estimated_ctx = int((available_for_ctx / mb_per_1k_ctx) * 1024)
    return max(DEFAULT_CTX_SIZE, min(estimated_ctx, 32768))


def get_file_size(path: str) -> int:
    """Get file size in bytes, return 0 if not found."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def compute_digest(path: str) -> str:
    """Compute SHA256 digest of a file (first 1MB for speed)."""
    sha = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            sha.update(f.read(1024 * 1024))
    except OSError:
        sha.update(path.encode())
    return f"sha256:{sha.hexdigest()}"


def sanitize_model_name(name: str) -> str:
    """Sanitize model name to prevent path traversal."""
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[/\\]', '_', name)
    sanitized = re.sub(r'\.\.', '_', sanitized)
    return sanitized[:256]
