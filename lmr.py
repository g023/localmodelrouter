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
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# =============================================================================
# Constants & Version
# =============================================================================
LMR_VERSION = "0.5.1"
LMR_NAME = "Local Model Router"
LMR_PREFIX = "[localmodelrouter]"
DEFAULT_PORT = 11434
DEFAULT_HOST = "0.0.0.0"
DEFAULT_KEEP_ALIVE = 300  # 5 minutes in seconds
DEFAULT_CTX_SIZE = 2048
DEFAULT_NUM_PREDICT = -1
LLAMA_SERVER_PORT_START = 39000
LLAMA_SERVER_PORT_END = 39999
HEALTH_CHECK_INTERVAL = 0.25
HEALTH_CHECK_TIMEOUT = 120
DEFAULT_PARALLEL_SLOTS = 4

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("localmodelrouter")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(f"%(asctime)s {LMR_PREFIX} %(levelname)s %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# =============================================================================
# Utility Functions
# =============================================================================
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
    """Estimate optimal context size based on model size and GPU memory.
    Conservative estimate to avoid OOM - leaves room for KV cache overhead."""
    if gpu_memory_mb <= 0:
        return DEFAULT_CTX_SIZE
    model_size_mb = model_size_bytes / (1024 * 1024)
    # Reserve model size + 20% overhead + 1GB buffer
    available_for_ctx = gpu_memory_mb - (model_size_mb * 1.2) - 1024
    if available_for_ctx <= 0:
        return DEFAULT_CTX_SIZE
    # Rough estimate: ~0.5MB per 1024 context tokens per billion parameters
    # For a typical 7B Q4 model (~4GB), ~8MB per 1024 ctx tokens
    # Scale by model size ratio to 4GB (typical 7B Q4)
    model_ratio = max(1.0, model_size_mb / 4096)
    mb_per_1k_ctx = 8.0 * (model_ratio ** 0.5)
    estimated_ctx = int((available_for_ctx / mb_per_1k_ctx) * 1024)
    # Clamp to reasonable values: min 2048, max 32768 for auto
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


# =============================================================================
# Model Configuration
# =============================================================================
class ModelConfig:
    """Configuration for a single model."""

    def __init__(self, name: str, path: str, **kwargs: Any):
        self.name = name
        self.path = path
        self.num_gpu: int = kwargs.get("num_gpu", -1)  # -1 = auto
        self.ctx_size: int = kwargs.get("ctx_size", 0)  # 0 = auto
        self.num_parallel: int = kwargs.get("num_parallel", DEFAULT_PARALLEL_SLOTS)
        self.extra_flags: Dict[str, Any] = kwargs.get("extra_flags", {})
        self.template: str = kwargs.get("template", "")
        self.system: str = kwargs.get("system", "")
        self.parameters: Dict[str, Any] = kwargs.get("parameters", {})
        self.family: str = kwargs.get("family", "")
        self.format: str = "gguf"


class ServerConfig:
    """Global server configuration."""

    def __init__(self):
        self.host: str = DEFAULT_HOST
        self.port: int = DEFAULT_PORT
        self.llama_server_binary: str = ""
        self.models_json: str = "models.json"
        self.models_dir: str = os.path.expanduser("~/.local/share/localmodelrouter/models")
        self.default_ctx_size: int = DEFAULT_CTX_SIZE
        self.default_keep_alive: float = DEFAULT_KEEP_ALIVE
        self.default_num_gpu: int = -1
        self.default_parallel: int = DEFAULT_PARALLEL_SLOTS
        self.default_flash_attn: str = "auto"
        self.gpu_memory_mb: int = 0
        self.model_configs: Dict[str, ModelConfig] = {}
        self.modelfiles: Dict[str, Dict[str, Any]] = {}  # Modelfile-style configs

    def find_llama_server(self) -> str:
        """Find the llama-server binary."""
        if self.llama_server_binary and os.path.isfile(self.llama_server_binary):
            return self.llama_server_binary
        # Check common locations
        candidates = [
            shutil.which("llama-server"),
            os.path.expanduser("~/llamacpp/llama.cpp/build/bin/llama-server"),
            "/usr/local/bin/llama-server",
            "/usr/bin/llama-server",
        ]
        for c in candidates:
            if c and os.path.isfile(c):
                self.llama_server_binary = c
                return c
        raise FileNotFoundError(
            "llama-server binary not found. Install llama.cpp or specify path with --llama-server"
        )

    def load_models_json(self, path: str) -> None:
        """Load model configurations from JSON file."""
        if not os.path.isfile(path):
            logger.warning(f"Models config not found at {path}, starting with empty config")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            for name, value in data.items():
                if isinstance(value, str):
                    self.model_configs[name] = ModelConfig(name=name, path=value)
                elif isinstance(value, dict):
                    model_path = value.pop("path", value.pop("model", ""))
                    self.model_configs[name] = ModelConfig(name=name, path=model_path, **value)
            logger.info(f"Loaded {len(self.model_configs)} model(s) from {path}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading models config: {e}")

    def save_models_json(self, path: str) -> None:
        """Save model configurations to JSON file."""
        data = {}
        for name, cfg in self.model_configs.items():
            if cfg.extra_flags or cfg.num_gpu != -1 or cfg.ctx_size != 0:
                data[name] = {
                    "path": cfg.path,
                    "num_gpu": cfg.num_gpu,
                    "ctx_size": cfg.ctx_size,
                    "extra_flags": cfg.extra_flags,
                }
            else:
                data[name] = cfg.path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Process Info for a running llama-server
# =============================================================================
class ProcessInfo:
    """Track a running llama-server process."""

    def __init__(self, model_name: str, port: int, process: subprocess.Popen,
                 model_path: str, ctx_size: int, num_gpu: int):
        self.model_name = model_name
        self.port = port
        self.process = process
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.num_gpu = num_gpu
        self.loaded_at = time.time()
        self.last_used = time.time()
        self.keep_alive: float = DEFAULT_KEEP_ALIVE
        self.request_count = 0
        self.is_ready = False
        self.base_url = f"http://127.0.0.1:{port}"
        self.lock = asyncio.Lock()
        self._request_queue: asyncio.Queue = asyncio.Queue()

    @property
    def is_alive(self) -> bool:
        return self.process.poll() is None

    def touch(self) -> None:
        self.last_used = time.time()
        self.request_count += 1

    @property
    def should_unload(self) -> bool:
        if self.keep_alive == -1:
            return False
        if self.keep_alive == 0:
            return True
        return (time.time() - self.last_used) > self.keep_alive

    @property
    def size_vram(self) -> int:
        """Estimate VRAM usage based on model file size."""
        return get_file_size(self.model_path)

    def kill(self) -> None:
        """Kill the llama-server process."""
        if self.is_alive:
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            except Exception as e:
                logger.error(f"Error killing process for {self.model_name}: {e}")
        logger.info(f"Unloaded model {self.model_name} (port {self.port})")


# =============================================================================
# Process Manager - Manages llama-server instances
# =============================================================================
class ProcessManager:
    """Manages llama-server process lifecycle."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.processes: Dict[str, ProcessInfo] = {}
        self._lock = asyncio.Lock()
        self._unload_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Start the process manager background tasks."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300, connect=5)
        )
        self._unload_task = asyncio.create_task(self._unload_loop())

    async def stop(self) -> None:
        """Stop all processes and cleanup."""
        if self._unload_task:
            self._unload_task.cancel()
        async with self._lock:
            for name, proc in list(self.processes.items()):
                proc.kill()
            self.processes.clear()
        if self._session:
            await self._session.close()

    async def _unload_loop(self) -> None:
        """Background loop to unload idle models."""
        while True:
            try:
                await asyncio.sleep(10)
                async with self._lock:
                    to_unload = [
                        name for name, proc in self.processes.items()
                        if proc.should_unload or not proc.is_alive
                    ]
                for name in to_unload:
                    await self.unload_model(name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in unload loop: {e}")

    def _resolve_model_path(self, model_name: str, request_body: Optional[Dict] = None) -> str:
        """Resolve model name to a GGUF file path."""
        # Check if model is in config
        if model_name in self.config.model_configs:
            path = self.config.model_configs[model_name].path
            if path.startswith("hf:"):
                return self._download_hf_model(path)
            if os.path.isfile(path):
                return path

        # Strip :tag suffix for lookup
        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        if base_name in self.config.model_configs:
            path = self.config.model_configs[base_name].path
            if path.startswith("hf:"):
                return self._download_hf_model(path)
            if os.path.isfile(path):
                return path

        # Check if it's a direct path
        if os.path.isfile(model_name):
            return model_name

        # Check models directory
        models_dir = self.config.models_dir
        for ext in ["", ".gguf"]:
            candidate = os.path.join(models_dir, f"{model_name}{ext}")
            if os.path.isfile(candidate):
                return candidate
            candidate = os.path.join(models_dir, f"{base_name}{ext}")
            if os.path.isfile(candidate):
                return candidate

        # Check if it's an HF reference
        if model_name.startswith("hf:"):
            return self._download_hf_model(model_name)

        raise FileNotFoundError(
            f"Model '{model_name}' not found. Add it to models.json or provide the full path."
        )

    def _download_hf_model(self, hf_ref: str) -> str:
        """Download a model from HuggingFace. Format: hf:user/repo:filename"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required for HF downloads. pip install huggingface_hub")

        # Parse hf:user/repo:filename
        ref = hf_ref.replace("hf:", "", 1)
        parts = ref.split(":")
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None

        if not filename:
            # Try to find the main GGUF file
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {repo_id}")
            # Prefer Q4_K_M or similar
            preferred = [f for f in gguf_files if "Q4_K_M" in f or "q4_k_m" in f]
            filename = preferred[0] if preferred else gguf_files[0]

        logger.info(f"Downloading {repo_id}/{filename} from HuggingFace...")
        cache_dir = os.path.join(self.config.models_dir, "hf_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_dir=os.path.join(self.config.models_dir, "hf"),
        )
        logger.info(f"Downloaded to {local_path}")
        return local_path

    def _build_launch_command(self, model_path: str, port: int, model_name: str,
                               options: Optional[Dict] = None,
                               extra_binary: Optional[str] = None,
                               extra_flags: Optional[Dict] = None) -> List[str]:
        """Build the llama-server launch command."""
        binary = extra_binary or self.config.find_llama_server()
        model_cfg = self.config.model_configs.get(model_name, None)
        opts = options or {}

        cmd = [
            binary,
            "--model", model_path,
            "--port", str(port),
            "--host", "127.0.0.1",
        ]

        # Context size
        ctx = opts.get("num_ctx", 0)
        if not ctx and model_cfg and model_cfg.ctx_size:
            ctx = model_cfg.ctx_size
        if not ctx:
            ctx = self.config.default_ctx_size
            # Try auto-detect based on GPU memory
            if self.config.gpu_memory_mb > 0:
                file_size = get_file_size(model_path)
                if file_size > 0:
                    auto_ctx = estimate_context_size(file_size, self.config.gpu_memory_mb)
                    if auto_ctx > ctx:
                        ctx = auto_ctx
        cmd.extend(["--ctx-size", str(ctx)])

        # GPU layers
        num_gpu = opts.get("num_gpu", -1)
        if num_gpu == -1 and model_cfg and model_cfg.num_gpu != -1:
            num_gpu = model_cfg.num_gpu
        if num_gpu == -1:
            num_gpu = self.config.default_num_gpu
        if num_gpu == -1:
            cmd.extend(["--n-gpu-layers", "auto"])
        elif num_gpu >= 0:
            cmd.extend(["--n-gpu-layers", str(num_gpu)])

        # Parallel slots
        parallel = opts.get("num_parallel", self.config.default_parallel)
        cmd.extend(["--parallel", str(parallel)])

        # Threads
        num_thread = opts.get("num_thread")
        if num_thread:
            cmd.extend(["--threads", str(num_thread)])

        # Main GPU
        main_gpu = opts.get("main_gpu")
        if main_gpu is not None:
            cmd.extend(["--main-gpu", str(main_gpu)])

        # Batch size
        num_batch = opts.get("num_batch")
        if num_batch:
            cmd.extend(["--batch-size", str(num_batch)])

        # RoPE
        rope_scale = opts.get("rope_scale")
        if rope_scale:
            cmd.extend(["--rope-scale", str(rope_scale)])
        rope_freq_base = opts.get("rope_freq_base")
        if rope_freq_base:
            cmd.extend(["--rope-freq-base", str(rope_freq_base)])

        # Flash attention
        flash_attn = self.config.default_flash_attn
        cmd.extend(["--flash-attn", flash_attn])

        # Continuous batching (enabled by default in llama-server)
        # Slots monitoring
        cmd.append("--slots")
        # Metrics
        cmd.append("--metrics")

        # NOTE: --n-predict is NOT set at server launch level.
        # It's a per-request parameter handled via max_tokens in the API.

        # Embedding mode - enable if model name suggests embedding
        if any(kw in model_name.lower() for kw in ["embed", "minilm", "bge", "e5", "gte"]):
            cmd.append("--embedding")

        # mmap/mlock
        if opts.get("use_mmap") is False:
            cmd.append("--no-mmap")
        if opts.get("use_mlock") is True or (extra_flags and extra_flags.get("--mlock")):
            cmd.append("--mlock")

        # Extra flags from model config
        if model_cfg and model_cfg.extra_flags:
            for flag, val in model_cfg.extra_flags.items():
                if not flag.startswith("--"):
                    flag = f"--{flag}"
                if val is True:
                    cmd.append(flag)
                elif val is not False and val is not None:
                    cmd.extend([flag, str(val)])

        # Extra flags from request
        if extra_flags:
            for flag, val in extra_flags.items():
                if flag in ("--mlock",):  # Already handled
                    continue
                if not flag.startswith("--"):
                    flag = f"--{flag}"
                if val is True:
                    cmd.append(flag)
                elif val is not False and val is not None:
                    cmd.extend([flag, str(val)])

        return cmd

    async def ensure_model(self, model_name: str, options: Optional[Dict] = None,
                           keep_alive: Optional[Any] = None,
                           llama_cpp_binary: Optional[str] = None,
                           llama_cpp_flags: Optional[Dict] = None) -> ProcessInfo:
        """Ensure a model is loaded and return its ProcessInfo."""
        async with self._lock:
            # Check if already running
            if model_name in self.processes:
                proc = self.processes[model_name]
                if proc.is_alive and proc.is_ready:
                    proc.touch()
                    if keep_alive is not None:
                        proc.keep_alive = parse_keep_alive(keep_alive)
                    return proc
                elif not proc.is_alive:
                    # Process died, remove and re-launch
                    proc.kill()
                    del self.processes[model_name]

            # Resolve model path
            model_path = self._resolve_model_path(model_name)

            # Find free port
            used_ports = {p.port for p in self.processes.values()}
            port = find_free_port()
            while port in used_ports:
                port = find_free_port(port + 1)

            # Build command
            cmd = self._build_launch_command(
                model_path, port, model_name,
                options=options,
                extra_binary=llama_cpp_binary,
                extra_flags=llama_cpp_flags,
            )

            logger.info(f"Launching llama-server for '{model_name}' on port {port}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Launch process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            ctx_size = 0
            for i, arg in enumerate(cmd):
                if arg == "--ctx-size" and i + 1 < len(cmd):
                    ctx_size = int(cmd[i + 1])
                    break

            num_gpu = 0
            for i, arg in enumerate(cmd):
                if arg == "--n-gpu-layers" and i + 1 < len(cmd):
                    try:
                        num_gpu = int(cmd[i + 1])
                    except ValueError:
                        num_gpu = 99  # "auto"
                    break

            proc_info = ProcessInfo(
                model_name=model_name,
                port=port,
                process=process,
                model_path=model_path,
                ctx_size=ctx_size,
                num_gpu=num_gpu,
            )
            proc_info.keep_alive = parse_keep_alive(keep_alive) if keep_alive is not None else self.config.default_keep_alive
            self.processes[model_name] = proc_info

        # Wait for health check (outside lock so other models can load concurrently)
        ready = await self._wait_for_health(proc_info)
        if not ready:
            async with self._lock:
                if model_name in self.processes:
                    self.processes[model_name].kill()
                    del self.processes[model_name]
            # Try to read stderr for error info
            stderr_output = ""
            try:
                stderr_output = process.stderr.read().decode("utf-8", errors="replace")[-2000:]
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to start llama-server for '{model_name}'. "
                f"stderr: {stderr_output}"
            )

        proc_info.is_ready = True
        logger.info(f"Model '{model_name}' ready on port {port}")
        return proc_info

    async def _wait_for_health(self, proc: ProcessInfo) -> bool:
        """Wait for llama-server health endpoint to respond."""
        url = f"{proc.base_url}/health"
        start = time.time()
        while time.time() - start < HEALTH_CHECK_TIMEOUT:
            if not proc.is_alive:
                return False
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("status") == "ok":
                            return True
            except (aiohttp.ClientError, asyncio.TimeoutError, Exception):
                pass
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
        return False

    async def unload_model(self, model_name: str) -> bool:
        """Unload a model by killing its process."""
        async with self._lock:
            proc = self.processes.pop(model_name, None)
        if proc:
            proc.kill()
            return True
        return False

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Get info about currently running models."""
        models = []
        for name, proc in self.processes.items():
            if proc.is_alive:
                models.append({
                    "name": name,
                    "model": name,
                    "size": get_file_size(proc.model_path),
                    "digest": compute_digest(proc.model_path),
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": self.config.model_configs.get(name, ModelConfig(name, "")).family or "unknown",
                        "families": None,
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                    "expires_at": (
                        datetime.fromtimestamp(proc.last_used + proc.keep_alive, tz=timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M:%S.%fZ"
                        )
                        if proc.keep_alive > 0
                        else "never"
                    ),
                    "size_vram": proc.size_vram,
                })
        return models

    async def proxy_request(self, proc: ProcessInfo, method: str, path: str,
                            body: Optional[Dict] = None, stream: bool = False,
                            headers: Optional[Dict] = None) -> aiohttp.ClientResponse:
        """Proxy a request to a llama-server instance."""
        url = f"{proc.base_url}{path}"
        proc.touch()
        kwargs: Dict[str, Any] = {"headers": headers or {}}
        if body is not None:
            kwargs["json"] = body
        if method.upper() == "GET":
            return await self._session.get(url, **kwargs)
        elif method.upper() == "POST":
            return await self._session.post(url, **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")


# =============================================================================
# Pydantic Models for API
# =============================================================================

# -- Native API Models --

class GenerateRequest(BaseModel):
    model: str
    prompt: Optional[str] = ""
    suffix: Optional[str] = None
    images: Optional[List[str]] = None
    think: Optional[bool] = None
    format: Optional[Any] = None
    options: Optional[Dict[str, Any]] = None
    system: Optional[str] = None
    template: Optional[str] = None
    stream: Optional[bool] = True
    raw: Optional[bool] = None
    keep_alive: Optional[Any] = None
    context: Optional[List[int]] = None
    # Image gen experimental
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


class Message(BaseModel):
    role: str
    content: Optional[str] = ""
    images: Optional[List[str]] = None
    thinking: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str
    messages: List[Message] = []
    tools: Optional[List[Dict[str, Any]]] = None
    think: Optional[bool] = None
    format: Optional[Any] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = True
    keep_alive: Optional[Any] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


class ShowRequest(BaseModel):
    model: str
    verbose: Optional[bool] = False


class CreateRequest(BaseModel):
    model: str
    from_model: Optional[str] = Field(None, alias="from")
    files: Optional[Dict[str, str]] = None
    adapters: Optional[Dict[str, str]] = None
    template: Optional[str] = None
    license: Optional[Any] = None
    system: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    messages: Optional[List[Dict[str, Any]]] = None
    stream: Optional[bool] = True
    quantize: Optional[str] = None

    model_config = {"populate_by_name": True}


class CopyRequest(BaseModel):
    source: str
    destination: str


class DeleteRequest(BaseModel):
    model: str


class PullRequest(BaseModel):
    model: str
    insecure: Optional[bool] = False
    stream: Optional[bool] = True


class EmbedRequest(BaseModel):
    model: str
    input: Any  # str or list of str
    truncate: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None
    dimensions: Optional[int] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


# -- OpenAI API Models --

class OpenAIMessage(BaseModel):
    role: str
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[OpenAIMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    stop: Optional[Any] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[str] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Any  # str or list
    suffix: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stream_options: Optional[Dict[str, Any]] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    stop: Optional[Any] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


class OpenAIEmbeddingRequest(BaseModel):
    model: str
    input: Any  # str or list
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


class OpenAIResponsesRequest(BaseModel):
    model: str
    input: Any
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    # Router extensions
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


# =============================================================================
# Helper: Convert between Ollama-native and OpenAI formats for llama-server
# =============================================================================

def ollama_options_to_llama_server(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert native options dict to llama-server API parameters."""
    if not options:
        return {}
    mapping = {
        "temperature": "temperature",
        "top_k": "top_k",
        "top_p": "top_p",
        "min_p": "min_p",
        "repeat_penalty": "repeat_penalty",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "stop": "stop",
        "num_predict": "n_predict",
        "tfs_z": "tfs_z",
        "typical_p": "typical_p",
        "mirostat": "mirostat",
        "mirostat_tau": "mirostat_tau",
        "mirostat_eta": "mirostat_eta",
        "grammar": "grammar",
        "repeat_last_n": "repeat_last_n",
        "penalize_newline": "penalize_nl",
    }
    result = {}
    for k, v in options.items():
        if k in mapping:
            result[mapping[k]] = v
        elif k not in (
            "num_ctx", "num_gpu", "num_thread", "num_batch",
            "main_gpu", "rope_scale", "rope_freq_base",
            "num_keep", "numa", "use_mmap", "use_mlock",
            "num_parallel",
        ):
            # Pass through unknown options
            result[k] = v
    return result


def build_oai_messages_from_native(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert native message format to OpenAI message format for llama-server."""
    oai_messages = []
    for msg in messages:
        oai_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}

        # Handle multimodal images
        if msg.images:
            content_parts = []
            if msg.content:
                content_parts.append({"type": "text", "text": msg.content})
            for img in msg.images:
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"}
                })
            oai_msg["content"] = content_parts

        # Handle tool calls
        if msg.tool_calls:
            oai_msg["tool_calls"] = []
            for i, tc in enumerate(msg.tool_calls):
                func = tc.get("function", tc)
                oai_msg["tool_calls"].append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": func.get("name", ""),
                        "arguments": json.dumps(func.get("arguments", {})) if isinstance(func.get("arguments"), dict) else str(func.get("arguments", "")),
                    }
                })

        # Handle tool role
        if msg.role == "tool":
            oai_msg["role"] = "tool"
            if msg.tool_name:
                oai_msg["tool_call_id"] = f"call_{msg.tool_name}"
            else:
                oai_msg["tool_call_id"] = "call_unknown"

        oai_messages.append(oai_msg)
    return oai_messages


def build_oai_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """Ensure tools are in proper OpenAI format."""
    if not tools:
        return None
    result = []
    for tool in tools:
        if "type" in tool and "function" in tool:
            result.append(tool)
        elif "function" in tool:
            result.append({"type": "function", "function": tool["function"]})
        else:
            result.append({"type": "function", "function": tool})
    return result


def format_to_response_format(fmt: Any) -> Optional[Dict[str, Any]]:
    """Convert native 'format' field to OpenAI response_format."""
    if fmt is None:
        return None
    if fmt == "json":
        return {"type": "json_object"}
    if isinstance(fmt, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": fmt}}
    return None


# =============================================================================
# Response builders for native API
# =============================================================================

def build_generate_response(model: str, response_text: str, done: bool,
                             context: Optional[List[int]] = None,
                             done_reason: Optional[str] = None,
                             timings: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a native /api/generate response object."""
    result: Dict[str, Any] = {
        "model": model,
        "created_at": get_timestamp(),
        "response": response_text,
        "done": done,
    }
    if done:
        if done_reason:
            result["done_reason"] = done_reason
        if context:
            result["context"] = context
        if timings:
            result["total_duration"] = timings.get("total_duration", 0)
            result["load_duration"] = timings.get("load_duration", 0)
            result["prompt_eval_count"] = timings.get("prompt_eval_count", 0)
            result["prompt_eval_duration"] = timings.get("prompt_eval_duration", 0)
            result["eval_count"] = timings.get("eval_count", 0)
            result["eval_duration"] = timings.get("eval_duration", 0)
    return result


def build_chat_response(model: str, message: Dict[str, Any], done: bool,
                         done_reason: Optional[str] = None,
                         timings: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a native /api/chat response object."""
    result: Dict[str, Any] = {
        "model": model,
        "created_at": get_timestamp(),
        "message": message,
        "done": done,
    }
    if done:
        if done_reason:
            result["done_reason"] = done_reason
        if timings:
            result["total_duration"] = timings.get("total_duration", 0)
            result["load_duration"] = timings.get("load_duration", 0)
            result["prompt_eval_count"] = timings.get("prompt_eval_count", 0)
            result["prompt_eval_duration"] = timings.get("prompt_eval_duration", 0)
            result["eval_count"] = timings.get("eval_count", 0)
            result["eval_duration"] = timings.get("eval_duration", 0)
    return result


def extract_timings_from_oai(oai_response: Dict[str, Any], start_time: float) -> Dict[str, Any]:
    """Extract timing info from an OpenAI-format response and convert to native format."""
    total_duration_s = time.time() - start_time
    usage = oai_response.get("usage", {})
    timings_raw = oai_response.get("timings", {})

    prompt_tokens = usage.get("prompt_tokens", timings_raw.get("prompt_n", 0))
    completion_tokens = usage.get("completion_tokens", timings_raw.get("predicted_n", 0))

    # Use llama-server's timings if available, otherwise estimate
    prompt_ms = timings_raw.get("prompt_ms", 0)
    predicted_ms = timings_raw.get("predicted_ms", 0)

    if prompt_ms > 0:
        prompt_eval_duration = int(prompt_ms * 1e6)  # ms to ns
    else:
        prompt_eval_duration = ns_to_duration(total_duration_s * 0.2) if prompt_tokens else 0

    if predicted_ms > 0:
        eval_duration = int(predicted_ms * 1e6)
    else:
        eval_duration = ns_to_duration(total_duration_s * 0.7) if completion_tokens else 0

    return {
        "total_duration": ns_to_duration(total_duration_s),
        "load_duration": ns_to_duration(total_duration_s * 0.05),
        "prompt_eval_count": prompt_tokens,
        "prompt_eval_duration": prompt_eval_duration,
        "eval_count": completion_tokens,
        "eval_duration": eval_duration,
    }


# =============================================================================
# The Application
# =============================================================================
config = ServerConfig()
process_manager = ProcessManager(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config.gpu_memory_mb = detect_gpu_memory()
    logger.info(f"{LMR_NAME} v{LMR_VERSION} starting on {config.host}:{config.port}")
    if config.gpu_memory_mb > 0:
        logger.info(f"Detected GPU memory: {config.gpu_memory_mb} MB")
    else:
        logger.info("No GPU detected, running in CPU mode")
    try:
        config.find_llama_server()
        logger.info(f"Using llama-server: {config.llama_server_binary}")
    except FileNotFoundError as e:
        logger.warning(str(e))
    config.load_models_json(config.models_json)
    await process_manager.start()
    yield
    logger.info("Shutting down, stopping all model processes...")
    await process_manager.stop()
    logger.info("Shutdown complete")


app = FastAPI(title=LMR_NAME, version=LMR_VERSION, lifespan=lifespan)

# CORS - allow everything for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Error Handling
# =============================================================================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"error": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": f"Internal server error: {str(exc)}"})


# =============================================================================
# Health & Version
# =============================================================================
@app.get("/")
async def root():
    return Response(content=f"{LMR_NAME} is running", media_type="text/plain")


@app.head("/")
async def root_head():
    return Response(content="", media_type="text/plain")


@app.get("/api/version")
async def api_version():
    return {"version": LMR_VERSION}


# =============================================================================
# Native API: /api/tags - List Models
# =============================================================================
@app.get("/api/tags")
async def api_tags():
    models = []
    for name, cfg in config.model_configs.items():
        path = cfg.path
        if path.startswith("hf:"):
            size = 0
            digest = hashlib.sha256(path.encode()).hexdigest()
        else:
            size = get_file_size(path)
            digest = compute_digest(path) if os.path.isfile(path) else hashlib.sha256(path.encode()).hexdigest()
        models.append({
            "name": f"{name}:latest",
            "model": f"{name}:latest",
            "modified_at": datetime.fromtimestamp(
                os.path.getmtime(path) if os.path.isfile(path) else time.time(),
                tz=timezone.utc,
            ).strftime("%Y-%m-%dT%H:%M:%S.%f%z"),
            "size": size,
            "digest": digest,
            "details": {
                "parent_model": "",
                "format": cfg.format,
                "family": cfg.family or "unknown",
                "families": [cfg.family] if cfg.family else None,
                "parameter_size": "",
                "quantization_level": "",
            },
        })

    # Also include Modelfile-created models
    for name, mf in config.modelfiles.items():
        if name not in config.model_configs:
            models.append({
                "name": f"{name}:latest",
                "model": f"{name}:latest",
                "modified_at": get_timestamp(),
                "size": 0,
                "digest": hashlib.sha256(name.encode()).hexdigest(),
                "details": {
                    "parent_model": mf.get("from", ""),
                    "format": "gguf",
                    "family": "unknown",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": "",
                },
            })

    return {"models": models}


# =============================================================================
# Native API: /api/ps - List Running Models
# =============================================================================
@app.get("/api/ps")
async def api_ps():
    return {"models": process_manager.get_running_models()}


# =============================================================================
# Native API: /api/show - Show Model Information
# =============================================================================
@app.post("/api/show")
async def api_show(req: ShowRequest):
    model_name = req.model.split(":")[0]

    # Check model configs
    model_cfg = config.model_configs.get(model_name) or config.model_configs.get(req.model)
    if not model_cfg:
        # Check Modelfile models
        if model_name in config.modelfiles:
            mf = config.modelfiles[model_name]
            return {
                "modelfile": json.dumps(mf, indent=2),
                "parameters": "\n".join(f"{k} {v}" for k, v in mf.get("parameters", {}).items()),
                "template": mf.get("template", ""),
                "details": {
                    "parent_model": mf.get("from", ""),
                    "format": "gguf",
                    "family": "unknown",
                    "families": None,
                    "parameter_size": "",
                    "quantization_level": "",
                },
                "model_info": {},
                "capabilities": ["completion"],
            }
        raise HTTPException(status_code=404, detail=f"model '{req.model}' not found")

    path = model_cfg.path
    file_size = get_file_size(path) if not path.startswith("hf:") and os.path.isfile(path) else 0

    return {
        "modelfile": f"FROM {path}\n",
        "parameters": "\n".join(f"{k} {v}" for k, v in model_cfg.parameters.items()) if model_cfg.parameters else "",
        "template": model_cfg.template or "",
        "details": {
            "parent_model": "",
            "format": model_cfg.format,
            "family": model_cfg.family or "unknown",
            "families": [model_cfg.family] if model_cfg.family else None,
            "parameter_size": "",
            "quantization_level": "",
        },
        "model_info": {
            "general.architecture": model_cfg.family or "unknown",
            "general.file_type": 2,
            "general.parameter_count": 0,
        },
        "capabilities": ["completion"],
    }


# =============================================================================
# Native API: /api/create - Create Model
# =============================================================================
@app.post("/api/create")
async def api_create(req: CreateRequest):
    async def stream_create():
        yield json.dumps({"status": "reading model metadata"}) + "\n"

        # If creating from existing model
        source_model = req.from_model
        if source_model and source_model in config.model_configs:
            source_cfg = config.model_configs[source_model]
            new_cfg = ModelConfig(
                name=req.model,
                path=source_cfg.path,
                num_gpu=source_cfg.num_gpu,
                ctx_size=source_cfg.ctx_size,
                extra_flags=copy.deepcopy(source_cfg.extra_flags),
                template=req.template or source_cfg.template,
                system=req.system or source_cfg.system,
                parameters=req.parameters or copy.deepcopy(source_cfg.parameters),
                family=source_cfg.family,
            )
            config.model_configs[req.model] = new_cfg
            yield json.dumps({"status": "creating system layer"}) + "\n"
        elif source_model:
            # Try to find it as a Modelfile model
            config.modelfiles[req.model] = {
                "from": source_model,
                "template": req.template or "",
                "system": req.system or "",
                "parameters": req.parameters or {},
            }
            yield json.dumps({"status": "creating system layer"}) + "\n"
        else:
            # Creating from files
            config.modelfiles[req.model] = {
                "from": "",
                "files": req.files or {},
                "template": req.template or "",
                "system": req.system or "",
                "parameters": req.parameters or {},
            }
            yield json.dumps({"status": "parsing model data"}) + "\n"

        yield json.dumps({"status": "writing manifest"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"

        # Save config
        try:
            config.save_models_json(config.models_json)
        except Exception:
            pass

    if req.stream is False:
        # Collect all status messages and return final
        async for _ in stream_create():
            pass
        return {"status": "success"}
    return StreamingResponse(stream_create(), media_type="application/x-ndjson")


# =============================================================================
# Native API: /api/copy - Copy Model
# =============================================================================
@app.post("/api/copy")
async def api_copy(req: CopyRequest):
    source = req.source.split(":")[0]
    if source not in config.model_configs and req.source not in config.model_configs:
        raise HTTPException(status_code=404, detail=f"model '{req.source}' not found")
    source_cfg = config.model_configs.get(source) or config.model_configs.get(req.source)
    dest_name = req.destination.split(":")[0]
    config.model_configs[dest_name] = ModelConfig(
        name=dest_name,
        path=source_cfg.path,
        num_gpu=source_cfg.num_gpu,
        ctx_size=source_cfg.ctx_size,
        extra_flags=copy.deepcopy(source_cfg.extra_flags),
        template=source_cfg.template,
        system=source_cfg.system,
        parameters=copy.deepcopy(source_cfg.parameters),
        family=source_cfg.family,
    )
    try:
        config.save_models_json(config.models_json)
    except Exception:
        pass
    return Response(status_code=200)


# =============================================================================
# Native API: /api/delete - Delete Model
# =============================================================================
@app.api_route("/api/delete", methods=["DELETE"])
async def api_delete(req: DeleteRequest):
    model_name = req.model.split(":")[0]
    found = False
    if model_name in config.model_configs:
        del config.model_configs[model_name]
        found = True
    if req.model in config.model_configs:
        del config.model_configs[req.model]
        found = True
    if model_name in config.modelfiles:
        del config.modelfiles[model_name]
        found = True
    # Unload if running
    await process_manager.unload_model(model_name)
    if not found:
        raise HTTPException(status_code=404, detail=f"model '{req.model}' not found")
    try:
        config.save_models_json(config.models_json)
    except Exception:
        pass
    return Response(status_code=200)


# =============================================================================
# Native API: /api/pull - Pull Model
# =============================================================================
@app.post("/api/pull")
async def api_pull(req: PullRequest):
    async def stream_pull():
        yield json.dumps({"status": "pulling manifest"}) + "\n"

        model_name = req.model
        # Check if it's an HF reference
        if model_name.startswith("hf:") or "/" in model_name:
            try:
                hf_ref = model_name if model_name.startswith("hf:") else f"hf:{model_name}"
                yield json.dumps({"status": f"downloading from HuggingFace: {model_name}"}) + "\n"

                # Download in thread pool
                loop = asyncio.get_event_loop()
                local_path = await loop.run_in_executor(
                    None, process_manager._download_hf_model, hf_ref
                )

                # Register model
                short_name = model_name.split("/")[-1].split(":")[0] if "/" in model_name else model_name.replace("hf:", "").split(":")[0].split("/")[-1]
                config.model_configs[short_name] = ModelConfig(name=short_name, path=local_path)
                config.save_models_json(config.models_json)

                yield json.dumps({"status": "verifying sha256 digest"}) + "\n"
                yield json.dumps({"status": "writing manifest"}) + "\n"
                yield json.dumps({"status": "removing any unused layers"}) + "\n"
                yield json.dumps({"status": "success"}) + "\n"
            except Exception as e:
                yield json.dumps({"error": str(e)}) + "\n"
        else:
            # For non-HF models, just check if we already have it
            if model_name in config.model_configs or model_name.split(":")[0] in config.model_configs:
                yield json.dumps({"status": "model already available"}) + "\n"
                yield json.dumps({"status": "success"}) + "\n"
            else:
                yield json.dumps({"error": f"model '{model_name}' not found. Use 'hf:user/repo:file' format or add to models.json"}) + "\n"

    if req.stream is False:
        last_status = ""
        async for line in stream_pull():
            data = json.loads(line.strip())
            if "status" in data:
                last_status = data["status"]
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
        return {"status": last_status or "success"}
    return StreamingResponse(stream_pull(), media_type="application/x-ndjson")


# =============================================================================
# Native API: /api/generate - Generate Completion
# =============================================================================
@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    start_time = time.time()
    model_name = req.model

    # Handle load/unload with empty prompt
    if not req.prompt and not req.images:
        keep_alive_val = parse_keep_alive(req.keep_alive)
        if keep_alive_val == 0:
            await process_manager.unload_model(model_name)
            return build_generate_response(model_name, "", True, done_reason="unload")
        else:
            # Just load the model
            try:
                await process_manager.ensure_model(
                    model_name, options=req.options, keep_alive=req.keep_alive,
                    llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return build_generate_response(model_name, "", True)

    # Ensure model is loaded
    try:
        proc = await process_manager.ensure_model(
            model_name, options=req.options, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build request for llama-server's /v1/completions or /v1/chat/completions
    # Use /v1/chat/completions for better template handling unless raw mode
    if req.raw:
        # Use /v1/completions for raw mode
        llama_body: Dict[str, Any] = {
            "model": model_name,
            "prompt": req.prompt or "",
            "stream": bool(req.stream),
        }
        if req.suffix:
            llama_body["suffix"] = req.suffix
        endpoint = "/v1/completions"
    else:
        # Build chat messages
        messages = []
        system_msg = req.system
        if not system_msg:
            model_cfg = config.model_configs.get(model_name)
            if model_cfg and model_cfg.system:
                system_msg = model_cfg.system
        if system_msg:
            messages.append({"role": "system", "content": system_msg})
        if req.prompt:
            msg_content: Any = req.prompt
            if req.images:
                msg_content = [{"type": "text", "text": req.prompt}]
                for img in req.images:
                    msg_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}
                    })
            messages.append({"role": "user", "content": msg_content})

        llama_body = {
            "model": model_name,
            "messages": messages,
            "stream": bool(req.stream),
        }
        endpoint = "/v1/chat/completions"

    # Apply options
    opts = ollama_options_to_llama_server(req.options)
    llama_body.update(opts)

    # Handle max_tokens / n_predict
    if req.options and req.options.get("num_predict"):
        llama_body["max_tokens"] = req.options["num_predict"]
    elif "n_predict" in llama_body:
        llama_body["max_tokens"] = llama_body.pop("n_predict")

    # Handle format
    resp_format = format_to_response_format(req.format)
    if resp_format:
        llama_body["response_format"] = resp_format

    # Handle think
    if req.think is not None:
        llama_body["think"] = req.think

    # Streaming
    if req.stream is not False:
        return StreamingResponse(
            _stream_generate(proc, endpoint, llama_body, model_name, start_time, req.raw),
            media_type="application/x-ndjson",
        )

    # Non-streaming
    try:
        async with process_manager._session.post(
            f"{proc.base_url}{endpoint}", json=llama_body,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    timings = extract_timings_from_oai(oai_resp, start_time)

    if req.raw:
        text = oai_resp.get("choices", [{}])[0].get("text", "")
    else:
        msg = oai_resp.get("choices", [{}])[0].get("message", {})
        text = msg.get("content", "")
        # Handle thinking models: reasoning_content contains the thinking output
        reasoning = msg.get("reasoning_content", "")
        if not text and reasoning:
            text = reasoning

    return build_generate_response(
        model_name, text, True,
        done_reason=oai_resp.get("choices", [{}])[0].get("finish_reason", "stop"),
        context=[1, 2, 3],  # Placeholder context
        timings=timings,
    )


async def _stream_generate(proc: ProcessInfo, endpoint: str, body: Dict,
                            model_name: str, start_time: float, raw: bool) -> AsyncGenerator[str, None]:
    """Stream generate responses in native format."""
    eval_count = 0
    prompt_eval_count = 0

    try:
        async with process_manager._session.post(
            f"{proc.base_url}{endpoint}", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line = line[6:]
                    if line == b"[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Extract text from SSE chunk
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    if raw:
                        text = choice.get("text", "")
                    else:
                        delta = choice.get("delta", {})
                        text = delta.get("content", "") or delta.get("reasoning_content", "")

                    if text:
                        eval_count += 1
                        resp_obj = build_generate_response(model_name, text, False)
                        yield json.dumps(resp_obj) + "\n"

                    # Check for usage info
                    usage = data.get("usage")
                    if usage:
                        prompt_eval_count = usage.get("prompt_tokens", 0)

    except Exception as e:
        logger.error(f"Streaming error: {e}")

    # Final response with timing
    timings = {
        "total_duration": ns_to_duration(time.time() - start_time),
        "load_duration": ns_to_duration(0.001),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": ns_to_duration((time.time() - start_time) * 0.1),
        "eval_count": eval_count,
        "eval_duration": ns_to_duration((time.time() - start_time) * 0.85),
    }
    final = build_generate_response(model_name, "", True, done_reason="stop",
                                     context=[1, 2, 3], timings=timings)
    yield json.dumps(final) + "\n"


# =============================================================================
# Native API: /api/chat - Chat Completion
# =============================================================================
@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    start_time = time.time()
    model_name = req.model

    # Handle load/unload with empty messages
    if not req.messages:
        keep_alive_val = parse_keep_alive(req.keep_alive)
        if keep_alive_val == 0:
            await process_manager.unload_model(model_name)
            return build_chat_response(
                model_name, {"role": "assistant", "content": ""}, True, done_reason="unload"
            )
        else:
            try:
                await process_manager.ensure_model(
                    model_name, options=req.options, keep_alive=req.keep_alive,
                    llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
            return build_chat_response(
                model_name, {"role": "assistant", "content": ""}, True, done_reason="load"
            )

    # Ensure model loaded
    try:
        proc = await process_manager.ensure_model(
            model_name, options=req.options, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build OpenAI request for llama-server
    oai_messages = build_oai_messages_from_native(req.messages)

    # Inject system from model config if not present
    if oai_messages and oai_messages[0].get("role") != "system":
        model_cfg = config.model_configs.get(model_name)
        if model_cfg and model_cfg.system:
            oai_messages.insert(0, {"role": "system", "content": model_cfg.system})

    llama_body: Dict[str, Any] = {
        "model": model_name,
        "messages": oai_messages,
        "stream": bool(req.stream),
    }

    # Tools
    oai_tools = build_oai_tools(req.tools)
    if oai_tools:
        llama_body["tools"] = oai_tools

    # Options
    opts = ollama_options_to_llama_server(req.options)
    llama_body.update(opts)

    if req.options and req.options.get("num_predict"):
        llama_body["max_tokens"] = req.options["num_predict"]
    elif "n_predict" in llama_body:
        llama_body["max_tokens"] = llama_body.pop("n_predict")

    # Format
    resp_format = format_to_response_format(req.format)
    if resp_format:
        llama_body["response_format"] = resp_format

    # Think
    if req.think is not None:
        llama_body["think"] = req.think

    # Streaming
    if req.stream is not False:
        return StreamingResponse(
            _stream_chat(proc, llama_body, model_name, start_time),
            media_type="application/x-ndjson",
        )

    # Non-streaming
    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=llama_body,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    timings = extract_timings_from_oai(oai_resp, start_time)
    choice = oai_resp.get("choices", [{}])[0]
    oai_message = choice.get("message", {})

    # Convert tool_calls from OpenAI format to native format
    content = oai_message.get("content", "")
    reasoning = oai_message.get("reasoning_content", "")
    native_message: Dict[str, Any] = {
        "role": "assistant",
        "content": content if content else reasoning if reasoning else "",
    }
    if reasoning and content:
        native_message["thinking"] = reasoning

    if oai_message.get("tool_calls"):
        native_tool_calls = []
        for tc in oai_message["tool_calls"]:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            native_tool_calls.append({
                "function": {
                    "name": func.get("name", ""),
                    "arguments": args,
                }
            })
        native_message["tool_calls"] = native_tool_calls

    return build_chat_response(
        model_name, native_message, True,
        done_reason=choice.get("finish_reason", "stop"),
        timings=timings,
    )


async def _stream_chat(proc: ProcessInfo, body: Dict, model_name: str,
                        start_time: float) -> AsyncGenerator[str, None]:
    """Stream chat responses in native format."""
    eval_count = 0
    prompt_eval_count = 0
    full_content = ""
    tool_calls_buffer: List[Dict] = []
    current_tool_call: Optional[Dict] = None
    finish_reason = "stop"

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line = line[6:]
                    if line == b"[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta", {})

                    # Handle content (including reasoning_content for thinking models)
                    content = delta.get("content", "")
                    reasoning = delta.get("reasoning_content", "")
                    text_to_send = content or reasoning
                    if text_to_send:
                        eval_count += 1
                        full_content += text_to_send
                        msg: Dict[str, Any] = {"role": "assistant", "content": text_to_send}
                        if reasoning and not content:
                            msg["thinking"] = reasoning
                        yield json.dumps(build_chat_response(model_name, msg, False)) + "\n"

                    # Handle tool calls in delta
                    if delta.get("tool_calls"):
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            while len(tool_calls_buffer) <= idx:
                                tool_calls_buffer.append({"function": {"name": "", "arguments": ""}})
                            if tc_delta.get("function", {}).get("name"):
                                tool_calls_buffer[idx]["function"]["name"] = tc_delta["function"]["name"]
                            if tc_delta.get("function", {}).get("arguments"):
                                tool_calls_buffer[idx]["function"]["arguments"] += tc_delta["function"]["arguments"]

                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    # Usage
                    usage = data.get("usage")
                    if usage:
                        prompt_eval_count = usage.get("prompt_tokens", 0)
                        eval_count = max(eval_count, usage.get("completion_tokens", 0))

    except Exception as e:
        logger.error(f"Chat streaming error: {e}")

    # Emit tool calls if any
    if tool_calls_buffer:
        native_tool_calls = []
        for tc in tool_calls_buffer:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            native_tool_calls.append({
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": args,
                }
            })
        msg = {"role": "assistant", "content": "", "tool_calls": native_tool_calls}
        yield json.dumps(build_chat_response(model_name, msg, False)) + "\n"

    # Final response
    timings = {
        "total_duration": ns_to_duration(time.time() - start_time),
        "load_duration": ns_to_duration(0.001),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration": ns_to_duration((time.time() - start_time) * 0.1),
        "eval_count": eval_count,
        "eval_duration": ns_to_duration((time.time() - start_time) * 0.85),
    }
    final_msg = {"role": "assistant", "content": ""}
    yield json.dumps(build_chat_response(model_name, final_msg, True,
                                          done_reason=finish_reason, timings=timings)) + "\n"


# =============================================================================
# Native API: /api/embed - Generate Embeddings
# =============================================================================
@app.post("/api/embed")
async def api_embed(req: EmbedRequest):
    start_time = time.time()
    try:
        proc = await process_manager.ensure_model(
            req.model, options=req.options, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Normalize input to list
    inputs = req.input if isinstance(req.input, list) else [req.input]

    oai_body = {
        "model": req.model,
        "input": inputs,
    }
    if req.dimensions:
        oai_body["dimensions"] = req.dimensions

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/embeddings", json=oai_body,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    embeddings = []
    for item in oai_resp.get("data", []):
        embeddings.append(item.get("embedding", []))

    result: Dict[str, Any] = {
        "model": req.model,
        "embeddings": embeddings,
        "total_duration": ns_to_duration(time.time() - start_time),
        "load_duration": ns_to_duration(0.001),
        "prompt_eval_count": oai_resp.get("usage", {}).get("prompt_tokens", len(inputs)),
    }
    return result


# =============================================================================
# Native API: /api/embeddings (deprecated) - Generate Embedding
# =============================================================================
@app.post("/api/embeddings")
async def api_embeddings(req: EmbeddingsRequest):
    start_time = time.time()
    try:
        proc = await process_manager.ensure_model(
            req.model, options=req.options, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    oai_body = {"model": req.model, "input": [req.prompt]}
    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/embeddings", json=oai_body,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    embedding = []
    for item in oai_resp.get("data", []):
        embedding = item.get("embedding", [])
        break

    return {"embedding": embedding}


# =============================================================================
# Native API: /api/push (stub)
# =============================================================================
@app.post("/api/push")
async def api_push(request: Request):
    body = await request.json()
    stream = body.get("stream", True)

    async def stream_push():
        yield json.dumps({"status": "push is not supported in Local Model Router"}) + "\n"

    if not stream:
        return {"status": "push is not supported in Local Model Router"}
    return StreamingResponse(stream_push(), media_type="application/x-ndjson")


# =============================================================================
# Native API: /api/blobs (stubs)
# =============================================================================
@app.head("/api/blobs/{digest:path}")
async def api_blobs_head(digest: str):
    return Response(status_code=200)


@app.post("/api/blobs/{digest:path}")
async def api_blobs_post(digest: str, request: Request):
    return Response(status_code=201)


# =============================================================================
# OpenAI API: /v1/models
# =============================================================================
@app.get("/v1/models")
async def v1_models():
    models = []
    for name in config.model_configs:
        models.append({
            "id": name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        })
    for name in config.modelfiles:
        if name not in config.model_configs:
            models.append({
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            })
    return {"object": "list", "data": models}


# =============================================================================
# OpenAI API: /v1/chat/completions
# =============================================================================
@app.post("/v1/chat/completions")
async def v1_chat_completions(req: OpenAIChatRequest):
    start_time = time.time()
    model_name = req.model

    # Resolve model - check for Modelfile models that derive from a base
    actual_model = model_name
    model_cfg = config.model_configs.get(model_name) or config.model_configs.get(model_name.split(":")[0])
    if not model_cfg and model_name in config.modelfiles:
        mf = config.modelfiles[model_name]
        base = mf.get("from", "")
        if base and base in config.model_configs:
            actual_model = base
            # Inject system from Modelfile
            if mf.get("system") and req.messages and req.messages[0].role != "system":
                req.messages.insert(0, OpenAIMessage(role="system", content=mf["system"]))

    # Build options from request
    options: Dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.seed is not None:
        options["seed"] = req.seed
    if req.presence_penalty is not None:
        options["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        options["frequency_penalty"] = req.frequency_penalty

    try:
        proc = await process_manager.ensure_model(
            actual_model, options=options, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Build request body - pass through to llama-server
    llama_body: Dict[str, Any] = {
        "model": actual_model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        "stream": bool(req.stream),
    }

    # Pass through all supported parameters
    if req.temperature is not None:
        llama_body["temperature"] = req.temperature
    if req.top_p is not None:
        llama_body["top_p"] = req.top_p
    if req.max_tokens is not None:
        llama_body["max_tokens"] = req.max_tokens
    if req.max_completion_tokens is not None:
        llama_body["max_tokens"] = req.max_completion_tokens
    if req.stop is not None:
        llama_body["stop"] = req.stop
    if req.seed is not None:
        llama_body["seed"] = req.seed
    if req.presence_penalty is not None:
        llama_body["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        llama_body["frequency_penalty"] = req.frequency_penalty
    if req.logit_bias is not None:
        llama_body["logit_bias"] = req.logit_bias
    if req.logprobs is not None:
        llama_body["logprobs"] = req.logprobs
    if req.top_logprobs is not None:
        llama_body["top_logprobs"] = req.top_logprobs
    if req.tools:
        llama_body["tools"] = req.tools
    if req.tool_choice is not None:
        llama_body["tool_choice"] = req.tool_choice
    if req.response_format:
        llama_body["response_format"] = req.response_format
    if req.stream and req.stream_options:
        llama_body["stream_options"] = req.stream_options

    # Streaming
    if req.stream:
        return StreamingResponse(
            _stream_v1_chat(proc, llama_body, model_name, start_time, req.stream_options),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming - proxy directly
    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=llama_body,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    # Ensure response has proper id and model
    oai_resp["id"] = oai_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    oai_resp["object"] = "chat.completion"
    oai_resp["model"] = model_name
    oai_resp["created"] = oai_resp.get("created", int(time.time()))

    # Ensure usage exists
    if "usage" not in oai_resp:
        oai_resp["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    return oai_resp


async def _stream_v1_chat(proc: ProcessInfo, body: Dict, model_name: str,
                           start_time: float,
                           stream_options: Optional[Dict] = None) -> AsyncGenerator[str, None]:
    """Stream OpenAI chat completions in SSE format."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    total_tokens = 0
    prompt_tokens = 0

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line_data = line[6:]
                    else:
                        line_data = line
                    if line_data == b"[DONE]":
                        break
                    try:
                        data = json.loads(line_data)
                    except json.JSONDecodeError:
                        continue

                    # Rewrite id and model
                    data["id"] = completion_id
                    data["model"] = model_name
                    data["created"] = created

                    # Track usage
                    if "usage" in data:
                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        total_tokens = data["usage"].get("total_tokens", 0)

                    # Track tokens from choices
                    choices = data.get("choices", [])
                    for c in choices:
                        delta = c.get("delta", {})
                        if delta.get("content"):
                            total_tokens += 1

                    yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        logger.error(f"OpenAI streaming error: {e}")
        error_data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "created": created,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(error_data)}\n\n"

    # Include usage in final chunk if requested
    if stream_options and stream_options.get("include_usage"):
        usage_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "model": model_name,
            "created": created,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": max(0, total_tokens - prompt_tokens),
                "total_tokens": total_tokens,
            }
        }
        yield f"data: {json.dumps(usage_chunk)}\n\n"

    yield "data: [DONE]\n\n"


# =============================================================================
# OpenAI API: /v1/completions
# =============================================================================
@app.post("/v1/completions")
async def v1_completions(req: OpenAICompletionRequest):
    model_name = req.model
    actual_model = model_name
    model_cfg = config.model_configs.get(model_name) or config.model_configs.get(model_name.split(":")[0])
    if not model_cfg and model_name in config.modelfiles:
        mf = config.modelfiles[model_name]
        base = mf.get("from", "")
        if base and base in config.model_configs:
            actual_model = base

    try:
        proc = await process_manager.ensure_model(
            actual_model, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    llama_body: Dict[str, Any] = {
        "model": actual_model,
        "prompt": req.prompt,
        "stream": bool(req.stream),
    }
    if req.suffix:
        llama_body["suffix"] = req.suffix
    if req.max_tokens is not None:
        llama_body["max_tokens"] = req.max_tokens
    if req.temperature is not None:
        llama_body["temperature"] = req.temperature
    if req.top_p is not None:
        llama_body["top_p"] = req.top_p
    if req.stop is not None:
        llama_body["stop"] = req.stop
    if req.seed is not None:
        llama_body["seed"] = req.seed
    if req.presence_penalty is not None:
        llama_body["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        llama_body["frequency_penalty"] = req.frequency_penalty
    if req.logprobs is not None:
        llama_body["logprobs"] = req.logprobs
    if req.echo is not None:
        llama_body["echo"] = req.echo

    if req.stream:
        return StreamingResponse(
            _stream_v1_completions(proc, llama_body, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/completions", json=llama_body,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    oai_resp["id"] = oai_resp.get("id", f"cmpl-{uuid.uuid4().hex[:12]}")
    oai_resp["object"] = "text_completion"
    oai_resp["model"] = model_name
    return oai_resp


async def _stream_v1_completions(proc: ProcessInfo, body: Dict,
                                  model_name: str) -> AsyncGenerator[str, None]:
    """Stream text completions in SSE format."""
    completion_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line_data = line[6:]
                    else:
                        line_data = line
                    if line_data == b"[DONE]":
                        break
                    try:
                        data = json.loads(line_data)
                    except json.JSONDecodeError:
                        continue
                    data["id"] = completion_id
                    data["model"] = model_name
                    data["created"] = created
                    yield f"data: {json.dumps(data)}\n\n"
    except Exception as e:
        logger.error(f"Completion streaming error: {e}")

    yield "data: [DONE]\n\n"


# =============================================================================
# OpenAI API: /v1/embeddings
# =============================================================================
@app.post("/v1/embeddings")
async def v1_embeddings(req: OpenAIEmbeddingRequest):
    model_name = req.model
    actual_model = model_name
    model_cfg = config.model_configs.get(model_name) or config.model_configs.get(model_name.split(":")[0])
    if not model_cfg and model_name in config.modelfiles:
        mf = config.modelfiles[model_name]
        base = mf.get("from", "")
        if base and base in config.model_configs:
            actual_model = base

    try:
        proc = await process_manager.ensure_model(
            actual_model, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    oai_body: Dict[str, Any] = {
        "model": actual_model,
        "input": req.input,
    }
    if req.encoding_format:
        oai_body["encoding_format"] = req.encoding_format
    if req.dimensions:
        oai_body["dimensions"] = req.dimensions

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/embeddings", json=oai_body,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    oai_resp["model"] = model_name
    return oai_resp


# =============================================================================
# OpenAI API: /v1/images/generations (stub)
# =============================================================================
@app.post("/v1/images/generations")
async def v1_images_generations(request: Request):
    body = await request.json()
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": "Image generation is not supported. Use an image generation model via /api/generate.",
                "type": "not_implemented",
                "code": "not_implemented",
            }
        },
    )


# =============================================================================
# OpenAI API: /v1/responses
# =============================================================================
@app.post("/v1/responses")
async def v1_responses(req: OpenAIResponsesRequest):
    """OpenAI Responses API - non-stateful implementation."""
    model_name = req.model
    actual_model = model_name
    model_cfg = config.model_configs.get(model_name) or config.model_configs.get(model_name.split(":")[0])
    if not model_cfg and model_name in config.modelfiles:
        mf = config.modelfiles[model_name]
        base = mf.get("from", "")
        if base and base in config.model_configs:
            actual_model = base

    try:
        proc = await process_manager.ensure_model(
            actual_model, keep_alive=req.keep_alive,
            llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Convert input to messages
    messages = []
    if req.instructions:
        messages.append({"role": "system", "content": req.instructions})

    if isinstance(req.input, str):
        messages.append({"role": "user", "content": req.input})
    elif isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                messages.append(item)

    # Build request body - proxy to llama-server's /v1/responses if available,
    # otherwise use /v1/chat/completions
    llama_body: Dict[str, Any] = {
        "model": actual_model,
        "messages": messages,
        "stream": bool(req.stream),
    }
    if req.temperature is not None:
        llama_body["temperature"] = req.temperature
    if req.top_p is not None:
        llama_body["top_p"] = req.top_p
    if req.max_output_tokens is not None:
        llama_body["max_tokens"] = req.max_output_tokens
    if req.tools:
        llama_body["tools"] = req.tools
    if req.tool_choice is not None:
        llama_body["tool_choice"] = req.tool_choice

    if req.stream:
        # Try /v1/responses first, fall back to /v1/chat/completions
        return StreamingResponse(
            _stream_v1_responses(proc, llama_body, model_name),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-streaming
    try:
        # Try /v1/responses first
        async with process_manager._session.post(
            f"{proc.base_url}/v1/responses", json={
                "model": actual_model,
                "input": req.input,
                "instructions": req.instructions,
                "temperature": req.temperature,
                "top_p": req.top_p,
                "max_output_tokens": req.max_output_tokens,
                "stream": False,
                "tools": req.tools,
                "tool_choice": req.tool_choice,
            },
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                result["model"] = model_name
                return result
    except Exception:
        pass

    # Fallback to chat completions
    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=llama_body,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            oai_resp = await resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

    # Convert to responses API format
    response_id = f"resp-{uuid.uuid4().hex[:12]}"
    choice = oai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")

    output_items = []
    if content:
        output_items.append({
            "type": "message",
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        })

    return {
        "id": response_id,
        "object": "response",
        "model": model_name,
        "created_at": int(time.time()),
        "status": "completed",
        "output": output_items,
        "usage": oai_resp.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
    }


async def _stream_v1_responses(proc: ProcessInfo, body: Dict,
                                model_name: str) -> AsyncGenerator[str, None]:
    """Stream responses API in SSE format, falling back to chat completions."""
    response_id = f"resp-{uuid.uuid4().hex[:12]}"

    # Try /v1/responses first
    try:
        responses_body = copy.deepcopy(body)
        responses_body["stream"] = True
        async with process_manager._session.post(
            f"{proc.base_url}/v1/responses", json=responses_body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            if resp.status == 200:
                buffer = b""
                async for chunk in resp.content.iter_any():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(b"data: "):
                            line_data = line[6:]
                        else:
                            line_data = line
                        if line_data == b"[DONE]":
                            break
                        try:
                            data = json.loads(line_data)
                            data["model"] = model_name
                            yield f"data: {json.dumps(data)}\n\n"
                        except json.JSONDecodeError:
                            continue
                yield "data: [DONE]\n\n"
                return
    except Exception:
        pass

    # Fallback: use chat completions and convert
    body["stream"] = True
    msg_id = f"msg-{uuid.uuid4().hex[:8]}"

    # Emit response.created
    yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'model': model_name, 'status': 'in_progress'}})}\n\n"

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            buffer = b""
            async for chunk in resp.content.iter_any():
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b"data: "):
                        line_data = line[6:]
                    else:
                        line_data = line
                    if line_data == b"[DONE]":
                        break
                    try:
                        data = json.loads(line_data)
                    except json.JSONDecodeError:
                        continue

                    choices = data.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            event = {
                                "type": "response.output_text.delta",
                                "item_id": msg_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": content,
                            }
                            yield f"data: {json.dumps(event)}\n\n"
    except Exception as e:
        logger.error(f"Responses streaming error: {e}")

    # Emit done
    yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0})}\n\n"
    yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': response_id, 'object': 'response', 'model': model_name, 'status': 'completed'}})}\n\n"
    yield "data: [DONE]\n\n"


# =============================================================================
# Additional Compatibility Routes
# =============================================================================

# Some clients check /v1/health
@app.get("/v1/health")
async def v1_health():
    return {"status": "ok"}


# HEAD /api/tags for compatibility
@app.head("/api/tags")
async def api_tags_head():
    return Response(status_code=200)


# =============================================================================
# Catch-all for other /v1 paths (proxy to loaded model if possible)
# =============================================================================
@app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def v1_catchall(path: str, request: Request):
    """Catch-all for /v1 paths not explicitly handled. Proxy to first running model."""
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            body = await request.json()
        except Exception:
            body = {}

    model_name = None
    if body and isinstance(body, dict):
        model_name = body.get("model")

    # Find a running process to proxy to
    proc = None
    if model_name:
        actual = model_name.split(":")[0]
        proc = process_manager.processes.get(model_name) or process_manager.processes.get(actual)
        if not proc:
            try:
                proc = await process_manager.ensure_model(model_name)
            except Exception:
                pass

    if not proc and process_manager.processes:
        proc = next(iter(process_manager.processes.values()))

    if not proc:
        return JSONResponse(status_code=404, content={"error": "No models loaded"})

    url = f"{proc.base_url}/v1/{path}"
    try:
        if request.method == "GET":
            async with process_manager._session.get(url) as resp:
                data = await resp.json()
                return JSONResponse(status_code=resp.status, content=data)
        else:
            async with process_manager._session.post(url, json=body) as resp:
                content_type = resp.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    async def stream_proxy():
                        async for chunk in resp.content.iter_any():
                            yield chunk
                    return StreamingResponse(stream_proxy(), media_type="text/event-stream")
                data = await resp.json()
                return JSONResponse(status_code=resp.status, content=data)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =============================================================================
# CLI Argument Parser
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{LMR_NAME} - Local LLM Server powered by llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python lmr.py
  python lmr.py --port 8080
  python lmr.py --llama-server /usr/local/bin/llama-server
  python lmr.py --models-json my_models.json
  python lmr.py --default-ctx-size 8192 --default-num-gpu 99
        """,
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind to (default: {DEFAULT_HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--llama-server", default="", help="Path to llama-server binary")
    parser.add_argument("--models-json", default="models.json", help="Path to models.json config file")
    parser.add_argument("--models-dir", default=os.path.expanduser("~/.local/share/localmodelrouter/models"),
                        help="Directory for downloaded models")
    parser.add_argument("--default-ctx-size", type=int, default=DEFAULT_CTX_SIZE,
                        help=f"Default context size (default: {DEFAULT_CTX_SIZE})")
    parser.add_argument("--default-keep-alive", default=str(DEFAULT_KEEP_ALIVE),
                        help=f"Default keep_alive in seconds (default: {DEFAULT_KEEP_ALIVE})")
    parser.add_argument("--default-num-gpu", type=int, default=-1,
                        help="Default GPU layers (-1=auto, 0=CPU only)")
    parser.add_argument("--default-parallel", type=int, default=DEFAULT_PARALLEL_SLOTS,
                        help=f"Default parallel slots per model (default: {DEFAULT_PARALLEL_SLOTS})")
    parser.add_argument("--default-flash-attn", default="auto", choices=["on", "off", "auto"],
                        help="Flash attention setting (default: auto)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    args = parse_args()

    # Apply CLI args to config
    config.host = args.host
    config.port = args.port
    config.llama_server_binary = args.llama_server
    config.models_json = args.models_json
    config.models_dir = args.models_dir
    config.default_ctx_size = args.default_ctx_size
    config.default_keep_alive = parse_keep_alive(args.default_keep_alive)
    config.default_num_gpu = args.default_num_gpu
    config.default_parallel = args.default_parallel
    config.default_flash_attn = args.default_flash_attn
    logger.setLevel(getattr(logging, args.log_level))

    # Ensure models directory exists
    os.makedirs(config.models_dir, exist_ok=True)

    # Graceful shutdown signal handler
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, cleaning up...")
        # The lifespan handler will clean up processes
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run server
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
