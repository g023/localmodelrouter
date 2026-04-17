"""Process management for llama-server instances."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import HTTPException

from lmr.config import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_PARALLEL_SLOTS,
    HEALTH_CHECK_INTERVAL,
    HEALTH_CHECK_TIMEOUT,
    ModelConfig,
    ServerConfig,
    logger,
)
from lmr.utils import (
    compute_digest,
    estimate_context_size,
    find_free_port,
    get_file_size,
    parse_keep_alive,
)


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


class ProcessManager:
    """Manages llama-server process lifecycle."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.processes: Dict[str, ProcessInfo] = {}
        self._lock = asyncio.Lock()
        self._unload_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        """Start the process manager background tasks."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300, connect=5)
        )
        self._unload_task = asyncio.create_task(self._unload_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        """Stop all processes and cleanup."""
        if self._unload_task:
            self._unload_task.cancel()
            try:
                await self._unload_task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        async with self._lock:
            for name in list(self.processes.keys()):
                proc = self.processes.pop(name)
                proc.kill()
        if self._session:
            await self._session.close()

    async def proxy_request_with_retry(self, model_name: str, endpoint: str, data: Dict[str, Any],
                                      options: Optional[Dict] = None, keep_alive: Optional[Any] = None,
                                      llama_cpp_binary: Optional[str] = None,
                                      llama_cpp_flags: Optional[Dict] = None,
                                      timeout: int = 300) -> Dict[str, Any]:
        """Proxy a request to a model with automatic retry on process failure."""
        max_retries = 2
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                proc = await self.ensure_model(
                    model_name, options=options, keep_alive=keep_alive,
                    llama_cpp_binary=llama_cpp_binary, llama_cpp_flags=llama_cpp_flags
                )

                async with self._session.post(
                    f"{proc.base_url}{endpoint}", json=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as resp:
                    if resp.status >= 400:
                        error_body = await resp.text()
                        if attempt < max_retries and resp.status in (502, 503, 422):
                            logger.warning(f"Request failed with status {resp.status}, retrying (attempt {attempt + 1})...")
                            async with self._lock:
                                if model_name in self.processes:
                                    self.processes[model_name].kill()
                                    del self.processes[model_name]
                            continue
                        else:
                            try:
                                return await resp.json()
                            except Exception:
                                raise HTTPException(status_code=resp.status, detail=error_body)
                    return await resp.json()

            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Request failed: {e}, retrying (attempt {attempt + 1})...")
                    async with self._lock:
                        if model_name in self.processes:
                            self.processes[model_name].kill()
                            del self.processes[model_name]
                    continue
                break
            except HTTPException:
                raise

        raise HTTPException(status_code=500, detail=f"Backend request failed after {max_retries + 1} attempts: {last_error}")

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

    async def _cleanup_loop(self) -> None:
        """Background loop to clean up dead processes."""
        while True:
            try:
                await asyncio.sleep(30)
                async with self._lock:
                    dead = [
                        name for name, proc in self.processes.items()
                        if not proc.is_alive
                    ]
                for name in dead:
                    logger.info(f"Cleaning up dead process for model '{name}'")
                    async with self._lock:
                        proc = self.processes.pop(name, None)
                    if proc:
                        proc.kill()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _resolve_model_path(self, model_name: str, request_body: Optional[Dict] = None) -> str:
        """Resolve model name to a GGUF file path."""
        if model_name in self.config.model_configs:
            path = self.config.model_configs[model_name].path
            if path.startswith("hf:"):
                return self._download_hf_model(path)
            if os.path.isfile(path):
                return path

        base_name = model_name.split(":")[0] if ":" in model_name else model_name
        if base_name in self.config.model_configs:
            path = self.config.model_configs[base_name].path
            if path.startswith("hf:"):
                return self._download_hf_model(path)
            if os.path.isfile(path):
                return path

        if os.path.isfile(model_name):
            return model_name

        models_dir = self.config.models_dir
        for ext in ["", ".gguf"]:
            candidate = os.path.join(models_dir, f"{model_name}{ext}")
            if os.path.isfile(candidate):
                return candidate
            candidate = os.path.join(models_dir, f"{base_name}{ext}")
            if os.path.isfile(candidate):
                return candidate

        hf_ref = self._parse_hf_reference(model_name)
        if hf_ref:
            return self._download_hf_model(hf_ref)

        raise FileNotFoundError(
            f"Model '{model_name}' not found. Add it to models.json or provide the full path."
        )

    def _parse_hf_reference(self, model_name: str) -> Optional[str]:
        """Parse various Hugging Face reference formats into hf:repo:filename format."""
        patterns = [
            (r'^https?://huggingface\.co/([^/]+)/([^/]+):(.+)$', lambda m: f"hf:{m.group(1)}/{m.group(2)}:{m.group(3)}"),
            (r'^https?://hf\.co/([^/]+)/([^/]+):(.+)$', lambda m: f"hf:{m.group(1)}/{m.group(2)}:{m.group(3)}"),
            (r'^huggingface\.co/([^/]+)/([^/]+):(.+)$', lambda m: f"hf:{m.group(1)}/{m.group(2)}:{m.group(3)}"),
            (r'^hf\.co/([^/]+)/([^/]+):(.+)$', lambda m: f"hf:{m.group(1)}/{m.group(2)}:{m.group(3)}"),
        ]
        for pattern, builder in patterns:
            match = re.match(pattern, model_name)
            if match:
                return builder(match)

        # owner/repo:filename (assume HF if contains / and :)
        match = re.match(r'^([^/]+)/([^/]+):(.+)$', model_name)
        if match and not os.path.exists(model_name):
            return f"hf:{match.group(1)}/{match.group(2)}:{match.group(3)}"

        if model_name.startswith("hf:"):
            return model_name

        return None

    def _download_hf_model(self, hf_ref: str) -> str:
        """Download a model from HuggingFace. Format: hf:user/repo:filename"""
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("huggingface_hub is required for HF downloads. pip install huggingface_hub")

        ref = hf_ref.replace("hf:", "", 1)
        parts = ref.split(":")
        repo_id = parts[0]
        filename = parts[1] if len(parts) > 1 else None

        if not filename:
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith(".gguf")]
            if not gguf_files:
                raise FileNotFoundError(f"No GGUF files found in {repo_id}")
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
            # Check cache first
            cached_ctx = self.config._ctx_size_cache.get(model_path)
            if cached_ctx:
                ctx = cached_ctx
            else:
                ctx = self.config.default_ctx_size
                if self.config.gpu_memory_mb > 0:
                    file_size = get_file_size(model_path)
                    if file_size > 0:
                        auto_ctx = estimate_context_size(file_size, self.config.gpu_memory_mb)
                        if auto_ctx > ctx:
                            ctx = auto_ctx
                # Cache the result
                self.config._ctx_size_cache[model_path] = ctx
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

        # Slots monitoring & metrics
        cmd.append("--slots")
        cmd.append("--metrics")

        # Embedding mode
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
                if flag in ("--mlock",):
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
            if model_name in self.processes:
                proc = self.processes[model_name]
                if proc.is_alive and proc.is_ready:
                    proc.touch()
                    if keep_alive is not None:
                        proc.keep_alive = parse_keep_alive(keep_alive)
                    return proc
                elif not proc.is_alive:
                    proc.kill()
                    del self.processes[model_name]

            model_path = self._resolve_model_path(model_name)

            used_ports = {p.port for p in self.processes.values()}
            port = find_free_port()
            while port in used_ports:
                port = find_free_port(port + 1)

            cmd = self._build_launch_command(
                model_path, port, model_name,
                options=options,
                extra_binary=llama_cpp_binary,
                extra_flags=llama_cpp_flags,
            )

            logger.info(f"Launching llama-server for '{model_name}' on port {port}")
            logger.debug(f"Command: {' '.join(cmd)}")

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
                        num_gpu = 99
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

        ready = await self._wait_for_health(proc_info)
        if not ready:
            async with self._lock:
                if model_name in self.processes:
                    self.processes[model_name].kill()
                    del self.processes[model_name]
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
        from datetime import datetime, timezone
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
