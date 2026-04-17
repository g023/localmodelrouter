"""Configuration constants, ModelConfig, and ServerConfig for LMR."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re
import shutil
from typing import Any, Dict, Optional

# =============================================================================
# Constants & Version
# =============================================================================
LMR_VERSION = "0.6.0"
LMR_NAME = "Local Model Router"
LMR_PREFIX = "[localmodelrouter]"
DEFAULT_PORT = 11434
DEFAULT_HOST = "0.0.0.0"
DEFAULT_KEEP_ALIVE = 300  # 5 minutes in seconds
DEFAULT_CTX_SIZE = 16384
DEFAULT_NUM_PREDICT = -1
LLAMA_SERVER_PORT_START = 39000
LLAMA_SERVER_PORT_END = 39999
HEALTH_CHECK_INTERVAL = 0.25
HEALTH_CHECK_TIMEOUT = 120
DEFAULT_PARALLEL_SLOTS = 4
MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100MB max request payload
MAX_MODEL_NAME_LENGTH = 256

# =============================================================================
# Logging
# =============================================================================
logger = logging.getLogger("localmodelrouter")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(f"%(asctime)s {LMR_PREFIX} %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# =============================================================================
# Model Configuration
# =============================================================================
class ModelConfig:
    """Configuration for a single model."""

    def __init__(self, name: str, path: str, **kwargs: Any):
        self.name = name
        self.path = path
        self.num_gpu: int = kwargs.get("num_gpu", -1)
        self.ctx_size: int = kwargs.get("ctx_size", 0)
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
        self.modelfiles: Dict[str, Dict[str, Any]] = {}
        self._ctx_size_cache: Dict[str, int] = {}  # Cache for context size results

    def find_llama_server(self) -> str:
        """Find the llama-server binary."""
        if self.llama_server_binary and os.path.isfile(self.llama_server_binary):
            return self.llama_server_binary
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
            if not isinstance(data, dict):
                logger.error(f"Invalid models config: expected dict, got {type(data).__name__}")
                return
            for name, value in data.items():
                if len(name) > MAX_MODEL_NAME_LENGTH:
                    logger.warning(f"Skipping model with name too long: {name[:50]}...")
                    continue
                if isinstance(value, str):
                    self.model_configs[name] = ModelConfig(name=name, path=value)
                elif isinstance(value, dict):
                    model_path = value.pop("path", value.pop("model", ""))
                    if not model_path:
                        logger.warning(f"Model '{name}' has no path, skipping")
                        continue
                    self.model_configs[name] = ModelConfig(name=name, path=model_path, **value)
                else:
                    logger.warning(f"Invalid config for model '{name}': expected str or dict")
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

    def validate(self) -> list[str]:
        """Validate configuration and return list of warnings."""
        warnings = []
        for name, cfg in self.model_configs.items():
            if not cfg.path:
                warnings.append(f"Model '{name}' has no path configured")
            elif not cfg.path.startswith("hf:") and not os.path.isfile(cfg.path):
                warnings.append(f"Model '{name}' path not found: {cfg.path}")
            if cfg.ctx_size < 0:
                warnings.append(f"Model '{name}' has invalid ctx_size: {cfg.ctx_size}")
            if cfg.num_parallel < 1:
                warnings.append(f"Model '{name}' has invalid num_parallel: {cfg.num_parallel}")
        return warnings
