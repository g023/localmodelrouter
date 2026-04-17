"""CLI argument parsing and main entry point for LMR."""

from __future__ import annotations

import argparse
import logging
import os

import uvicorn

from lmr.config import (
    DEFAULT_CTX_SIZE,
    DEFAULT_HOST,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_PORT,
    LMR_NAME,
    ServerConfig,
    logger,
)
from lmr.utils import parse_keep_alive


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
    parser.add_argument("--default-keep-alive", default="300",
                        help="Default keep_alive in seconds (default: 300)")
    parser.add_argument("--default-num-gpu", type=int, default=-1,
                        help="Default GPU layers (-1=auto, 0=CPU only)")
    parser.add_argument("--default-parallel", type=int, default=DEFAULT_PARALLEL_SLOTS,
                        help=f"Default parallel slots per model (default: {DEFAULT_PARALLEL_SLOTS})")
    parser.add_argument("--default-flash-attn", default="auto", choices=["on", "off", "auto"],
                        help="Flash attention setting (default: auto)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ServerConfig()
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

    os.makedirs(config.models_dir, exist_ok=True)

    from lmr.app import create_app
    app, process_manager = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=args.log_level.lower(),
        access_log=True,
    )
