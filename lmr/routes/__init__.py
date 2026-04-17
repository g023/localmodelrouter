"""Route registration for LMR."""

from fastapi import FastAPI

from lmr.routes.native import register_native_routes
from lmr.routes.openai_api import register_openai_routes


def register_all_routes(app: FastAPI, config, process_manager) -> None:
    """Register all API routes on the FastAPI app."""
    register_native_routes(app, config, process_manager)
    register_openai_routes(app, config, process_manager)
