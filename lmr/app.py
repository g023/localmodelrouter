"""FastAPI application factory for LMR."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from lmr.config import LMR_NAME, LMR_VERSION, ServerConfig, logger
from lmr.process import ProcessManager
from lmr.routes import register_all_routes
from lmr.utils import detect_gpu_memory


def create_app(config: ServerConfig) -> tuple[FastAPI, ProcessManager]:
    """Create and configure the FastAPI application."""
    process_manager = ProcessManager(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
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

        # Validate config and warn
        warnings = config.validate()
        for w in warnings:
            logger.warning(f"Config: {w}")

        await process_manager.start()
        yield
        logger.info("Shutting down, stopping all model processes...")
        await process_manager.stop()
        logger.info("Shutdown complete")

    app = FastAPI(title=LMR_NAME, version=LMR_VERSION, lifespan=lifespan)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
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

    # Health & Version routes
    @app.get("/")
    async def root():
        return Response(content=f"{LMR_NAME} is running", media_type="text/plain")

    @app.head("/")
    async def root_head():
        return Response(content="", media_type="text/plain")

    @app.get("/api/version")
    async def api_version():
        return {"version": LMR_VERSION}

    # Register all API routes
    register_all_routes(app, config, process_manager)

    return app, process_manager
