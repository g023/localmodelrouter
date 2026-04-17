"""OpenAI-compatible API routes (/v1/*) for LMR."""

from __future__ import annotations

import copy
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from lmr.config import ServerConfig, logger
from lmr.models import (
    OpenAIChatRequest,
    OpenAICompletionRequest,
    OpenAIEmbeddingRequest,
    OpenAIMessage,
    OpenAIResponsesRequest,
)
from lmr.process import ProcessInfo, ProcessManager


def register_openai_routes(app: FastAPI, config: ServerConfig, process_manager: ProcessManager) -> None:
    """Register all OpenAI /v1/* routes."""

    # =========================================================================
    # /v1/models
    # =========================================================================
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

    # =========================================================================
    # /v1/chat/completions
    # =========================================================================
    @app.post("/v1/chat/completions")
    async def v1_chat_completions(req: OpenAIChatRequest):
        start_time = time.time()
        model_name = req.model

        actual_model = model_name
        model_cfg = config.model_configs.get(model_name) or config.model_configs.get(model_name.split(":")[0])
        if not model_cfg and model_name in config.modelfiles:
            mf = config.modelfiles[model_name]
            base = mf.get("from", "")
            if base and base in config.model_configs:
                actual_model = base
                if mf.get("system") and req.messages and req.messages[0].role != "system":
                    req.messages.insert(0, OpenAIMessage(role="system", content=mf["system"]))

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

        llama_body: Dict[str, Any] = {
            "model": actual_model,
            "messages": [m.model_dump(exclude_none=True) for m in req.messages],
            "stream": bool(req.stream),
        }

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

        if req.stream:
            return StreamingResponse(
                _stream_v1_chat(process_manager, proc, llama_body, model_name, start_time, req.stream_options),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                actual_model, "/v1/chat/completions", llama_body,
                options=options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        oai_resp["id"] = oai_resp.get("id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
        oai_resp["object"] = "chat.completion"
        oai_resp["model"] = model_name
        oai_resp["created"] = oai_resp.get("created", int(time.time()))
        if "usage" not in oai_resp:
            oai_resp["usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return oai_resp

    # =========================================================================
    # /v1/completions
    # =========================================================================
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
                _stream_v1_completions(process_manager, proc, llama_body, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                actual_model, "/v1/completions", llama_body,
                keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        oai_resp["id"] = oai_resp.get("id", f"cmpl-{uuid.uuid4().hex[:12]}")
        oai_resp["object"] = "text_completion"
        oai_resp["model"] = model_name
        return oai_resp

    # =========================================================================
    # /v1/embeddings
    # =========================================================================
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

        oai_body: Dict[str, Any] = {"model": actual_model, "input": req.input}
        if req.encoding_format:
            oai_body["encoding_format"] = req.encoding_format
        if req.dimensions:
            oai_body["dimensions"] = req.dimensions

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                actual_model, "/v1/embeddings", oai_body,
                keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                timeout=120
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        oai_resp["model"] = model_name
        return oai_resp

    # =========================================================================
    # /v1/images/generations (stub)
    # =========================================================================
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

    # =========================================================================
    # /v1/responses
    # =========================================================================
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
            return StreamingResponse(
                _stream_v1_responses(process_manager, proc, llama_body, model_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        # Non-streaming: try /v1/responses first, fall back to chat completions
        try:
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

        try:
            async with process_manager._session.post(
                f"{proc.base_url}/v1/chat/completions", json=llama_body,
                timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                oai_resp = await resp.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

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

    # =========================================================================
    # /v1/health
    # =========================================================================
    @app.get("/v1/health")
    async def v1_health():
        return {"status": "ok"}

    # =========================================================================
    # Catch-all for other /v1 paths
    # =========================================================================
    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def v1_catchall(path: str, request: Request):
        """Catch-all for /v1 paths not explicitly handled."""
        body = None
        if request.method in ("POST", "PUT", "PATCH"):
            try:
                body = await request.json()
            except Exception:
                body = {}

        model_name = None
        if body and isinstance(body, dict):
            model_name = body.get("model")

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
# Streaming helpers
# =============================================================================

async def _stream_v1_chat(process_manager: ProcessManager, proc: ProcessInfo,
                           body: Dict, model_name: str, start_time: float,
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
                    data["id"] = completion_id
                    data["model"] = model_name
                    data["created"] = created
                    if "usage" in data:
                        prompt_tokens = data["usage"].get("prompt_tokens", 0)
                        total_tokens = data["usage"].get("total_tokens", 0)
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


async def _stream_v1_completions(process_manager: ProcessManager, proc: ProcessInfo,
                                  body: Dict, model_name: str) -> AsyncGenerator[str, None]:
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


async def _stream_v1_responses(process_manager: ProcessManager, proc: ProcessInfo,
                                body: Dict, model_name: str) -> AsyncGenerator[str, None]:
    """Stream responses API in SSE format, falling back to chat completions."""
    response_id = f"resp-{uuid.uuid4().hex[:12]}"

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

    body["stream"] = True
    msg_id = f"msg-{uuid.uuid4().hex[:8]}"
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

    yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0})}\n\n"
    yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': response_id, 'object': 'response', 'model': model_name, 'status': 'completed'}})}\n\n"
    yield "data: [DONE]\n\n"
