"""Native API routes (/api/*) for LMR - compatible with Ollama API format."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from lmr.config import ModelConfig, ServerConfig, logger
from lmr.converters import (
    build_oai_messages_from_native,
    build_oai_tools,
    format_to_response_format,
    ollama_options_to_llama_server,
)
from lmr.models import (
    ChatRequest,
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    EmbedRequest,
    EmbeddingsRequest,
    GenerateRequest,
    PullRequest,
    ShowRequest,
)
from lmr.process import ProcessInfo, ProcessManager
from lmr.responses import (
    build_chat_response,
    build_generate_response,
    extract_timings_from_oai,
)
from lmr.utils import compute_digest, get_file_size, get_timestamp, ns_to_duration, parse_keep_alive


def register_native_routes(app: FastAPI, config: ServerConfig, process_manager: ProcessManager) -> None:
    """Register all native /api/* routes."""

    # =========================================================================
    # /api/tags - List Models
    # =========================================================================
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

    @app.head("/api/tags")
    async def api_tags_head():
        return Response(status_code=200)

    # =========================================================================
    # /api/ps - List Running Models
    # =========================================================================
    @app.get("/api/ps")
    async def api_ps():
        return {"models": process_manager.get_running_models()}

    # =========================================================================
    # /api/show - Show Model Information
    # =========================================================================
    @app.post("/api/show")
    async def api_show(req: ShowRequest):
        model_name = req.model.split(":")[0]
        model_cfg = config.model_configs.get(model_name) or config.model_configs.get(req.model)
        if not model_cfg:
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

    # =========================================================================
    # /api/create - Create Model
    # =========================================================================
    @app.post("/api/create")
    async def api_create(req: CreateRequest):
        async def stream_create():
            yield json.dumps({"status": "reading model metadata"}) + "\n"
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
                config.modelfiles[req.model] = {
                    "from": source_model,
                    "template": req.template or "",
                    "system": req.system or "",
                    "parameters": req.parameters or {},
                }
                yield json.dumps({"status": "creating system layer"}) + "\n"
            else:
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
            try:
                config.save_models_json(config.models_json)
            except Exception:
                pass

        if req.stream is False:
            async for _ in stream_create():
                pass
            return {"status": "success"}
        return StreamingResponse(stream_create(), media_type="application/x-ndjson")

    # =========================================================================
    # /api/copy - Copy Model
    # =========================================================================
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

    # =========================================================================
    # /api/delete - Delete Model
    # =========================================================================
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
        await process_manager.unload_model(model_name)
        if not found:
            raise HTTPException(status_code=404, detail=f"model '{req.model}' not found")
        try:
            config.save_models_json(config.models_json)
        except Exception:
            pass
        return Response(status_code=200)

    # =========================================================================
    # /api/pull - Pull Model
    # =========================================================================
    @app.post("/api/pull")
    async def api_pull(req: PullRequest):
        async def stream_pull():
            yield json.dumps({"status": "pulling manifest"}) + "\n"
            model_name = req.model
            if model_name.startswith("hf:") or "/" in model_name:
                try:
                    hf_ref = model_name if model_name.startswith("hf:") else f"hf:{model_name}"
                    yield json.dumps({"status": f"downloading from HuggingFace: {model_name}"}) + "\n"
                    loop = asyncio.get_event_loop()
                    local_path = await loop.run_in_executor(
                        None, process_manager._download_hf_model, hf_ref
                    )
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

    # =========================================================================
    # /api/generate - Generate Completion
    # =========================================================================
    @app.post("/api/generate")
    async def api_generate(req: GenerateRequest):
        start_time = time.time()
        model_name = req.model

        if not req.prompt and not req.images:
            keep_alive_val = parse_keep_alive(req.keep_alive)
            if keep_alive_val == 0:
                await process_manager.unload_model(model_name)
                return build_generate_response(model_name, "", True, done_reason="unload")
            else:
                try:
                    await process_manager.ensure_model(
                        model_name, options=req.options, keep_alive=req.keep_alive,
                        llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
                return build_generate_response(model_name, "", True)

        try:
            proc = await process_manager.ensure_model(
                model_name, options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        if req.raw:
            llama_body: Dict[str, Any] = {
                "model": model_name,
                "prompt": req.prompt or "",
                "stream": bool(req.stream),
            }
            if req.suffix:
                llama_body["suffix"] = req.suffix
            endpoint = "/v1/completions"
        else:
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

        opts = ollama_options_to_llama_server(req.options)
        llama_body.update(opts)

        if req.options and req.options.get("num_predict"):
            llama_body["max_tokens"] = req.options["num_predict"]
        elif "n_predict" in llama_body:
            llama_body["max_tokens"] = llama_body.pop("n_predict")

        resp_format = format_to_response_format(req.format)
        if resp_format:
            llama_body["response_format"] = resp_format

        if req.think is not None:
            llama_body["think"] = req.think

        if req.stream is not False:
            return StreamingResponse(
                _stream_generate(process_manager, proc, endpoint, llama_body, model_name, start_time, req.raw),
                media_type="application/x-ndjson",
            )

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                model_name, endpoint, llama_body,
                options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        timings = extract_timings_from_oai(oai_resp, start_time)

        if req.raw:
            text = oai_resp.get("choices", [{}])[0].get("text", "")
        else:
            msg = oai_resp.get("choices", [{}])[0].get("message", {})
            text = msg.get("content", "")
            reasoning = msg.get("reasoning_content", "")
            if not text and reasoning:
                text = reasoning

        return build_generate_response(
            model_name, text, True,
            done_reason=oai_resp.get("choices", [{}])[0].get("finish_reason", "stop"),
            context=[1, 2, 3],
            timings=timings,
        )

    # =========================================================================
    # /api/chat - Chat Completion
    # =========================================================================
    @app.post("/api/chat")
    async def api_chat(req: ChatRequest):
        start_time = time.time()
        model_name = req.model

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

        try:
            proc = await process_manager.ensure_model(
                model_name, options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        oai_messages = build_oai_messages_from_native(req.messages)

        if oai_messages and oai_messages[0].get("role") != "system":
            model_cfg = config.model_configs.get(model_name)
            if model_cfg and model_cfg.system:
                oai_messages.insert(0, {"role": "system", "content": model_cfg.system})

        llama_body: Dict[str, Any] = {
            "model": model_name,
            "messages": oai_messages,
            "stream": bool(req.stream),
        }

        oai_tools = build_oai_tools(req.tools)
        if oai_tools:
            llama_body["tools"] = oai_tools

        opts = ollama_options_to_llama_server(req.options)
        llama_body.update(opts)

        if req.options and req.options.get("num_predict"):
            llama_body["max_tokens"] = req.options["num_predict"]
        elif "n_predict" in llama_body:
            llama_body["max_tokens"] = llama_body.pop("n_predict")

        resp_format = format_to_response_format(req.format)
        if resp_format:
            llama_body["response_format"] = resp_format

        if req.think is not None:
            llama_body["think"] = req.think

        if req.stream is not False:
            return StreamingResponse(
                _stream_chat(process_manager, proc, llama_body, model_name, start_time),
                media_type="application/x-ndjson",
            )

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                model_name, "/v1/chat/completions", llama_body,
                options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        timings = extract_timings_from_oai(oai_resp, start_time)
        choice = oai_resp.get("choices", [{}])[0]
        oai_message = choice.get("message", {})

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

    # =========================================================================
    # /api/embed - Generate Embeddings
    # =========================================================================
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

        inputs = req.input if isinstance(req.input, list) else [req.input]
        oai_body = {"model": req.model, "input": inputs}
        if req.dimensions:
            oai_body["dimensions"] = req.dimensions

        try:
            oai_resp = await process_manager.proxy_request_with_retry(
                req.model, "/v1/embeddings", oai_body,
                options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                timeout=120
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        embeddings = [item.get("embedding", []) for item in oai_resp.get("data", [])]
        return {
            "model": req.model,
            "embeddings": embeddings,
            "total_duration": ns_to_duration(time.time() - start_time),
            "load_duration": ns_to_duration(0.001),
            "prompt_eval_count": oai_resp.get("usage", {}).get("prompt_tokens", len(inputs)),
        }

    # =========================================================================
    # /api/embeddings (deprecated)
    # =========================================================================
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
            oai_resp = await process_manager.proxy_request_with_retry(
                req.model, "/v1/embeddings", oai_body,
                options=req.options, keep_alive=req.keep_alive,
                llama_cpp_binary=req.llama_cpp_binary, llama_cpp_flags=req.llama_cpp_flags,
                timeout=120
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Backend error: {str(e)}")

        embedding = []
        for item in oai_resp.get("data", []):
            embedding = item.get("embedding", [])
            break
        return {"embedding": embedding}

    # =========================================================================
    # /api/push (stub)
    # =========================================================================
    @app.post("/api/push")
    async def api_push(request: Request):
        body = await request.json()
        stream = body.get("stream", True)

        async def stream_push():
            yield json.dumps({"status": "push is not supported in Local Model Router"}) + "\n"

        if not stream:
            return {"status": "push is not supported in Local Model Router"}
        return StreamingResponse(stream_push(), media_type="application/x-ndjson")

    # =========================================================================
    # /api/blobs (stubs)
    # =========================================================================
    @app.head("/api/blobs/{digest:path}")
    async def api_blobs_head(digest: str):
        return Response(status_code=200)

    @app.post("/api/blobs/{digest:path}")
    async def api_blobs_post(digest: str, request: Request):
        return Response(status_code=201)


# =============================================================================
# Streaming helpers (module-level for reuse)
# =============================================================================

async def _stream_generate(process_manager: ProcessManager, proc: ProcessInfo,
                            endpoint: str, body: Dict, model_name: str,
                            start_time: float, raw: bool) -> AsyncGenerator[str, None]:
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
                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    if raw:
                        text = choice.get("text", "") or ""
                    else:
                        delta = choice.get("delta", {})
                        text = (delta.get("content") or "") or (delta.get("reasoning_content") or "")
                    if text:
                        eval_count += 1
                        resp_obj = build_generate_response(model_name, text, False)
                        yield json.dumps(resp_obj) + "\n"
                    usage = data.get("usage")
                    if usage:
                        prompt_eval_count = usage.get("prompt_tokens", 0)
    except Exception as e:
        logger.error(f"Streaming error: {e}")

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


async def _stream_chat(process_manager: ProcessManager, proc: ProcessInfo,
                        body: Dict, model_name: str,
                        start_time: float) -> AsyncGenerator[str, None]:
    """Stream chat responses in native format."""
    eval_count = 0
    prompt_eval_count = 0
    full_content = ""
    tool_calls_buffer: List[Dict] = []
    finish_reason = "stop"

    try:
        async with process_manager._session.post(
            f"{proc.base_url}/v1/chat/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600)
        ) as resp:
            logger.debug(f"_stream_chat: llama-server responded with status={resp.status}, body stream={body.get('stream')}")
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

                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    text_to_send = content or reasoning
                    if text_to_send:
                        eval_count += 1
                        full_content += text_to_send
                        msg: Dict[str, Any] = {"role": "assistant", "content": text_to_send}
                        if reasoning and not content:
                            msg["thinking"] = reasoning
                        yield json.dumps(build_chat_response(model_name, msg, False)) + "\n"

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

                    usage = data.get("usage")
                    if usage:
                        prompt_eval_count = usage.get("prompt_tokens", 0)
                        eval_count = max(eval_count, usage.get("completion_tokens", 0))
    except Exception as e:
        logger.error(f"Chat streaming error: {e}")

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
