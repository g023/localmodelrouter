"""Format conversion helpers between Ollama-native and OpenAI formats."""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from lmr.models import Message


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
    # Server-level options that are not per-request parameters
    server_opts = {
        "num_ctx", "num_gpu", "num_thread", "num_batch",
        "main_gpu", "rope_scale", "rope_freq_base",
        "num_keep", "numa", "use_mmap", "use_mlock",
        "num_parallel",
    }
    result = {}
    for k, v in options.items():
        if k in mapping:
            result[mapping[k]] = v
        elif k not in server_opts:
            result[k] = v
    return result


def build_oai_messages_from_native(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert native message format to OpenAI message format for llama-server."""
    oai_messages = []
    for msg in messages:
        oai_msg: Dict[str, Any] = {"role": msg.role, "content": msg.content or ""}

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
