"""Response builder functions for native API format."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from lmr.utils import get_timestamp, ns_to_duration


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

    prompt_ms = timings_raw.get("prompt_ms", 0)
    predicted_ms = timings_raw.get("predicted_ms", 0)

    if prompt_ms > 0:
        prompt_eval_duration = int(prompt_ms * 1e6)
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
