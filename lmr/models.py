"""Pydantic request/response models for LMR API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Native API Models
# =============================================================================

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
    width: Optional[int] = None
    height: Optional[int] = None
    steps: Optional[int] = None
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
    input: Any
    truncate: Optional[bool] = True
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None
    dimensions: Optional[int] = None
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None


# =============================================================================
# OpenAI API Models
# =============================================================================

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
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Any
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
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None


class OpenAIEmbeddingRequest(BaseModel):
    model: str
    input: Any
    encoding_format: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None
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
    llama_cpp_binary: Optional[str] = None
    llama_cpp_flags: Optional[Dict[str, Any]] = None
    keep_alive: Optional[Any] = None
