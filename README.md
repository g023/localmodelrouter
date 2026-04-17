# Local Model Router (LMR) v0.6.0

[![Version](https://img.shields.io/badge/version-0.6.0-blue.svg)](https://github.com/g023/localmodelrouter)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance local LLM server providing drop-in API compatibility with Ollama and OpenAI, built on [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server`. Features automatic VRAM management, Hugging Face integration, and modular architecture.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

Local Model Router (LMR) is a Python-based server that bridges client applications to local Large Language Models running via llama.cpp. It offers:

- **Full Ollama API compatibility** - Drop-in replacement for Ollama server
- **Complete OpenAI API compatibility** - Works with OpenAI SDK and tools
- **Multi-model support** - Load and manage multiple models simultaneously
- **Automatic VRAM management** - LRU eviction and context optimization
- **Hugging Face integration** - Direct downloads with quantization hints
- **Streaming responses** - Real-time token streaming for both APIs
- **GPU acceleration** - Automatic GPU memory detection and optimization

Unlike Ollama which bundles its own inference engine, LMR leverages the battle-tested llama.cpp backend while providing familiar APIs.

## Features

### Core Capabilities
- **Modular package architecture** - Clean `lmr/` package with thin `lmr.py` entry point
- **Zero dependencies on Ollama** - Pure Python + llama.cpp
- **Concurrent model serving** - Multiple models loaded simultaneously with VRAM management
- **Dynamic port allocation** - Automatic port management for llama-server instances
- **Health monitoring** - Automatic process health checks and restarts
- **Graceful shutdown** - Proper cleanup of all child processes

### API Compatibility
- **Ollama API**: `/api/generate`, `/api/chat`, `/api/tags`, `/api/show`, `/api/create`, `/api/copy`, `/api/delete`, `/api/pull`, `/api/push`, `/api/ps`, `/api/embed`, `/api/embeddings`, `/api/version`
- **OpenAI API**: `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/v1/images/generations`, `/v1/responses`, `/v1/health`
- **Streaming support** - Server-Sent Events (SSE) for real-time responses
- **Tool/function calling** - Full support for tool use in chat completions

### Model Management
- **Flexible model configuration** - JSON-based model registry
- **Hugging Face integration** - Direct downloads with `hf:owner/repo:quant` format (e.g., `hf:g023/Qwen3-1.77B-g023-GGUF:Q8_0`)
- **Local file support** - Direct paths to GGUF files
- **Model templating** - Custom chat templates and system prompts
- **Parameter overrides** - Per-model GPU layers, context size, etc.
- **Keep-alive control** - Configurable model unloading timeouts

### Performance & Optimization
- **VRAM management** - LRU eviction when memory is insufficient, embedding context capping
- **GPU memory detection** - Automatic context size estimation based on available VRAM
- **Parallel processing** - Multiple concurrent requests per model
- **Flash attention** - Hardware-accelerated attention when available
- **Context size optimization** - Conservative memory management with multi-model awareness
- **Async I/O** - Non-blocking request handling with aiohttp

## Architecture

LMR follows a multi-process architecture with automatic resource management:

```
┌─────────────────┐    ┌──────────────────┐
│   Client Apps   │────│   LMR Server     │
│ (OpenAI SDK,    │    │   (FastAPI)      │
│  Ollama CLI,    │    │                  │
│  Custom tools)  │    │ ┌─────────────┐  │
└─────────────────┘    │ │ Process     │  │
                       │ │ Manager     │  │
                       └─┼─────────────┼──┘
                         │             │
                ┌────────▼────────┐    │
                │ llama-server    │    │
                │ (Model A)       │    │
                │ Port 39000      │    │
                └─────────────────┘    │
                                       │
                ┌────────▼────────┐    │
                │ llama-server    │    │
                │ (Model B)       │    │
                │ Port 39001      │    │
                └─────────────────┘    │
```

### Project Structure

```
lmr.py                    # Thin entry point (imports from lmr package)
lmr/
├── __init__.py           # Package init, version exports (LMR_VERSION="0.6.0")
├── __main__.py           # python -m lmr support
├── app.py                # FastAPI app factory, lifespan, error handlers
├── cli.py                # CLI argument parsing, main()
├── config.py             # ServerConfig, ModelConfig, constants
├── converters.py         # Format conversion (Ollama ↔ OpenAI ↔ llama-server)
├── models.py             # Pydantic request/response models
├── process.py            # ProcessManager, llama-server lifecycle, VRAM management
├── responses.py          # Response builders (generate, chat, timings)
├── utils.py              # Utility functions (GPU detection, parsing, etc.)
└── routes/
    ├── __init__.py       # Route registration
    ├── native.py         # Ollama /api/* endpoints
    └── openai_api.py     # OpenAI /v1/* endpoints
models.json               # Model configuration
examples/                 # Usage examples (7 scripts)
├── chat_basic.py         # Basic chat completion
├── chat_streaming.py     # Streaming chat
├── openai_sdk.py         # OpenAI SDK usage
├── model_management.py   # Model CRUD operations
├── embeddings.py         # Embedding generation
├── tool_calling.py       # Function calling
└── multi_model.py        # Multi-model usage
test_lmr.py               # Test suite (39 tests)
README.md                 # This file
```

### Key Components

1. **FastAPI Server** (`lmr/app.py`) - Main HTTP server handling all API requests
2. **Process Manager** (`lmr/process.py`) - Manages llama-server process lifecycle with VRAM management
3. **Model Registry** (`lmr/config.py`) - JSON-based configuration of available models
4. **Request Router** (`lmr/routes/`) - Translates between Ollama/OpenAI formats and llama-server protocol
5. **Health Monitor** (`lmr/process.py`) - Background task checking process health and unloading idle models

### Data Flow

1. **Request Reception** - FastAPI receives API call (e.g., `/api/chat`)
2. **Model Resolution** - Process Manager checks if model is loaded, downloads from HF if needed
3. **VRAM Check** - Estimates memory requirements, evicts LRU models if necessary
4. **Process Launch** - If not loaded, launches llama-server with optimized parameters
5. **Request Translation** - Converts Ollama/OpenAI format to llama-server JSON
6. **Proxying** - Forwards request to appropriate llama-server instance
7. **Response Translation** - Converts llama-server response back to requested format
8. **Streaming** - Handles Server-Sent Events for real-time responses

### Process Lifecycle & VRAM Management

- **Load on Demand** - Models loaded when first requested, with automatic HF downloads
- **VRAM Management** - LRU eviction when insufficient memory, embedding models capped at 8192 context
- **Keep-Alive** - Models stay loaded for configurable time after last use
- **Health Checks** - Periodic verification of process health
- **Graceful Shutdown** - Proper termination of all child processes on exit

## Installation

### Prerequisites

- **Python 3.8+** with pip
- **llama.cpp** with `llama-server` binary compiled
- **GGUF model files** (download from Hugging Face or convert existing models)

### Install LMR

```bash
# Clone repository
git clone https://github.com/g023/localmodelrouter.git
cd localmodelrouter

# Install dependencies
pip install fastapi uvicorn pydantic requests huggingface_hub aiohttp
```

### Install llama.cpp

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# Build with server support
mkdir build && cd build
cmake .. -DLLAMA_BUILD_SERVER=ON
make -j$(nproc)

# Verify installation
./bin/llama-server --help
```

### Download Models

```bash
# Option 1: Direct GGUF download
wget https://huggingface.co/microsoft/WizardLM-2-8x22B-GGUF/resolve/main/WizardLM-2-8x22B.Q4_K_M.gguf

# Option 2: Use LMR's Hugging Face integration (automatic)
# Models will be downloaded on first use with hf: prefix
# Example: "hf:mistralai/Mistral-7B-Instruct-v0.1-GGUF:Q4_K_M"
# Example: "g023/Qwen3-1.77B-g023-GGUF:Q8_0"
```

## Configuration

### Model Configuration (`models.json`)

LMR uses a JSON file to define available models. Create `models.json` in the same directory as `lmr.py`:

```json
{
  "llama3.2-3b": "/path/to/llama-3.2-3b-instruct.Q4_K_M.gguf",
  "codellama": {
    "path": "/path/to/codellama-7b-instruct.Q5_K_M.gguf",
    "num_gpu": 35,
    "ctx_size": 4096,
    "num_parallel": 8,
    "extra_flags": {"--flash-attn": true}
  },
  "mistral-7b": {
    "path": "hf:mistralai/Mistral-7B-Instruct-v0.1-GGUF:Q4_K_M",
    "ctx_size": 8192,
    "system": "You are a helpful assistant."
  },
  "qwen-embedding": {
    "path": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
    "ctx_size": 8192
  }
}
```

#### Configuration Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `path` | string | Path to GGUF file or `hf:owner/repo:quant` for HF download | Required |
| `num_gpu` | int | GPU layers (-1 = auto, 0 = CPU only) | -1 |
| `ctx_size` | int | Context size in tokens (0 = auto, capped at 8192 for embeddings) | 0 |
| `num_parallel` | int | Parallel processing slots | 4 |
| `extra_flags` | object | Additional llama-server flags | {} |
| `template` | string | Chat template override | "" |
| `system` | string | System message | "" |
| `parameters` | object | Model parameters | {} |
| `family` | string | Model family | "" |

### Server Configuration

LMR supports command-line configuration:

```bash
# Basic usage
python lmr.py

# Custom port and config
python lmr.py --port 8080 --models-json /path/to/models.json

# Custom llama-server binary
python lmr.py --llama-server /custom/path/to/llama-server

# Debug logging
python lmr.py --log-level DEBUG
```

#### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--port` | Server port | 11434 |
| `--host` | Server host | 0.0.0.0 |
| `--models-json` | Path to models.json | models.json |
| `--llama-server` | Path to llama-server binary | Auto-detected |
| `--models-dir` | Directory for downloaded models | ~/.local/share/localmodelrouter/models |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |

## Usage

### Starting the Server

```bash
# Default configuration
python lmr.py

# Custom configuration
python lmr.py --port 8080 --models-json my-models.json --log-level DEBUG
```

The server will start and listen for requests. It automatically detects GPU memory and configures models accordingly.

### Basic Testing

```bash
# Check server status
curl http://localhost:11434/

# List available models
curl http://localhost:11434/api/tags

# Get server version
curl http://localhost:11434/api/version
```

### Using with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # LMR doesn't require authentication
)

response = client.chat.completions.create(
    model="g023/Qwen3-1.77B-g023-GGUF:Q8_0",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Model Management

```bash
# Show model details
curl -X POST http://localhost:11434/api/show -d '{"model": "g023/Qwen3-1.77B-g023-GGUF:Q8_0"}'

# Create a new model variant
curl -X POST http://localhost:11434/api/create \
  -d '{
    "model": "llama3.2-3b-custom",
    "from": "g023/Qwen3-1.77B-g023-GGUF:Q8_0",
    "system": "You are a helpful coding assistant."
  }'

# Copy a model
curl -X POST http://localhost:11434/api/copy \
  -d '{"source": "g023/Qwen3-1.77B-g023-GGUF:Q8_0", "destination": "turbo-g023-backup"}'

# Delete a model
curl -X DELETE http://localhost:11434/api/delete -d '{"model": "turbo-g023-backup"}'

# List running models
curl http://localhost:11434/api/ps
```

## API Reference

### Ollama-Compatible API Endpoints

#### Core Endpoints

- `GET /` - Server status
- `GET /api/version` - Server version
- `GET /api/tags` - List available models
- `GET /api/ps` - List running models with VRAM usage

#### Model Management

- `POST /api/show` - Show model details
- `POST /api/create` - Create new model
- `POST /api/copy` - Copy model
- `DELETE /api/delete` - Delete model
- `POST /api/pull` - Pull model (stub implementation)

#### Inference

- `POST /api/generate` - Text generation
- `POST /api/chat` - Chat completion
- `POST /api/embed` - Single embedding
- `POST /api/embeddings` - Batch embeddings

#### Other

- `POST /api/push` - Push model (stub)
- `HEAD /api/blobs/{digest}` - Check blob existence
- `POST /api/blobs/{digest}` - Create blob

### OpenAI-Compatible API Endpoints

- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Chat completions with tool calling
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Embeddings
- `POST /v1/images/generations` - Image generation (stub)
- `POST /v1/responses` - Responses API (stub)
- `GET /v1/health` - Health check
- `/{path}` - Catch-all for other OpenAI endpoints

### Request/Response Formats

#### Chat Completion (Ollama)

**Request:**
```json
{
  "model": "g023/Qwen3-1.77B-g023-GGUF:Q8_0",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "options": {
    "temperature": 0.7,
    "num_predict": 100
  }
}
```

**Response:**
```json
{
  "model": "g023/Qwen3-1.77B-g023-GGUF:Q8_0",
  "created_at": "2024-01-01T12:00:00Z",
  "message": {
    "role": "assistant",
    "content": "Hello! How can I help you today?"
  },
  "done": true,
  "total_duration": 1500000000,
  "load_duration": 500000000,
  "prompt_eval_count": 10,
  "prompt_eval_duration": 100000000,
  "eval_count": 20,
  "eval_duration": 1000000000
}
```

#### Chat Completion (OpenAI)

**Request:**
```json
{
  "model": "g023/Qwen3-1.77B-g023-GGUF:Q8_0",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 100
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "g023/Qwen3-1.77B-g023-GGUF:Q8_0",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

#### Embeddings

**Request:**
```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

**Response:**
```json
{
  "model": "Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
  "embeddings": [[0.1, 0.2, ..., 0.9]],
  "total_duration": 50000000,
  "load_duration": 10000000,
  "prompt_eval_count": 9
}
```

## Examples

LMR includes some comprehensive examples in the `examples/` directory:

### Basic Chat (`examples/chat_basic.py`)
Demonstrates basic chat completion with both Ollama and OpenAI APIs.

### Streaming Chat (`examples/chat_streaming.py`)
Shows real-time token streaming for chat completions.

### OpenAI SDK (`examples/openai_sdk.py`)
Using the official OpenAI Python SDK with LMR.

### Model Management (`examples/model_management.py`)
Creating, copying, and deleting models programmatically.

### Embeddings (`examples/embeddings.py`)
Generating embeddings with both native and OpenAI endpoints.

### Tool Calling (`examples/tool_calling.py`)
Function calling with tools, using Hugging Face model auto-download.

### Multi-Model (`examples/multi_model.py`)
Loading and using multiple models simultaneously with VRAM management.

Run any example with:
```bash
python examples/chat_basic.py
```

## Troubleshooting

### Common Issues

**"llama-server binary not found"**
- Ensure llama.cpp is compiled with `-DLLAMA_BUILD_SERVER=ON`
- Use `--llama-server /path/to/llama-server` to specify custom path

**"Model not found"**
- Check `models.json` configuration
- For HF models, ensure correct format: `hf:owner/repo:quant`
- Verify model file exists or can be downloaded

**"CUDA out of memory"**
- Reduce context size with `ctx_size` in model config
- Enable VRAM management (automatic LRU eviction)
- Embedding models are automatically capped at 8192 context

**"Failed to start llama-server"**
- Check stderr output in server logs
- Verify model file is valid GGUF
- Ensure sufficient system memory

**Streaming not working**
- Ensure `import aiohttp` is available (fixed in v0.6.0)
- Check for null content deltas (fixed in v0.6.0)

### Debugging

Enable debug logging:
```bash
python lmr.py --log-level DEBUG
```

Check running processes:
```bash
curl http://localhost:11434/api/ps
```

View server logs for detailed error messages and VRAM management actions.

### Performance Tuning

- **GPU Memory**: LMR auto-detects GPU memory and optimizes context sizes
- **Parallel Slots**: Increase `num_parallel` for higher concurrency
- **Flash Attention**: Enable with `extra_flags: {"--flash-attn": true}`
- **Context Size**: Balance between memory usage and capability

## Development

### Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest test_lmr.py -v

# Run with coverage
python -m pytest test_lmr.py --cov=lmr --cov-report=html
```

All 39 tests should pass, covering API endpoints, model management, and streaming.

### Project Structure

The codebase is organized into a clean modular package:

- `lmr/` - Main package
- `examples/` - Usage examples
- `test_lmr.py` - Test suite


### Version History

- **v0.6.0** - Modular refactor, VRAM management, HF quantization hints, embedding fixes
- **v0.5.0** - Initial modular architecture
- **v0.4.0** - OpenAI API compatibility
- **v0.3.0** - Streaming support
- **v0.2.0** - Multi-model support
- **v0.1.0** - Basic Ollama compatibility

## License

MIT License - see [LICENSE](LICENSE) for details.

