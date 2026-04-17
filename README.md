# Local Model Router (LMR)

[![Version](https://img.shields.io/badge/version-0.5.1-blue.svg)](https://github.com/g023/localmodelrouter)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A full-featured, single-file local LLM server that provides drop-in API compatibility with both Ollama and OpenAI, using [llama.cpp](https://github.com/ggerganov/llama.cpp)'s `llama-server` as the inference backend.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Overview

Local Model Router (LMR) is a Python-based server that acts as a bridge between client applications and local Large Language Models (LLMs) running via llama.cpp. It provides:

- **Full Ollama API compatibility** - Drop-in replacement for Ollama server
- **Complete OpenAI API compatibility** - Works with OpenAI SDK and tools
- **Multi-model support** - Load and manage multiple models simultaneously
- **Automatic model lifecycle** - Load on demand, unload when idle
- **Streaming responses** - Real-time token streaming for both APIs
- **GPU acceleration** - Automatic GPU memory detection and optimization
- **Hugging Face integration** - Direct model downloads from HF Hub

Unlike Ollama which bundles its own inference engine, LMR leverages the high-performance llama.cpp backend while providing a familiar API surface.

## Features

### Core Capabilities
- **Single-file deployment** - Everything in one `lmr.py` file (~1500 lines)
- **Zero dependencies on Ollama** - Pure Python + llama.cpp
- **Concurrent model serving** - Multiple models loaded simultaneously
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
- **Hugging Face integration** - Direct downloads with `hf:` prefix
- **Local file support** - Direct paths to GGUF files
- **Model templating** - Custom chat templates and system prompts
- **Parameter overrides** - Per-model GPU layers, context size, etc.
- **Keep-alive control** - Configurable model unloading timeouts

### Performance & Optimization
- **GPU memory detection** - Automatic context size estimation
- **Parallel processing** - Multiple concurrent requests per model
- **Flash attention** - Hardware-accelerated attention when available
- **Context size optimization** - Conservative memory management
- **Async I/O** - Non-blocking request handling with aiohttp

## Architecture

LMR follows a multi-process architecture:

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
                ┌────────▼────────┐   │
                │ llama-server    │   │
                │ (Model A)       │   │
                │ Port 39000      │   │
                └─────────────────┘   │
                                      │
                ┌────────▼────────┐   │
                │ llama-server    │   │
                │ (Model B)       │   │
                │ Port 39001      │   │
                └─────────────────┘   │
```

### Key Components

1. **FastAPI Server** - Main HTTP server handling all API requests
2. **Process Manager** - Manages llama-server process lifecycle
3. **Model Registry** - JSON-based configuration of available models
4. **Request Router** - Translates between Ollama/OpenAI formats and llama-server protocol
5. **Health Monitor** - Background task checking process health and unloading idle models

### Data Flow

1. **Request Reception** - FastAPI receives API call (e.g., `/api/chat`)
2. **Model Resolution** - Process Manager checks if model is loaded
3. **Process Launch** - If not loaded, launches llama-server with model
4. **Request Translation** - Converts Ollama/OpenAI format to llama-server JSON
5. **Proxying** - Forwards request to appropriate llama-server instance
6. **Response Translation** - Converts llama-server response back to requested format
7. **Streaming** - Handles Server-Sent Events for real-time responses

### Process Lifecycle

- **Load on Demand** - Models loaded when first requested
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
# Clone or download lmr.py
wget https://raw.githubusercontent.com/g023/localmodelrouter/main/lmr.py
chmod +x lmr.py

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
    "path": "hf:mistralai/Mistral-7B-Instruct-v0.1-GGUF:mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "ctx_size": 8192,
    "system": "You are a helpful assistant."
  }
}
```

#### Configuration Options

| Field | Type | Description | Default |
|-------|------|-------------|---------|
| `path` | string | Path to GGUF file or `hf:owner/repo:filename` for HF download | Required |
| `num_gpu` | int | GPU layers (-1 = auto, 0 = CPU only) | -1 |
| `ctx_size` | int | Context size in tokens (0 = auto) | 0 |
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
    model="llama3.2-3b",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Model Management

```bash
# Show model details
curl -X POST http://localhost:11434/api/show -d '{"model": "llama3.2-3b"}'

# Create a new model variant
curl -X POST http://localhost:11434/api/create \
  -d '{
    "model": "llama3.2-3b-custom",
    "from": "llama3.2-3b",
    "system": "You are a helpful coding assistant."
  }'

# Copy a model
curl -X POST http://localhost:11434/api/copy \
  -d '{"source": "llama3.2-3b", "destination": "llama3.2-3b-backup"}'

# Delete a model
curl -X DELETE http://localhost:11434/api/delete -d '{"model": "llama3.2-3b-backup"}'

# List running models
curl http://localhost:11434/api/ps
```

## API Reference

### Ollama-Compatible API Endpoints

#### Core Endpoints

- `GET /` - Server status
- `GET /api/version` - Server version
- `GET /api/tags` - List available models
- `GET /api/ps` - List running models

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
- `POST /v1/chat/completions` - Chat completions
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
  "model": "llama3.2-3b",
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
  "model": "llama3.2-3b",
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
  "model": "llama3.2-3b",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "llama3.2-3b",
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

## Examples

### Basic Chat

```python
import requests

# Ollama-style
response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2-3b",
    "messages": [{"role": "user", "content": "Write a haiku about coding"}],
    "stream": False
})
print(response.json()["message"]["content"])
```

### Streaming Chat

```python
import json

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "llama3.2-3b", 
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}, stream=True)

for line in response.iter_lines():
    if line:
        chunk = json.loads(line.decode())
        if chunk.get("done"):
            break
        content = chunk.get("message", {}).get("content", "")
        print(content, end="", flush=True)
```

### OpenAI SDK Integration

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="dummy")

# Chat completion
response = client.chat.completions.create(
    model="llama3.2-3b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing simply."}
    ],
    max_tokens=200,
    temperature=0.7
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="llama3.2-3b",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Model Management

```python
# Create a specialized model
requests.post("http://localhost:11434/api/create", json={
    "model": "coder-assistant",
    "from": "codellama",
    "system": "You are an expert coding assistant. Provide clear, efficient code solutions.",
    "parameters": {
        "temperature": 0.2,
        "top_p": 0.9
    }
})

# Use the specialized model
response = requests.post("http://localhost:11434/api/chat", json={
    "model": "coder-assistant",
    "messages": [{"role": "user", "content": "Write a Python function to reverse a string"}]
})
```

### Function Calling

```python
# OpenAI-style function calling
response = client.chat.completions.create(
    model="llama3.2-3b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    functions=[{
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            },
            "required": ["location"]
        }
    }],
    function_call="auto"
)

if response.choices[0].message.function_call:
    print(f"Function: {response.choices[0].message.function_call.name}")
    print(f"Args: {response.choices[0].message.function_call.arguments}")
```

## Development

### Project Structure

```
lmr.py                 # Main server file (~1500 lines)
models.json           # Model configuration
test_lmr.py          # Comprehensive test suite
README.md            # This file
```

### Running Tests

```bash
# Run the comprehensive test suite
python test_lmr.py

# Test specific endpoints
curl -X POST http://localhost:11434/api/chat \
  -d '{"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}'
```

### Adding New Features

1. **New API Endpoints**: Add FastAPI route handlers in the main file
2. **Model Configuration**: Extend `ModelConfig` class for new parameters
3. **Process Management**: Modify `ProcessManager` for new lifecycle features
4. **Response Translation**: Update format conversion functions

### Code Style

- **Type Hints**: All functions use full type annotations
- **Async/Await**: Asynchronous request handling throughout
- **Error Handling**: Comprehensive exception handling with proper HTTP status codes
- **Logging**: Structured logging with configurable levels
- **Docstrings**: Comprehensive documentation for all classes and functions

### Debugging

```bash
# Enable debug logging
python lmr.py --log-level DEBUG

# Check running processes
ps aux | grep llama-server

# Monitor GPU usage
nvidia-smi

# Check server logs
tail -f /tmp/lmr.log  # If logging to file
```

## Troubleshooting

### Common Issues

#### "llama-server binary not found"
**Solution**: Install llama.cpp and ensure `llama-server` is in PATH, or specify path with `--llama-server`

```bash
# Check if installed
which llama-server

# Specify custom path
python lmr.py --llama-server /path/to/llama-server
```

#### "Model file not found"
**Solution**: Verify model path in `models.json` or use Hugging Face integration

```json
{
  "my-model": "hf:owner/repo/model.Q4_K_M.gguf"
}
```

#### "CUDA out of memory"
**Solution**: Reduce context size or GPU layers

```json
{
  "my-model": {
    "path": "/path/to/model.gguf",
    "ctx_size": 2048,
    "num_gpu": 20
  }
}
```

#### "Connection refused"
**Solution**: Ensure server is running and accessible

```bash
# Check if server is running
curl http://localhost:11434/api/version

# Check port usage
netstat -tlnp | grep 11434
```

#### "Model loading slow"
**Solution**: Check GPU utilization and model size

```bash
# Monitor loading
python lmr.py --log-level DEBUG

# Check GPU memory
nvidia-smi
```

### Performance Tuning

1. **GPU Memory**: Ensure sufficient VRAM for model + context
2. **Context Size**: Balance between capability and memory usage
3. **Parallel Slots**: Increase for concurrent requests
4. **Flash Attention**: Enable for supported GPUs
5. **Model Quantization**: Use Q4_K_M for balance of speed/quality

### Logs and Monitoring

```bash
# Server logs show detailed information
python lmr.py --log-level DEBUG

# Check running models
curl http://localhost:11434/api/ps

# Monitor llama-server processes
ps aux | grep llama-server
```

### Compatibility

- **Python**: 3.8+ required
- **llama.cpp**: Latest version recommended
- **GPU**: CUDA 11.0+ or ROCm 5.0+
- **Models**: GGUF format only

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The underlying inference engine
- [Ollama](https://github.com/jmorganca/ollama) - API design inspiration
- [FastAPI](https://github.com/tiangolo/fastapi) - Web framework
- [OpenAI](https://platform.openai.com/docs/api-reference) - API specification

---

**Local Model Router** - Bringing the power of local LLMs to your applications with familiar APIs.

