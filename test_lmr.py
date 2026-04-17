#!/usr/bin/env python3
"""Comprehensive test suite for Local Model Router."""
import json
import sys
import time
import requests

BASE = "http://localhost:11435"
MODEL = "qwen3-tiny"
PASS = 0
FAIL = 0
ERRORS = []

def test(name, func):
    global PASS, FAIL, ERRORS
    try:
        result = func()
        if result:
            PASS += 1
            print(f"  ✓ {name}")
        else:
            FAIL += 1
            ERRORS.append(name)
            print(f"  ✗ {name}")
    except Exception as e:
        FAIL += 1
        ERRORS.append(f"{name}: {e}")
        print(f"  ✗ {name} - ERROR: {e}")

# ========== Basic Endpoints ==========
print("\n=== Basic Endpoints ===")

def test_root():
    r = requests.get(f"{BASE}/")
    return r.status_code == 200 and "running" in r.text.lower()
test("GET /", test_root)

def test_head():
    r = requests.head(f"{BASE}/")
    return r.status_code == 200
test("HEAD /", test_head)

def test_version():
    r = requests.get(f"{BASE}/api/version")
    d = r.json()
    return r.status_code == 200 and "version" in d
test("GET /api/version", test_version)

def test_tags():
    r = requests.get(f"{BASE}/api/tags")
    d = r.json()
    return r.status_code == 200 and "models" in d and len(d["models"]) > 0
test("GET /api/tags", test_tags)

def test_tags_format():
    r = requests.get(f"{BASE}/api/tags")
    m = r.json()["models"][0]
    required = ["name", "model", "modified_at", "size", "digest", "details"]
    return all(k in m for k in required)
test("GET /api/tags format", test_tags_format)

def test_ps_empty():
    r = requests.get(f"{BASE}/api/ps")
    d = r.json()
    return r.status_code == 200 and "models" in d
test("GET /api/ps", test_ps_empty)

# ========== OpenAI Endpoints (Basic) ==========
print("\n=== OpenAI Basic Endpoints ===")

def test_v1_models():
    r = requests.get(f"{BASE}/v1/models")
    d = r.json()
    return r.status_code == 200 and "data" in d and len(d["data"]) > 0
test("GET /v1/models", test_v1_models)

def test_v1_models_format():
    r = requests.get(f"{BASE}/v1/models")
    m = r.json()["data"][0]
    return all(k in m for k in ["id", "object", "created", "owned_by"])
test("GET /v1/models format", test_v1_models_format)

def test_v1_health():
    r = requests.get(f"{BASE}/v1/health")
    return r.status_code == 200
test("GET /v1/health", test_v1_health)

# ========== Model Management ==========
print("\n=== Model Management ===")

def test_show():
    r = requests.post(f"{BASE}/api/show", json={"model": MODEL})
    d = r.json()
    return r.status_code == 200 and "details" in d
test("POST /api/show", test_show)

def test_show_404():
    r = requests.post(f"{BASE}/api/show", json={"model": "nonexistent-model-xyz"})
    return r.status_code == 404
test("POST /api/show 404", test_show_404)

def test_copy():
    r = requests.post(f"{BASE}/api/copy", json={"source": MODEL, "destination": "test-copy"})
    return r.status_code == 200
test("POST /api/copy", test_copy)

def test_copy_shows():
    r = requests.get(f"{BASE}/api/tags")
    names = [m["name"] for m in r.json()["models"]]
    return "test-copy:latest" in names
test("POST /api/copy shows in tags", test_copy_shows)

def test_delete():
    r = requests.delete(f"{BASE}/api/delete", json={"model": "test-copy"})
    return r.status_code == 200
test("DELETE /api/delete", test_delete)

def test_delete_gone():
    r = requests.get(f"{BASE}/api/tags")
    names = [m["name"] for m in r.json()["models"]]
    return "test-copy:latest" not in names
test("DELETE /api/delete removes from tags", test_delete_gone)

def test_create():
    r = requests.post(f"{BASE}/api/create", json={
        "model": "test-created",
        "from": MODEL,
        "system": "You are a test assistant.",
        "stream": False
    })
    return r.status_code == 200
test("POST /api/create", test_create)

def test_create_cleanup():
    requests.delete(f"{BASE}/api/delete", json={"model": "test-created"})
    return True
test("Cleanup created model", test_create_cleanup)

# ========== Chat Completion (Native) ==========
print("\n=== Native Chat (loads model - may take time) ===")

def test_chat_nonstream():
    r = requests.post(f"{BASE}/api/chat", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say the word 'hello'"}],
        "stream": False,
        "options": {"num_predict": 100, "temperature": 0.5}
    }, timeout=120)
    d = r.json()
    return (r.status_code == 200 and d.get("done") == True and 
            "message" in d and d["message"]["role"] == "assistant")
test("POST /api/chat non-streaming", test_chat_nonstream)

def test_chat_nonstream_timings():
    r = requests.post(f"{BASE}/api/chat", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count to 3"}],
        "stream": False,
        "options": {"num_predict": 100}
    }, timeout=120)
    d = r.json()
    timing_keys = ["total_duration", "load_duration", "prompt_eval_count", 
                   "prompt_eval_duration", "eval_count", "eval_duration"]
    return all(k in d for k in timing_keys)
test("POST /api/chat timings", test_chat_nonstream_timings)

def test_chat_has_content():
    r = requests.post(f"{BASE}/api/chat", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with exactly: HELLO WORLD"}],
        "stream": False,
        "options": {"num_predict": 200, "temperature": 0.3}
    }, timeout=120)
    d = r.json()
    content = d.get("message", {}).get("content", "")
    return len(content) > 0
test("POST /api/chat has content", test_chat_has_content)

def test_chat_streaming():
    r = requests.post(f"{BASE}/api/chat", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi"}],
        "stream": True,
        "options": {"num_predict": 50}
    }, timeout=120, stream=True)
    chunks = []
    for line in r.iter_lines():
        if line:
            chunks.append(json.loads(line))
    # Should have at least one streaming chunk + final done
    has_done = any(c.get("done") == True for c in chunks)
    has_content = any(c.get("message", {}).get("content", "") for c in chunks)
    return has_done and (has_content or len(chunks) > 1)
test("POST /api/chat streaming", test_chat_streaming)

def test_chat_load():
    r = requests.post(f"{BASE}/api/chat", json={
        "model": MODEL,
        "messages": []
    }, timeout=120)
    d = r.json()
    return d.get("done") == True and d.get("done_reason") == "load"
test("POST /api/chat load model", test_chat_load)

# ========== Generate (Native) ==========
print("\n=== Native Generate ===")

def test_generate_nonstream():
    r = requests.post(f"{BASE}/api/generate", json={
        "model": MODEL,
        "prompt": "Write the number 42",
        "stream": False,
        "options": {"num_predict": 100, "temperature": 0.5}
    }, timeout=120)
    d = r.json()
    return r.status_code == 200 and d.get("done") == True and "response" in d
test("POST /api/generate non-streaming", test_generate_nonstream)

def test_generate_streaming():
    r = requests.post(f"{BASE}/api/generate", json={
        "model": MODEL,
        "prompt": "Count to 3",
        "stream": True,
        "options": {"num_predict": 80}
    }, timeout=120, stream=True)
    chunks = []
    for line in r.iter_lines():
        if line:
            chunks.append(json.loads(line))
    has_done = any(c.get("done") == True for c in chunks)
    return has_done and len(chunks) > 1
test("POST /api/generate streaming", test_generate_streaming)

def test_generate_load():
    r = requests.post(f"{BASE}/api/generate", json={"model": MODEL}, timeout=120)
    d = r.json()
    return d.get("done") == True
test("POST /api/generate load model", test_generate_load)

# ========== PS with running model ==========
print("\n=== PS with model loaded ===")

def test_ps_with_model():
    r = requests.get(f"{BASE}/api/ps")
    d = r.json()
    return len(d.get("models", [])) > 0
test("GET /api/ps shows loaded model", test_ps_with_model)

def test_ps_format():
    r = requests.get(f"{BASE}/api/ps")
    models = r.json().get("models", [])
    if not models:
        return False
    m = models[0]
    return all(k in m for k in ["name", "model", "size", "digest", "details", "size_vram"])
test("GET /api/ps format", test_ps_format)

# ========== OpenAI Chat Completions ==========
print("\n=== OpenAI Chat Completions ===")

def test_v1_chat_nonstream():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 100,
        "temperature": 0.5
    }, timeout=120)
    d = r.json()
    return (r.status_code == 200 and "choices" in d and len(d["choices"]) > 0 
            and "id" in d and "object" in d and d["object"] == "chat.completion")
test("POST /v1/chat/completions non-streaming", test_v1_chat_nonstream)

def test_v1_chat_usage():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 50
    }, timeout=120)
    d = r.json()
    return "usage" in d and "prompt_tokens" in d.get("usage", {})
test("POST /v1/chat/completions usage", test_v1_chat_usage)

def test_v1_chat_streaming():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 50,
        "stream": True
    }, timeout=120, stream=True)
    chunks = []
    for line in r.iter_lines():
        if line:
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return len(chunks) > 0
test("POST /v1/chat/completions streaming", test_v1_chat_streaming)

def test_v1_chat_with_system():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"}
        ],
        "max_tokens": 50,
    }, timeout=120)
    return r.status_code == 200 and "choices" in r.json()
test("POST /v1/chat/completions with system", test_v1_chat_with_system)

def test_v1_chat_seed():
    r = requests.post(f"{BASE}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 30,
        "seed": 42,
        "temperature": 0
    }, timeout=120)
    return r.status_code == 200
test("POST /v1/chat/completions with seed", test_v1_chat_seed)

# ========== OpenAI Completions ==========
print("\n=== OpenAI Completions ===")

def test_v1_completions():
    r = requests.post(f"{BASE}/v1/completions", json={
        "model": MODEL,
        "prompt": "The sky is",
        "max_tokens": 30
    }, timeout=120)
    d = r.json()
    return r.status_code == 200 and "choices" in d
test("POST /v1/completions", test_v1_completions)

def test_v1_completions_streaming():
    r = requests.post(f"{BASE}/v1/completions", json={
        "model": MODEL,
        "prompt": "Count: 1, 2, 3",
        "max_tokens": 30,
        "stream": True
    }, timeout=120, stream=True)
    chunks = []
    for line in r.iter_lines():
        if line:
            line = line.decode() if isinstance(line, bytes) else line
            if line.startswith("data: ") and line != "data: [DONE]":
                chunks.append(json.loads(line[6:]))
    return len(chunks) > 0
test("POST /v1/completions streaming", test_v1_completions_streaming)

# ========== Pull ==========
print("\n=== Pull ===")

def test_pull_existing():
    r = requests.post(f"{BASE}/api/pull", json={"model": MODEL, "stream": False}, timeout=30)
    d = r.json()
    return "status" in d
test("POST /api/pull existing model", test_pull_existing)

# ========== Push (stub) ==========
print("\n=== Push ===")

def test_push_stub():
    r = requests.post(f"{BASE}/api/push", json={"model": "test", "stream": False}, timeout=10)
    return r.status_code == 200
test("POST /api/push stub", test_push_stub)

# ========== Blobs (stubs) ==========
print("\n=== Blobs ===")

def test_blob_head():
    r = requests.head(f"{BASE}/api/blobs/sha256:abc123")
    return r.status_code == 200
test("HEAD /api/blobs/:digest", test_blob_head)

def test_blob_post():
    r = requests.post(f"{BASE}/api/blobs/sha256:abc123")
    return r.status_code == 201
test("POST /api/blobs/:digest", test_blob_post)

# ========== Images Generations (stub) ==========
print("\n=== Images ===")

def test_images_stub():
    r = requests.post(f"{BASE}/v1/images/generations", json={"prompt": "a cat"})
    return r.status_code == 501
test("POST /v1/images/generations stub", test_images_stub)

# ========== Summary ==========
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed out of {PASS+FAIL} tests")
if ERRORS:
    print(f"\nFailed tests:")
    for e in ERRORS:
        print(f"  - {e}")
print(f"{'='*50}")
sys.exit(0 if FAIL == 0 else 1)
