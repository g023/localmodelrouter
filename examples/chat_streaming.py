#!/usr/bin/env python3
"""Streaming chat example using native /api/chat endpoint."""
import json
import requests

BASE = "http://localhost:11434"
MODEL = "g023/Qwen3-1.77B-g023-GGUF:Q8_0"

# Streaming chat
resp = requests.post(f"{BASE}/api/chat", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "Count from 1 to 5."}],
    "stream": True,
    "options": {"num_predict": 100},
}, stream=True)

print("Streaming response: ", end="", flush=True)
for line in resp.iter_lines():
    if line:
        chunk = json.loads(line)
        content = chunk.get("message", {}).get("content", "")
        if content:
            print(content, end="", flush=True)
        if chunk.get("done"):
            print(f"\n\nDone. Tokens: {chunk.get('eval_count', '?')}")
            break
