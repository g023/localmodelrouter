#!/usr/bin/env python3
"""OpenAI SDK compatible example using /v1/chat/completions."""
import json
import requests

BASE = "http://localhost:11434"
MODEL = "g023/Qwen3-1.77B-g023-GGUF:Q8_0"

# Non-streaming - works with any OpenAI SDK client
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": "Write a Python hello world function."},
    ],
    "max_tokens": 100,
    "temperature": 0.7,
})
data = resp.json()
content = data["choices"][0]["message"]["content"]
print(f"Response:\n{content}")
print(f"\nUsage: {data.get('usage', {})}")

# Streaming
print("\n--- Streaming ---")
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 30,
    "stream": True,
}, stream=True)

for line in resp.iter_lines():
    if line:
        text = line.decode()
        if text.startswith("data: ") and text != "data: [DONE]":
            chunk = json.loads(text[6:])
            delta = chunk["choices"][0].get("delta", {})
            if "content" in delta:
                print(delta["content"], end="", flush=True)
print()
