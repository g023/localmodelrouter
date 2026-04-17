#!/usr/bin/env python3
"""Basic chat example using native /api/chat endpoint."""
import json
import requests

BASE = "http://localhost:11434"
MODEL = "qwen3-tiny"

# Non-streaming chat
resp = requests.post(f"{BASE}/api/chat", json={
    "model": MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Be concise."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    "stream": False,
    "options": {"num_predict": 50},
})
data = resp.json()
print(f"Response: {data['message']['content']}")
print(f"Tokens: prompt={data.get('prompt_eval_count', '?')}, completion={data.get('eval_count', '?')}")
print(f"Duration: {data.get('total_duration', 0) / 1e9:.2f}s")
