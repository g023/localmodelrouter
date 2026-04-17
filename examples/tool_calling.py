#!/usr/bin/env python3
"""Tool calling example using OpenAI /v1/chat/completions."""
import json
import requests

BASE = "http://localhost:11434"
MODEL = "qwen3-tiny"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"},
                },
                "required": ["city"],
            },
        },
    }
]

print("=== Tool Calling ===")
resp = requests.post(f"{BASE}/v1/chat/completions", json={
    "model": MODEL,
    "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
    "tools": tools,
    "tool_choice": "auto",
    "max_tokens": 200,
})

data = resp.json()
msg = data["choices"][0]["message"]
print(f"Role: {msg['role']}")
print(f"Content: {msg.get('content', '(none)')}")
if msg.get("tool_calls"):
    for tc in msg["tool_calls"]:
        print(f"Tool call: {tc['function']['name']}({tc['function']['arguments']})")
else:
    print("(No tool calls made - model may not support tool calling)")
