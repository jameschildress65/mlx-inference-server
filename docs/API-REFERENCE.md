# MLX Server V3 - API Reference

**Version**: 3.0.0-alpha
**Date**: 2025-12-24
**Base URL**: http://localhost:11440 (main), http://localhost:11441 (admin)

---

## Overview

V3 provides two API interfaces:
- **Main API** (port 11440): OpenAI-compatible completions
- **Admin API** (port 11441): Model management

**OpenAPI Spec**: See `docs/OPENAPI-SPEC.yaml`

---

## Main API Endpoints

### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "version": "3.0.0-alpha"
}
```

**Use**: Monitor server availability

---

### Text Completion

```http
POST /v1/completions
Content-Type: application/json
```

**Request**:
```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "prompt": "What is machine learning?",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 1.0,
  "stream": false
}
```

**Response** (Non-Streaming):
```json
{
  "id": "cmpl-mlx-v3",
  "object": "text_completion",
  "created": 1703289600,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [
    {
      "text": "Machine learning is a branch of artificial intelligence...",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 42,
    "total_tokens": 42
  }
}
```

**Response** (Streaming, `stream: true`):
```
data: {"choices":[{"text":"Machine","index":0,"finish_reason":null}]}

data: {"choices":[{"text":" learning","index":0,"finish_reason":null}]}

data: {"choices":[{"text":" is","index":0,"finish_reason":null}]}

...

data: {"choices":[{"text":".","index":0,"finish_reason":"stop"}]}
```

**Parameters**:

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | HuggingFace model path |
| `prompt` | string | Yes | - | Text prompt |
| `max_tokens` | integer | No | 100 | Max tokens to generate (1-2048) |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 1.0 | Nucleus sampling (0.0-1.0) |
| `stream` | boolean | No | false | Enable SSE streaming |

**Status Codes**:
- `200`: Success
- `503`: No model loaded
- `500`: Worker error

---

### Chat Completion

```http
POST /v1/chat/completions
Content-Type: application/json
```

**Request**:
```json
{
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "max_tokens": 50,
  "temperature": 0.7
}
```

**Response**:
```json
{
  "id": "chatcmpl-mlx-v3",
  "object": "chat.completion",
  "created": 1703289600,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The answer is 4."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 5,
    "total_tokens": 5
  }
}
```

**Parameters**: Same as `/v1/completions`, but uses `messages` array instead of `prompt`

---

### List Models

```http
GET /v1/models
```

**Response**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen2.5-7B-Instruct-4bit",
      "object": "model",
      "created": 0,
      "owned_by": "mlx-server-v3"
    }
  ]
}
```

**Note**: Only shows currently loaded model

---

## Admin API Endpoints

### Admin Health Check

```http
GET /admin/health
```

**Response**:
```json
{
  "status": "healthy",
  "worker_status": "healthy",
  "version": "3.0.0-alpha"
}
```

**Worker Status Values**:
- `"no_worker"`: No model loaded
- `"healthy"`: Worker running and responsive
- `"dead"`: Worker crashed

---

### Server Status

```http
GET /admin/status
```

**Response**:
```json
{
  "status": "running",
  "version": "3.0.0-alpha",
  "ports": {
    "main": 11440,
    "admin": 11441
  },
  "model": {
    "loaded": true,
    "name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "memory_gb": 4.2
  },
  "worker": {
    "healthy": true,
    "status": "healthy"
  },
  "config": {
    "machine_type": "high-memory",
    "total_ram_gb": 64.0,
    "idle_timeout_seconds": 600
  }
}
```

---

### Load Model

```http
POST /admin/load?model_path=<model>
```

**Parameters**:
- `model_path` (query): HuggingFace model path

**Example**:
```bash
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"
```

**Response**:
```json
{
  "status": "success",
  "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "memory_gb": 4.2,
  "load_time": 2.3
}
```

**Status Codes**:
- `200`: Success
- `500`: Load failed

**Notes**:
- Kills existing worker if model already loaded
- Spawns new worker subprocess
- Waits for ready signal (timeout: 120s)

---

### Unload Model

```http
POST /admin/unload
```

**Example**:
```bash
curl -X POST http://localhost:11441/admin/unload
```

**Response**:
```json
{
  "status": "success",
  "model_name": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "memory_freed_gb": 4.2,
  "unload_time": 0.5
}
```

**Status Codes**:
- `200`: Success
- `400`: No model loaded

**Notes**:
- Terminates worker subprocess
- Frees all model memory (0 MB residual)
- Graceful shutdown (SIGTERM → SIGKILL)

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

**Common Errors**:

| Status | Error | Cause |
|--------|-------|-------|
| 400 | "No model loaded" | Trying to unload when no model |
| 422 | "Validation error" | Invalid request format |
| 500 | "Worker error: ..." | Worker crashed or failed |
| 503 | "No model loaded" | Trying to generate without model |

---

## Code Examples

### Python

```python
import requests

# Load model
requests.post(
    "http://localhost:11441/admin/load",
    params={"model_path": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
)

# Generate completion
response = requests.post(
    "http://localhost:11440/v1/completions",
    json={
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "prompt": "What is MLX?",
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json()["choices"][0]["text"])

# Unload model
requests.post("http://localhost:11441/admin/unload")
```

### JavaScript

```javascript
// Load model
await fetch(
  "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit",
  { method: "POST" }
);

// Generate completion
const response = await fetch("http://localhost:11440/v1/completions", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "mlx-community/Qwen2.5-7B-Instruct-4bit",
    prompt: "What is MLX?",
    max_tokens: 100,
    temperature: 0.7
  })
});

const data = await response.json();
console.log(data.choices[0].text);
```

### Streaming (Python)

```python
import requests
import json

response = requests.post(
    "http://localhost:11440/v1/completions",
    json={
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "prompt": "Count to 10:",
        "max_tokens": 50,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            text = data['choices'][0]['text']
            print(text, end='', flush=True)
```

### curl

```bash
# Health check
curl http://localhost:11440/health

# Load model
curl -X POST "http://localhost:11441/admin/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit"

# Generate
curl -X POST http://localhost:11440/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "prompt": "What is MLX?",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Streaming
curl -X POST http://localhost:11440/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "prompt": "Count to 5:",
    "max_tokens": 30,
    "stream": true
  }'

# Unload
curl -X POST http://localhost:11441/admin/unload
```

---

## OpenAI Compatibility

V3 is compatible with OpenAI Python client:

```python
from openai import OpenAI

# Point to V3
client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:11440/v1"
)

# Use OpenAI API
response = client.completions.create(
    model="mlx-community/Qwen2.5-7B-Instruct-4bit",
    prompt="What is MLX?",
    max_tokens=100
)

print(response.choices[0].text)
```

**Supported**:
- ✅ `client.completions.create()`
- ✅ `client.chat.completions.create()`
- ✅ Streaming

**Not Supported**:
- ❌ `client.models.list()` (returns only loaded model)
- ❌ Embeddings
- ❌ Fine-tuning
- ❌ Assistants API

---

## Rate Limits

**Current**: None (no rate limiting)

**Concurrency**: 1 request at a time (serialized by lock)

**Future**: Phase 4 will add:
- Worker pool (concurrent requests)
- Rate limiting per client
- Request quotas

---

## Deprecation Policy

**V3 Alpha**: API may change without notice

**V3 Beta**: 30-day deprecation notice for breaking changes

**V3 Stable**: Semantic versioning (major.minor.patch)

---

**Document Version**: 1.0
**Last Updated**: 2025-12-24
**Maintainer**: MLX Server V3 Team
