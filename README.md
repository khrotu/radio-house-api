# radio-house-api

An OpenAI-compatible local server.

## Quick start

```powershell
pip install -r requirements.txt
python rh_server.py
```

Or with uvicorn:

```powershell
uvicorn rh_server:app --host 127.0.0.1 --port 8002
```

## Models

The current model list is not finalised for the final release. Expect models to not work, be removed, or be added. `claude-sonnet-5` and `gpt-5.6-terra` are rate-limited to roughly 5 requests per 30–40 minutes, and can only use emulated tool calls.

| ID | Context |
| --- | --- |
| `claude-sonnet-5` | 977k |
| `deepseek-v4-flash-0731` | 1M |
| `deepseek-v4-pro-0813` | 1M |
| `gemma-4-26b-a4b-it` | 256k |
| `gemma-4-31b-it` | 256k |
| `glm-5.2` | 1M |
| `glm-5.3-flash` | 1M |
| `gpt-5.6-terra` | 1025k |
| `gpt-oss-120b` | 128k |
| `inkling-small` | 524k |
| `minimax-m3` | 1M |
| `muse-glimmer-30b` | 131k |
| `nemotron-3.5-lightning` | 262k |
| `qwen3.8-27b` | 262k |
| `qwen3.8-flash` | 1M |

## Endpoints

- `POST /v1/chat/completions` — chat completions. `stream: true` for SSE. Supports `tools`, `tool_choice`, and `reasoning_effort`.
- `GET  /v1/models` — list models.
- `GET  /v1/models/{id}` — retrieve one model.
- `GET  /health` — liveness plus live credential-pool stats.
- `POST /admin/reload` — reload `credentials.json` from disk without restarting.

`/chat/completions` and `/api/v1/chat/completions` are accepted as aliases of `/v1/chat/completions`.
