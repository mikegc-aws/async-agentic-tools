# Voice interface (Nova 2 Sonic)

Experimental voice chat for Strands Async Tools using **Amazon Nova 2 Sonic**. The agent can keep talking and answering follow-up questions while async tools (e.g. a subagent doing web research) run in the background.

## Requirements

- **Repo root**: Run from the repository root so `strands_async_tools` and `strands_tools` are on the path. The agent’s workspace (files it can read and where the subagent writes reports) is **`voice/workspace`** — created automatically and scoped to the voice package.
- **AWS**: Credentials with Bedrock access (e.g. `us-east-1`).
- **Hardware**: Microphone and speakers.
- **Model**: Access to `amazon.nova-2-sonic-v1:0`.
- **Tavily**: `TAVILY_API_KEY` for the subagent’s web search/extract tools.

## Run

From the repository root:

```bash
TAVILY_API_KEY=tvly-xxx AWS_REGION=us-east-1 uv run python -m voice.voice
```

## Env vars

| Variable           | Default                               | Description                    |
|--------------------|----------------------------------------|--------------------------------|
| `AWS_REGION`       | `us-east-1`                            | AWS region for Bedrock         |
| `NOVA_SONIC_VOICE` | `tiffany`                              | Nova Sonic voice name          |
| `NOVA_SONIC_MODEL` | `amazon.nova-2-sonic-v1:0`             | Nova Sonic model ID            |
| `SUBAGENT_MODEL`   | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Subagent Bedrock model  |
| `TAVILY_API_KEY`   | (required)                             | Tavily API key                 |
| `LOG_FILE`         | `voice_debug.log`                     | Debug log path (under voice/ when run from repo root) |
| `LOG_LEVEL`        | `WARNING`                              | Console log level              |

## Contents

- **`voice.py`** — Main entry: BidiAgent + Nova Sonic, async result injection, console transcript.
- **`subagent.py`** — Background agent (web search, file I/O, etc.) exposed as the async tool `handle_task`.
- **`echo_cancel.py`** — Echo-cancelled audio I/O using LiveKit WebRTC APM (drop-in for `BidiAudioIO`).

This is experimental, scrappy code for trying out async tool calling with voice.
