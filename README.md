# Cloudflare Workers AI for Home Assistant

A Home Assistant custom integration that brings [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) to your smart home.

## What it does

- **Conversation Agent** — Chat with LLMs that can control your Home Assistant devices via the Assist API
- **Text-to-Speech** — Generate speech using Deepgram Aura-2 or MeloTTS
- **Speech-to-Text** — Transcribe audio using Whisper or Deepgram Nova-3
- **AI Tasks** — Generate text, structured data, and images for use in automations
- **Image Generation** — Create images with FLUX, Stable Diffusion, and other models

## Features

- **Any model** — Pick from the Workers AI catalog or type any model ID
- **Tool calling** — The LLM can control lights, check sensors, get the time, and more
- **AI Gateway support** — Route through [Cloudflare AI Gateway](https://developers.cloudflare.com/ai-gateway/) for analytics, caching, and rate limiting
- **Reasoning toggle** — Enable step-by-step thinking for models that support it (Kimi K2.5, QwQ, etc.)
- **Streaming** — Real-time token display for chat responses

## Installation

### HACS (recommended)

1. Open HACS in your Home Assistant
2. Click the three dots menu (top right) → **Custom repositories**
3. Add this repository URL and select **Integration** as the category
4. Click **Add**
5. Find "Cloudflare Workers AI" in the HACS store and click **Install**
6. Restart Home Assistant

### Manual

Copy the `custom_components/cloudflare_ai/` directory to your HA config's `custom_components/` folder and restart.

## Setup

1. Go to **Settings → Devices & Services → Add Integration**
2. Search for "Cloudflare Workers AI"
3. Enter your:
   - **Account ID** — found in the Cloudflare dashboard sidebar
   - **API Token** — create one with "Workers AI Read" permission at [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)
   - (Optional) **AI Gateway** settings if you want to route through a gateway
4. The integration auto-creates a conversation agent, TTS engine, STT engine, and AI task entity
5. You can add more agents or change models via the integration page

## Usage

### Voice Assistant Pipeline

Go to **Settings → Voice assistants** and create a pipeline using:
- **Conversation agent:** Cloudflare AI Conversation
- **Speech-to-text:** Cloudflare AI STT
- **Text-to-speech:** Cloudflare AI TTS

### Chat

Use the **Assist** panel (bottom left) and select your Cloudflare AI agent.

### AI Tasks

Use `ai_task.generate_data` and `ai_task.generate_image` in automations and scripts:

```yaml
# Generate text
action: ai_task.generate_data
data:
  entity_id: ai_task.cloudflare_ai_task
  task_name: notification
  instructions: "Write a funny reminder that the garage door is open"
response_variable: result

# Generate an image
action: ai_task.generate_image
data:
  entity_id: ai_task.cloudflare_ai_task
  task_name: weather art
  instructions: "A watercolor painting of a rainy day in Stockholm"
response_variable: image
```

## Supported Models

Any Workers AI model can be used — the dropdowns show popular options but you can type any model ID.

**Conversation**
- `@cf/moonshotai/kimi-k2.5` (default — fast, tool calling, 256k context)
- `@cf/meta/llama-3.3-70b-instruct-fp8-fast`
- `@cf/meta/llama-4-scout-17b-16e-instruct`
- `@cf/openai/gpt-oss-120b`
- `@cf/qwen/qwen3-30b-a3b-fp8`
- `@cf/mistralai/mistral-small-3.1-24b-instruct`
- [All text-generation models →](https://developers.cloudflare.com/workers-ai/models/?feature=text-generation)

**Text-to-Speech**
- `@cf/deepgram/aura-2-en` (default — MP3, 40 voices)
- `@cf/deepgram/aura-2-es` (Spanish)
- `@cf/myshell-ai/melotts` (multi-lingual)
- [All TTS models →](https://developers.cloudflare.com/workers-ai/models/?feature=text-to-speech)

**Speech-to-Text**
- `@cf/openai/whisper-large-v3-turbo` (default)
- `@cf/deepgram/nova-3`
- [All ASR models →](https://developers.cloudflare.com/workers-ai/models/?feature=automatic-speech-recognition)

**Image Generation**
- `@cf/black-forest-labs/flux-1-schnell` (default — fast, 1024×1024)
- `@cf/black-forest-labs/flux-2-klein-9b`
- `@cf/black-forest-labs/flux-2-dev`
- `@cf/leonardo/lucid-origin`
- `@cf/stabilityai/stable-diffusion-xl-base-1.0`
- [All image models →](https://developers.cloudflare.com/workers-ai/models/?feature=text-to-image)
