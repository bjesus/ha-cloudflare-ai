# Cloudflare Workers AI for Home Assistant

A Home Assistant custom integration that brings Cloudflare Workers AI to your smart home, providing:

- **Conversation Agent** — Chat with LLMs (Llama 3.3, Mistral, Qwen, etc.) that can control your Home Assistant devices via the Assist API
- **Text-to-Speech** — Generate speech using Deepgram Aura-2 (MP3) or MeloTTS (WAV)
- **Speech-to-Text** — Transcribe audio using Whisper or Deepgram Nova-3

## Features

- **Any model** — Pick from the Workers AI catalog or type any model ID
- **AI Gateway support** — Route through Cloudflare AI Gateway for analytics, caching, and rate limiting
- **Tool calling** — The LLM can control lights, check sensors, get the time, etc.
- **Streaming** — Real-time token display for chat responses
- **Compressed audio** — Aura-2 returns MP3 (~9KB) instead of WAV (~220KB)
- **Zero dependencies** — Uses Home Assistant's built-in HTTP client

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
4. The integration creates a conversation agent, TTS engine, and STT engine with sensible defaults
5. You can add more agents or change models via the integration's **Configure** button

## Usage

### Voice Assistant Pipeline

Go to **Settings → Voice assistants** and create a pipeline using:
- **Conversation agent:** Cloudflare AI Conversation
- **Speech-to-text:** Cloudflare AI STT
- **Text-to-speech:** Cloudflare AI TTS

### Chat

Use the **Assist** panel (bottom left) and select your Cloudflare AI agent.

## Supported Models

Any Workers AI model can be used. Known models with tested support:

**Chat (text-generation)**
- `@cf/meta/llama-3.3-70b-instruct-fp8-fast` (default, supports tool calling)
- `@cf/meta/llama-4-scout-17b-16e-instruct`
- `@cf/qwen/qwen3-30b-a3b-fp8`
- `@cf/mistralai/mistral-small-3.1-24b-instruct`
- Any model from [Workers AI text-generation](https://developers.cloudflare.com/workers-ai/models/)

**TTS**
- `@cf/deepgram/aura-2-en` (default, MP3, 40 voices)
- `@cf/deepgram/aura-2-es` (Spanish)
- `@cf/myshell-ai/melotts` (multi-lingual, WAV)

**STT**
- `@cf/openai/whisper-large-v3-turbo` (default)
- `@cf/deepgram/nova-3` (10 languages)
- `@cf/openai/whisper`
