"""Constants for the Cloudflare Workers AI integration."""

DOMAIN = "cloudflare_ai"

# Config keys
CONF_ACCOUNT_ID = "account_id"
CONF_API_TOKEN = "api_token"
CONF_USE_AI_GATEWAY = "use_ai_gateway"
CONF_GATEWAY_ID = "gateway_id"
CONF_GATEWAY_API_TOKEN = "gateway_api_token"
CONF_CHAT_MODEL = "chat_model"
CONF_TTS_MODEL = "tts_model"
CONF_STT_MODEL = "stt_model"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_PROMPT = "prompt"
CONF_VOICE = "voice"
CONF_ENABLE_THINKING = "enable_thinking"

# API URLs
CF_API_BASE = "https://api.cloudflare.com/client/v4"
CF_AI_GATEWAY_BASE = "https://gateway.ai.cloudflare.com/v1"

# Defaults
DEFAULT_CHAT_MODEL = "@cf/moonshotai/kimi-k2.5"
DEFAULT_TTS_MODEL = "@cf/deepgram/aura-2-en"
DEFAULT_STT_MODEL = "@cf/openai/whisper-large-v3-turbo"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TTS_VOICE = "luna"
DEFAULT_ENABLE_THINKING = False
DEFAULT_PROMPT = """You are a helpful voice assistant for Home Assistant.
Answer in plain text. Be brief and concise."""

# Retry settings
MAX_RETRIES = 2
RETRY_BACKOFF_BASE = 1.0  # seconds

# Tool calling
MAX_TOOL_ITERATIONS = 10

# Subentry types
SUBENTRY_CONVERSATION = "conversation"
SUBENTRY_TTS = "tts"
SUBENTRY_STT = "stt"

# Known models by task type
CHAT_MODELS = [
    "@cf/moonshotai/kimi-k2.5",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/qwen/qwq-32b",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/google/gemma-3-12b-it",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/zai-org/glm-4.7-flash",
    "@cf/meta/llama-3.1-70b-instruct",
    "@cf/meta/llama-3.1-8b-instruct-fast",
    "@cf/deepseek/deepseek-r1-distill-qwen-32b",
]

# Models known to support function calling
FUNCTION_CALLING_MODELS = [
    "@cf/moonshotai/kimi-k2.5",
    "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
    "@cf/meta/llama-4-scout-17b-16e-instruct",
    "@cf/qwen/qwen3-30b-a3b-fp8",
    "@cf/mistralai/mistral-small-3.1-24b-instruct",
    "@cf/openai/gpt-oss-120b",
    "@cf/openai/gpt-oss-20b",
    "@cf/nvidia/nemotron-3-120b-a12b",
    "@cf/zai-org/glm-4.7-flash",
]

TTS_MODELS = [
    "@cf/deepgram/aura-2-en",
    "@cf/deepgram/aura-2-es",
    "@cf/myshell-ai/melotts",
    "@cf/deepgram/aura-1",
]

STT_MODELS = [
    "@cf/deepgram/nova-3",
    "@cf/openai/whisper-large-v3-turbo",
    "@cf/openai/whisper",
    "@cf/openai/whisper-tiny-en",
]

# Aura-2 voices
AURA2_VOICES = [
    "amalthea", "andromeda", "apollo", "arcas", "aries", "asteria", "athena",
    "atlas", "aurora", "callista", "cora", "cordelia", "delia", "draco",
    "electra", "harmonia", "helena", "hera", "hermes", "hyperion", "iris",
    "janus", "juno", "jupiter", "luna", "mars", "minerva", "neptune",
    "odysseus", "ophelia", "orion", "orpheus", "pandora", "phoebe", "pluto",
    "saturn", "thalia", "theia", "vesta", "zeus",
]

# Aura-1 voices (subset)
AURA1_VOICES = [
    "asteria", "luna", "stella", "athena", "hera", "orion", "arcas",
    "perseus", "angus", "orpheus", "helios", "zeus",
]

# MeloTTS supported languages
MELOTTS_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
}

# Nova-3 supported languages
NOVA3_LANGUAGES = {
    "en": "English",
    "en-US": "English (US)",
    "en-AU": "English (AU)",
    "en-GB": "English (GB)",
    "en-IN": "English (IN)",
    "en-NZ": "English (NZ)",
    "es": "Spanish",
    "es-419": "Spanish (Latin America)",
    "fr": "French",
    "fr-CA": "French (Canada)",
    "de": "German",
    "de-CH": "German (Switzerland)",
    "hi": "Hindi",
    "ru": "Russian",
    "pt": "Portuguese",
    "pt-BR": "Portuguese (Brazil)",
    "pt-PT": "Portuguese (Portugal)",
    "ja": "Japanese",
    "it": "Italian",
    "nl": "Dutch",
    "multi": "Multilingual (auto-detect)",
}

# Whisper supports many languages - subset of most common
WHISPER_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pl": "Polish",
    "tr": "Turkish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "he": "Hebrew",
    "uk": "Ukrainian",
    "cs": "Czech",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "th": "Thai",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "ms": "Malay",
}
