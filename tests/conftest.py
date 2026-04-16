"""Shared fixtures for Cloudflare Workers AI tests."""

from __future__ import annotations

import base64
import io
import pathlib
import wave
from collections.abc import Generator
from unittest.mock import AsyncMock, patch

import pytest
from homeassistant import loader
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import (
    MockConfigEntry,
)

from custom_components.cloudflare_ai.const import (
    CONF_ACCOUNT_ID,
    CONF_API_TOKEN,
    CONF_CHAT_MODEL,
    CONF_ENABLE_THINKING,
    CONF_IMAGE_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TTS_MODEL,
    CONF_USE_AI_GATEWAY,
    CONF_VOICE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DOMAIN,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
)

TEST_ACCOUNT_ID = "test_account_id_123"
TEST_API_TOKEN = "test_api_token_456"
TEST_GATEWAY_ID = "test-gateway"
TEST_GATEWAY_TOKEN = "test_gw_token_789"

# Path to the custom component
_COMPONENT_DIR = (
    pathlib.Path(__file__).parent.parent / "custom_components" / "cloudflare_ai"
)


@pytest.fixture(autouse=True)
async def auto_enable_custom_integrations(
    hass: HomeAssistant,
) -> None:
    """Enable custom integrations for all tests.

    The test hass pre-blocks custom components with an empty cache.
    We pop it and force a re-scan, then pre-load the integration
    modules into HA's module caches.
    """
    # Remove the empty custom components cache
    hass.data.pop(loader.DATA_CUSTOM_COMPONENTS, None)

    # Re-scan: this finds our integration via the custom_components namespace
    custom = await loader.async_get_custom_components(hass)

    # Cache in the integrations lookup dict
    cache = hass.data.get(loader.DATA_INTEGRATIONS)
    if cache is not None:
        for domain, integration in custom.items():
            cache[domain] = integration

    # Pre-import the integration modules into HA's component cache
    # so that _load_platform can find config_flow, conversation, etc.
    import importlib

    comp_cache = hass.data.setdefault(loader.DATA_COMPONENTS, {})
    for domain, integration in custom.items():
        # Import the main module
        mod = importlib.import_module(integration.pkg_path)
        comp_cache[domain] = mod
        # Import known platform modules
        for platform in ("ai_task", "config_flow", "conversation", "tts", "stt"):
            try:
                pmod = importlib.import_module(f"{integration.pkg_path}.{platform}")
                comp_cache[f"{domain}.{platform}"] = pmod
            except ImportError:
                pass


@pytest.fixture
async def setup_ha_components(hass: HomeAssistant) -> None:
    """Set up core HA components needed by our integration.

    The 'conversation' dependency requires 'homeassistant' to be set up
    with exposed_entities. This fixture ensures both are available.
    """
    from homeassistant.setup import async_setup_component

    await async_setup_component(hass, "homeassistant", {})
    await async_setup_component(hass, "conversation", {})
    await hass.async_block_till_done()


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Create a mock config entry with subentries."""
    entry = MockConfigEntry(
        version=1,
        minor_version=1,
        domain=DOMAIN,
        title="Cloudflare Workers AI",
        data={
            CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
            CONF_API_TOKEN: TEST_API_TOKEN,
            CONF_USE_AI_GATEWAY: False,
        },
        unique_id=TEST_ACCOUNT_ID,
        subentries_data=[
            {
                "subentry_type": SUBENTRY_CONVERSATION,
                "title": "Cloudflare AI Conversation",
                "unique_id": None,
                "data": {
                    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
                    CONF_MAX_TOKENS: 1024,
                    CONF_TEMPERATURE: 0.6,
                    CONF_PROMPT: "You are a helpful assistant.",
                    CONF_LLM_HASS_API: ["assist"],
                },
            },
            {
                "subentry_type": SUBENTRY_AI_TASK,
                "title": "Cloudflare AI Task",
                "unique_id": None,
                "data": {
                    CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
                    CONF_MAX_TOKENS: 1024,
                    CONF_TEMPERATURE: 0.6,
                    CONF_ENABLE_THINKING: DEFAULT_ENABLE_THINKING,
                    CONF_IMAGE_MODEL: DEFAULT_IMAGE_MODEL,
                },
            },
            {
                "subentry_type": SUBENTRY_TTS,
                "title": "Cloudflare AI TTS",
                "unique_id": None,
                "data": {
                    CONF_TTS_MODEL: DEFAULT_TTS_MODEL,
                    CONF_VOICE: "luna",
                },
            },
            {
                "subentry_type": SUBENTRY_STT,
                "title": "Cloudflare AI STT",
                "unique_id": None,
                "data": {
                    CONF_STT_MODEL: DEFAULT_STT_MODEL,
                },
            },
        ],
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_validate_credentials() -> Generator[AsyncMock]:
    """Mock the validate_credentials method."""
    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ) as mock:
        yield mock


@pytest.fixture
def mock_run_model() -> Generator[AsyncMock]:
    """Mock the run_model method."""
    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.run_model",
        new_callable=AsyncMock,
    ) as mock:
        yield mock


@pytest.fixture
def mock_run_model_binary() -> Generator[AsyncMock]:
    """Mock the run_model_binary method."""
    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.run_model_binary",
        new_callable=AsyncMock,
    ) as mock:
        yield mock


@pytest.fixture
def mock_stream_model() -> Generator[AsyncMock]:
    """Mock the stream_model method."""
    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.stream_model",
    ) as mock:
        yield mock


def make_wav_audio(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Generate a small WAV file for testing."""
    num_frames = int(sample_rate * duration_ms / 1000)
    frames = b"\x00" * (num_frames * channels * sample_width)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(frames)
    return buf.getvalue()


# Sample API responses
SAMPLE_CHAT_RESPONSE = {
    "response": "Hello! How can I help you?",
    "usage": {
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "total_tokens": 28,
    },
}

SAMPLE_TOOL_CALL_RESPONSE = {
    "response": None,
    "tool_calls": [
        {
            "name": "GetDateTime",
            "arguments": {},
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 19,
        "total_tokens": 119,
    },
}

SAMPLE_TOOL_RESULT_RESPONSE = {
    "response": "The current date is March 18, 2026.",
    "usage": {
        "prompt_tokens": 150,
        "completion_tokens": 12,
        "total_tokens": 162,
    },
}

SAMPLE_STT_WHISPER_RESPONSE = {
    "text": "Hello, how can I help you with Home Assistant?",
}

SAMPLE_STT_NOVA_RESPONSE = {
    "results": {
        "channels": [{"alternatives": [{"transcript": "Hello, how can I help you?"}]}]
    }
}

SAMPLE_TTS_BASE64_RESPONSE = {
    "audio": base64.b64encode(make_wav_audio()).decode("utf-8"),
}
