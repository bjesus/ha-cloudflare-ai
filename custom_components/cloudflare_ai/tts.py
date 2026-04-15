"""Text-to-Speech entity for Cloudflare Workers AI."""

from __future__ import annotations

import logging
from typing import Any

from homeassistant.components.tts import (
    TextToSpeechEntity,
    TtsAudioType,
    Voice,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant, callback
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
    CloudflareAIError,
)
from .const import (
    AURA1_VOICES,
    AURA2_VOICES,
    CONF_TTS_MODEL,
    CONF_VOICE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    MELOTTS_LANGUAGES,
    SUBENTRY_TTS,
)
from .entity import CloudflareAIBaseEntity

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model profiles
#
# Each TTS model on Workers AI has a different API contract:
#   - Input field name ("text" vs "prompt")
#   - Whether it accepts a voice/speaker param
#   - Whether it accepts an encoding param for compressed output
#   - What audio format it actually returns
#
# We define profiles for known models and a sensible fallback for unknown ones.
# ---------------------------------------------------------------------------

class _ModelProfile:
    """Describes how to call a specific TTS model and what it returns."""

    def __init__(
        self,
        *,
        text_field: str = "text",
        voice_field: str | None = "speaker",
        supports_encoding: bool = False,
        preferred_encoding: str | None = None,
        default_output_format: str = "mp3",
        languages: list[str] | None = None,
        voices: list[str] | None = None,
    ) -> None:
        self.text_field = text_field
        self.voice_field = voice_field
        self.supports_encoding = supports_encoding
        self.preferred_encoding = preferred_encoding
        self.default_output_format = default_output_format
        self.languages = languages or ["en"]
        self.voices = voices


_AURA2_PROFILE = _ModelProfile(
    text_field="text",
    voice_field="speaker",
    supports_encoding=True,
    preferred_encoding="mp3",
    default_output_format="mp3",
    languages=["en"],
    voices=AURA2_VOICES,
)

_AURA2_ES_PROFILE = _ModelProfile(
    text_field="text",
    voice_field="speaker",
    supports_encoding=True,
    preferred_encoding="mp3",
    default_output_format="mp3",
    languages=["es"],
    voices=AURA2_VOICES,
)

_AURA1_PROFILE = _ModelProfile(
    text_field="text",
    voice_field="speaker",
    supports_encoding=True,
    preferred_encoding="mp3",
    default_output_format="mp3",
    languages=["en"],
    voices=AURA1_VOICES,
)

_MELOTTS_PROFILE = _ModelProfile(
    text_field="prompt",
    voice_field=None,
    supports_encoding=False,
    default_output_format="wav",  # MeloTTS always returns base64 WAV
    languages=list(MELOTTS_LANGUAGES.keys()),
    voices=None,
)

# Map known model IDs to profiles
_KNOWN_PROFILES: dict[str, _ModelProfile] = {
    "@cf/deepgram/aura-2-en": _AURA2_PROFILE,
    "@cf/deepgram/aura-2-es": _AURA2_ES_PROFILE,
    "@cf/deepgram/aura-1": _AURA1_PROFILE,
    "@cf/myshell-ai/melotts": _MELOTTS_PROFILE,
}

# Fallback for unknown models: assume Deepgram-style "text" input with MP3
# encoding support. This is the most common pattern on Workers AI.
_DEFAULT_PROFILE = _ModelProfile(
    text_field="text",
    voice_field="speaker",
    supports_encoding=True,
    preferred_encoding="mp3",
    default_output_format="mp3",
)


def _get_profile(model: str) -> _ModelProfile:
    """Get the profile for a model, falling back to defaults for unknown models."""
    if model in _KNOWN_PROFILES:
        return _KNOWN_PROFILES[model]

    # Heuristic matching for partial model names (e.g. future aura-2-fr)
    model_lower = model.lower()
    if "melotts" in model_lower or "melo" in model_lower:
        return _MELOTTS_PROFILE
    if "aura-2" in model_lower:
        return _AURA2_PROFILE
    if "aura-1" in model_lower or "aura" in model_lower:
        return _AURA1_PROFILE

    _LOGGER.info(
        "Unknown TTS model %s, using default profile (text field, MP3 encoding)",
        model,
    )
    return _DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up TTS entities."""
    entities = []
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type == SUBENTRY_TTS:
            entities.append(
                CloudflareTTSEntity(config_entry, subentry)
            )
    async_add_entities(entities)


class CloudflareTTSEntity(TextToSpeechEntity, CloudflareAIBaseEntity):
    """Cloudflare Workers AI TTS entity."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the TTS entity."""
        super().__init__(config_entry, subentry, CONF_TTS_MODEL)
        self._model = subentry.data.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL)
        self._voice = subentry.data.get(CONF_VOICE, DEFAULT_TTS_VOICE)
        self._profile = _get_profile(self._model)

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._profile.languages[0]

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return self._profile.languages

    @property
    def supported_options(self) -> list[str]:
        """Return list of supported options."""
        return ["voice"] if self._profile.voice_field else []

    @property
    def default_options(self) -> dict[str, Any]:
        """Return default options."""
        if self._profile.voice_field:
            return {"voice": self._voice}
        return {}

    @callback
    def async_get_supported_voices(self, language: str) -> list[Voice] | None:
        """Return supported voices for a language."""
        if self._profile.voices is None:
            return None
        return [
            Voice(voice_id=v, name=v.capitalize())
            for v in self._profile.voices
        ]

    async def async_get_tts_audio(
        self,
        message: str,
        language: str,
        options: dict[str, Any],
    ) -> TtsAudioType:
        """Generate TTS audio from text."""
        client: CloudflareAIClient = self._config_entry.runtime_data
        profile = self._profile
        voice = options.get("voice", self._voice)

        # Build input using the model's profile
        input_data: dict[str, Any] = {
            profile.text_field: message,
        }

        # Add voice/speaker if the model supports it
        if profile.voice_field and voice:
            input_data[profile.voice_field] = voice

        # Add language hint for models that use it (MeloTTS uses "lang")
        if profile.text_field == "prompt":
            # MeloTTS-style: uses "lang" param
            input_data["lang"] = language or "en"

        # Request compressed encoding when the model supports it
        if profile.supports_encoding and profile.preferred_encoding:
            input_data["encoding"] = profile.preferred_encoding

        try:
            audio_bytes = await client.run_model_binary(
                self._model, input_data, timeout=30.0
            )
        except CloudflareAIAuthError as err:
            _LOGGER.error("TTS auth error: %s", err)
            self._config_entry.async_start_reauth(self.hass)
            return (None, None)
        except CloudflareAIError as err:
            _LOGGER.error("TTS error with model %s: %s", self._model, err)
            return (None, None)

        # Detect actual format from the audio bytes if possible
        output_format = self._detect_audio_format(audio_bytes, profile)
        return (output_format, audio_bytes)

    @staticmethod
    def _detect_audio_format(data: bytes, profile: _ModelProfile) -> str:
        """Detect audio format from the first bytes, falling back to profile default."""
        if len(data) < 4:
            return profile.default_output_format

        # RIFF header = WAV
        if data[:4] == b"RIFF":
            return "wav"
        # OGG container
        if data[:4] == b"OggS":
            return "ogg"
        # FLAC
        if data[:4] == b"fLaC":
            return "flac"
        # MP3 sync bytes (0xFF 0xFB, 0xFF 0xF3, 0xFF 0xF2, or ID3 tag)
        if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            return "mp3"

        return profile.default_output_format
