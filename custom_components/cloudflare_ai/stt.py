"""Speech-to-Text entity for Cloudflare Workers AI."""

from __future__ import annotations

import base64
import io
import logging
import wave
from collections.abc import AsyncIterable
from typing import Any

from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
    SpeechToTextEntity,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
    CloudflareAIError,
)
from .const import (
    CONF_STT_MODEL,
    DEFAULT_STT_MODEL,
    DOMAIN,
    NOVA3_LANGUAGES,
    SUBENTRY_STT,
    WHISPER_LANGUAGES,
)
from .entity import CloudflareAIBaseEntity

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model profiles
#
# STT models on Workers AI differ in two key ways:
#   1. Input method: base64 JSON body (Whisper) vs raw audio binary (Nova-3)
#   2. Response structure: flat {"text": ...} (Whisper) vs nested Deepgram format
#
# We define profiles for known models and a sensible fallback.
# ---------------------------------------------------------------------------

class _STTModelProfile:
    """Describes how to send audio to a specific STT model and parse its response."""

    def __init__(
        self,
        *,
        input_mode: str = "base64",  # "base64" (JSON body) or "raw" (binary body)
        response_mode: str = "flat",  # "flat" ({"text": ...}) or "deepgram" (nested)
        languages: list[str] | None = None,
    ) -> None:
        self.input_mode = input_mode
        self.response_mode = response_mode
        self.languages = languages or list(WHISPER_LANGUAGES.keys())


_WHISPER_PROFILE = _STTModelProfile(
    input_mode="base64",
    response_mode="flat",
    languages=list(WHISPER_LANGUAGES.keys()),
)

_NOVA3_PROFILE = _STTModelProfile(
    input_mode="raw",
    response_mode="deepgram",
    languages=list(NOVA3_LANGUAGES.keys()),
)

_KNOWN_PROFILES: dict[str, _STTModelProfile] = {
    "@cf/openai/whisper-large-v3-turbo": _WHISPER_PROFILE,
    "@cf/openai/whisper": _WHISPER_PROFILE,
    "@cf/openai/whisper-tiny-en": _WHISPER_PROFILE,
    "@cf/openai/whisper-sherpa": _WHISPER_PROFILE,
    "@cf/deepgram/nova-3": _NOVA3_PROFILE,
}

# Fallback: base64 JSON input with flat response is the most common pattern
# on Workers AI for ASR models.
_DEFAULT_PROFILE = _STTModelProfile(
    input_mode="base64",
    response_mode="flat",
)


def _get_profile(model: str) -> _STTModelProfile:
    """Get the profile for a model, falling back to defaults for unknown models."""
    if model in _KNOWN_PROFILES:
        return _KNOWN_PROFILES[model]

    model_lower = model.lower()
    if "whisper" in model_lower:
        return _WHISPER_PROFILE
    if "nova" in model_lower or "deepgram" in model_lower:
        return _NOVA3_PROFILE

    _LOGGER.info(
        "Unknown STT model %s, using default profile (base64 input, flat response)",
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
    """Set up STT entities."""
    entities = []
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type == SUBENTRY_STT:
            entities.append(
                CloudflareSTTEntity(config_entry, subentry)
            )
    async_add_entities(entities)


class CloudflareSTTEntity(SpeechToTextEntity, CloudflareAIBaseEntity):
    """Cloudflare Workers AI STT entity."""

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the STT entity."""
        super().__init__(config_entry, subentry, CONF_STT_MODEL)
        self._model = subentry.data.get(CONF_STT_MODEL, DEFAULT_STT_MODEL)
        self._profile = _get_profile(self._model)

    @property
    def supported_languages(self) -> list[str]:
        """Return supported languages."""
        return self._profile.languages

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return supported audio formats."""
        return [AudioFormats.WAV, AudioFormats.OGG]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return supported audio codecs."""
        return [AudioCodecs.PCM, AudioCodecs.OPUS]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return supported bit rates."""
        return [
            AudioBitRates.BITRATE_8,
            AudioBitRates.BITRATE_16,
            AudioBitRates.BITRATE_24,
            AudioBitRates.BITRATE_32,
        ]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return supported sample rates."""
        return [
            AudioSampleRates.SAMPLERATE_8000,
            AudioSampleRates.SAMPLERATE_16000,
            AudioSampleRates.SAMPLERATE_44100,
            AudioSampleRates.SAMPLERATE_48000,
        ]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return supported channels."""
        return [AudioChannels.CHANNEL_MONO, AudioChannels.CHANNEL_STEREO]

    async def async_process_audio_stream(
        self,
        metadata: SpeechMetadata,
        stream: AsyncIterable[bytes],
    ) -> SpeechResult:
        """Process an audio stream to text."""
        client: CloudflareAIClient = self._config_entry.runtime_data

        # Collect all audio bytes
        audio_chunks: list[bytes] = []
        async for chunk in stream:
            audio_chunks.append(chunk)
        audio_data = b"".join(audio_chunks)

        if not audio_data:
            return SpeechResult(text="", result=SpeechResultState.ERROR)

        language = metadata.language or "en"

        try:
            if self._profile.input_mode == "raw":
                text = await self._transcribe_raw(
                    client, audio_data, metadata, language
                )
            else:
                text = await self._transcribe_base64(
                    client, audio_data, metadata, language
                )
        except CloudflareAIAuthError as err:
            _LOGGER.error("STT auth error: %s", err)
            self._config_entry.async_start_reauth(self.hass)
            return SpeechResult(text="", result=SpeechResultState.ERROR)
        except CloudflareAIError as err:
            _LOGGER.error("STT error with model %s: %s", self._model, err)
            return SpeechResult(text="", result=SpeechResultState.ERROR)

        return SpeechResult(
            text=text or "",
            result=SpeechResultState.SUCCESS if text else SpeechResultState.ERROR,
        )

    async def _transcribe_raw(
        self,
        client: CloudflareAIClient,
        audio_data: bytes,
        metadata: SpeechMetadata,
        language: str,
    ) -> str | None:
        """Transcribe by sending raw audio bytes (Deepgram Nova-style)."""
        content_type = self._content_type_from_metadata(metadata)

        # Ensure raw PCM is wrapped in a WAV container
        if metadata.format == AudioFormats.WAV and metadata.codec == AudioCodecs.PCM:
            audio_data = self._ensure_wav_header(audio_data, metadata)

        result = await client.run_model(
            self._model,
            {},
            raw_audio=audio_data,
            audio_content_type=content_type,
            timeout=60.0,
        )

        return self._extract_text(result)

    async def _transcribe_base64(
        self,
        client: CloudflareAIClient,
        audio_data: bytes,
        metadata: SpeechMetadata,
        language: str,
    ) -> str | None:
        """Transcribe by sending base64-encoded audio in JSON (Whisper-style)."""
        # Ensure raw PCM is wrapped in a WAV container
        if metadata.format == AudioFormats.WAV and metadata.codec == AudioCodecs.PCM:
            audio_data = self._ensure_wav_header(audio_data, metadata)

        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        input_data: dict[str, Any] = {
            "audio": audio_b64,
            "language": language,
        }

        result = await client.run_model(
            self._model, input_data, timeout=60.0
        )

        return self._extract_text(result)

    def _extract_text(self, result: Any) -> str | None:
        """Extract transcribed text from the model response.

        Handles both flat Whisper responses and nested Deepgram responses.
        """
        if not isinstance(result, dict):
            return None

        # Flat format: {"text": "transcription"}
        if "text" in result:
            return result["text"]

        # Deepgram format: {"results": {"channels": [{"alternatives": [{"transcript": ...}]}]}}
        results = result.get("results", {})
        channels = results.get("channels", [])
        if channels:
            alternatives = channels[0].get("alternatives", [])
            if alternatives:
                return alternatives[0].get("transcript", "")

        # Last resort: look for any string value that looks like a transcription
        for key in ("transcript", "transcription", "output"):
            if key in result and isinstance(result[key], str):
                return result[key]

        return None

    @staticmethod
    def _content_type_from_metadata(metadata: SpeechMetadata) -> str:
        """Determine the Content-Type header from audio metadata."""
        if metadata.format == AudioFormats.OGG:
            return "audio/ogg"
        return "audio/wav"

    @staticmethod
    def _ensure_wav_header(
        audio_data: bytes,
        metadata: SpeechMetadata,
    ) -> bytes:
        """Ensure audio data has a WAV header, wrapping raw PCM if needed."""
        if audio_data[:4] == b"RIFF":
            return audio_data

        sample_rate = metadata.sample_rate or 16000
        channels = 1 if metadata.channel == AudioChannels.CHANNEL_MONO else 2
        sample_width = (metadata.bit_rate or 16) // 8

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        return buf.getvalue()
