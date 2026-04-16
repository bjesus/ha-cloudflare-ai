"""Tests for the Cloudflare Workers AI TTS entity."""

from __future__ import annotations

from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant

from .conftest import make_wav_audio


def _import_tts():
    """Lazy import to avoid triggering HA component import chain at collection time."""
    from custom_components.cloudflare_ai.tts import (
        CloudflareTTSEntity,
        _get_profile,
    )

    return CloudflareTTSEntity, _get_profile


class TestModelProfiles:
    """Test TTS model profile selection."""

    def test_aura2_en_profile(self) -> None:
        _, _get_profile = _import_tts()
        profile = _get_profile("@cf/deepgram/aura-2-en")
        assert profile.text_field == "text"
        assert profile.voice_field == "speaker"
        assert profile.supports_encoding is True
        assert profile.preferred_encoding == "mp3"
        assert profile.default_output_format == "mp3"
        assert "en" in profile.languages

    def test_melotts_profile(self) -> None:
        _, _get_profile = _import_tts()
        profile = _get_profile("@cf/myshell-ai/melotts")
        assert profile.text_field == "prompt"
        assert profile.voice_field is None
        assert profile.supports_encoding is False
        assert profile.default_output_format == "wav"
        assert "en" in profile.languages
        assert "es" in profile.languages

    def test_unknown_model_default_profile(self) -> None:
        _, _get_profile = _import_tts()
        profile = _get_profile("@cf/some/new-tts-model")
        assert profile.text_field == "text"
        assert profile.supports_encoding is True
        assert profile.preferred_encoding == "mp3"

    def test_partial_aura2_match(self) -> None:
        """Future aura-2 variants should match aura2 profile."""
        _, _get_profile = _import_tts()
        profile = _get_profile("@cf/deepgram/aura-2-fr")
        assert profile.supports_encoding is True
        assert profile.preferred_encoding == "mp3"


class TestFormatDetection:
    """Test audio format detection from bytes."""

    def test_detect_wav(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        wav = make_wav_audio()
        assert (
            CloudflareTTSEntity._detect_audio_format(wav, _get_profile("@cf/test"))
            == "wav"
        )

    def test_detect_mp3_sync_bytes(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        mp3 = bytes([0xFF, 0xFB]) + b"\x00" * 50
        assert (
            CloudflareTTSEntity._detect_audio_format(mp3, _get_profile("@cf/test"))
            == "mp3"
        )

    def test_detect_mp3_id3(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        mp3 = b"ID3" + b"\x00" * 50
        assert (
            CloudflareTTSEntity._detect_audio_format(mp3, _get_profile("@cf/test"))
            == "mp3"
        )

    def test_detect_ogg(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        ogg = b"OggS" + b"\x00" * 50
        assert (
            CloudflareTTSEntity._detect_audio_format(ogg, _get_profile("@cf/test"))
            == "ogg"
        )

    def test_detect_flac(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        flac = b"fLaC" + b"\x00" * 50
        assert (
            CloudflareTTSEntity._detect_audio_format(flac, _get_profile("@cf/test"))
            == "flac"
        )

    def test_fallback_to_profile_default(self) -> None:
        CloudflareTTSEntity, _get_profile = _import_tts()
        unknown = b"\x00\x01\x02\x03" + b"\x00" * 50
        profile = _get_profile("@cf/deepgram/aura-2-en")
        assert CloudflareTTSEntity._detect_audio_format(unknown, profile) == "mp3"


class TestTTSEntity:
    """Test TTS entity behavior."""

    async def test_tts_aura2_mp3(
        self,
        hass: HomeAssistant,
        mock_config_entry,
        mock_validate_credentials,
        mock_run_model_binary: AsyncMock,
        setup_ha_components,
    ) -> None:
        """Test Aura-2 TTS returns MP3."""
        mp3_audio = bytes([0xFF, 0xFB]) + b"\x00" * 100
        mock_run_model_binary.return_value = mp3_audio

        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

        state = hass.states.get("tts.cloudflare_ai_tts")
        assert state is not None

    async def test_tts_melotts_wav(
        self,
        hass: HomeAssistant,
        mock_config_entry,
        mock_validate_credentials,
        mock_run_model_binary: AsyncMock,
        setup_ha_components,
    ) -> None:
        """Test MeloTTS returns WAV."""
        CloudflareTTSEntity, _get_profile = _import_tts()
        wav_audio = make_wav_audio()
        mock_run_model_binary.return_value = wav_audio
        profile = _get_profile("@cf/myshell-ai/melotts")
        fmt = CloudflareTTSEntity._detect_audio_format(wav_audio, profile)
        assert fmt == "wav"
