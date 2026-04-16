"""Tests for the Cloudflare Workers AI STT entity."""

from __future__ import annotations

from .conftest import make_wav_audio


def _import_stt():
    """Lazy import to avoid triggering HA component import chain at collection time."""
    from custom_components.cloudflare_ai.stt import (
        CloudflareSTTEntity,
        _get_profile,
    )

    return CloudflareSTTEntity, _get_profile


class TestSTTModelProfiles:
    """Test STT model profile selection."""

    def test_whisper_profile(self) -> None:
        _, _get_profile = _import_stt()
        profile = _get_profile("@cf/openai/whisper-large-v3-turbo")
        assert profile.input_mode == "base64"
        assert profile.response_mode == "flat"

    def test_nova3_profile(self) -> None:
        _, _get_profile = _import_stt()
        profile = _get_profile("@cf/deepgram/nova-3")
        assert profile.input_mode == "raw"
        assert profile.response_mode == "deepgram"

    def test_unknown_model_defaults_to_base64(self) -> None:
        _, _get_profile = _import_stt()
        profile = _get_profile("@cf/some/new-stt-model")
        assert profile.input_mode == "base64"
        assert profile.response_mode == "flat"

    def test_partial_whisper_match(self) -> None:
        _, _get_profile = _import_stt()
        profile = _get_profile("@cf/openai/whisper-v4-ultra")
        assert profile.input_mode == "base64"

    def test_partial_nova_match(self) -> None:
        _, _get_profile = _import_stt()
        profile = _get_profile("@cf/deepgram/nova-4")
        assert profile.input_mode == "raw"


class TestResponseParsing:
    """Test STT response text extraction."""

    def test_flat_response(self) -> None:
        """Test Whisper-style flat response."""
        CloudflareSTTEntity, _ = _import_stt()
        entity = object.__new__(CloudflareSTTEntity)
        assert entity._extract_text({"text": "Hello world"}) == "Hello world"

    def test_deepgram_response(self) -> None:
        """Test Deepgram-style nested response."""
        CloudflareSTTEntity, _ = _import_stt()
        result = {
            "results": {
                "channels": [{"alternatives": [{"transcript": "Hello Deepgram"}]}]
            }
        }
        entity = object.__new__(CloudflareSTTEntity)
        assert entity._extract_text(result) == "Hello Deepgram"

    def test_fallback_transcript_key(self) -> None:
        """Test fallback to common key names."""
        CloudflareSTTEntity, _ = _import_stt()
        entity = object.__new__(CloudflareSTTEntity)
        assert entity._extract_text({"transcript": "Fallback text"}) == "Fallback text"

    def test_empty_response(self) -> None:
        CloudflareSTTEntity, _ = _import_stt()
        entity = object.__new__(CloudflareSTTEntity)
        assert entity._extract_text({}) is None

    def test_non_dict_response(self) -> None:
        CloudflareSTTEntity, _ = _import_stt()
        entity = object.__new__(CloudflareSTTEntity)
        assert entity._extract_text("not a dict") is None
        assert entity._extract_text(None) is None


class TestWavHeader:
    """Test WAV header handling."""

    def test_already_has_header(self) -> None:
        """Audio with RIFF header should pass through unchanged."""
        from homeassistant.components.stt import (
            AudioBitRates,
            AudioChannels,
            AudioCodecs,
            AudioFormats,
            AudioSampleRates,
            SpeechMetadata,
        )

        CloudflareSTTEntity, _ = _import_stt()
        wav = make_wav_audio()
        assert wav[:4] == b"RIFF"

        metadata = SpeechMetadata(
            language="en",
            format=AudioFormats.WAV,
            codec=AudioCodecs.PCM,
            bit_rate=AudioBitRates.BITRATE_16,
            sample_rate=AudioSampleRates.SAMPLERATE_16000,
            channel=AudioChannels.CHANNEL_MONO,
        )
        result = CloudflareSTTEntity._ensure_wav_header(wav, metadata)
        assert result == wav

    def test_raw_pcm_gets_header(self) -> None:
        """Raw PCM data should get a WAV header."""
        from homeassistant.components.stt import (
            AudioBitRates,
            AudioChannels,
            AudioCodecs,
            AudioFormats,
            AudioSampleRates,
            SpeechMetadata,
        )

        CloudflareSTTEntity, _ = _import_stt()
        raw_pcm = b"\x00" * 3200
        metadata = SpeechMetadata(
            language="en",
            format=AudioFormats.WAV,
            codec=AudioCodecs.PCM,
            bit_rate=AudioBitRates.BITRATE_16,
            sample_rate=AudioSampleRates.SAMPLERATE_16000,
            channel=AudioChannels.CHANNEL_MONO,
        )
        result = CloudflareSTTEntity._ensure_wav_header(raw_pcm, metadata)
        assert result[:4] == b"RIFF"
        assert len(result) > len(raw_pcm)
