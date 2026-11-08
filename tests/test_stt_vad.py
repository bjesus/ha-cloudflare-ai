"""Whisper large-v3-turbo requests turn on vad_filter.

Without it, near-silent audio makes Whisper invent subtitle credits
(for example "詞・作曲・編曲 李宗盛"), which the agent then answers as if
the user had said it. With vad_filter the model returns an empty string.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from .test_stt import _import_stt


def _metadata():
    from homeassistant.components.stt import (
        AudioBitRates,
        AudioChannels,
        AudioCodecs,
        AudioFormats,
        AudioSampleRates,
        SpeechMetadata,
    )

    return SpeechMetadata(
        language="zh",
        format=AudioFormats.WAV,
        codec=AudioCodecs.PCM,
        bit_rate=AudioBitRates.BITRATE_16,
        sample_rate=AudioSampleRates.SAMPLERATE_16000,
        channel=AudioChannels.CHANNEL_MONO,
    )


async def _sent_input(model: str, language: str = "zh") -> dict:
    CloudflareSTTEntity, _ = _import_stt()
    client = SimpleNamespace(run_model=AsyncMock(return_value={"text": "ok"}))
    fake_self = SimpleNamespace(
        _model=model,
        _ensure_wav_header=CloudflareSTTEntity._ensure_wav_header,
        _extract_text=lambda result: result["text"],
    )
    await CloudflareSTTEntity._transcribe_base64(
        fake_self, client, b"audio", _metadata(), language
    )
    return client.run_model.call_args.args[1]


async def test_turbo_request_enables_vad_filter() -> None:
    sent = await _sent_input("@cf/openai/whisper-large-v3-turbo")
    assert sent.get("vad_filter") is True


async def test_turbo_request_enables_vad_filter_for_english() -> None:
    sent = await _sent_input("@cf/openai/whisper-large-v3-turbo", "en")
    assert sent.get("vad_filter") is True


async def test_legacy_whisper_request_has_no_vad_filter() -> None:
    sent = await _sent_input("@cf/openai/whisper")
    assert "vad_filter" not in sent
