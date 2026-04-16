"""Tests for the Cloudflare Workers AI client wrapper."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from custom_components.cloudflare_ai.client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
)
from custom_components.cloudflare_ai.const import CF_AI_GATEWAY_BASE


def _mock_cf(httpx_client: AsyncMock | None = None) -> MagicMock:
    """Create a mock AsyncCloudflare instance."""
    cf = MagicMock()
    cf.ai = MagicMock()
    cf.ai.run = AsyncMock()
    cf.ai.with_raw_response = MagicMock()
    cf.ai.with_raw_response.run = AsyncMock()
    cf.ai.with_streaming_response = MagicMock()
    cf.ai.models = MagicMock()
    cf.ai.models.list = AsyncMock()
    cf._client = httpx_client or AsyncMock(spec=httpx.AsyncClient)
    return cf


@pytest.fixture
def mock_cf() -> MagicMock:
    """Create a mock AsyncCloudflare."""
    return _mock_cf()


@pytest.fixture
def client_direct(mock_cf: MagicMock) -> CloudflareAIClient:
    """Create a direct API client."""
    return CloudflareAIClient(
        cf=mock_cf,
        account_id="test_account",
        api_token="test_token",
    )


@pytest.fixture
def client_gateway(mock_cf: MagicMock) -> CloudflareAIClient:
    """Create an AI Gateway client."""
    return CloudflareAIClient(
        cf=mock_cf,
        account_id="test_account",
        api_token="test_token",
        gateway_id="my-gateway",
        gateway_api_token="gw_token",
    )


def _make_response(
    status_code: int = 200,
    json_data: dict | None = None,
    content: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    if json_data is not None:
        resp.json.return_value = json_data
        resp.content = json.dumps(json_data).encode()
    else:
        resp.content = content
    resp.text = content.decode("utf-8", errors="replace") if content else ""
    return resp


class TestURLConstruction:
    """Test URL construction for gateway."""

    def test_gateway_url(self, client_gateway: CloudflareAIClient) -> None:
        url = client_gateway._gateway_url("@cf/meta/llama-3.3-70b")
        assert (
            url
            == f"{CF_AI_GATEWAY_BASE}/test_account/my-gateway/workers-ai/@cf/meta/llama-3.3-70b"
        )

    def test_use_gateway_flag(
        self,
        client_direct: CloudflareAIClient,
        client_gateway: CloudflareAIClient,
    ) -> None:
        assert client_direct.use_gateway is False
        assert client_gateway.use_gateway is True


class TestDirectAPI:
    """Test direct API calls via SDK."""

    async def test_run_model_calls_sdk(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """run_model delegates to cf.ai.run for direct API."""
        mock_cf.ai.run.return_value = {"response": "hello"}
        result = await client_direct.run_model(
            "@cf/test/model", {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert result == {"response": "hello"}
        mock_cf.ai.run.assert_called_once()

    async def test_run_model_unwraps_non_dict(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Non-dict SDK results are wrapped."""
        mock_cf.ai.run.return_value = "plain text"
        result = await client_direct.run_model("@cf/test/model", {})
        assert result == {"response": "plain text"}

    async def test_run_model_auth_error(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """SDK AuthenticationError is converted to CloudflareAIAuthError."""
        from cloudflare import AuthenticationError

        mock_cf.ai.run.side_effect = AuthenticationError.__new__(AuthenticationError)
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.run_model("@cf/test/model", {})


class TestGatewayAPI:
    """Test AI Gateway calls via SDK's httpx client."""

    async def test_gateway_json(
        self, client_gateway: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Gateway calls use cf._client.post with gateway URL."""
        mock_cf._client.post = AsyncMock(
            return_value=_make_response(
                json_data={"result": {"response": "hello"}, "success": True}
            )
        )
        result = await client_gateway.run_model("@cf/test/model", {"messages": []})
        assert result == {"response": "hello"}
        # Verify gateway URL was used
        call_url = mock_cf._client.post.call_args[0][0]
        assert "gateway.ai.cloudflare.com" in call_url
        assert "my-gateway" in call_url

    async def test_gateway_auth_headers(
        self, client_gateway: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Gateway calls include cf-aig-authorization header."""
        mock_cf._client.post = AsyncMock(
            return_value=_make_response(json_data={"response": "ok"})
        )
        await client_gateway.run_model("@cf/test/model", {})
        headers = mock_cf._client.post.call_args[1]["headers"]
        assert headers["cf-aig-authorization"] == "Bearer gw_token"
        assert headers["Authorization"] == "Bearer test_token"


class TestBinaryResponse:
    """Test run_model_binary for TTS."""

    async def test_binary_audio_direct(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Direct API TTS returns raw audio bytes."""
        mp3_bytes = bytes([0xFF, 0xFB]) + b"\x00" * 100
        raw_resp = MagicMock()
        raw_resp.headers = {"content-type": "audio/mpeg"}
        raw_resp.read = AsyncMock(return_value=mp3_bytes)
        mock_cf.ai.with_raw_response.run = AsyncMock(return_value=raw_resp)

        result = await client_direct.run_model_binary("@cf/test/tts", {"text": "hi"})
        assert result == mp3_bytes

    async def test_base64_json_direct(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Direct API TTS with base64 JSON response (MeloTTS style)."""
        import base64

        raw_audio = b"RIFF" + b"\x00" * 50
        b64 = base64.b64encode(raw_audio).decode()
        raw_resp = MagicMock()
        raw_resp.headers = {"content-type": "application/json"}
        raw_resp.read = AsyncMock(
            return_value=json.dumps(
                {"result": {"audio": b64}, "success": True}
            ).encode()
        )
        mock_cf.ai.with_raw_response.run = AsyncMock(return_value=raw_resp)

        result = await client_direct.run_model_binary("@cf/test/tts", {"prompt": "hi"})
        assert result == raw_audio

    async def test_binary_gateway(
        self, client_gateway: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        """Gateway TTS returns raw audio bytes."""
        mp3_bytes = bytes([0xFF, 0xFB]) + b"\x00" * 100
        mock_cf._client.post = AsyncMock(
            return_value=_make_response(content=mp3_bytes, content_type="audio/mpeg")
        )
        result = await client_gateway.run_model_binary("@cf/test/tts", {"text": "hi"})
        assert result == mp3_bytes


class TestValidateCredentials:
    """Test credential validation."""

    async def test_validate_success(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        mock_cf.ai.models.list = AsyncMock(return_value=[])
        result = await client_direct.validate_credentials()
        assert result is True

    async def test_validate_auth_error(
        self, client_direct: CloudflareAIClient, mock_cf: MagicMock
    ) -> None:
        from cloudflare import AuthenticationError

        mock_cf.ai.models.list = AsyncMock(
            side_effect=AuthenticationError.__new__(AuthenticationError)
        )
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.validate_credentials()


class TestEnvelopeUnwrap:
    """Test CF API response envelope unwrapping."""

    def test_unwrap_envelope(self) -> None:
        result = CloudflareAIClient._unwrap(
            {"result": {"response": "hello"}, "success": True}
        )
        assert result == {"response": "hello"}

    def test_no_unwrap_without_success(self) -> None:
        data = {"response": "hello", "tool_calls": []}
        result = CloudflareAIClient._unwrap(data)
        assert result == data
