"""Tests for the Cloudflare Workers AI HTTP client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from custom_components.cloudflare_ai.client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
)
from custom_components.cloudflare_ai.const import CF_AI_GATEWAY_BASE, CF_API_BASE


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Create a mock httpx AsyncClient."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def client_direct(mock_httpx_client: AsyncMock) -> CloudflareAIClient:
    """Create a direct API client."""
    return CloudflareAIClient(
        httpx_client=mock_httpx_client,
        account_id="test_account",
        api_token="test_token",
    )


@pytest.fixture
def client_gateway(mock_httpx_client: AsyncMock) -> CloudflareAIClient:
    """Create an AI Gateway client."""
    return CloudflareAIClient(
        httpx_client=mock_httpx_client,
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
    """Test URL construction for direct vs gateway."""

    def test_direct_url(self, client_direct: CloudflareAIClient) -> None:
        url = client_direct._direct_url("@cf/meta/llama-3.3-70b")
        assert (
            url == f"{CF_API_BASE}/accounts/test_account/ai/run/@cf/meta/llama-3.3-70b"
        )

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


class TestEnvelopeUnwrap:
    """Test CF API response envelope unwrapping."""

    async def test_unwrap_result_envelope(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Both direct and gateway responses with result envelope are unwrapped."""
        mock_httpx_client.post.return_value = _make_response(
            json_data={
                "result": {"response": "hello"},
                "success": True,
                "errors": [],
                "messages": [],
            }
        )
        result = await client_direct.run_model("@cf/test/model", {"text": "hi"})
        assert result == {"response": "hello"}

    async def test_no_unwrap_without_success(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Responses without success key are not unwrapped."""
        mock_httpx_client.post.return_value = _make_response(
            json_data={"response": "hello", "tool_calls": []}
        )
        result = await client_direct.run_model("@cf/test/model", {"text": "hi"})
        assert result == {"response": "hello", "tool_calls": []}


class TestErrorHandling:
    """Test HTTP error handling."""

    async def test_auth_error_401(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(status_code=401)
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.run_model("@cf/test/model", {})

    async def test_auth_error_403(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        mock_httpx_client.post.return_value = _make_response(status_code=403)
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.run_model("@cf/test/model", {})

    async def test_no_retry_on_auth_error(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Auth errors should not be retried."""
        mock_httpx_client.post.return_value = _make_response(status_code=401)
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.run_model("@cf/test/model", {})
        # Should only be called once (no retries)
        assert mock_httpx_client.post.call_count == 1

    async def test_retry_on_server_error(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Server errors (5xx) should be retried."""
        mock_httpx_client.post.side_effect = [
            _make_response(status_code=500),
            _make_response(status_code=500),
            _make_response(json_data={"response": "ok"}),
        ]
        with patch("custom_components.cloudflare_ai.client.asyncio.sleep"):
            result = await client_direct.run_model("@cf/test/model", {})
        assert result == {"response": "ok"}
        assert mock_httpx_client.post.call_count == 3


class TestValidateCredentials:
    """Test credential validation."""

    async def test_validate_success(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        mock_httpx_client.get.return_value = _make_response(
            json_data={"result": [], "success": True}
        )
        result = await client_direct.validate_credentials()
        assert result is True

    async def test_validate_auth_error(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        mock_httpx_client.get.return_value = _make_response(status_code=401)
        with pytest.raises(CloudflareAIAuthError):
            await client_direct.validate_credentials()


class TestBinaryResponse:
    """Test run_model_binary for TTS."""

    async def test_binary_audio_response(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Direct binary audio response."""
        audio_bytes = b"RIFF" + b"\x00" * 100
        mock_httpx_client.post.return_value = _make_response(
            content=audio_bytes,
            content_type="audio/mpeg",
        )
        result = await client_direct.run_model_binary("@cf/test/tts", {"text": "hi"})
        assert result == audio_bytes

    async def test_base64_json_response(
        self,
        client_direct: CloudflareAIClient,
        mock_httpx_client: AsyncMock,
    ) -> None:
        """Base64-encoded audio in JSON response (MeloTTS style)."""
        import base64

        raw_audio = b"RIFF" + b"\x00" * 50
        b64_audio = base64.b64encode(raw_audio).decode()
        mock_httpx_client.post.return_value = _make_response(
            json_data={
                "result": {"audio": b64_audio},
                "success": True,
            },
            content_type="application/json",
        )
        result = await client_direct.run_model_binary("@cf/test/tts", {"text": "hi"})
        assert result == raw_audio
