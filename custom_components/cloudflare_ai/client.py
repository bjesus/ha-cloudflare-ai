"""Client wrapper around the official Cloudflare Python SDK."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from cloudflare import APIConnectionError, AsyncCloudflare, AuthenticationError

from .const import CF_AI_GATEWAY_BASE, CF_API_BASE

_LOGGER = logging.getLogger(__name__)


class CloudflareAIError(Exception):
    """Base error for Cloudflare AI."""


class CloudflareAIAuthError(CloudflareAIError):
    """Authentication error."""


class CloudflareAIConnectionError(CloudflareAIError):
    """Connection error."""


class CloudflareAIClient:
    """Client for Cloudflare Workers AI.

    Wraps the official cloudflare SDK for direct API calls, and uses the
    SDK's underlying httpx client for AI Gateway calls (which have a
    different URL pattern).
    """

    def __init__(
        self,
        cf: AsyncCloudflare,
        account_id: str,
        api_token: str,
        gateway_id: str | None = None,
        gateway_api_token: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._cf = cf
        self._account_id = account_id
        self._api_token = api_token
        self._gateway_id = gateway_id
        self._gateway_api_token = gateway_api_token

    @property
    def use_gateway(self) -> bool:
        """Return True if AI Gateway is configured."""
        return self._gateway_id is not None

    def _gateway_url(self, model: str) -> str:
        """Build the AI Gateway URL for a model."""
        return (
            f"{CF_AI_GATEWAY_BASE}/{self._account_id}"
            f"/{self._gateway_id}/workers-ai/{model}"
        )

    def _gateway_headers(self) -> dict[str, str]:
        """Build headers for AI Gateway requests."""
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        if self._gateway_api_token:
            headers["cf-aig-authorization"] = f"Bearer {self._gateway_api_token}"
        return headers

    def _endpoint(self, model: str) -> tuple[str, dict[str, str]]:
        """Return the URL and headers for a model, direct or via AI Gateway.

        The URL is built here rather than through ``cf.ai.run`` because the
        SDK (5.x) percent-encodes the slashes in model names, which the
        Workers AI API rejects with "No route for that URI".
        """
        if self.use_gateway:
            return self._gateway_url(model), self._gateway_headers()
        url = f"{CF_API_BASE}/accounts/{self._account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
            "Content-Type": "application/json",
        }
        return url, headers

    async def run_model(
        self,
        model: str,
        input_data: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Run a model and return the parsed response dict.

        Used for: conversation, STT (Whisper), AI task, image generation.
        """
        try:
            return await self._http_json(model, input_data, timeout)
        except httpx.TransportError as err:
            raise CloudflareAIConnectionError(str(err)) from err
        except CloudflareAIError:
            raise
        except Exception as err:
            raise CloudflareAIError(str(err)) from err

    async def run_model_binary(
        self,
        model: str,
        input_data: dict[str, Any],
        timeout: float = 60.0,
    ) -> bytes:
        """Run a model and return raw binary (for TTS).

        Handles both raw audio responses (Aura-2) and JSON with base64
        audio (MeloTTS).
        """
        try:
            return await self._http_binary(model, input_data, timeout)
        except httpx.TransportError as err:
            raise CloudflareAIConnectionError(str(err)) from err
        except CloudflareAIError:
            raise
        except Exception as err:
            raise CloudflareAIError(str(err)) from err

    async def run_model_raw_audio(
        self,
        model: str,
        audio_data: bytes,
        content_type: str = "audio/wav",
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Send raw audio bytes to an STT model (Nova-3 style).

        Uses the SDK's underlying httpx client since the SDK doesn't
        support raw binary POST with custom Content-Type.
        """
        try:
            url, headers = self._endpoint(model)
            headers["Content-Type"] = content_type
            resp = await self._cf._client.post(
                url, content=audio_data, headers=headers, timeout=timeout
            )
            self._check_http(resp)
            return self._unwrap(resp.json())
        except (CloudflareAIAuthError, CloudflareAIError):
            raise
        except Exception as err:
            raise CloudflareAIError(str(err)) from err

    async def stream_model(
        self,
        model: str,
        input_data: dict[str, Any],
        timeout: float = 120.0,
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream model output as parsed SSE events.

        Yields dicts from the SSE stream. Events are in OpenAI format:
          {"choices": [{"delta": {"content": "text"}}]}
        """
        try:
            async for event in self._http_stream(model, input_data, timeout):
                yield event
        except httpx.TransportError as err:
            raise CloudflareAIConnectionError(str(err)) from err

    async def validate_credentials(self) -> bool:
        """Validate credentials by listing AI models."""
        try:
            await self._cf.ai.models.list(account_id=self._account_id)
        except AuthenticationError as err:
            raise CloudflareAIAuthError(str(err)) from err
        except APIConnectionError as err:
            raise CloudflareAIConnectionError(str(err)) from err
        except Exception as err:
            raise CloudflareAIConnectionError(str(err)) from err
        else:
            return True

    # ----------------------------------------------------------------
    # HTTP calls (direct API and AI Gateway share one path)
    # ----------------------------------------------------------------

    async def _http_json(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """POST JSON to a model and return the unwrapped result."""
        url, headers = self._endpoint(model)
        resp = await self._cf._client.post(
            url, json=input_data, headers=headers, timeout=timeout
        )
        self._check_http(resp)
        return self._unwrap(resp.json())

    async def _http_binary(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> bytes:
        """POST to a TTS model and return audio bytes."""
        url, headers = self._endpoint(model)
        resp = await self._cf._client.post(
            url, json=input_data, headers=headers, timeout=timeout
        )
        self._check_http(resp)

        content_type = resp.headers.get("content-type", "")
        if "audio/" in content_type:
            return resp.content

        try:
            data = self._unwrap(resp.json())
            if isinstance(data, dict) and "audio" in data:
                return base64.b64decode(data["audio"])
        except (json.JSONDecodeError, KeyError):
            pass

        return resp.content

    async def _http_stream(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> AsyncGenerator[dict[str, Any]]:
        """POST to a model and yield parsed SSE events."""
        input_data = {**input_data, "stream": True}
        url, headers = self._endpoint(model)
        request = self._cf._client.build_request(
            "POST", url, json=input_data, headers=headers, timeout=timeout
        )
        resp = await self._cf._client.send(request, stream=True)
        try:
            self._check_http(resp)
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    _LOGGER.debug("Failed to parse SSE: %s", payload)
        finally:
            await resp.aclose()

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _unwrap(data: dict[str, Any]) -> dict[str, Any]:
        """Unwrap the CF API {result: ..., success: true} envelope."""
        if isinstance(data, dict) and "result" in data and "success" in data:
            return data["result"]
        return data

    @staticmethod
    def _check_http(resp: Any) -> None:
        """Check an httpx response for errors."""
        if resp.status_code == 401:
            raise CloudflareAIAuthError("Authentication failed")
        if resp.status_code == 403:
            raise CloudflareAIAuthError("Insufficient permissions")
        if resp.status_code >= 400:
            raise CloudflareAIError(f"API error {resp.status_code}: {resp.text[:200]}")
