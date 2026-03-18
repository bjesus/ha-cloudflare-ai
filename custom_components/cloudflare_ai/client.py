"""HTTP client for Cloudflare Workers AI API."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import httpx

from collections.abc import AsyncGenerator

from .const import (
    CF_AI_GATEWAY_BASE,
    CF_API_BASE,
    MAX_RETRIES,
    RETRY_BACKOFF_BASE,
)

_LOGGER = logging.getLogger(__name__)


class CloudflareAIError(Exception):
    """Base error for Cloudflare AI."""


class CloudflareAIAuthError(CloudflareAIError):
    """Authentication error."""


class CloudflareAIConnectionError(CloudflareAIError):
    """Connection error."""


class CloudflareAIClient:
    """Client for Cloudflare Workers AI API."""

    def __init__(
        self,
        httpx_client: httpx.AsyncClient,
        account_id: str,
        api_token: str,
        gateway_id: str | None = None,
        gateway_api_token: str | None = None,
    ) -> None:
        """Initialize the client."""
        self._client = httpx_client
        self._account_id = account_id
        self._api_token = api_token
        self._gateway_id = gateway_id
        self._gateway_api_token = gateway_api_token

    @property
    def use_gateway(self) -> bool:
        """Return True if AI Gateway is configured."""
        return self._gateway_id is not None

    def _direct_url(self, model: str) -> str:
        """Get the direct API URL for a model."""
        return f"{CF_API_BASE}/accounts/{self._account_id}/ai/run/{model}"

    def _gateway_url(self, model: str) -> str:
        """Get the AI Gateway URL for a model."""
        return (
            f"{CF_AI_GATEWAY_BASE}/{self._account_id}"
            f"/{self._gateway_id}/workers-ai/{model}"
        )

    def _get_url(self, model: str) -> str:
        """Get the appropriate URL for a model."""
        if self.use_gateway:
            return self._gateway_url(model)
        return self._direct_url(model)

    def _get_headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        """Get request headers."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.use_gateway:
            headers["cf-aig-authorization"] = f"Bearer {self._gateway_api_token or self._api_token}"
            headers["Authorization"] = f"Bearer {self._api_token}"
        else:
            headers["Authorization"] = f"Bearer {self._api_token}"

        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def run_model(
        self,
        model: str,
        input_data: dict[str, Any],
        *,
        stream: bool = False,
        raw_audio: bytes | None = None,
        audio_content_type: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any] | httpx.Response:
        """Run an AI model with retry logic.

        For JSON input, pass input_data.
        For raw audio input (like STT), pass raw_audio and audio_content_type.
        If stream=True, returns the httpx.Response for streaming consumption.
        """
        url = self._get_url(model)
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                if stream:
                    return await self._request_stream(
                        url, input_data, timeout=timeout
                    )
                if raw_audio is not None:
                    return await self._request_raw_audio(
                        url, raw_audio, audio_content_type or "audio/wav",
                        timeout=timeout,
                    )
                return await self._request_json(url, input_data, timeout=timeout)
            except CloudflareAIAuthError:
                raise
            except (CloudflareAIError, httpx.HTTPError) as err:
                last_error = err
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    _LOGGER.debug(
                        "Cloudflare AI request failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        attempt + 1, MAX_RETRIES + 1, wait, err,
                    )
                    await asyncio.sleep(wait)

        raise CloudflareAIConnectionError(
            f"Failed after {MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error

    async def _request_json(
        self,
        url: str,
        input_data: dict[str, Any],
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Make a JSON request."""
        headers = self._get_headers()
        response = await self._client.post(
            url,
            headers=headers,
            json=input_data,
            timeout=timeout,
        )
        self._check_response(response)
        data = response.json()

        # Both direct API and AI Gateway (Workers AI provider) wrap results
        # in {"result": ..., "success": true}
        if isinstance(data, dict) and "result" in data and "success" in data:
            return data["result"]
        return data

    async def _request_stream(
        self,
        url: str,
        input_data: dict[str, Any],
        timeout: float = 60.0,
    ) -> httpx.Response:
        """Make a streaming request. Returns the response for streaming."""
        headers = self._get_headers()
        input_data = {**input_data, "stream": True}

        # Use send() for streaming so we can return the response
        request = self._client.build_request(
            "POST",
            url,
            headers=headers,
            json=input_data,
            timeout=timeout,
        )
        response = await self._client.send(request, stream=True)
        self._check_response(response)
        return response

    async def stream_model(
        self,
        model: str,
        input_data: dict[str, Any],
        timeout: float = 120.0,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream model output as parsed SSE events.

        Yields dicts from the Workers AI SSE stream. Each dict may contain:
          - {"response": "text_chunk", ...} — text delta
          - {"response": "", "usage": {...}} — final usage stats
        The [DONE] sentinel closes the generator.
        """
        url = self._get_url(model)
        headers = self._get_headers()
        input_data = {**input_data, "stream": True}

        request = self._client.build_request(
            "POST", url, headers=headers, json=input_data, timeout=timeout,
        )
        response = await self._client.send(request, stream=True)
        try:
            self._check_response(response)
            async for event in self._parse_sse(response):
                yield event
        finally:
            await response.aclose()

    @staticmethod
    async def _parse_sse(
        response: httpx.Response,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Parse SSE lines from an httpx streaming response.

        Workers AI SSE format:
            data: {"response":"token","p":"..."}
            data: {"response":"","usage":{...}}
            data: [DONE]
        """
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    payload = line[6:]
                    if payload == "[DONE]":
                        return
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        _LOGGER.debug("Failed to parse SSE payload: %s", payload)

    async def _request_raw_audio(
        self,
        url: str,
        audio_data: bytes,
        content_type: str,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        """Make a request with raw audio body (for STT models like Nova-3)."""
        headers = self._get_headers({"Content-Type": content_type})
        response = await self._client.post(
            url,
            headers=headers,
            content=audio_data,
            timeout=timeout,
        )
        self._check_response(response)
        data = response.json()
        if isinstance(data, dict) and "result" in data and "success" in data:
            return data["result"]
        return data

    async def run_model_binary(
        self,
        model: str,
        input_data: dict[str, Any],
        timeout: float = 60.0,
    ) -> bytes:
        """Run a model and return raw binary response (for TTS)."""
        url = self._get_url(model)
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                headers = self._get_headers()
                response = await self._client.post(
                    url,
                    headers=headers,
                    json=input_data,
                    timeout=timeout,
                )
                self._check_response(response)

                # TTS models may return binary audio or JSON with base64
                content_type = response.headers.get("content-type", "")
                if "audio/" in content_type:
                    return response.content
                # JSON response with base64-encoded audio
                data = response.json()
                if isinstance(data, dict) and "result" in data and "success" in data:
                    data = data["result"]
                if isinstance(data, dict) and "audio" in data:
                    return base64.b64decode(data["audio"])
                return response.content
            except CloudflareAIAuthError:
                raise
            except (CloudflareAIError, httpx.HTTPError) as err:
                last_error = err
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(wait)

        raise CloudflareAIConnectionError(
            f"Failed after {MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error

    async def validate_credentials(self) -> bool:
        """Validate that the credentials work by listing models."""
        url = f"{CF_API_BASE}/accounts/{self._account_id}/ai/models/search"
        headers = {
            "Authorization": f"Bearer {self._api_token}",
        }
        try:
            response = await self._client.get(
                url, headers=headers, timeout=10.0
            )
            self._check_response(response)
            return True
        except CloudflareAIAuthError:
            raise
        except Exception as err:
            raise CloudflareAIConnectionError(
                f"Failed to validate credentials: {err}"
            ) from err

    def _check_response(self, response: httpx.Response) -> None:
        """Check the HTTP response for errors."""
        if response.status_code == 401:
            raise CloudflareAIAuthError("Invalid API token")
        if response.status_code == 403:
            raise CloudflareAIAuthError(
                "API token does not have required permissions"
            )
        if response.status_code == 429:
            raise CloudflareAIError("Rate limited by Cloudflare")
        if response.status_code >= 500:
            raise CloudflareAIError(
                f"Cloudflare server error: {response.status_code}"
            )
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            raise CloudflareAIError(
                f"Cloudflare API error {response.status_code}: {detail}"
            )
