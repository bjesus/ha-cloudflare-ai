"""Client wrapper around the official Cloudflare Python SDK."""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from cloudflare import APIConnectionError, AsyncCloudflare, AuthenticationError

from .const import CF_AI_GATEWAY_BASE

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
            if self.use_gateway:
                return await self._gateway_json(model, input_data, timeout)
            return await self._sdk_run(model, input_data, timeout)
        except AuthenticationError as err:
            raise CloudflareAIAuthError(str(err)) from err
        except APIConnectionError as err:
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
            if self.use_gateway:
                return await self._gateway_binary(model, input_data, timeout)
            return await self._sdk_run_binary(model, input_data, timeout)
        except AuthenticationError as err:
            raise CloudflareAIAuthError(str(err)) from err
        except APIConnectionError as err:
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
            if self.use_gateway:
                url = self._gateway_url(model)
                headers = self._gateway_headers()
                headers["Content-Type"] = content_type
            else:
                url = (
                    "https://api.cloudflare.com/client/v4"
                    f"/accounts/{self._account_id}/ai/run/{model}"
                )
                headers = {
                    "Authorization": f"Bearer {self._api_token}",
                    "Content-Type": content_type,
                }

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
            if self.use_gateway:
                async for event in self._gateway_stream(model, input_data, timeout):
                    yield event
            else:
                async for event in self._sdk_stream(model, input_data, timeout):
                    yield event
        except AuthenticationError as err:
            raise CloudflareAIAuthError(str(err)) from err
        except APIConnectionError as err:
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
    # Direct Workers AI (via official SDK)
    # ----------------------------------------------------------------

    async def _sdk_run(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Run a model via the SDK."""
        sdk_kwargs: dict[str, Any] = {
            "account_id": self._account_id,
            "timeout": timeout,
        }
        extra: dict[str, Any] = {}

        sdk_params = {
            "messages",
            "tools",
            "max_tokens",
            "temperature",
            "stream",
            "prompt",
            "text",
            "audio",
            "image",
            "image_b64",
            "lang",
            "source_lang",
            "target_lang",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "seed",
            "guidance",
            "height",
            "width",
            "num_steps",
            "strength",
            "negative_prompt",
            "lora",
            "raw",
            "response_format",
            "input_text",
            "max_length",
            "ignore_eos",
            "functions",
        }
        for key, value in input_data.items():
            if key in sdk_params:
                sdk_kwargs[key] = value
            else:
                extra[key] = value

        if extra:
            sdk_kwargs["extra_body"] = extra

        result = await self._cf.ai.run(model, **sdk_kwargs)

        if isinstance(result, dict):
            return result
        return {"response": str(result) if result else ""}

    async def _sdk_run_binary(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> bytes:
        """Run a model via SDK and return raw bytes (for TTS)."""
        sdk_kwargs: dict[str, Any] = {
            "account_id": self._account_id,
            "timeout": timeout,
        }
        extra: dict[str, Any] = {}

        sdk_params = {"text", "prompt"}
        for key, value in input_data.items():
            if key in sdk_params:
                sdk_kwargs[key] = value
            else:
                extra[key] = value

        if extra:
            sdk_kwargs["extra_body"] = extra

        raw_response = await self._cf.ai.with_raw_response.run(model, **sdk_kwargs)
        body = await raw_response.read()

        content_type = raw_response.headers.get("content-type", "")
        if "audio/" in content_type:
            return body

        # JSON response with base64 audio (MeloTTS style)
        try:
            data = json.loads(body)
            data = self._unwrap(data)
            if isinstance(data, dict) and "audio" in data:
                return base64.b64decode(data["audio"])
        except (json.JSONDecodeError, KeyError):
            pass

        return body

    async def _sdk_stream(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream via SDK."""
        sdk_kwargs: dict[str, Any] = {
            "account_id": self._account_id,
            "timeout": timeout,
            "stream": True,
        }
        extra: dict[str, Any] = {}

        sdk_params = {
            "messages",
            "tools",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
        }
        for key, value in input_data.items():
            if key == "stream":
                continue
            if key in sdk_params:
                sdk_kwargs[key] = value
            else:
                extra[key] = value

        if extra:
            sdk_kwargs["extra_body"] = extra

        async with self._cf.ai.with_streaming_response.run(
            model, **sdk_kwargs
        ) as response:
            async for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    _LOGGER.debug("Failed to parse SSE: %s", payload)

    # ----------------------------------------------------------------
    # AI Gateway (via SDK's underlying httpx client)
    # ----------------------------------------------------------------

    async def _gateway_json(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        """Run a model via AI Gateway."""
        resp = await self._cf._client.post(
            self._gateway_url(model),
            json=input_data,
            headers=self._gateway_headers(),
            timeout=timeout,
        )
        self._check_http(resp)
        return self._unwrap(resp.json())

    async def _gateway_binary(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> bytes:
        """Run a TTS model via AI Gateway, returning audio bytes."""
        resp = await self._cf._client.post(
            self._gateway_url(model),
            json=input_data,
            headers=self._gateway_headers(),
            timeout=timeout,
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

    async def _gateway_stream(
        self, model: str, input_data: dict[str, Any], timeout: float
    ) -> AsyncGenerator[dict[str, Any]]:
        """Stream via AI Gateway."""
        input_data = {**input_data, "stream": True}
        request = self._cf._client.build_request(
            "POST",
            self._gateway_url(model),
            json=input_data,
            headers=self._gateway_headers(),
            timeout=timeout,
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
