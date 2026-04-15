"""AI Task entity for Cloudflare Workers AI."""

from __future__ import annotations

import json
import logging
from typing import Any

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
    CloudflareAIError,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_ENABLE_THINKING,
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    SUBENTRY_AI_TASK,
)
from .entity import CloudflareAIBaseEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_AI_TASK:
            async_add_entities(
                [CloudflareAITaskEntity(config_entry, subentry)],
                config_subentry_id=subentry.subentry_id,
            )


class CloudflareAITaskEntity(ai_task.AITaskEntity, CloudflareAIBaseEntity):
    """Cloudflare Workers AI task entity."""

    _attr_supported_features = ai_task.AITaskEntityFeature.GENERATE_DATA

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the AI task entity."""
        super().__init__(config_entry, subentry, CONF_CHAT_MODEL)

    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        client: CloudflareAIClient = self._config_entry.runtime_data
        options = self._subentry.data

        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = float(options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))
        enable_thinking = bool(
            options.get(CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING)
        )

        # Build messages from chat log
        messages: list[dict[str, Any]] = []
        for content in chat_log.content:
            if isinstance(content, conversation.SystemContent):
                messages.append({
                    "role": "system",
                    "content": content.content or "",
                })
            elif isinstance(content, conversation.UserContent):
                messages.append({
                    "role": "user",
                    "content": content.content or "",
                })
            elif isinstance(content, conversation.AssistantContent):
                messages.append({
                    "role": "assistant",
                    "content": content.content or "",
                })

        request_body: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            },
        }

        try:
            response_data = await client.run_model(
                model, request_body, timeout=120.0
            )
        except CloudflareAIAuthError as err:
            _LOGGER.error("AI task auth error: %s", err)
            self._config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="auth_failed",
            ) from err
        except CloudflareAIError as err:
            _LOGGER.error("AI task error: %s", err)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="api_error",
                translation_placeholders={"error": str(err)},
            ) from err

        # Extract text from response
        text = self._extract_text(response_data)

        # Add the assistant response to the chat log
        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=text,
            )
        )

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )

        # Parse structured output
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            _LOGGER.error(
                "Failed to parse structured JSON response: %s. Response: %s",
                err,
                text,
            )
            raise HomeAssistantError(
                "Error parsing structured AI response"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )

    @staticmethod
    def _extract_text(data: Any) -> str:
        """Extract text from the model response."""
        if isinstance(data, dict):
            # OpenAI-compatible format
            if "choices" in data:
                return (
                    data["choices"][0]
                    .get("message", {})
                    .get("content", "")
                )
            # Workers AI native format
            if "response" in data:
                return data["response"] or ""
        return str(data) if data else ""
