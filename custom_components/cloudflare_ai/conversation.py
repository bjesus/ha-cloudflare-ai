"""Conversation entity for Cloudflare Workers AI."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Literal

from homeassistant.components import conversation
from homeassistant.components.conversation import (
    AssistantContent,
    ChatLog,
    ConversationEntity,
    ConversationInput,
    ConversationResult,
    ToolResultContent,
)
from homeassistant.config_entries import ConfigEntry, ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
    CloudflareAIError,
)
from .const import (
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    FUNCTION_CALLING_MODELS,
    MAX_TOOL_ITERATIONS,
    SUBENTRY_CONVERSATION,
)
from .entity import CloudflareAIBaseEntity

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    entities = []
    for subentry_id, subentry in config_entry.subentries.items():
        if subentry.subentry_type == SUBENTRY_CONVERSATION:
            entities.append(
                CloudflareConversationEntity(config_entry, subentry)
            )
    async_add_entities(entities)


def _format_tool(tool: llm.Tool) -> dict[str, Any]:
    """Format an HA LLM tool as an OpenAI-compatible function definition."""
    parameters = {}
    if tool.parameters and tool.parameters.schema:
        # Convert voluptuous schema to JSON-compatible dict
        parameters = _vol_schema_to_json(tool.parameters)
    else:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": parameters,
        },
    }


def _vol_schema_to_json(schema: Any) -> dict[str, Any]:
    """Convert a voluptuous schema to JSON Schema dict.

    This is a simplified conversion that handles the common cases
    from Home Assistant LLM tools.
    """
    import voluptuous as vol

    if not hasattr(schema, "schema"):
        return {"type": "object", "properties": {}}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for key in schema.schema:
        if isinstance(key, vol.Required):
            name = str(key.schema)
            required.append(name)
        elif isinstance(key, vol.Optional):
            name = str(key.schema)
        elif isinstance(key, str):
            name = key
        else:
            name = str(key)

        validator = schema.schema[key]
        properties[name] = _validator_to_json(validator, name)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = required
    return result


def _validator_to_json(validator: Any, name: str = "") -> dict[str, Any]:
    """Convert a voluptuous validator to a JSON Schema property."""
    import voluptuous as vol

    if validator is str or validator == str:
        return {"type": "string"}
    if validator is int or validator == int:
        return {"type": "integer"}
    if validator is float or validator == float:
        return {"type": "number"}
    if validator is bool or validator == bool:
        return {"type": "boolean"}
    if isinstance(validator, vol.All):
        # Use the first validator that gives a useful type
        for v in validator.validators:
            result = _validator_to_json(v, name)
            if result.get("type") != "string":
                return result
        return _validator_to_json(validator.validators[0], name)
    if isinstance(validator, vol.In):
        return {"type": "string", "enum": list(validator.container)}
    if isinstance(validator, vol.Coerce):
        return _validator_to_json(validator.type, name)

    return {"type": "string"}


class CloudflareConversationEntity(ConversationEntity, CloudflareAIBaseEntity):
    """Cloudflare Workers AI conversation agent."""

    _attr_supports_streaming = True

    def __init__(
        self,
        config_entry: ConfigEntry,
        subentry: ConfigSubentry,
    ) -> None:
        """Initialize the conversation entity."""
        super().__init__(config_entry, subentry, CONF_CHAT_MODEL)
        self._subentry = subentry
        model = subentry.data.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        self._model = model
        self._supports_tools = model in FUNCTION_CALLING_MODELS

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return supported languages."""
        return MATCH_ALL

    @property
    def supported_features(self) -> int:
        """Return supported features."""
        if self._supports_tools:
            return conversation.ConversationEntityFeature.CONTROL
        return 0

    async def async_added_to_hass(self) -> None:
        """Register as a conversation agent when added to hass."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self._config_entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """Unregister as a conversation agent when removed."""
        conversation.async_unset_agent(self.hass, self._config_entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: ConversationInput,
        chat_log: ChatLog,
    ) -> ConversationResult:
        """Handle an incoming chat message."""
        client: CloudflareAIClient = self._config_entry.runtime_data
        options = self._subentry.data

        # Provide LLM data (tools + system prompt) if configured
        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT, DEFAULT_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        # Build tools list
        tools: list[dict[str, Any]] | None = None
        if self._supports_tools and chat_log.llm_api:
            tools = [_format_tool(tool) for tool in chat_log.llm_api.tools]
            _LOGGER.debug("Sending %d tools to model %s", len(tools), self._model)

        # Convert chat log to OpenAI-compatible messages
        messages = self._build_messages(chat_log)

        model = options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        max_tokens = int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS))
        temperature = float(options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE))

        # Tool-calling loop.
        # When tools are configured, use non-streaming requests so we get
        # structured tool_calls. When no tools are needed (or after tools
        # are resolved), stream the final text response for real-time UX.
        try:
            for _iteration in range(MAX_TOOL_ITERATIONS):
                request_body: dict[str, Any] = {
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                if tools:
                    request_body["tools"] = tools

                if tools:
                    # Non-streaming: need structured tool_calls
                    response_data = await client.run_model(
                        model, request_body, timeout=120.0
                    )
                    assistant_message = self._parse_response(response_data)
                    self._trace_usage(chat_log, response_data)

                    if assistant_message.get("tool_calls"):
                        self._append_tool_call_messages(
                            messages, assistant_message
                        )
                        tool_results = await self._execute_tool_calls(
                            assistant_message["tool_calls"],
                            chat_log,
                            user_input,
                        )
                        messages.extend(tool_results)
                        continue

                    # Model responded with text, not tools — done
                    chat_log.async_add_assistant_content_without_tools(
                        AssistantContent(
                            agent_id=user_input.agent_id,
                            content=assistant_message.get("content", ""),
                        )
                    )
                    break

                # No tools — stream the response for real-time UX
                async for _content in chat_log.async_add_delta_content_stream(
                    user_input.agent_id,
                    self._stream_response(client, model, request_body),
                ):
                    pass  # content is accumulated by chat_log
                break
            else:
                chat_log.async_add_assistant_content_without_tools(
                    AssistantContent(
                        agent_id=user_input.agent_id,
                        content="I'm sorry, I was unable to complete the request after multiple attempts.",
                    )
                )

        except CloudflareAIAuthError as err:
            _LOGGER.error("Authentication error: %s", err)
            self._config_entry.async_start_reauth(self.hass)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="auth_failed",
            ) from err
        except CloudflareAIError as err:
            _LOGGER.error("Cloudflare AI error: %s", err)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="api_error",
                translation_placeholders={"error": str(err)},
            ) from err

        intent_response = intent.IntentResponse(language=user_input.language)
        assert chat_log.content[-1].content is not None
        intent_response.async_set_speech(chat_log.content[-1].content)
        return ConversationResult(
            response=intent_response,
            conversation_id=chat_log.conversation_id,
            continue_conversation=chat_log.continue_conversation,
        )

    def _build_messages(self, chat_log: ChatLog) -> list[dict[str, Any]]:
        """Convert chat log content to OpenAI-compatible messages."""
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
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content.content or "",
                }
                if content.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.tool_args),
                            },
                        }
                        for tc in content.tool_calls
                    ]
                messages.append(msg)
            elif isinstance(content, ToolResultContent):
                messages.append({
                    "role": "tool",
                    "tool_call_id": content.tool_call_id,
                    "content": json.dumps(content.tool_result),
                })

        return messages

    def _parse_response(self, data: Any) -> dict[str, Any]:
        """Parse the model response into an assistant message dict."""
        # Workers AI text-generation returns:
        # {"response": "text"} (non-streaming, non-tool)
        # or OpenAI-compatible format with choices
        if isinstance(data, dict):
            # OpenAI-compatible format (from gateway or newer models)
            if "choices" in data:
                choice = data["choices"][0]
                return choice.get("message", {"content": "", "role": "assistant"})

            # Workers AI native format
            # Response may contain both "response" (text) and "tool_calls"
            if "response" in data:
                result: dict[str, Any] = {
                    "role": "assistant",
                    "content": data["response"] or "",
                }
                if data.get("tool_calls"):
                    result["tool_calls"] = data["tool_calls"]
                return result

            # Direct tool_calls format from some CF models
            if "tool_calls" in data:
                return {
                    "role": "assistant",
                    "content": data.get("content", ""),
                    "tool_calls": data["tool_calls"],
                }

        # Fallback
        return {
            "role": "assistant",
            "content": str(data) if data else "",
        }

    @staticmethod
    def _trace_usage(chat_log: ChatLog, response_data: Any) -> None:
        """Track token usage from the model response."""
        if not isinstance(response_data, dict):
            return
        usage = response_data.get("usage")
        if not usage:
            return
        stats: dict[str, int] = {}
        if "prompt_tokens" in usage:
            stats["input_tokens"] = usage["prompt_tokens"]
        if "completion_tokens" in usage:
            stats["output_tokens"] = usage["completion_tokens"]
        if stats:
            chat_log.async_trace({"stats": stats})

    async def _stream_response(
        self,
        client: CloudflareAIClient,
        model: str,
        request_body: dict[str, Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream a Workers AI response and yield chat_log delta dicts.

        Yields dicts consumed by chat_log.async_add_delta_content_stream():
          {"role": "assistant"}  — starts a new assistant content block
          {"content": "text"}   — text delta
        """
        yield {"role": "assistant"}

        usage: dict[str, Any] | None = None
        async for event in client.stream_model(model, request_body):
            text = event.get("response", "")
            if text:
                yield {"content": text}
            # Capture usage from the last event
            if "usage" in event:
                usage = event["usage"]

        # Report final token usage
        if usage:
            stats: dict[str, int] = {}
            if "prompt_tokens" in usage:
                stats["input_tokens"] = usage["prompt_tokens"]
            if "completion_tokens" in usage:
                stats["output_tokens"] = usage["completion_tokens"]
            if stats:
                yield {"stats": stats}

    @staticmethod
    def _append_tool_call_messages(
        messages: list[dict[str, Any]],
        assistant_message: dict[str, Any],
    ) -> None:
        """Normalize and append an assistant tool-call message to the history."""
        normalized_tcs = []
        for tc in assistant_message["tool_calls"]:
            if "function" not in tc:
                # CF native format -> normalize to OpenAI format
                normalized_tcs.append({
                    "id": tc.get("id", tc.get("name", "call")),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("arguments", {}))
                            if not isinstance(tc.get("arguments"), str)
                            else tc.get("arguments", "{}"),
                    },
                })
            else:
                normalized_tcs.append(tc)

        messages.append({
            "role": "assistant",
            "content": assistant_message.get("content", ""),
            "tool_calls": normalized_tcs,
        })

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        chat_log: ChatLog,
        user_input: ConversationInput,
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return tool result messages."""
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            # CF Workers AI uses {"name": ..., "arguments": ...} directly
            # OpenAI uses {"function": {"name": ..., "arguments": ...}}
            if "function" in tc:
                function = tc["function"]
                tool_name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                tool_call_id = tc.get("id", tool_name)
            else:
                tool_name = tc.get("name", "")
                raw_args = tc.get("arguments", "{}")
                tool_call_id = tc.get("id", tool_name)
            try:
                tool_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except json.JSONDecodeError:
                tool_args = {}

            _LOGGER.debug(
                "Executing tool %s with args: %s", tool_name, tool_args
            )

            if chat_log.llm_api:
                try:
                    tool_input = llm.ToolInput(
                        tool_name=tool_name,
                        tool_args=tool_args,
                    )
                    tool_response = await chat_log.llm_api.async_call_tool(
                        tool_input
                    )
                    result_str = json.dumps(tool_response)
                except Exception as err:
                    _LOGGER.error("Tool call %s failed: %s", tool_name, err)
                    result_str = json.dumps({"error": str(err)})
            else:
                result_str = json.dumps({"error": "No LLM API configured"})

            results.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result_str,
            })

        return results
