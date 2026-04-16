"""Tests for the Cloudflare Workers AI conversation entity."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from homeassistant.core import HomeAssistant

from .conftest import (
    SAMPLE_CHAT_RESPONSE,
    SAMPLE_TOOL_CALL_RESPONSE,
    SAMPLE_TOOL_RESULT_RESPONSE,
)


async def test_conversation_entity_created(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    mock_run_model: AsyncMock,
    setup_ha_components,
) -> None:
    """Test that the conversation entity is created on setup."""
    mock_run_model.return_value = SAMPLE_CHAT_RESPONSE

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("conversation.cloudflare_ai_conversation")
    assert state is not None


async def test_simple_chat_response(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    mock_run_model: AsyncMock,
    setup_ha_components,
) -> None:
    """Test a simple chat without tool calls."""
    mock_run_model.return_value = SAMPLE_CHAT_RESPONSE

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.services.async_call(
        "conversation",
        "process",
        {
            "text": "Hello!",
            "agent_id": "conversation.cloudflare_ai_conversation",
        },
        blocking=True,
        return_response=True,
    )
    assert result is not None
    speech = result["response"]["speech"]["plain"]["speech"]
    assert "Hello" in speech


async def test_chat_with_tool_call(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    mock_run_model: AsyncMock,
    setup_ha_components,
) -> None:
    """Test conversation with a tool call and follow-up response."""
    # First call returns tool call, second returns text
    mock_run_model.side_effect = [
        SAMPLE_TOOL_CALL_RESPONSE,
        SAMPLE_TOOL_RESULT_RESPONSE,
    ]

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    # Mock the LLM tool execution
    with patch(
        "homeassistant.helpers.llm.APIInstance.async_call_tool",
        new_callable=AsyncMock,
        return_value={"date": "2026-03-18", "time": "20:00:00"},
    ):
        result = await hass.services.async_call(
            "conversation",
            "process",
            {
                "text": "What time is it?",
                "agent_id": "conversation.cloudflare_ai_conversation",
            },
            blocking=True,
            return_response=True,
        )

    assert result is not None
    speech = result["response"]["speech"]["plain"]["speech"]
    assert "March 18" in speech or "2026" in speech


async def test_chat_auth_error_triggers_reauth(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    mock_run_model: AsyncMock,
    setup_ha_components,
) -> None:
    """Test that auth errors trigger reauth."""
    from custom_components.cloudflare_ai.client import CloudflareAIAuthError

    mock_run_model.side_effect = CloudflareAIAuthError("Invalid token")

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    result = await hass.services.async_call(
        "conversation",
        "process",
        {
            "text": "Hello",
            "agent_id": "conversation.cloudflare_ai_conversation",
        },
        blocking=True,
        return_response=True,
    )
    assert result is not None
    assert result["response"]["response_type"] == "error"


class TestResponseParsing:
    """Test conversation response parsing."""

    def test_parse_workers_ai_native(self) -> None:
        """Test parsing Workers AI native format."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        result = entity._parse_response(
            {
                "response": "Hello!",
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        assert result["content"] == "Hello!"
        assert result["role"] == "assistant"

    def test_parse_with_tool_calls(self) -> None:
        """Test parsing response with tool calls."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        result = entity._parse_response(
            {
                "response": None,
                "tool_calls": [{"name": "GetDateTime", "arguments": {}}],
            }
        )
        assert result["content"] == ""
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "GetDateTime"

    def test_parse_openai_format(self) -> None:
        """Test parsing OpenAI-compatible format."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        result = entity._parse_response(
            {"choices": [{"message": {"role": "assistant", "content": "Hi there!"}}]}
        )
        assert result["content"] == "Hi there!"

    def test_parse_fallback(self) -> None:
        """Test fallback for unknown format."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        result = entity._parse_response("raw string response")
        assert result["content"] == "raw string response"
        assert result["role"] == "assistant"


class TestToolCallParsing:
    """Test tool call format handling."""

    def test_cf_native_tool_format(self) -> None:
        """Test CF native tool call format: {name, arguments}."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        messages: list[dict] = []
        entity._append_tool_call_messages(
            messages,
            {
                "content": "",
                "tool_calls": [
                    {"name": "GetDateTime", "arguments": {}},
                ],
            },
        )
        assert len(messages) == 1
        tc = messages[0]["tool_calls"][0]
        assert tc["function"]["name"] == "GetDateTime"
        assert tc["type"] == "function"
        assert tc["id"] == "GetDateTime"

    def test_openai_tool_format(self) -> None:
        """Test OpenAI tool call format: {function: {name, arguments}}."""
        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        entity = object.__new__(CloudflareConversationEntity)
        messages: list[dict] = []
        entity._append_tool_call_messages(
            messages,
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "GetDateTime",
                            "arguments": "{}",
                        },
                    },
                ],
            },
        )
        assert len(messages) == 1
        tc = messages[0]["tool_calls"][0]
        assert tc["function"]["name"] == "GetDateTime"
        assert tc["id"] == "call_123"


class TestTokenUsageTracking:
    """Test token usage tracking."""

    def test_trace_usage_with_stats(self) -> None:
        """Test that usage data is traced."""
        from unittest.mock import MagicMock

        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        chat_log = MagicMock()
        CloudflareConversationEntity._trace_usage(
            chat_log,
            {
                "response": "hello",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                },
            },
        )
        chat_log.async_trace.assert_called_once_with(
            {
                "stats": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                }
            }
        )

    def test_trace_usage_no_usage(self) -> None:
        """Test that missing usage is handled gracefully."""
        from unittest.mock import MagicMock

        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        chat_log = MagicMock()
        CloudflareConversationEntity._trace_usage(chat_log, {"response": "hello"})
        chat_log.async_trace.assert_not_called()

    def test_trace_usage_not_dict(self) -> None:
        """Test that non-dict responses are handled."""
        from unittest.mock import MagicMock

        from custom_components.cloudflare_ai.conversation import (
            CloudflareConversationEntity,
        )

        chat_log = MagicMock()
        CloudflareConversationEntity._trace_usage(chat_log, "not a dict")
        chat_log.async_trace.assert_not_called()
