"""Tests for the Cloudflare Workers AI task entity."""

from __future__ import annotations

from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant


async def test_ai_task_entity_created(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    mock_run_model: AsyncMock,
    setup_ha_components,
) -> None:
    """Test that the AI task entity is created on setup."""
    mock_run_model.return_value = {"response": "test"}

    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    state = hass.states.get("ai_task.cloudflare_ai_task")
    assert state is not None


class TestExtractText:
    """Test AI task response text extraction."""

    def test_workers_ai_format(self) -> None:
        """Test Workers AI native format."""
        from custom_components.cloudflare_ai.ai_task import CloudflareAITaskEntity

        entity = object.__new__(CloudflareAITaskEntity)
        assert entity._extract_text({"response": "Hello!"}) == "Hello!"

    def test_openai_format(self) -> None:
        """Test OpenAI-compatible format."""
        from custom_components.cloudflare_ai.ai_task import CloudflareAITaskEntity

        entity = object.__new__(CloudflareAITaskEntity)
        result = entity._extract_text(
            {"choices": [{"message": {"content": "Hi there!"}}]}
        )
        assert result == "Hi there!"

    def test_none_response(self) -> None:
        """Test None response field."""
        from custom_components.cloudflare_ai.ai_task import CloudflareAITaskEntity

        entity = object.__new__(CloudflareAITaskEntity)
        assert entity._extract_text({"response": None}) == ""

    def test_non_dict(self) -> None:
        """Test non-dict response."""
        from custom_components.cloudflare_ai.ai_task import CloudflareAITaskEntity

        entity = object.__new__(CloudflareAITaskEntity)
        assert entity._extract_text("raw text") == "raw text"

    def test_empty(self) -> None:
        """Test empty response."""
        from custom_components.cloudflare_ai.ai_task import CloudflareAITaskEntity

        entity = object.__new__(CloudflareAITaskEntity)
        assert entity._extract_text(None) == ""
