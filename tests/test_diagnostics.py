"""Tests for the Cloudflare Workers AI diagnostics."""

from __future__ import annotations

from homeassistant.core import HomeAssistant

from custom_components.cloudflare_ai.const import (
    CONF_API_TOKEN,
)
from custom_components.cloudflare_ai.diagnostics import (
    async_get_config_entry_diagnostics,
)


async def test_diagnostics_redacts_tokens(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    setup_ha_components,
) -> None:
    """Test that diagnostics redacts API tokens."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    diag = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    # API token should be redacted
    assert diag["config_entry"][CONF_API_TOKEN] == "**REDACTED**"

    # Subentries should be present
    assert len(diag["subentries"]) >= 3
    for sub in diag["subentries"]:
        assert "subentry_type" in sub
        assert "data" in sub


async def test_diagnostics_includes_subentries(
    hass: HomeAssistant,
    mock_config_entry,
    mock_validate_credentials,
    setup_ha_components,
) -> None:
    """Test that diagnostics includes all subentry data."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    diag = await async_get_config_entry_diagnostics(hass, mock_config_entry)

    subentry_types = {s["subentry_type"] for s in diag["subentries"]}
    assert "conversation" in subentry_types
    assert "tts" in subentry_types
    assert "stt" in subentry_types
    assert "ai_task_data" in subentry_types
