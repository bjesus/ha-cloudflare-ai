"""Tests for the Cloudflare Workers AI integration setup."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant

from custom_components.cloudflare_ai.const import DOMAIN

from .conftest import TEST_ACCOUNT_ID


async def test_setup_entry_success(
    hass: HomeAssistant, mock_config_entry, mock_validate_credentials,
    setup_ha_components,
) -> None:
    """Test successful setup of a config entry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert mock_config_entry.runtime_data is not None


async def test_setup_entry_auth_failed(
    hass: HomeAssistant, mock_config_entry, setup_ha_components,
) -> None:
    """Test setup fails with ConfigEntryAuthFailed on auth error."""
    from custom_components.cloudflare_ai.client import CloudflareAIAuthError

    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        side_effect=CloudflareAIAuthError("Invalid token"),
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_ERROR


async def test_setup_entry_not_ready(
    hass: HomeAssistant, mock_config_entry, setup_ha_components,
) -> None:
    """Test setup retries with ConfigEntryNotReady on connection error."""
    from custom_components.cloudflare_ai.client import CloudflareAIConnectionError

    with patch(
        "custom_components.cloudflare_ai.client.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        side_effect=CloudflareAIConnectionError("Connection failed"),
    ):
        await hass.config_entries.async_setup(mock_config_entry.entry_id)
        await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.SETUP_RETRY


async def test_unload_entry(
    hass: HomeAssistant, mock_config_entry, mock_validate_credentials,
    setup_ha_components,
) -> None:
    """Test unloading a config entry."""
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.LOADED

    await hass.config_entries.async_unload(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED
