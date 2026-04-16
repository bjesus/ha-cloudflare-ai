"""Tests for the Cloudflare Workers AI config flow."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from custom_components.cloudflare_ai.const import (
    CONF_ACCOUNT_ID,
    CONF_API_TOKEN,
    CONF_GATEWAY_API_TOKEN,
    CONF_GATEWAY_ID,
    CONF_USE_AI_GATEWAY,
    DOMAIN,
)

from .conftest import TEST_ACCOUNT_ID, TEST_API_TOKEN


@pytest.fixture(autouse=True)
async def mock_setup_entry(hass: HomeAssistant):
    """Prevent full setup during config flow tests."""
    # Set up conversation component (required dependency)
    # by setting up homeassistant and conversation components
    from homeassistant.setup import async_setup_component

    await async_setup_component(hass, "homeassistant", {})
    await async_setup_component(hass, "conversation", {})
    await hass.async_block_till_done()

    with patch(
        "custom_components.cloudflare_ai.async_setup_entry",
        return_value=True,
    ):
        yield


async def test_user_step_success(hass: HomeAssistant) -> None:
    """Test successful user step."""
    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "user"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
                CONF_API_TOKEN: TEST_API_TOKEN,
                CONF_USE_AI_GATEWAY: False,
            },
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["title"] == "Cloudflare Workers AI"
        assert result["data"][CONF_ACCOUNT_ID] == TEST_ACCOUNT_ID
        assert result["data"][CONF_API_TOKEN] == TEST_API_TOKEN
        assert result["data"][CONF_USE_AI_GATEWAY] is False
        # Should create 4 default subentries (conversation, ai_task, tts, stt)
        assert len(result.get("subentries", [])) == 4


async def test_user_step_with_gateway(hass: HomeAssistant) -> None:
    """Test user step with AI Gateway enabled."""
    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
                CONF_API_TOKEN: TEST_API_TOKEN,
                CONF_USE_AI_GATEWAY: True,
                CONF_GATEWAY_ID: "my-gateway",
                CONF_GATEWAY_API_TOKEN: "gw-token",
            },
        )
        assert result["type"] is FlowResultType.CREATE_ENTRY
        assert result["data"][CONF_USE_AI_GATEWAY] is True
        assert result["data"][CONF_GATEWAY_ID] == "my-gateway"
        assert result["data"][CONF_GATEWAY_API_TOKEN] == "gw-token"


async def test_user_step_invalid_auth(hass: HomeAssistant) -> None:
    """Test user step with invalid credentials."""
    from custom_components.cloudflare_ai.client import CloudflareAIAuthError

    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        side_effect=CloudflareAIAuthError("Invalid token"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
                CONF_API_TOKEN: "bad_token",
                CONF_USE_AI_GATEWAY: False,
            },
        )
        assert result["type"] is FlowResultType.FORM
        assert result["errors"]["base"] == "invalid_auth"


async def test_user_step_cannot_connect(hass: HomeAssistant) -> None:
    """Test user step when API is unreachable."""
    from custom_components.cloudflare_ai.client import CloudflareAIConnectionError

    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        side_effect=CloudflareAIConnectionError("Connection failed"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
                CONF_API_TOKEN: TEST_API_TOKEN,
                CONF_USE_AI_GATEWAY: False,
            },
        )
        assert result["type"] is FlowResultType.FORM
        assert result["errors"]["base"] == "cannot_connect"


async def test_user_step_duplicate_entry(
    hass: HomeAssistant, mock_config_entry
) -> None:
    """Test that duplicate accounts are rejected."""
    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: TEST_ACCOUNT_ID,
                CONF_API_TOKEN: TEST_API_TOKEN,
                CONF_USE_AI_GATEWAY: False,
            },
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "already_configured"


async def test_reauth_flow(hass: HomeAssistant, mock_config_entry) -> None:
    """Test the reauth flow."""
    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await mock_config_entry.start_reauth_flow(hass)
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "reauth_confirm"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_API_TOKEN: "new_token_123"},
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reauth_successful"
        assert mock_config_entry.data[CONF_API_TOKEN] == "new_token_123"


async def test_reconfigure_flow(hass: HomeAssistant, mock_config_entry) -> None:
    """Test the main entry reconfigure flow."""
    with patch(
        "custom_components.cloudflare_ai.config_flow.CloudflareAIClient.validate_credentials",
        new_callable=AsyncMock,
        return_value=True,
    ):
        result = await mock_config_entry.start_reconfigure_flow(hass)
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "reconfigure"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                CONF_ACCOUNT_ID: "new_account_id",
                CONF_API_TOKEN: "new_token",
                CONF_USE_AI_GATEWAY: True,
                CONF_GATEWAY_ID: "my-gw",
                CONF_GATEWAY_API_TOKEN: "gw-token",
            },
        )
        assert result["type"] is FlowResultType.ABORT
        assert result["reason"] == "reconfigure_successful"
        assert mock_config_entry.data[CONF_ACCOUNT_ID] == "new_account_id"
        assert mock_config_entry.data[CONF_USE_AI_GATEWAY] is True
        assert mock_config_entry.data[CONF_GATEWAY_ID] == "my-gw"
