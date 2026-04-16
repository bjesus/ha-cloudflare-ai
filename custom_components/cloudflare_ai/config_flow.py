"""Config flow for Cloudflare Workers AI."""

from __future__ import annotations

import logging
from typing import Any

import voluptuous as vol
from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TemplateSelectorConfig,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .client import (
    CloudflareAIAuthError,
    CloudflareAIClient,
    CloudflareAIConnectionError,
)
from .const import (
    AURA1_VOICES,
    AURA2_VOICES,
    CHAT_MODELS,
    CONF_ACCOUNT_ID,
    CONF_API_TOKEN,
    CONF_CHAT_MODEL,
    CONF_ENABLE_THINKING,
    CONF_GATEWAY_API_TOKEN,
    CONF_GATEWAY_ID,
    CONF_IMAGE_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_STT_MODEL,
    CONF_TEMPERATURE,
    CONF_TTS_MODEL,
    CONF_USE_AI_GATEWAY,
    CONF_VOICE,
    DEFAULT_CHAT_MODEL,
    DEFAULT_ENABLE_THINKING,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_STT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_VOICE,
    DOMAIN,
    IMAGE_MODELS,
    STT_MODELS,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
    SUBENTRY_STT,
    SUBENTRY_TTS,
    TTS_MODELS,
)

_LOGGER = logging.getLogger(__name__)


class CloudflareAIConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Cloudflare Workers AI."""

    VERSION = 1
    MINOR_VERSION = 1

    @classmethod
    @callback
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this integration."""
        return {
            SUBENTRY_CONVERSATION: CloudflareAIConversationSubentryFlow,
            SUBENTRY_AI_TASK: CloudflareAITaskSubentryFlow,
            SUBENTRY_TTS: CloudflareAITTSSubentryFlow,
            SUBENTRY_STT: CloudflareAISTTSubentryFlow,
        }

    async def async_step_user(
        self,
        user_input: dict[str, Any] | None = None,
    ) -> ConfigFlowResult:
        """Handle the initial step — collect credentials."""
        errors: dict[str, str] = {}

        if user_input is not None:
            # Prevent duplicate entries for the same account
            self._async_abort_entries_match(
                {CONF_ACCOUNT_ID: user_input[CONF_ACCOUNT_ID]}
            )

            # Validate credentials
            httpx_client = get_async_client(self.hass)
            client = CloudflareAIClient(
                httpx_client=httpx_client,
                account_id=user_input[CONF_ACCOUNT_ID],
                api_token=user_input[CONF_API_TOKEN],
            )
            try:
                await client.validate_credentials()
            except CloudflareAIAuthError:
                errors["base"] = "invalid_auth"
            except CloudflareAIConnectionError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected error during validation")
                errors["base"] = "unknown"

            if not errors:
                # Store config data
                data = {
                    CONF_ACCOUNT_ID: user_input[CONF_ACCOUNT_ID],
                    CONF_API_TOKEN: user_input[CONF_API_TOKEN],
                    CONF_USE_AI_GATEWAY: user_input.get(CONF_USE_AI_GATEWAY, False),
                }
                if data[CONF_USE_AI_GATEWAY]:
                    data[CONF_GATEWAY_ID] = user_input.get(CONF_GATEWAY_ID, "")
                    data[CONF_GATEWAY_API_TOKEN] = user_input.get(
                        CONF_GATEWAY_API_TOKEN, ""
                    )

                return self.async_create_entry(
                    title="Cloudflare Workers AI",
                    data=data,
                    subentries=[
                        {
                            "subentry_type": SUBENTRY_CONVERSATION,
                            "title": "Cloudflare AI Conversation",
                            "data": {
                                CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
                                CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                                CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
                                CONF_ENABLE_THINKING: DEFAULT_ENABLE_THINKING,
                                CONF_PROMPT: DEFAULT_PROMPT,
                                CONF_LLM_HASS_API: ["assist"],
                            },
                        },
                        {
                            "subentry_type": SUBENTRY_AI_TASK,
                            "title": "Cloudflare AI Task",
                            "data": {
                                CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
                                CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
                                CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
                                CONF_ENABLE_THINKING: DEFAULT_ENABLE_THINKING,
                                CONF_IMAGE_MODEL: DEFAULT_IMAGE_MODEL,
                            },
                        },
                        {
                            "subentry_type": SUBENTRY_TTS,
                            "title": "Cloudflare AI TTS",
                            "data": {
                                CONF_TTS_MODEL: DEFAULT_TTS_MODEL,
                                CONF_VOICE: DEFAULT_TTS_VOICE,
                            },
                        },
                        {
                            "subentry_type": SUBENTRY_STT,
                            "title": "Cloudflare AI STT",
                            "data": {
                                CONF_STT_MODEL: DEFAULT_STT_MODEL,
                            },
                        },
                    ],
                )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_ACCOUNT_ID): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.TEXT)
                    ),
                    vol.Required(CONF_API_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                    vol.Optional(CONF_USE_AI_GATEWAY, default=False): bool,
                    vol.Optional(CONF_GATEWAY_ID): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.TEXT)
                    ),
                    vol.Optional(CONF_GATEWAY_API_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_reauth(self, entry_data: dict[str, Any]) -> ConfigFlowResult:
        """Handle reauth."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reauth confirmation."""
        errors: dict[str, str] = {}

        if user_input is not None:
            httpx_client = get_async_client(self.hass)
            entry = self.hass.config_entries.async_get_entry(self.context["entry_id"])
            assert entry is not None
            client = CloudflareAIClient(
                httpx_client=httpx_client,
                account_id=entry.data[CONF_ACCOUNT_ID],
                api_token=user_input[CONF_API_TOKEN],
            )
            try:
                await client.validate_credentials()
            except CloudflareAIAuthError:
                errors["base"] = "invalid_auth"
            except CloudflareAIConnectionError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected error during reauth")
                errors["base"] = "unknown"

            if not errors:
                return self.async_update_reload_and_abort(
                    entry,
                    data={**entry.data, CONF_API_TOKEN: user_input[CONF_API_TOKEN]},
                )

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=vol.Schema(
                {
                    vol.Required(CONF_API_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle reconfiguration of the main entry (change credentials)."""
        errors: dict[str, str] = {}
        entry = self._get_reconfigure_entry()

        if user_input is not None:
            httpx_client = get_async_client(self.hass)
            client = CloudflareAIClient(
                httpx_client=httpx_client,
                account_id=user_input[CONF_ACCOUNT_ID],
                api_token=user_input[CONF_API_TOKEN],
            )
            try:
                await client.validate_credentials()
            except CloudflareAIAuthError:
                errors["base"] = "invalid_auth"
            except CloudflareAIConnectionError:
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected error during reconfigure")
                errors["base"] = "unknown"

            if not errors:
                data = {
                    CONF_ACCOUNT_ID: user_input[CONF_ACCOUNT_ID],
                    CONF_API_TOKEN: user_input[CONF_API_TOKEN],
                    CONF_USE_AI_GATEWAY: user_input.get(CONF_USE_AI_GATEWAY, False),
                }
                if data[CONF_USE_AI_GATEWAY]:
                    data[CONF_GATEWAY_ID] = user_input.get(CONF_GATEWAY_ID, "")
                    data[CONF_GATEWAY_API_TOKEN] = user_input.get(
                        CONF_GATEWAY_API_TOKEN, ""
                    )
                return self.async_update_reload_and_abort(entry, data=data)

        return self.async_show_form(
            step_id="reconfigure",
            data_schema=vol.Schema(
                {
                    vol.Required(
                        CONF_ACCOUNT_ID,
                        default=entry.data.get(CONF_ACCOUNT_ID, ""),
                    ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Required(CONF_API_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                    vol.Optional(
                        CONF_USE_AI_GATEWAY,
                        default=entry.data.get(CONF_USE_AI_GATEWAY, False),
                    ): bool,
                    vol.Optional(
                        CONF_GATEWAY_ID,
                        default=entry.data.get(CONF_GATEWAY_ID, ""),
                    ): TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT)),
                    vol.Optional(CONF_GATEWAY_API_TOKEN): TextSelector(
                        TextSelectorConfig(type=TextSelectorType.PASSWORD)
                    ),
                }
            ),
            errors=errors,
        )


class CloudflareAIConversationSubentryFlow(ConfigSubentryFlow):
    """Subentry flow for conversation configuration."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return True if creating a new subentry."""
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle new subentry creation."""
        self.options = {
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_ENABLE_THINKING: DEFAULT_ENABLE_THINKING,
            CONF_PROMPT: DEFAULT_PROMPT,
        }
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration of existing subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the shared form step."""
        if user_input is not None:
            self.options.update(user_input)
            title = self.options.pop("name", "Cloudflare AI Conversation")
            if self._is_new:
                return self.async_create_entry(title=title, data=self.options)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=self.options,
                title=title,
            )

        # Get available LLM APIs
        apis: list[SelectOptionDict] = [
            SelectOptionDict(label=api.name, value=api.id)
            for api in llm.async_get_apis(self.hass)
        ]

        chat_model_options = [
            SelectOptionDict(label=m.split("/")[-1], value=m) for m in CHAT_MODELS
        ]

        schema: dict[vol.Optional | vol.Required, Any] = {}
        if self._is_new:
            schema[vol.Optional("name", default="Cloudflare AI Conversation")] = str

        schema.update(
            {
                vol.Optional(
                    CONF_CHAT_MODEL,
                    default=self.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=chat_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(
                    CONF_LLM_HASS_API,
                    default=self.options.get(CONF_LLM_HASS_API),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=apis,
                        multiple=True,
                    )
                ),
                vol.Optional(
                    CONF_PROMPT,
                    default=self.options.get(CONF_PROMPT, DEFAULT_PROMPT),
                ): TemplateSelector(TemplateSelectorConfig()),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=self.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1, max=8192, step=1, mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=self.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Optional(
                    CONF_ENABLE_THINKING,
                    default=self.options.get(
                        CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING
                    ),
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


class CloudflareAITTSSubentryFlow(ConfigSubentryFlow):
    """Subentry flow for TTS configuration."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle new subentry creation."""
        self.options = {
            CONF_TTS_MODEL: DEFAULT_TTS_MODEL,
            CONF_VOICE: DEFAULT_TTS_VOICE,
        }
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the shared form step."""
        if user_input is not None:
            self.options.update(user_input)
            title = self.options.pop("name", "Cloudflare AI TTS")
            if self._is_new:
                return self.async_create_entry(title=title, data=self.options)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=self.options,
                title=title,
            )

        tts_model_options = [
            SelectOptionDict(label=m.split("/")[-1], value=m) for m in TTS_MODELS
        ]

        all_voices = sorted(set(AURA2_VOICES + AURA1_VOICES))
        voice_options = [
            SelectOptionDict(label=v.capitalize(), value=v) for v in all_voices
        ]

        schema: dict[vol.Optional | vol.Required, Any] = {}
        if self._is_new:
            schema[vol.Optional("name", default="Cloudflare AI TTS")] = str

        schema.update(
            {
                vol.Optional(
                    CONF_TTS_MODEL,
                    default=self.options.get(CONF_TTS_MODEL, DEFAULT_TTS_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=tts_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(
                    CONF_VOICE,
                    default=self.options.get(CONF_VOICE, DEFAULT_TTS_VOICE),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=voice_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


class CloudflareAISTTSubentryFlow(ConfigSubentryFlow):
    """Subentry flow for STT configuration."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle new subentry creation."""
        self.options = {
            CONF_STT_MODEL: DEFAULT_STT_MODEL,
        }
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the shared form step."""
        if user_input is not None:
            self.options.update(user_input)
            title = self.options.pop("name", "Cloudflare AI STT")
            if self._is_new:
                return self.async_create_entry(title=title, data=self.options)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=self.options,
                title=title,
            )

        stt_model_options = [
            SelectOptionDict(label=m.split("/")[-1], value=m) for m in STT_MODELS
        ]

        schema: dict[vol.Optional | vol.Required, Any] = {}
        if self._is_new:
            schema[vol.Optional("name", default="Cloudflare AI STT")] = str

        schema.update(
            {
                vol.Optional(
                    CONF_STT_MODEL,
                    default=self.options.get(CONF_STT_MODEL, DEFAULT_STT_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=stt_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


class CloudflareAITaskSubentryFlow(ConfigSubentryFlow):
    """Subentry flow for AI task configuration."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        return self.source == "user"

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle new subentry creation."""
        self.options = {
            CONF_CHAT_MODEL: DEFAULT_CHAT_MODEL,
            CONF_MAX_TOKENS: DEFAULT_MAX_TOKENS,
            CONF_TEMPERATURE: DEFAULT_TEMPERATURE,
            CONF_ENABLE_THINKING: DEFAULT_ENABLE_THINKING,
            CONF_IMAGE_MODEL: DEFAULT_IMAGE_MODEL,
        }
        return await self.async_step_init(user_input)

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle reconfiguration."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init(user_input)

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Handle the shared form step."""
        if user_input is not None:
            self.options.update(user_input)
            title = self.options.pop("name", "Cloudflare AI Task")
            if self._is_new:
                return self.async_create_entry(title=title, data=self.options)
            return self.async_update_and_abort(
                self._get_entry(),
                self._get_reconfigure_subentry(),
                data=self.options,
                title=title,
            )

        chat_model_options = [
            SelectOptionDict(label=m.split("/")[-1], value=m) for m in CHAT_MODELS
        ]
        image_model_options = [
            SelectOptionDict(label=m.split("/")[-1], value=m) for m in IMAGE_MODELS
        ]

        schema: dict[vol.Optional | vol.Required, Any] = {}
        if self._is_new:
            schema[vol.Optional("name", default="Cloudflare AI Task")] = str

        schema.update(
            {
                vol.Optional(
                    CONF_CHAT_MODEL,
                    default=self.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=chat_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(
                    CONF_IMAGE_MODEL,
                    default=self.options.get(CONF_IMAGE_MODEL, DEFAULT_IMAGE_MODEL),
                ): SelectSelector(
                    SelectSelectorConfig(
                        options=image_model_options,
                        mode=SelectSelectorMode.DROPDOWN,
                        custom_value=True,
                    )
                ),
                vol.Optional(
                    CONF_MAX_TOKENS,
                    default=self.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1, max=8192, step=1, mode=NumberSelectorMode.BOX
                    )
                ),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=self.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.0, max=2.0, step=0.1, mode=NumberSelectorMode.SLIDER
                    )
                ),
                vol.Optional(
                    CONF_ENABLE_THINKING,
                    default=self.options.get(
                        CONF_ENABLE_THINKING, DEFAULT_ENABLE_THINKING
                    ),
                ): bool,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )
