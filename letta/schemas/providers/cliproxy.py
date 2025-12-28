"""
CLIProxyAPI Provider - OpenAI-compatible proxy for LLM calls only.
Embeddings are NOT supported - use OpenAI directly for embeddings.
"""
import os
from typing import Literal

from pydantic import Field

from letta.constants import LLM_MAX_CONTEXT_WINDOW
from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider

logger = get_logger(__name__)

# Models available through CLIProxyAPI
CLIPROXY_MODELS = {
    # GPT-5.2 variants
    "gpt-5.2-xhigh": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.2-high": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.2-medium": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.2-low": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.2-minimal": {"context_window": 272000, "max_tokens": 128000},
    # GPT-5.1 Codex variants
    "gpt-5.1-codex-max-xhigh": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-max-high": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-max-medium": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-max-low": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-high": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-medium": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-low": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-mini-medium": {"context_window": 272000, "max_tokens": 128000},
    "gpt-5.1-codex-mini-high": {"context_window": 272000, "max_tokens": 128000},
    # GPT-5 Codex
    "gpt-5-codex": {"context_window": 272000, "max_tokens": 128000},
    # Gemini via proxy
    "gemini-3-pro-preview": {"context_window": 180000, "max_tokens": 64000},
    "gemini-3-flash-preview": {"context_window": 180000, "max_tokens": 64000},
    "gemini-2.5-pro": {"context_window": 180000, "max_tokens": 64000},
    "gemini-2.5-flash": {"context_window": 180000, "max_tokens": 64000},
    # Copilot proxied models
    "copilot-claude-sonnet-4.5": {"context_window": 200000, "max_tokens": 64000},
    "copilot-claude-opus-4.5": {"context_window": 200000, "max_tokens": 64000},
    "copilot-claude-haiku-4.5": {"context_window": 200000, "max_tokens": 64000},
    "copilot-gpt-5.1-codex": {"context_window": 272000, "max_tokens": 128000},
    "copilot-gpt-5.1": {"context_window": 272000, "max_tokens": 128000},
    "copilot-gpt-5": {"context_window": 272000, "max_tokens": 128000},
    "copilot-gpt-5-codex": {"context_window": 272000, "max_tokens": 128000},
    "copilot-gpt-4.1": {"context_window": 1047576, "max_tokens": 32768},
    "copilot-gemini-3-pro-preview": {"context_window": 180000, "max_tokens": 64000},
    # Qwen
    "qwen3-coder-plus": {"context_window": 128000, "max_tokens": 32768},
    "qwen3-coder-flash": {"context_window": 128000, "max_tokens": 32768},
}


class CLIProxyProvider(Provider):
    """
    CLIProxyAPI Provider - an OpenAI-compatible API proxy for LLM calls.
    
    This provider handles LLM inference through your CLIProxyAPI instance.
    Embeddings should be configured separately using the standard OpenAI provider.
    
    Environment variables:
        CLIPROXY_API_KEY: API key for your CLIProxyAPI instance  
        CLIPROXY_BASE_URL: Base URL for your CLIProxyAPI instance
    """
    
    # Use cliproxy provider type to ensure we use our own credentials
    provider_type: Literal[ProviderType.cliproxy] = Field(
        ProviderType.cliproxy, 
        description="The type of the provider (CLIProxy - OpenAI-compatible API)."
    )
    provider_category: ProviderCategory = Field(
        ProviderCategory.base, 
        description="The category of the provider"
    )
    api_key: str | None = Field(
        default_factory=lambda: os.environ.get("CLIPROXY_API_KEY"),
        description="API key for CLIProxyAPI"
    )
    base_url: str = Field(
        default_factory=lambda: os.environ.get("CLIPROXY_BASE_URL"),
        description="Base URL for CLIProxyAPI"
    )
    
    async def check_api_key(self):
        """Validate the API key by attempting to list models."""
        from letta.llm_api.openai import openai_get_model_list_async
        
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else self.api_key
        try:
            await openai_get_model_list_async(self.base_url, api_key=api_key)
        except Exception as e:
            logger.warning(f"CLIProxyAPI key check failed (may be fine if no auth required): {e}")
    
    async def _get_models_async(self) -> list[dict]:
        """Fetch available models from CLIProxyAPI."""
        from letta.llm_api.openai import openai_get_model_list_async
        
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else self.api_key
        
        try:
            response = await openai_get_model_list_async(self.base_url, api_key=api_key)
            data = response.get("data", response)
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning(f"Failed to fetch models from CLIProxyAPI, using hardcoded list: {e}")
        
        # Fallback: return our known models
        return [{"id": model} for model in CLIPROXY_MODELS.keys()]
    
    async def list_llm_models_async(self) -> list[LLMConfig]:
        """List available LLM models from CLIProxyAPI."""
        data = await self._get_models_async()
        return self._list_llm_models(data)
    
    async def list_embedding_models_async(self) -> list[EmbeddingConfig]:
        """
        CLIProxyAPI does NOT support embeddings.
        Return empty list - embeddings should use the standard OpenAI provider.
        """
        return []
    
    def _list_llm_models(self, data: list[dict]) -> list[LLMConfig]:
        """Convert model data to LLMConfig objects."""
        configs = []
        
        for model in data:
            if "id" not in model:
                continue
            
            model_name = model["id"]
            
            # Get context window from our known models or use defaults
            model_info = CLIPROXY_MODELS.get(model_name, {})
            context_window = model_info.get("context_window", 128000)
            max_tokens = model_info.get("max_tokens", 16384)
            
            config = LLMConfig(
                model=model_name,
                model_endpoint_type="openai",  # OpenAI-compatible API
                model_endpoint=self.base_url,
                context_window=context_window,
                handle=self.get_handle(model_name, base_name="cliproxy"),
                max_tokens=max_tokens,
                provider_name=self.name,
                provider_category=self.provider_category,
            )
            
            configs.append(config)
        
        return configs
    
    def get_model_context_window_size(self, model_name: str) -> int | None:
        """Get context window size for a model."""
        if model_name in CLIPROXY_MODELS:
            return CLIPROXY_MODELS[model_name]["context_window"]
        return 128000  # Default
    
    def get_model_context_window(self, model_name: str) -> int | None:
        return self.get_model_context_window_size(model_name)
    
    async def get_model_context_window_async(self, model_name: str) -> int | None:
        return self.get_model_context_window_size(model_name)
