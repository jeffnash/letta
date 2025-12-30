"""
CLIProxyAPI Provider - OpenAI-compatible proxy for LLM calls only.
Embeddings are NOT supported - use OpenAI directly for embeddings.

This provider dynamically fetches available models from CLIProxyAPI,
including passthru routes and any other configured models.
"""
import os
import time
from typing import ClassVar, Literal

from pydantic import Field

from letta.log import get_logger
from letta.schemas.embedding_config import EmbeddingConfig
from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.base import Provider

logger = get_logger(__name__)

# Default values for models that don't provide metadata
DEFAULT_CONTEXT_WINDOW = 128000
DEFAULT_MAX_TOKENS = 32000

# Cache TTL in seconds (5 minutes)
MODEL_CACHE_TTL_SECONDS = 300


class CLIProxyProvider(Provider):
    """
    CLIProxyAPI Provider - an OpenAI-compatible API proxy for LLM calls.
    
    This provider dynamically fetches available models from CLIProxyAPI,
    including passthru routes and any other configured models.
    Embeddings should be configured separately using the standard OpenAI provider.
    
    Environment variables:
        CLIPROXY_API_KEY: API key for your CLIProxyAPI instance  
        CLIPROXY_BASE_URL: Base URL for your CLIProxyAPI instance
    """
    
    # Class-level cache for model data (shared across instances with same base_url)
    _model_cache: ClassVar[dict[str, tuple[list[dict], float]]] = {}
    
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
    
    def _get_cache_key(self) -> str:
        """Generate a cache key based on base_url."""
        base = (self.base_url or "").strip()
        return base if base else "default"
    
    def _get_cached_models(self) -> list[dict] | None:
        """Get cached models if still valid."""
        cache_key = self._get_cache_key()
        if cache_key in self._model_cache:
            models, timestamp = self._model_cache[cache_key]
            if time.time() - timestamp < MODEL_CACHE_TTL_SECONDS:
                return models
        return None
    
    def _set_cached_models(self, models: list[dict]):
        """Cache the model list."""
        cache_key = self._get_cache_key()
        self._model_cache[cache_key] = (models, time.time())
    
    async def _get_models_async(self) -> list[dict]:
        """Fetch available models from CLIProxyAPI with caching."""
        from letta.llm_api.openai import openai_get_model_list_async
        
        # Check cache first
        cached = self._get_cached_models()
        if cached is not None:
            return cached
        
        api_key = await self.api_key_enc.get_plaintext_async() if self.api_key_enc else self.api_key
        
        try:
            response = await openai_get_model_list_async(self.base_url, api_key=api_key)
            data = response.get("data", response)
            if isinstance(data, list):
                self._set_cached_models(data)
                return data
        except Exception as e:
            logger.warning(f"Failed to fetch models from CLIProxyAPI: {e}")
            # Return stale cache if available
            cache_key = self._get_cache_key()
            if cache_key in self._model_cache:
                models, _ = self._model_cache[cache_key]
                logger.info("Using stale cached models from CLIProxyAPI")
                return models
        
        # No cache, no API response - return empty list
        return []
    
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
        """Convert model data to LLMConfig objects using dynamic metadata from API."""
        configs = []
        
        for model in data:
            if "id" not in model:
                continue
            
            model_name = model["id"]
            
            # Get context window from API response, with sensible defaults
            # CLIProxyAPI returns: context_window, context_length, max_completion_tokens, max_tokens
            context_window = (
                model.get("context_window") or 
                model.get("context_length") or 
                DEFAULT_CONTEXT_WINDOW
            )
            max_tokens = (
                model.get("max_completion_tokens") or 
                model.get("max_tokens") or 
                DEFAULT_MAX_TOKENS
            )
            
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
        """Get context window size for a model.
        
        For dynamic models, returns default. The actual value is set
        when the model is fetched via list_llm_models_async().
        """
        return DEFAULT_CONTEXT_WINDOW
    
    def get_model_context_window(self, model_name: str) -> int | None:
        return self.get_model_context_window_size(model_name)
    
    async def get_model_context_window_async(self, model_name: str) -> int | None:
        """Async version that can fetch from API if needed."""
        # Try to get from cached models first
        cached = self._get_cached_models()
        if cached:
            for model in cached:
                if model.get("id") == model_name:
                    return (
                        model.get("context_window") or 
                        model.get("context_length") or 
                        DEFAULT_CONTEXT_WINDOW
                    )
        
        # Fetch fresh if not cached
        models = await self._get_models_async()
        for model in models:
            if model.get("id") == model_name:
                return (
                    model.get("context_window") or 
                    model.get("context_length") or 
                    DEFAULT_CONTEXT_WINDOW
                )
        
        return DEFAULT_CONTEXT_WINDOW
