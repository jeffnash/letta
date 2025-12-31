"""
CLIProxyAPI Provider - OpenAI-compatible proxy for LLM calls only.
Embeddings are NOT supported - use OpenAI directly for embeddings.

This provider dynamically fetches available models from CLIProxyAPI,
including passthru routes and any other configured models.
"""
import math
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

# Apply a safety buffer to avoid hitting provider hard limits exactly.
# Can be configured via env vars to avoid double-buffering when upstream already buffers.
CONTEXT_WINDOW_SAFETY_FACTOR = float(os.environ.get("LETTA_CLIPROXY_CONTEXT_WINDOW_SAFETY_FACTOR", "0.95"))
MAX_TOKENS_SAFETY_FACTOR = float(os.environ.get("LETTA_CLIPROXY_MAX_TOKENS_SAFETY_FACTOR", "0.95"))

def _apply_safety_limit(value: int, safety_factor: float) -> int:
    if not isinstance(value, int):
        value = int(value)
    if not math.isfinite(safety_factor) or safety_factor <= 0 or safety_factor > 1:
        safety_factor = 1.0
    return max(1, int(math.floor(value * safety_factor)))


def _models_dev_fallback_limits(model_id: str, owned_by: str) -> tuple[int, int]:
    """Fallback limits when CLIProxyAPI /v1/models omits metadata.

    Strategy:
    1) Prefer an explicit mapping based on CLIProxyAPI's owned_by, since it indicates the
       underlying provider family (e.g. openai, copilot, antigravity, passthru).
    2) Fall back to a conservative default when unknown.

    Notes:
    - This is intentionally defensive: even if some providers/models are missing here,
      the result should never be *larger* than reality.
    - Once CLIProxyAPI consistently returns context_length/max_completion_tokens for every
      model (including passthru), this function becomes mostly unused.
    """

    owned_by_norm = (owned_by or "").strip().lower()
    model_norm = (model_id or "").strip()

    # Passthru models should ideally include explicit limits in the /v1/models payload.
    # If they don't, default conservatively.
    if owned_by_norm == "passthru":
        return DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

    # Antigravity (Gemini/Vertex-mediated routes) frequently includes Anthropic models.
    # These ids match CLIProxyAPI's /v1/models entries (no provider prefix).
    if owned_by_norm == "antigravity":
        # Vertex Anthropic via Gemini / thinking
        if model_norm in {
            "gemini-claude-opus-4-5-thinking",
            "gemini-claude-sonnet-4-5-thinking",
        }:
            return 200000, 64000
        # Non-thinking Sonnet/Opus via Gemini can still have large context.
        if model_norm in {
            "gemini-claude-opus-4-5",
            "gemini-claude-sonnet-4-5",
        }:
            return 200000, 64000
        # Gemini-first models (when present under antigravity)
        if model_norm == "gemini-3-pro-preview":
            return 1000000, 64000
        if model_norm in {"gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"}:
            return 1048576, 65536

    # OpenAI (direct)
    if owned_by_norm == "openai":
        if model_norm.startswith("gpt-5"):
            # GPT-5.x and Codex variants on OpenAI are 400k/128k on models.dev
            return 400000, 128000
        if model_norm.startswith("gpt-4.1"):
            return 1047576, 32768
        if model_norm.startswith("gpt-4o"):
            return 128000, 16384

    # GitHub Copilot
    if owned_by_norm == "copilot":
        if model_norm in {
            "gpt-5",
            "gpt-5.1",
            "gpt-5-codex",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "copilot-gpt-5",
            "copilot-gpt-5.1",
            "copilot-gpt-5-codex",
            "copilot-gpt-5.1-codex",
            "copilot-gpt-5.1-codex-max",
        }:
            return 128000, 128000
        if model_norm in {"gpt-5.2", "copilot-gpt-5.2"}:
            return 128000, 64000
        if model_norm in {"gpt-5-mini", "copilot-gpt-5-mini"}:
            return 128000, 64000
        if model_norm in {"gpt-4.1", "copilot-gpt-4.1"}:
            return 128000, 16384
        if model_norm in {"gpt-4o", "copilot-gpt-4o"}:
            return 64000, 16384
        if model_norm.startswith("gemini-3") or model_norm == "gemini-2.5-pro":
            return 128000, 64000
        if model_norm == "gemini-2.0-flash-001":
            return 1000000, 8192

    # Google (AI Studio)
    if owned_by_norm == "google":
        if model_norm == "gemini-3-pro-preview":
            return 1000000, 64000
        if model_norm in {"gemini-3-flash-preview", "gemini-2.5-pro", "gemini-2.5-flash"}:
            return 1048576, 65536

    # Catch-all: unknown owner/provider.
    return DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_TOKENS

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
            # Prefer metadata from /v1/models when provided.
            # NOTE: some CLIProxyAPI deployments historically returned "thin" model objects
            # (id/object/created/owned_by only). In that case, fall back to models.dev-derived
            # defaults based on owned_by + model id.
            raw_context_window = model.get("context_window") or model.get("context_length")
            raw_max_tokens = (
                model.get("max_completion_tokens")
                or model.get("max_output_tokens")
                or model.get("outputTokenLimit")
                or model.get("output_token_limit")
                or model.get("max_tokens")
            )

            if raw_context_window is None or raw_max_tokens is None:
                fallback_ctx, fallback_out = _models_dev_fallback_limits(
                    model_id=str(model_name),
                    owned_by=str(model.get("owned_by") or ""),
                )
                raw_context_window = raw_context_window or fallback_ctx
                raw_max_tokens = raw_max_tokens or fallback_out

            raw_context_window = raw_context_window or DEFAULT_CONTEXT_WINDOW
            raw_max_tokens = raw_max_tokens or DEFAULT_MAX_TOKENS

            context_window = _apply_safety_limit(
                int(raw_context_window),
                CONTEXT_WINDOW_SAFETY_FACTOR,
            )
            max_tokens = _apply_safety_limit(int(raw_max_tokens), MAX_TOKENS_SAFETY_FACTOR)
            
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
        return _apply_safety_limit(DEFAULT_CONTEXT_WINDOW, CONTEXT_WINDOW_SAFETY_FACTOR)
    
    def get_model_context_window(self, model_name: str) -> int | None:
        return self.get_model_context_window_size(model_name)
    
    async def get_model_context_window_async(self, model_name: str) -> int | None:
        """Async version that can fetch from API if needed."""
        # Try to get from cached models first
        cached = self._get_cached_models()
        if cached:
            for model in cached:
                if model.get("id") == model_name:
                    raw = (
                        model.get("context_window")
                        or model.get("context_length")
                        or DEFAULT_CONTEXT_WINDOW
                    )
                    return _apply_safety_limit(int(raw), CONTEXT_WINDOW_SAFETY_FACTOR)
        
        # Fetch fresh if not cached
        models = await self._get_models_async()
        for model in models:
            if model.get("id") == model_name:
                raw = (
                    model.get("context_window")
                    or model.get("context_length")
                    or DEFAULT_CONTEXT_WINDOW
                )
                return _apply_safety_limit(int(raw), CONTEXT_WINDOW_SAFETY_FACTOR)
        
        return DEFAULT_CONTEXT_WINDOW
