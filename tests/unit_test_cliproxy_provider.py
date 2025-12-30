"""
Unit tests for CLIProxyProvider dynamic model fetching.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from letta.schemas.providers.cliproxy import (
    CLIProxyProvider,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_TOKENS,
)


class TestCLIProxyProvider:
    """Tests for CLIProxyProvider dynamic model handling."""

    @pytest.fixture
    def provider(self):
        """Create a CLIProxyProvider instance for testing."""
        return CLIProxyProvider(
            name="cliproxy-test",
            base_url="http://localhost:8317",
            api_key="test-key",
        )

    @pytest.fixture
    def sample_models_response(self):
        """Sample response from CLIProxyAPI /v1/models endpoint."""
        return [
            {
                "id": "gpt-5.2-medium",
                "object": "model",
                "owned_by": "openai",
                "context_window": 272000,
                "max_tokens": 128000,
            },
            {
                "id": "zai-glm-4.7",
                "object": "model",
                "owned_by": "passthru",
                "context_window": 128000,
                "max_tokens": 32000,
            },
            {
                "id": "model-without-metadata",
                "object": "model",
                "owned_by": "unknown",
                # No context_window or max_tokens
            },
        ]

    def test_list_llm_models_uses_dynamic_context_window(self, provider, sample_models_response):
        """Test that _list_llm_models uses context_window from API response."""
        configs = provider._list_llm_models(sample_models_response)

        assert len(configs) == 3

        # Check model with explicit context_window
        gpt_model = next(c for c in configs if c.model == "gpt-5.2-medium")
        assert gpt_model.context_window == 272000
        assert gpt_model.max_tokens == 128000

        # Check passthru model
        passthru_model = next(c for c in configs if c.model == "zai-glm-4.7")
        assert passthru_model.context_window == 128000
        assert passthru_model.max_tokens == 32000

    def test_list_llm_models_uses_defaults_when_missing(self, provider, sample_models_response):
        """Test that _list_llm_models uses defaults when metadata is missing."""
        configs = provider._list_llm_models(sample_models_response)

        # Check model without metadata uses defaults
        unknown_model = next(c for c in configs if c.model == "model-without-metadata")
        assert unknown_model.context_window == DEFAULT_CONTEXT_WINDOW
        assert unknown_model.max_tokens == DEFAULT_MAX_TOKENS

    def test_list_llm_models_handles_context_length_alias(self, provider):
        """Test that _list_llm_models handles context_length as alias for context_window."""
        models = [
            {
                "id": "model-with-context-length",
                "object": "model",
                "owned_by": "test",
                "context_length": 200000,  # Using context_length instead of context_window
                "max_completion_tokens": 50000,  # Using max_completion_tokens instead of max_tokens
            }
        ]

        configs = provider._list_llm_models(models)
        assert len(configs) == 1
        assert configs[0].context_window == 200000
        assert configs[0].max_tokens == 50000

    def test_list_llm_models_generates_correct_handles(self, provider, sample_models_response):
        """Test that _list_llm_models generates correct handles."""
        configs = provider._list_llm_models(sample_models_response)

        for config in configs:
            assert config.handle.startswith("cliproxy/")
            assert config.model in config.handle

    def test_list_llm_models_skips_models_without_id(self, provider):
        """Test that _list_llm_models skips models without id field."""
        models = [
            {"object": "model", "owned_by": "test"},  # No id
            {"id": "valid-model", "object": "model", "owned_by": "test"},
        ]

        configs = provider._list_llm_models(models)
        assert len(configs) == 1
        assert configs[0].model == "valid-model"

    def test_get_model_context_window_size_returns_default(self, provider):
        """Test that get_model_context_window_size returns default for unknown models."""
        result = provider.get_model_context_window_size("unknown-model")
        assert result == DEFAULT_CONTEXT_WINDOW

    def test_cache_key_uses_base_url(self, provider):
        """Test that cache key is based on base_url."""
        key = provider._get_cache_key()
        assert key == "http://localhost:8317"

    def test_cache_key_default_when_no_base_url(self):
        """Test that cache key defaults when no base_url."""
        provider = CLIProxyProvider(name="cliproxy-test", base_url="", api_key="test")
        key = provider._get_cache_key()
        assert key == "default"

    @pytest.mark.asyncio
    async def test_get_models_async_caches_results(self, provider, sample_models_response):
        """Test that _get_models_async caches results."""
        with patch("letta.llm_api.openai.openai_get_model_list_async") as mock_fetch:
            mock_fetch.return_value = {"data": sample_models_response}

            # First call should fetch from network
            result1 = await provider._get_models_async()
            assert len(result1) == 3
            assert mock_fetch.call_count == 1

            # Second call should use cache
            result2 = await provider._get_models_async()
            assert len(result2) == 3
            assert mock_fetch.call_count == 1  # No additional call

    @pytest.mark.asyncio
    async def test_get_models_async_returns_empty_on_failure(self, provider):
        """Test that _get_models_async returns empty list on API failure."""
        # Clear any cached data
        provider._model_cache.clear()

        with patch("letta.llm_api.openai.openai_get_model_list_async") as mock_fetch:
            mock_fetch.side_effect = Exception("API error")

            result = await provider._get_models_async()
            assert result == []

    @pytest.mark.asyncio
    async def test_get_model_context_window_async_fetches_from_api(self, provider, sample_models_response):
        """Test that get_model_context_window_async can fetch from API."""
        with patch("letta.llm_api.openai.openai_get_model_list_async") as mock_fetch:
            mock_fetch.return_value = {"data": sample_models_response}

            result = await provider.get_model_context_window_async("zai-glm-4.7")
            assert result == 128000

    @pytest.mark.asyncio
    async def test_get_model_context_window_async_returns_default_for_unknown(self, provider, sample_models_response):
        """Test that get_model_context_window_async returns default for unknown models."""
        with patch("letta.llm_api.openai.openai_get_model_list_async") as mock_fetch:
            mock_fetch.return_value = {"data": sample_models_response}

            result = await provider.get_model_context_window_async("unknown-model")
            assert result == DEFAULT_CONTEXT_WINDOW

    @pytest.mark.asyncio
    async def test_list_embedding_models_async_returns_empty(self, provider):
        """Test that list_embedding_models_async returns empty list (not supported)."""
        result = await provider.list_embedding_models_async()
        assert result == []


class TestCLIProxyProviderPassthruModels:
    """Tests specifically for passthru model handling."""

    @pytest.fixture
    def provider(self):
        return CLIProxyProvider(
            name="cliproxy-test",
            base_url="http://localhost:8317",
            api_key="test-key",
        )

    def test_passthru_model_with_full_metadata(self, provider):
        """Test passthru model with all metadata fields."""
        models = [
            {
                "id": "zai-glm-4.7",
                "object": "model",
                "owned_by": "passthru",
                "type": "claude",
                "display_name": "zai-glm-4.7",
                "context_window": 128000,
                "max_tokens": 32000,
            }
        ]

        configs = provider._list_llm_models(models)
        assert len(configs) == 1

        config = configs[0]
        assert config.model == "zai-glm-4.7"
        assert config.context_window == 128000
        assert config.max_tokens == 32000
        assert "cliproxy" in config.handle

    def test_passthru_model_uses_context_length_fallback(self, provider):
        """Test passthru model using context_length instead of context_window."""
        models = [
            {
                "id": "custom-passthru",
                "object": "model",
                "owned_by": "passthru",
                "context_length": 256000,  # Using context_length
                "max_completion_tokens": 64000,  # Using max_completion_tokens
            }
        ]

        configs = provider._list_llm_models(models)
        assert len(configs) == 1

        config = configs[0]
        assert config.context_window == 256000
        assert config.max_tokens == 64000

    def test_multiple_passthru_models(self, provider):
        """Test handling multiple passthru models."""
        models = [
            {
                "id": "passthru-model-1",
                "object": "model",
                "owned_by": "passthru",
                "context_window": 100000,
                "max_tokens": 20000,
            },
            {
                "id": "passthru-model-2",
                "object": "model",
                "owned_by": "passthru",
                "context_window": 200000,
                "max_tokens": 40000,
            },
        ]

        configs = provider._list_llm_models(models)
        assert len(configs) == 2

        model1 = next(c for c in configs if c.model == "passthru-model-1")
        model2 = next(c for c in configs if c.model == "passthru-model-2")

        assert model1.context_window == 100000
        assert model2.context_window == 200000
