"""
Tests for model policy and selector resolution.

Tests the server-side model group definitions and selector resolution algorithm.
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from letta.schemas.model_policy import (
    ModelPolicy,
    ModelSelectorRequest,
    ModelSelectorResponse,
    expand_selector,
    resolve_selector,
)
from letta.server.rest_api.dependencies import HeaderParams
from letta.server.rest_api.routers.v1.llms import resolve_model_selector as resolve_model_selector_endpoint
from letta.server.server import SyncServer


class TestExpandSelector:
    """Tests for the expand_selector function."""

    def test_expand_concrete_handles(self):
        """Concrete handles are kept as-is."""
        selector = ["openai/gpt-5.2", "copilot/gpt-5-mini"]
        groups = {}
        
        result = expand_selector(selector, groups, None, "default/model")
        
        assert result == ["openai/gpt-5.2", "copilot/gpt-5-mini"]

    def test_expand_inherit_with_parent(self):
        """'inherit' expands to parent model handle when provided."""
        selector = ["inherit"]
        groups = {}
        
        result = expand_selector(selector, groups, "parent/model", "default/model")
        
        assert result == ["parent/model"]

    def test_expand_inherit_without_parent(self):
        """'inherit' is skipped when no parent model is provided."""
        selector = ["inherit", "openai/gpt-5.2"]
        groups = {}
        
        result = expand_selector(selector, groups, None, "default/model")
        
        assert result == ["openai/gpt-5.2"]

    def test_expand_any(self):
        """'any' expands to the default model."""
        selector = ["any"]
        groups = {}
        
        result = expand_selector(selector, groups, None, "default/model")
        
        assert result == ["default/model"]

    def test_expand_group(self):
        """'group:X' expands to the group's contents."""
        selector = ["group:fast"]
        groups = {
            "fast": ["copilot/gpt-5-mini", "openai/gpt-4.1-mini"],
        }
        
        result = expand_selector(selector, groups, None, "default/model")
        
        assert result == ["copilot/gpt-5-mini", "openai/gpt-4.1-mini"]

    def test_expand_nested_groups(self):
        """Groups can reference other groups."""
        selector = ["group:default"]
        groups = {
            "fast": ["copilot/gpt-5-mini"],
            "strong": ["openai/gpt-5.2"],
            "default": ["group:strong", "group:fast"],
        }
        
        result = expand_selector(selector, groups, None, "fallback/model")
        
        assert result == ["openai/gpt-5.2", "copilot/gpt-5-mini"]

    def test_expand_repeated_group_references_do_not_trip_cycle_detection(self):
        """Repeated group references across siblings don't falsely raise circular errors."""
        selector = ["group:a", "group:a"]
        groups = {
            "a": ["group:b"],
            "b": ["copilot/gpt-5-mini"],
        }

        result = expand_selector(selector, groups, None, "fallback/model")

        assert result == ["copilot/gpt-5-mini", "copilot/gpt-5-mini"]

    def test_expand_circular_group_detection(self):
        """Circular group references raise ValueError."""
        selector = ["group:a"]
        groups = {
            "a": ["group:b"],
            "b": ["group:a"],  # Circular!
        }
        
        with pytest.raises(ValueError, match="Circular group reference"):
            expand_selector(selector, groups, None, "default/model")

    def test_expand_unknown_group_skipped(self):
        """Unknown groups are silently skipped."""
        selector = ["group:unknown", "openai/gpt-5.2"]
        groups = {}
        
        result = expand_selector(selector, groups, None, "default/model")
        
        assert result == ["openai/gpt-5.2"]

    def test_expand_mixed_selector(self):
        """Mixed selectors with groups, inherit, any, and handles."""
        selector = ["group:fast", "inherit", "openai/gpt-5.2", "any"]
        groups = {
            "fast": ["copilot/gpt-5-mini"],
        }
        
        result = expand_selector(
            selector, groups, "parent/model", "default/model"
        )
        
        assert result == [
            "copilot/gpt-5-mini",
            "parent/model",
            "openai/gpt-5.2",
            "default/model",
        ]


class TestResolveSelector:
    """Tests for the resolve_selector function."""

    def test_resolve_first_available(self):
        """Returns the first available handle in the expanded chain."""
        selector = ["openai/gpt-5.2", "copilot/gpt-5-mini"]
        available = {"copilot/gpt-5-mini", "openai/gpt-4.1"}
        groups = {}
        
        response = resolve_selector(
            selector, available, groups, None, "default/model"
        )
        
        # gpt-5.2 not available, so falls back to gpt-5-mini
        assert response.resolved_handle == "copilot/gpt-5-mini"
        assert response.expansion_chain == [
            "openai/gpt-5.2",
            "copilot/gpt-5-mini",
            "default/model",
        ]

    def test_resolve_with_group(self):
        """Resolves through group expansion."""
        selector = ["group:fast"]
        available = {"openai/gpt-4.1-mini"}
        groups = {
            "fast": ["copilot/gpt-5-mini", "openai/gpt-4.1-mini"],
        }
        
        response = resolve_selector(
            selector, available, groups, None, "default/model"
        )
        
        # First in group not available, second is
        assert response.resolved_handle == "openai/gpt-4.1-mini"

    def test_resolve_with_inherit(self):
        """Resolves 'inherit' to parent model."""
        selector = ["group:fast", "inherit"]
        available = {"parent/model"}
        groups = {
            "fast": ["copilot/gpt-5-mini"],
        }
        
        response = resolve_selector(
            selector, available, groups, "parent/model", "default/model"
        )
        
        assert response.resolved_handle == "parent/model"

    def test_resolve_with_any_fallback(self):
        """Falls back to 'any' (default model) when nothing else available."""
        selector = ["group:fast", "inherit", "any"]
        available = {"default/model"}
        groups = {
            "fast": ["copilot/gpt-5-mini"],
        }
        
        response = resolve_selector(
            selector, available, groups, None, "default/model"
        )
        
        assert response.resolved_handle == "default/model"

    def test_resolve_appends_default_model_when_missing(self):
        """Always appends default_model as a final fallback if not already present."""
        selector = ["openai/gpt-5.2"]
        available = {"fallback/model"}
        groups = {}

        response = resolve_selector(
            selector, available, groups, None, "fallback/model"
        )

        assert response.resolved_handle == "fallback/model"
        assert response.expansion_chain == ["openai/gpt-5.2", "fallback/model"]

    def test_resolve_no_available_raises(self):
        """Raises ValueError when no model in chain is available."""
        selector = ["openai/gpt-5.2", "copilot/gpt-5-mini"]
        available = {"completely/different-model"}
        groups = {}
        
        with pytest.raises(ValueError, match="No available model found"):
            resolve_selector(selector, available, groups, None, "also/unavailable")

    def test_resolve_empty_selector_raises(self):
        """Empty selector with no matches raises ValueError."""
        selector = []
        available = {"some/model"}
        groups = {}
        
        with pytest.raises(ValueError, match="No available model found"):
            resolve_selector(selector, available, groups, None, "unavailable/default")


class TestModelPolicy:
    """Tests for ModelPolicy schema."""

    def test_model_policy_creation(self):
        """ModelPolicy can be created with groups."""
        policy = ModelPolicy(
            default_model="copilot/gpt-5-mini",
            groups={
                "fast": ["copilot/gpt-5-mini", "openai/gpt-4.1-mini"],
                "strong": ["openai/gpt-5.2"],
            },
        )
        
        assert policy.default_model == "copilot/gpt-5-mini"
        assert "fast" in policy.groups
        assert "strong" in policy.groups

    def test_model_policy_empty_groups(self):
        """ModelPolicy works with empty groups."""
        policy = ModelPolicy(
            default_model="fallback/model",
            groups={},
        )
        
        assert policy.default_model == "fallback/model"
        assert policy.groups == {}


class TestModelSelectorRequest:
    """Tests for ModelSelectorRequest schema."""

    def test_request_with_selector(self):
        """Request can be created with selector list."""
        request = ModelSelectorRequest(
            selector=["group:fast", "inherit", "any"],
            parent_model_handle="parent/model",
        )
        
        assert request.selector == ["group:fast", "inherit", "any"]
        assert request.parent_model_handle == "parent/model"

    def test_request_without_parent(self):
        """Request works without parent model handle."""
        request = ModelSelectorRequest(
            selector=["group:fast", "any"],
        )
        
        assert request.selector == ["group:fast", "any"]
        assert request.parent_model_handle is None


class TestModelSelectorResponse:
    """Tests for ModelSelectorResponse schema."""

    def test_response_creation(self):
        """Response can be created with resolved handle and chain."""
        response = ModelSelectorResponse(
            resolved_handle="openai/gpt-5.2",
            expansion_chain=["group:strong", "openai/gpt-5.2"],
        )
        
        assert response.resolved_handle == "openai/gpt-5.2"
        assert response.expansion_chain == ["group:strong", "openai/gpt-5.2"]


class TestDefaultModelSelection:
    """Tests for default model selection in server."""

    def test_default_model_from_metadata(self):
        """Prefers explicit default metadata over heuristics."""
        server = SyncServer.__new__(SyncServer)
        models = [
            SimpleNamespace(handle="openai/gpt-5.2"),
            SimpleNamespace(handle="openai/gpt-4.1", metadata={"default": True}),
        ]
        available_handles = {m.handle for m in models if m.handle}

        result = server._get_default_model(available_handles, models)

        assert result == "openai/gpt-4.1"

    def test_default_model_from_is_default_flag(self):
        """Prefers explicit is_default flag over heuristics."""
        server = SyncServer.__new__(SyncServer)
        models = [
            SimpleNamespace(handle="openai/gpt-5.2"),
            SimpleNamespace(handle="openai/gpt-4.1", is_default=True),
        ]
        available_handles = {m.handle for m in models if m.handle}

        result = server._get_default_model(available_handles, models)

        assert result == "openai/gpt-4.1"


class TestResolveModelSelectorAsync:
    """Tests for resolve_model_selector_async behavior."""

    @pytest.mark.asyncio
    async def test_resolve_selector_single_fetch(self):
        """Ensure model list is fetched only once per resolve call."""
        server = SyncServer.__new__(SyncServer)
        calls = {"count": 0}

        async def list_llm_models_async(actor):
            calls["count"] += 1
            return [SimpleNamespace(handle="openai/gpt-5.2")]

        server.list_llm_models_async = list_llm_models_async

        result = await server.resolve_model_selector_async(
            selector=["group:strong", "any"],
            parent_model_handle=None,
            actor=object(),
        )

        assert result.resolved_handle == "openai/gpt-5.2"
        assert calls["count"] == 1


class DummyUserManager:
    async def get_actor_or_default_async(self, actor_id=None):
        return object()


class DummyServer:
    def __init__(self, error: Exception):
        self.user_manager = DummyUserManager()
        self._error = error

    async def resolve_model_selector_async(self, selector, parent_model_handle, actor):
        raise self._error


class TestResolveModelSelectorEndpoint:
    """Tests for the resolve_model_selector endpoint error handling."""

    @pytest.mark.asyncio
    async def test_value_error_returns_422(self):
        """ValueError from resolver is converted to HTTP 422."""
        server = DummyServer(ValueError("No available model found"))
        request = ModelSelectorRequest(selector=["group:fast"], parent_model_handle=None)
        headers = HeaderParams(actor_id="user-1")

        with pytest.raises(HTTPException) as exc_info:
            await resolve_model_selector_endpoint(
                request=request,
                server=server,
                headers=headers,
            )

        assert exc_info.value.status_code == 422
        assert "No available model found" in str(exc_info.value.detail)
