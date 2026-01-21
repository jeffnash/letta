"""
Unit tests for HeaderParams and related REST API dependencies.

These tests are pure unit tests that don't require database fixtures.
"""

import pytest

from letta.server.rest_api.dependencies import (
    CANCELLATION_CHECK_INTERVAL_DEFAULT,
    CANCELLATION_CHECK_INTERVAL_INTERACTIVE,
    LETTA_SOURCE_INTERACTIVE,
    HeaderParams,
)


class TestHeaderParamsConstants:
    """Test that constants are defined with expected values."""

    def test_interactive_interval_is_faster_than_default(self):
        """Interactive clients should have faster polling for better responsiveness."""
        assert CANCELLATION_CHECK_INTERVAL_INTERACTIVE < CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_interactive_interval_value(self):
        """Test the specific value for interactive interval."""
        assert CANCELLATION_CHECK_INTERVAL_INTERACTIVE == 0.1

    def test_default_interval_value(self):
        """Test the specific value for default interval."""
        assert CANCELLATION_CHECK_INTERVAL_DEFAULT == 0.5

    def test_letta_source_interactive_value(self):
        """Test the specific value for letta-code client identifier."""
        assert LETTA_SOURCE_INTERACTIVE == "letta-code"


class TestHeaderParamsGetCancellationCheckInterval:
    """Unit tests for HeaderParams.get_cancellation_check_interval method."""

    def test_returns_interactive_interval_for_letta_code_source(self):
        """Test that letta-code clients get the faster interactive interval."""
        headers = HeaderParams(letta_source="letta-code")
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_INTERACTIVE
        assert interval == 0.1

    def test_returns_default_interval_for_none_source(self):
        """Test that None letta_source gets the default interval."""
        headers = HeaderParams(letta_source=None)
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_DEFAULT
        assert interval == 0.5

    def test_returns_default_interval_for_empty_string_source(self):
        """Test that empty string letta_source gets the default interval."""
        headers = HeaderParams(letta_source="")
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_returns_default_interval_for_unknown_source(self):
        """Test that unknown client sources get the default interval."""
        headers = HeaderParams(letta_source="some-other-client")
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_returns_default_interval_for_sdk_client(self):
        """Test that SDK clients (not letta-code) get the default interval."""
        headers = HeaderParams(letta_source="letta-sdk")
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_case_sensitive_matching(self):
        """Test that source matching is case-sensitive."""
        # "Letta-Code" with different case should not match
        headers_upper = HeaderParams(letta_source="Letta-Code")
        headers_mixed = HeaderParams(letta_source="LETTA-CODE")
        
        assert headers_upper.get_cancellation_check_interval() == CANCELLATION_CHECK_INTERVAL_DEFAULT
        assert headers_mixed.get_cancellation_check_interval() == CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_exact_match_required(self):
        """Test that exact match is required (no partial matching)."""
        headers_prefix = HeaderParams(letta_source="letta-code-v2")
        headers_suffix = HeaderParams(letta_source="my-letta-code")
        headers_contains = HeaderParams(letta_source="foo-letta-code-bar")
        
        assert headers_prefix.get_cancellation_check_interval() == CANCELLATION_CHECK_INTERVAL_DEFAULT
        assert headers_suffix.get_cancellation_check_interval() == CANCELLATION_CHECK_INTERVAL_DEFAULT
        assert headers_contains.get_cancellation_check_interval() == CANCELLATION_CHECK_INTERVAL_DEFAULT

    def test_uses_constant_for_comparison(self):
        """Test that the method uses the LETTA_SOURCE_INTERACTIVE constant."""
        # This verifies the implementation uses the constant, not a hardcoded string
        headers = HeaderParams(letta_source=LETTA_SOURCE_INTERACTIVE)
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_INTERACTIVE


class TestHeaderParamsInitialization:
    """Test HeaderParams initialization with various field combinations."""

    def test_default_initialization(self):
        """Test HeaderParams with default values."""
        headers = HeaderParams()
        
        assert headers.actor_id is None
        assert headers.user_agent is None
        assert headers.project_id is None
        assert headers.letta_source is None
        assert headers.sdk_version is None
        assert headers.experimental_params is None

    def test_initialization_with_letta_source(self):
        """Test HeaderParams initialization with letta_source."""
        headers = HeaderParams(letta_source="letta-code")
        
        assert headers.letta_source == "letta-code"

    def test_initialization_with_all_fields(self):
        """Test HeaderParams initialization with all fields populated."""
        headers = HeaderParams(
            actor_id="user-123",
            user_agent="Mozilla/5.0",
            project_id="project-456",
            letta_source="letta-code",
            sdk_version="1.0.0",
        )
        
        assert headers.actor_id == "user-123"
        assert headers.user_agent == "Mozilla/5.0"
        assert headers.project_id == "project-456"
        assert headers.letta_source == "letta-code"
        assert headers.sdk_version == "1.0.0"

    def test_get_cancellation_check_interval_with_other_fields_populated(self):
        """Test that other fields don't affect the cancellation interval logic."""
        headers = HeaderParams(
            actor_id="user-123",
            user_agent="Mozilla/5.0",
            project_id="project-456",
            letta_source="letta-code",
            sdk_version="1.0.0",
        )
        
        # Should still return interactive interval based on letta_source
        interval = headers.get_cancellation_check_interval()
        
        assert interval == CANCELLATION_CHECK_INTERVAL_INTERACTIVE


class TestCancellationIntervalReturnType:
    """Test that get_cancellation_check_interval returns the correct type."""

    def test_returns_float(self):
        """Test that the method returns a float."""
        headers = HeaderParams(letta_source="letta-code")
        
        interval = headers.get_cancellation_check_interval()
        
        assert isinstance(interval, float)

    def test_interactive_interval_is_positive(self):
        """Test that interactive interval is a positive number."""
        headers = HeaderParams(letta_source="letta-code")
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval > 0

    def test_default_interval_is_positive(self):
        """Test that default interval is a positive number."""
        headers = HeaderParams(letta_source=None)
        
        interval = headers.get_cancellation_check_interval()
        
        assert interval > 0
