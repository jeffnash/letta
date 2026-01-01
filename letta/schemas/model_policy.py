"""
Model Policy Schema

Defines server-side model groups and selector resolution for subagent model selection.
This enables a unified mechanism where subagents specify ordered "model selector" lists
that the server resolves to concrete, available model handles.

Example selector:
    ["group:fast", "inherit", "any"]

Example policy:
    {
        "default_model": "copilot/gpt-5-mini",
        "groups": {
            "fast": ["copilot/gpt-5-mini", "openai/gpt-4.1-mini"],
            "strong": ["openai/gpt-5.2", "openai/gpt-4.1"],
            "planning": ["openai/gpt-5.2", "group:strong", "group:fast"],
            "default": ["group:strong", "group:fast"]
        }
    }
"""

from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field

from letta.schemas.letta_base import LettaBase


class ModelPolicy(LettaBase):
    """
    Server-defined model groups and default model.
    
    Groups are ordered lists of model handles or references to other groups.
    The server uses these to resolve model selectors to concrete handles.
    """
    
    default_model: str = Field(
        ...,
        description="Default model handle to use when 'any' is specified or as final fallback"
    )
    groups: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of group name to ordered list of handles/group references"
    )


class ModelSelectorRequest(BaseModel):
    """
    Request to resolve a model selector to a concrete handle.
    
    The selector is an ordered list where each entry can be:
    - group:X - Reference to a server-defined group
    - inherit - Use the parent model handle
    - any - Use the server default model
    - <handle> - A concrete model handle (e.g., "openai/gpt-5.2")
    """
    
    selector: List[str] = Field(
        ...,
        description="Ordered list of selector entries to resolve",
        examples=[["group:fast", "inherit", "any"]]
    )
    parent_model_handle: Optional[str] = Field(
        None,
        description="Parent agent's model handle, used to resolve 'inherit'"
    )


class ModelSelectorResponse(BaseModel):
    """
    Response containing the resolved model handle.
    """
    
    resolved_handle: str = Field(
        ...,
        description="The concrete model handle that was resolved"
    )
    expansion_chain: List[str] = Field(
        default_factory=list,
        description="The fully expanded selector chain (for debugging)"
    )


def expand_selector(
    selector: List[str],
    groups: Dict[str, List[str]],
    parent_model_handle: Optional[str],
    default_model: str,
    visited: Optional[Set[str]] = None,
) -> List[str]:
    """
    Expand a model selector into a flat list of concrete handles.
    
    Args:
        selector: List of selector entries (group:X, inherit, any, or handles)
        groups: Server-defined group mappings
        parent_model_handle: Parent agent's model handle for 'inherit'
        default_model: Server default model for 'any'
        visited: Set of visited groups (for cycle detection)
    
    Returns:
        Flattened list of concrete model handles
    
    Raises:
        ValueError: If a circular group reference is detected
    """
    if visited is None:
        visited = set()
    
    result: List[str] = []
    
    for entry in selector:
        entry = entry.strip()
        
        if entry.startswith("group:"):
            group_name = entry[6:]  # Remove "group:" prefix
            
            # Cycle detection
            if group_name in visited:
                raise ValueError(f"Circular group reference detected: {group_name}")
            
            if group_name in groups:
                # Create a branch-local visited set to avoid mutating siblings
                new_visited = visited | {group_name}
                # Recursively expand the group
                expanded = expand_selector(
                    groups[group_name],
                    groups,
                    parent_model_handle,
                    default_model,
                    new_visited,
                )
                result.extend(expanded)
            # If group doesn't exist, skip it silently
            
        elif entry == "inherit":
            if parent_model_handle:
                result.append(parent_model_handle)
            # If no parent, skip
            
        elif entry == "any":
            result.append(default_model)
            
        else:
            # Concrete handle
            result.append(entry)
    
    return result


def resolve_selector(
    selector: List[str],
    available_handles: Set[str],
    groups: Dict[str, List[str]],
    parent_model_handle: Optional[str],
    default_model: str,
) -> ModelSelectorResponse:
    """
    Resolve a model selector to the first available handle.
    
    Args:
        selector: List of selector entries
        available_handles: Set of currently available model handles
        groups: Server-defined group mappings
        parent_model_handle: Parent agent's model handle
        default_model: Server default model
    
    Returns:
        ModelSelectorResponse with resolved handle and expansion chain
    
    Raises:
        ValueError: If no available model is found in the expanded chain
    """
    # Expand the selector to a flat list
    expansion_chain = expand_selector(
        selector,
        groups,
        parent_model_handle,
        default_model,
    )
    
    # Ensure default_model is always a final fallback candidate
    if default_model and default_model not in expansion_chain:
        expansion_chain.append(default_model)
    
    # Find first available
    for handle in expansion_chain:
        if handle in available_handles:
            return ModelSelectorResponse(
                resolved_handle=handle,
                expansion_chain=expansion_chain,
            )
    
    # No available model found
    available_sample = list(available_handles)[:10]
    raise ValueError(
        f"No available model found in selector chain. "
        f"Expanded chain: {expansion_chain}. "
        f"Available models (sample): {available_sample}"
    )
