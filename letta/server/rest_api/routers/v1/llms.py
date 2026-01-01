from typing import TYPE_CHECKING, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.model import EmbeddingModel, Model
from letta.schemas.model_policy import ModelPolicy, ModelSelectorRequest, ModelSelectorResponse
from letta.server.rest_api.dependencies import HeaderParams, get_headers, get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer

router = APIRouter(prefix="/models", tags=["models", "llms"])


@router.get("/", response_model=List[Model], operation_id="list_models")
async def list_llm_models(
    provider_category: Optional[List[ProviderCategory]] = Query(None),
    provider_name: Optional[str] = Query(None),
    provider_type: Optional[ProviderType] = Query(None),
    server: "SyncServer" = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    List available LLM models using the asynchronous implementation for improved performance.

    Returns Model format which extends LLMConfig with additional metadata fields.
    Legacy LLMConfig fields are marked as deprecated but still available for backward compatibility.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)

    models = await server.list_llm_models_async(
        provider_category=provider_category,
        provider_name=provider_name,
        provider_type=provider_type,
        actor=actor,
    )

    # Convert all models to the new Model schema
    return [Model.from_llm_config(model) for model in models]


@router.get("/embedding", response_model=List[EmbeddingModel], operation_id="list_embedding_models")
async def list_embedding_models(
    server: "SyncServer" = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    List available embedding models using the asynchronous implementation for improved performance.

    Returns EmbeddingModel format which extends EmbeddingConfig with additional metadata fields.
    Legacy EmbeddingConfig fields are marked as deprecated but still available for backward compatibility.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    models = await server.list_embedding_models_async(actor=actor)

    # Convert all models to the new EmbeddingModel schema
    return [EmbeddingModel.from_embedding_config(model) for model in models]


@router.get("/policy", response_model=ModelPolicy, operation_id="get_model_policy")
async def get_model_policy(
    server: "SyncServer" = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Get server-defined model groups and default model.

    Returns a policy object containing:
    - default_model: The default model handle used for 'any' selector
    - groups: Map of group names to ordered lists of model handles/group references

    Groups can be used in model selectors like "group:fast" or "group:strong".
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    return await server.get_model_policy_async(actor=actor)


@router.post("/resolve", response_model=ModelSelectorResponse, operation_id="resolve_model_selector")
async def resolve_model_selector(
    request: ModelSelectorRequest,
    server: "SyncServer" = Depends(get_letta_server),
    headers: HeaderParams = Depends(get_headers),
):
    """
    Resolve a model selector to a concrete, available model handle.

    The selector is an ordered list where each entry can be:
    - group:X - Reference to a server-defined group (e.g., "group:fast")
    - inherit - Use the parent model handle (requires parent_model_handle)
    - any - Use the server default model
    - <handle> - A concrete model handle (e.g., "openai/gpt-5.2")

    The server expands all groups, flattens the list, and returns the first
    model handle that is currently available.
    """
    actor = await server.user_manager.get_actor_or_default_async(actor_id=headers.actor_id)
    try:
        return await server.resolve_model_selector_async(
            selector=request.selector,
            parent_model_handle=request.parent_model_handle,
            actor=actor,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
