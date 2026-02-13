from typing import TYPE_CHECKING, Optional

from fastapi import APIRouter, Depends, Query

from letta.server.rest_api.dependencies import get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer


router = APIRouter(prefix="/providers", tags=["providers", "admin"])


@router.post("/sync-models", tags=["admin"], operation_id="admin_sync_provider_models")
async def sync_provider_models(
    provider_name: Optional[str] = Query(
        None,
        description="Optional provider name filter (e.g. 'cliproxy'). If omitted, syncs all persisted providers.",
    ),
    clear_cliproxy_cache: bool = Query(
        True,
        description="Clear in-memory CLIProxy model cache before sync when provider_name is 'cliproxy' or omitted.",
    ),
    server: "SyncServer" = Depends(get_letta_server),
):
    """
    Trigger provider model synchronization to the database without restarting the server.

    This invokes the same sync path used at startup, which updates existing model metadata
    (including `max_context_window`) and inserts/removes provider models as needed.
    """
    provider_filter = provider_name.strip() if provider_name else None

    cleared_cliproxy_cache_entries = 0
    should_clear_cliproxy_cache = clear_cliproxy_cache and (
        provider_filter is None or provider_filter.lower() == "cliproxy"
    )
    if should_clear_cliproxy_cache:
        from letta.schemas.providers.cliproxy import CLIProxyProvider

        cleared_cliproxy_cache_entries = len(CLIProxyProvider._model_cache)
        CLIProxyProvider._model_cache.clear()

    summary = await server._sync_provider_models_async(provider_name=provider_filter)
    return {
        "status": "ok",
        "provider_filter": provider_filter or "all",
        "cleared_cliproxy_cache_entries": cleared_cliproxy_cache_entries,
        "sync_summary": summary,
    }
