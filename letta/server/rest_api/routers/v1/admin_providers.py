from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Query

from letta.orm.errors import NoResultFound
from letta.server.rest_api.dependencies import get_letta_server

if TYPE_CHECKING:
    from letta.server.server import SyncServer


router = APIRouter(prefix="/providers", tags=["admin", "providers"])


@router.post("/sync", operation_id="admin_sync_provider_models")
async def admin_sync_provider_models(
    server: "SyncServer" = Depends(get_letta_server),
    provider_name: str | None = Query(
        None,
        description="If set, only sync this provider name (e.g. 'cliproxy')",
    ),
):
    """Force a provider->model sync into the DB.

    This re-runs the same provider model sync performed on startup, which is useful for
    base providers (including CLIProxy) when upstream model metadata changes.
    """

    # NOTE: This uses the server's internal sync routine. It is intentionally exposed
    # via an admin route for operational use (e.g. on Railway) to avoid restarts.
    try:
        return await server._sync_provider_models_async(provider_name=provider_name)
    except NoResultFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
