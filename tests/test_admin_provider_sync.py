from fastapi import FastAPI
from fastapi.testclient import TestClient

from letta.orm.errors import NoResultFound


def test_admin_provider_sync_route(monkeypatch):
    # Import router here so monkeypatching is isolated to this test.
    from letta.server.rest_api.routers.v1.admin_providers import router as admin_providers_router
    from letta.server.rest_api.dependencies import get_letta_server

    class DummyServer:
        def __init__(self):
            self.called = 0

        async def _sync_provider_models_async(self, *, provider_name=None):
            self.called += 1
            self.provider_name = provider_name
            return {
                "status": "ok",
                "provider_name": provider_name,
                "providers_attempted": [provider_name] if provider_name else [],
                "providers_synced": [provider_name] if provider_name else [],
                "llm_models_synced": 1,
                "embedding_models_synced": 0,
                "duration_ms": 1.0,
            }

    dummy = DummyServer()

    app = FastAPI()
    app.include_router(admin_providers_router, prefix="/v1/admin")
    app.dependency_overrides[get_letta_server] = lambda: dummy

    client = TestClient(app)
    resp = client.post("/v1/admin/providers/sync")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert dummy.called == 1
    assert dummy.provider_name is None

    resp = client.post("/v1/admin/providers/sync?provider_name=cliproxy")
    assert resp.status_code == 200
    assert dummy.called == 2
    assert dummy.provider_name == "cliproxy"
    assert resp.json()["provider_name"] == "cliproxy"


def test_admin_provider_sync_route_returns_404_when_provider_not_found():
    from letta.server.rest_api.routers.v1.admin_providers import router as admin_providers_router
    from letta.server.rest_api.dependencies import get_letta_server

    class DummyServer:
        async def _sync_provider_models_async(self, *, provider_name=None):
            raise NoResultFound(f"Provider '{provider_name}' not found in persisted providers")

    app = FastAPI()
    app.include_router(admin_providers_router, prefix="/v1/admin")
    app.dependency_overrides[get_letta_server] = lambda: DummyServer()

    client = TestClient(app)
    resp = client.post("/v1/admin/providers/sync?provider_name=does-not-exist")

    assert resp.status_code == 404
    assert "does-not-exist" in resp.json()["detail"]
