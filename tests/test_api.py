import os
os.environ.setdefault("HUGGINGFACE_API_KEY", "test")

from fastapi.testclient import TestClient
from src.backend.fastapi_logic import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_query_without_session_is_400():
    r = client.post("/query", json={"query": "hi", "session_id": "nope"})
    assert r.status_code == 400


def test_assurance_without_session_is_400():
    r = client.post("/assurance", json={"session_id": "nope"})
    assert r.status_code == 400
