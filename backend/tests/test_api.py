from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_schema() -> None:
    payload = {"messages": [{"role": "user", "content": "Hiring a Java developer"}]}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "reply" in body
    assert "recommendations" in body
    assert "end_of_conversation" in body
    assert isinstance(body["recommendations"], list)
