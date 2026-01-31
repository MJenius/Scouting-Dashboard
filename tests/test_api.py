
from fastapi import status

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["status"] == "healthy"
    assert response.json()["database"] == "connected"

def test_get_players_empty(client):
    """Test getting players when DB is empty."""
    response = client.get("/players/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == []

def test_get_players_with_data(client, valid_player_payload):
    """Test getting players with data in DB."""
    response = client.get("/players/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Test Player"

def test_get_player_by_id(client, valid_player_payload):
    """Test getting a single player by ID."""
    player_id = valid_player_payload.id
    response = client.get(f"/players/{player_id}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["name"] == "Test Player"

def test_get_player_not_found(client):
    """Test getting a non-existent player."""
    response = client.get("/players/99999")
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_similarity_search(client, valid_player_payload):
    """Test the similarity endpoint (mocks engine)."""
    payload = {
        "player_id": valid_player_payload.id,
        "league": "all",
        "top_n": 5
    }
    response = client.post("/analysis/similarity", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["target"]["name"] == "Test Player"
    assert len(data["matches"]) > 0
    assert data["matches"][0]["name"] == "Sim Player"
