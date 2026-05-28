import pytest
import asyncio
from app.ingestion.mock_provider import MockOddsClient
from app.models.schemas.odds import OddsEvent

@pytest.mark.asyncio
async def test_mock_odds_client_streaming():
    fixture_path = "tests/fixtures/sample_odds_payload.json"
    client = MockOddsClient(fixture_path)
    
    events = []
    async for event in client.stream_live_odds(polling_interval=0.1):
        events.append(event)
        
    assert len(events) == 2
    assert isinstance(events[0], OddsEvent)
    assert events[0].id == "event_001"
    assert events[0].home_team == "New York Rangers"
    assert len(events[0].bookmakers) == 3

def test_mock_odds_client_load_data():
    fixture_path = "tests/fixtures/sample_odds_payload.json"
    client = MockOddsClient(fixture_path)
    data = client.load_data()
    assert len(data) == 2
    assert data[1].id == "event_002"
