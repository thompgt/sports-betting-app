import json
import asyncio
from typing import AsyncGenerator, List, Optional
from pathlib import Path
from app.models.schemas.odds import OddsEvent
from app.ingestion.base import OddsProvider

class MockOddsClient(OddsProvider):
    def __init__(self, fixture_path: str):
        self.fixture_path = Path(fixture_path)
        if not self.fixture_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture_path}")

    def load_data(self) -> List[OddsEvent]:
        with open(self.fixture_path, "r") as f:
            data = json.load(f)
            return [OddsEvent(**event) for event in data]

    async def get_odds(self, sport_keys: Optional[List[str]] = None) -> List[OddsEvent]:
        events = self.load_data()
        if sport_keys:
            events = [e for e in events if e.sport_key in sport_keys]
        return events

    async def stream_live_odds(self, polling_interval: int = 1) -> AsyncGenerator[OddsEvent, None]:
        """Simulates real-time data streaming by yielding events sequentially."""
        events = self.load_data()
        for event in events:
            yield event
            await asyncio.sleep(polling_interval)
