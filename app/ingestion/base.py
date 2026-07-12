from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.schemas.odds import OddsEvent

class OddsProvider(ABC):
    """
    Vendor-agnostic odds source. Concrete implementations (mock fixtures today,
    a real bookmaker/aggregator API later) adapt their own schema to OddsEvent.

    get_odds() is a pull/snapshot call - it returns the current board for the
    requested sports. The continuous detection loop owns polling cadence
    (sleeping between calls), not the provider, so providers stay simple and
    swappable.
    """

    @abstractmethod
    async def get_odds(self, sport_keys: Optional[List[str]] = None) -> List[OddsEvent]:
        """Return the current odds snapshot for the requested sports (or all configured sports)."""
        raise NotImplementedError

    async def close(self) -> None:
        """Optional cleanup hook (e.g. close HTTP sessions). No-op by default."""
        return None
