from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class Outcome(BaseModel):
    name: str
    price: float  # American or Decimal, client will handle

class Market(BaseModel):
    key: str
    outcomes: List[Outcome]

class Bookmaker(BaseModel):
    key: str
    title: str
    last_update: datetime
    markets: List[Market]

class OddsEvent(BaseModel):
    id: str
    sport_key: str
    commence_time: datetime
    home_team: str
    away_team: str
    bookmakers: List[Bookmaker]
