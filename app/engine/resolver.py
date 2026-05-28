from typing import List, Optional, Dict
from uuid import UUID
from datetime import datetime
import re
from pydantic import BaseModel, ConfigDict
from rapidfuzz import process, fuzz

class TeamCanonical(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    name: str
    aliases: List[str] = []

class GameCanonical(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: UUID
    home_team_id: UUID
    away_team_id: UUID
    start_time: datetime

class EntityResolver:
    def __init__(self, teams: List[TeamCanonical], games: List[GameCanonical]):
        self.teams = teams
        self.games = games
        self._exact_team_map: Dict[str, UUID] = {}
        self._team_names: List[str] = []
        self._initialize_indices()

    def _sanitize(self, name: str) -> str:
        """Sanitizes strings by lowercasing and removing extra whitespace."""
        return re.sub(r'\s+', ' ', name.strip().lower())

    def _initialize_indices(self) -> None:
        """Builds lookup indices for Tier 1 and Tier 2 matching."""
        for team in self.teams:
            sanitized_name = self._sanitize(team.name)
            self._exact_team_map[sanitized_name] = team.id
            if sanitized_name not in self._team_names:
                self._team_names.append(sanitized_name)
            
            for alias in team.aliases:
                sanitized_alias = self._sanitize(alias)
                self._exact_team_map[sanitized_alias] = team.id
                if sanitized_alias not in self._team_names:
                    self._team_names.append(sanitized_alias)

    def resolve_team(self, name: str, threshold: float = 85.0) -> Optional[UUID]:
        """
        Resolves a team name to its canonical UUID using a two-tier strategy.
        Tier 1: Exact hash map lookup.
        Tier 2: Fuzzy matching using token sort ratio.
        """
        if not name:
            return None
            
        sanitized = self._sanitize(name)
        
        # Tier 1: Exact Match
        if sanitized in self._exact_team_map:
            return self._exact_team_map[sanitized]
        
        # Tier 2: Fuzzy Match fallback
        if not self._team_names:
            return None
            
        match = process.extractOne(
            sanitized, 
            self._team_names, 
            scorer=fuzz.token_sort_ratio
        )
        
        if match and match[1] >= threshold:
            matched_name = match[0]
            return self._exact_team_map[matched_name]
            
        return None

    def resolve_game(self, home_name: str, away_name: str, event_time: datetime) -> Optional[UUID]:
        """
        Resolves a game to its canonical UUID.
        Requires both teams to resolve and the time window to be within 6 hours.
        """
        home_id = self.resolve_team(home_name)
        away_id = self.resolve_team(away_name)
        
        if not home_id or not away_id:
            return None
            
        for game in self.games:
            if game.home_team_id == home_id and game.away_team_id == away_id:
                # 6-hour window check (21600 seconds)
                time_diff = abs((game.start_time - event_time).total_seconds())
                if time_diff <= 21600:
                    return game.id
                    
        return None
