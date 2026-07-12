import json
from datetime import datetime
from uuid import UUID
from sqlalchemy.orm import Session
from app.storage.models import CanonicalTeam, CanonicalGame
from app.engine.resolver import EntityResolver, TeamCanonical, GameCanonical

def seed_if_empty(session: Session, seed_path: str) -> None:
    """Loads canonical teams/games from a seed JSON file into the DB if the tables are empty."""
    if session.query(CanonicalTeam).first() is not None:
        return

    with open(seed_path, "r") as f:
        data = json.load(f)

    for team in data.get("teams", []):
        session.add(CanonicalTeam(
            id=team["id"],
            name=team["name"],
            aliases_json=json.dumps(team.get("aliases", []))
        ))

    for game in data.get("games", []):
        session.add(CanonicalGame(
            id=game["id"],
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            sport=game.get("sport", ""),
            start_time=datetime.fromisoformat(game["start_time"])
        ))

    session.commit()

def load_resolver(session: Session) -> EntityResolver:
    """Builds an EntityResolver from the canonical teams/games stored in the DB."""
    teams = [
        TeamCanonical(
            id=UUID(team.id),
            name=team.name,
            aliases=json.loads(team.aliases_json)
        )
        for team in session.query(CanonicalTeam).all()
    ]
    games = [
        GameCanonical(
            id=UUID(game.id),
            home_team_id=UUID(game.home_team_id),
            away_team_id=UUID(game.away_team_id),
            start_time=game.start_time
        )
        for game in session.query(CanonicalGame).all()
    ]
    return EntityResolver(teams, games)
