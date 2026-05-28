import pytest
from uuid import uuid4
from datetime import datetime, timedelta
from app.engine.resolver import EntityResolver, TeamCanonical, GameCanonical

@pytest.fixture
def resolver_data():
    # Canonical Teams
    rangers_id = uuid4()
    knicks_id = uuid4()
    lakers_id = uuid4()
    
    teams = [
        TeamCanonical(id=rangers_id, name="New York Rangers", aliases=["NY Rangers", "Rangers"]),
        TeamCanonical(id=knicks_id, name="New York Knicks", aliases=["NY Knicks"]),
        TeamCanonical(id=lakers_id, name="Los Angeles Lakers", aliases=["LA Lakers", "Lakers"]),
    ]
    
    # Canonical Games
    game_id = uuid4()
    base_time = datetime(2026, 5, 27, 20, 0, 0)
    
    games = [
        GameCanonical(
            id=game_id, 
            home_team_id=rangers_id, 
            away_team_id=lakers_id, 
            start_time=base_time
        )
    ]
    
    return EntityResolver(teams, games), rangers_id, knicks_id, lakers_id, game_id, base_time

def test_resolve_team_exact(resolver_data):
    resolver, rangers_id, _, _, _, _ = resolver_data
    assert resolver.resolve_team("New York Rangers") == rangers_id
    assert resolver.resolve_team("NY Rangers") == rangers_id
    assert resolver.resolve_team("rangers") == rangers_id

def test_resolve_team_fuzzy(resolver_data):
    resolver, rangers_id, _, _, _, _ = resolver_data
    # Fuzzy match with slight typo
    assert resolver.resolve_team("New York Rangerz") == rangers_id
    # Case and spacing
    assert resolver.resolve_team("  ny   RANGERS  ") == rangers_id

def test_resolve_team_no_cross_resolve(resolver_data):
    resolver, rangers_id, knicks_id, _, _, _ = resolver_data
    # NY Rangers should NOT resolve to NY Knicks even though they are similar
    # token_sort_ratio handles this well by comparing tokens
    assert resolver.resolve_team("NY Rangers") != knicks_id
    assert resolver.resolve_team("NY Knicks") != rangers_id

def test_resolve_game_success(resolver_data):
    resolver, _, _, _, game_id, base_time = resolver_data
    # Match within 6-hour window
    assert resolver.resolve_game("NY Rangers", "LA Lakers", base_time + timedelta(hours=2)) == game_id
    assert resolver.resolve_game("Rangers", "Lakers", base_time - timedelta(hours=5)) == game_id

def test_resolve_game_time_window_failure(resolver_data):
    resolver, _, _, _, _, base_time = resolver_data
    # Outside 6-hour window (7 hours)
    assert resolver.resolve_game("NY Rangers", "LA Lakers", base_time + timedelta(hours=7)) is None

def test_resolve_game_team_failure(resolver_data):
    resolver, _, _, _, _, base_time = resolver_data
    # One team doesn't exist
    assert resolver.resolve_game("NY Rangers", "Mars Martians", base_time) is None
    # Wrong away team
    assert resolver.resolve_game("NY Rangers", "NY Knicks", base_time) is None

def test_resolve_team_low_confidence_failure(resolver_data):
    resolver, _, _, _, _, _ = resolver_data
    # Completely different name
    assert resolver.resolve_team("Boston Celtics") is None
