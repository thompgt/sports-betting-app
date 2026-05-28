import pytest
import math
from uuid import uuid4
from datetime import datetime
from app.engine.math_utils import american_to_decimal, strip_vig_power_method, decimal_to_implied_prob
from app.engine.resolver import EntityResolver, TeamCanonical

def test_extreme_odds():
    # Longshot: +1,000,000
    longshot_decimal = american_to_decimal(1000000)
    assert longshot_decimal == 10001.0
    assert decimal_to_implied_prob(longshot_decimal) == 1/10001.0
    
    # Heavy Favorite: -1,000,000
    fav_decimal = american_to_decimal(-1000000)
    assert fav_decimal == 1.0001
    assert math.isclose(decimal_to_implied_prob(fav_decimal), 0.9999, rel_tol=1e-7)

def test_high_vig_multi_way():
    # Extreme vig (e.g., sum of implied probs = 1.5)
    probs = [0.6, 0.5, 0.4] # Sum = 1.5
    fair_probs = strip_vig_power_method(probs)
    assert math.isclose(sum(fair_probs), 1.0, abs_tol=1e-9)
    # Power method should still distribute vig fairly
    assert all(0 < p < 1 for p in fair_probs)

def test_resolver_diverse_naming():
    st_louis_id = uuid4()
    teams = [
        TeamCanonical(id=st_louis_id, name="St. Louis Cardinals", aliases=["Saint Louis Cardinals", "SL Cardinals"])
    ]
    resolver = EntityResolver(teams, [])
    
    # Test different punctuations and case
    assert resolver.resolve_team("st louis cardinals") == st_louis_id
    assert resolver.resolve_team("SAINT LOUIS CARDINALS") == st_louis_id
    assert resolver.resolve_team("St. Louis Cardinals") == st_louis_id

def test_resolver_soccer_names():
    fc_id = uuid4()
    teams = [
        TeamCanonical(id=fc_id, name="Liverpool FC", aliases=["Liverpool"])
    ]
    resolver = EntityResolver(teams, [])
    
    assert resolver.resolve_team("liverpool") == fc_id
    assert resolver.resolve_team("Liverpool F.C.") == fc_id # Fuzzy match should handle "."

def test_math_undervalued_market():
    # Case where bookie odds actually sum to < 1.0 (Arbitrage or Error)
    # Probs: 0.4, 0.4 -> Sum 0.8
    probs = [0.4, 0.4]
    fair_probs = strip_vig_power_method(probs)
    # Should normalize to 1.0 (0.5, 0.5)
    assert fair_probs == [0.5, 0.5]
