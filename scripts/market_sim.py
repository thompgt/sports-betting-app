"""
A synthetic multi-bookmaker market, used by the demo notebook and by
scripts/seed_demo_db.py.

Nothing here is a real odds feed. It produces plausible-looking prices --
several books quoting both sides of a moneyline, each with its own margin and
its own pricing noise -- so the detection pipeline has something to chew on.
Treat every number it emits as made up.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

BOOKMAKERS = [
    # (key, title, price bias) -- bias shifts a book's view of the home side.
    # A negative bias means the book quotes the home side at a lower implied
    # probability (i.e. longer odds), which is what can create detectable +EV.
    ("pinnacle", "Pinnacle", 0.000),
    ("draftkings", "DraftKings", 0.004),
    ("fanduel", "FanDuel", -0.006),
    ("betmgm", "BetMGM", 0.014),
    ("caesars", "Caesars", -0.012),
    ("betrivers", "BetRivers", -0.010),
]

# Book-level overround (vig). Sharp, low-margin books run tighter.
BOOK_VIG = {
    "Pinnacle": 1.014,
    "DraftKings": 1.042,
    "FanDuel": 1.045,
    "BetMGM": 1.050,
    "Caesars": 1.020,
    "BetRivers": 1.018,
}

# Per-poll pricing noise (std-dev, in probability units). Wider dispersion
# between books is what an edge detector is looking for in the first place.
PRICE_NOISE_SD = 0.020


def entity_uuid(n: int) -> str:
    return f"{n:08d}-0000-4000-8000-{n:012d}"


def build_universe(now: datetime):
    """Canonical teams/games plus the market parameters used to simulate prices."""
    teams = [
        ("New York Rangers", ["NY Rangers", "Rangers"]),
        ("Los Angeles Kings", ["LA Kings", "Kings"]),
        ("New York Knicks", ["NY Knicks", "Knicks"]),
        ("Boston Celtics", ["Celtics", "BOS Celtics"]),
        ("Kansas City Chiefs", ["KC Chiefs", "Chiefs"]),
        ("Buffalo Bills", ["Bills", "BUF Bills"]),
    ]
    team_records = [
        {"id": entity_uuid(i + 1), "name": name, "aliases": aliases}
        for i, (name, aliases) in enumerate(teams)
    ]
    by_name = {t["name"]: t["id"] for t in team_records}

    games = [
        {
            "id": entity_uuid(101),
            "home": "New York Rangers",
            "away": "Los Angeles Kings",
            # the fixture deliberately uses aliases to exercise the resolver
            "home_feed_name": "NY Rangers",
            "away_feed_name": "LA Kings",
            "sport": "icehockey_nhl",
            "start_time": now + timedelta(hours=4),
            "true_home_prob": 0.56,
        },
        {
            "id": entity_uuid(102),
            "home": "New York Knicks",
            "away": "Boston Celtics",
            "home_feed_name": "Knicks",
            "away_feed_name": "Boston Celtics",
            "sport": "basketball_nba",
            "start_time": now + timedelta(hours=7),
            "true_home_prob": 0.41,
        },
        {
            "id": entity_uuid(103),
            "home": "Kansas City Chiefs",
            "away": "Buffalo Bills",
            "home_feed_name": "KC Chiefs",
            "away_feed_name": "Bills",
            "sport": "americanfootball_nfl",
            # tips off shortly; the closing snapshot below backdates it so the
            # EdgeAuditor closes this market out and records CLV
            "start_time": now + timedelta(minutes=20),
            "true_home_prob": 0.52,
        },
    ]

    seed_payload = {
        "teams": team_records,
        "games": [
            {
                "id": g["id"],
                "home_team_id": by_name[g["home"]],
                "away_team_id": by_name[g["away"]],
                "sport": g["sport"],
                "start_time": g["start_time"].replace(microsecond=0).isoformat(),
            }
            for g in games
        ],
    }
    return seed_payload, games


def prob_to_american(prob: float) -> int:
    """Convert an implied probability to the nearest American price."""
    prob = min(max(prob, 0.02), 0.98)
    decimal = 1.0 / prob
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))


def build_cycle_fixture(games, cycle: int, sim_time: datetime, rng: random.Random):
    """One polling snapshot: every book quotes both sides of every h2h market."""
    events = []
    for idx, game in enumerate(games):
        # the "true" price drifts slowly over the session (a random walk)
        drift = rng.gauss(0, 0.006) * cycle
        true_home = min(max(game["true_home_prob"] + drift, 0.15), 0.85)

        bookmakers = []
        for key, title, bias in BOOKMAKERS:
            noise = rng.gauss(0, PRICE_NOISE_SD)
            book_home = min(max(true_home + bias + noise, 0.10), 0.90)
            book_away = 1.0 - book_home

            vig = BOOK_VIG[title]
            quoted_home = book_home * vig
            quoted_away = book_away * vig

            bookmakers.append({
                "key": key,
                "title": title,
                "last_update": (sim_time - timedelta(seconds=rng.randint(5, 90))).isoformat(),
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": game["home_feed_name"], "price": prob_to_american(quoted_home)},
                        {"name": game["away_feed_name"], "price": prob_to_american(quoted_away)},
                    ],
                }],
            })

        events.append({
            "id": f"sim_{idx:03d}_{cycle:03d}",
            "sport_key": game["sport"],
            "commence_time": game["start_time"].replace(microsecond=0).isoformat(),
            "home_team": game["home_feed_name"],
            "away_team": game["away_feed_name"],
            "bookmakers": bookmakers,
        })
    return events
