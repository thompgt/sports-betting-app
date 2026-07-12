# Architecture Documentation

## Project Overview
LineEdge is a continuously-running edge-detection service for sports betting markets, structured like an algorithmic trading bot's poll/decide loop: it repeatedly pulls odds, prices the "true" market, and surfaces positive-EV opportunities. It does **not** place bets — detection and alerting only.

## Tech Stack
- **Config:** [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - typed, env-var-driven settings.
- **Validation:** [Pydantic v2](https://docs.pydantic.dev/latest/) - data validation for odds payloads.
- **Database:** [SQLite](https://www.sqlite.org/index.html) via [SQLAlchemy](https://www.sqlalchemy.org/) 2.x.
- **Entity matching:** [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) for fuzzy team-name resolution.
- **Dashboard:** [Streamlit](https://streamlit.io/) + [Plotly](https://plotly.com/python/).
- **Logging:** stdlib `logging` with a rotating file handler.

There is no HTTP API layer, no user/auth model, and no bet-execution path — the system's output is rows in SQLite, read by the dashboard.

## Directory Layout
```text
sports-betting-app/
├── app/
│   ├── core/                # Settings (config.py) and logging setup (logging.py)
│   ├── engine/               # Pure math/matching logic: resolver.py, math_utils.py
│   ├── ingestion/             # OddsProvider interface (base.py) + concrete providers
│   │   └── seed_data/         # Canonical team/game seed data loaded into the DB on first run
│   ├── models/schemas/        # Pydantic schemas for odds payloads
│   ├── services/               # EdgeDetectionService: the continuous poll/detect/persist loop
│   ├── storage/                # SQLAlchemy models, DatabaseManager, repository (seed/load), EdgeAuditor (CLV)
│   ├── ui/                     # Streamlit dashboard
│   └── main.py                 # Entry point (long-running service, or --once for a single poll)
├── docs/
├── tests/
└── requirements.txt
```

## Data Flow
```text
[ OddsProvider.get_odds() ] --poll interval--> [ EntityResolver ] --> [ No-Vig Power Method devig ]
                                                                              |
                                                                              v
                                                                     [ EV calc + EdgeCache dedupe ]
                                                                              |
                                                                              v
                                                                    [ SQLite: detected_edges ]
                                                                       |                |
                                                                       v                v
                                                            [ Streamlit Dashboard ]  [ EdgeAuditor:
                                                                                       CLV closeout
                                                                                       once game starts ]
```

`EdgeDetectionService.run_forever()` (`app/services/detection_service.py`) is the core loop: on each cycle it calls `poll_once()`, which pulls a fresh odds snapshot, resolves/prices/detects edges per event, and closes out markets (via `EdgeAuditor`) for any canonical game whose start time has passed — recording the last-seen price as the closing line and computing CLV%. Errors during a cycle trigger exponential backoff (capped) rather than crashing the process, since this runs unattended.

## Key Modules
- **`app/engine/resolver.py`** — `EntityResolver`: two-tier team matching (exact hashmap, then rapidfuzz token-sort-ratio fallback), plus game resolution with a 6-hour start-time tolerance.
- **`app/engine/math_utils.py`** — odds conversion, `strip_vig_power_method` (Newton-Raphson devig), `calculate_ev`, `kelly_criterion`.
- **`app/ingestion/base.py`** — `OddsProvider` ABC (`get_odds()` snapshot pull). `app/ingestion/mock_provider.py::MockOddsClient` is the current implementation, backed by a JSON fixture; a real bookmaker/aggregator API can implement the same interface without touching the engine or service.
- **`app/storage/repository.py`** — seeds canonical teams/games into the DB from `app/ingestion/seed_data/canonical_entities.json` on first run, and loads them into an `EntityResolver` (replaces the old hardcoded-in-script approach).
- **`app/storage/auditor.py`** — `EdgeAuditor.close_out_market(game_id, market_type, bookmaker_name, closing_odds)`: marks matching active edges inactive, records the closing line, and computes `clv_pct`.
- **`app/core/config.py`** — `Settings` (env-prefixed `LINEEDGE_*`, `.env` supported): poll interval, EV threshold, DB URL, cache TTL, log level, etc.

## Running
```bash
$env:PYTHONPATH = "."; python app/main.py          # continuous service
$env:PYTHONPATH = "."; python app/main.py --once    # single poll cycle, for smoke tests/CI
streamlit run app/ui/dashboard.py                   # dashboard, reads the same DB
```
