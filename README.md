# Sports Betting App

A high-performance sports betting analysis and execution engine built with Python 3.11+, FastAPI, and Pydantic v2.

## Purpose
This application is designed to ingest sports data from multiple sources, resolve entities (teams and games) to a canonical format, and perform advanced mathematical analysis to identify value bets using proven quantitative methods.

## Key Features
- **Entity Resolution Module:** Two-tier matching (Exact + Fuzzy) to link disparate data sources to a single source of truth.
- **Quantitative Engine:** Support for American/Decimal odds conversion, advanced devigging using the Power Method, EV calculation, and fractional Kelly Criterion staking.
- **Architectural Integrity:** Domain-driven design with clear separation between API schemas, domain models, and business logic.

## Tech Stack
- **Framework:** FastAPI
- **Validation:** Pydantic v2
- **Logic:** RapidFuzz, NumPy-ready math utilities
- **Database:** SQLite (planned)
- **Logging:** Loguru

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) or `pip`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/thompgt/sports-betting-app.git
   cd sports-betting-app
   ```
2. Install dependencies:
   ```bash
   pip install fastapi pydantic rapidfuzz loguru sqlalchemy pytest
   ```

### Running Tests
To ensure the engine is operating correctly:
```bash
$env:PYTHONPATH = "."; pytest
```

## Module Overview

### Entity Resolution (`app/engine/resolver.py`)
Handles the mapping of strings like "NY Rangers" and "New York Rangers" to a single UUID. It uses a 6-hour time window for game matching to ensure temporal accuracy across timezones.

### Mathematical Utilities (`app/engine/math_utils.py`)
Contains the quantitative core. The `strip_vig_power_method` is particularly robust, using an iterative Newton-Raphson solver to handle multi-way markets (e.g., Soccer 3-way or UFC) more accurately than simple multiplicative devigging.
