# Architecture Documentation

## Project Overview
A sports betting application built with a modern Python stack, focusing on performance, type safety, and maintainability.

## Tech Stack
- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) - High-performance web framework.
- **Validation:** [Pydantic v2](https://docs.pydantic.dev/latest/) - Data validation and settings management.
- **Logging:** [Loguru](https://github.com/Delgan/loguru) - Structured logging.
- **Database:** [SQLite](https://www.sqlite.org/index.html) - Lightweight relational database.
- **ORM:** [SQLAlchemy](https://www.sqlalchemy.org/) - SQL Toolkit and Object-Relational Mapper.

## Directory Layout
```text
sports-betting-app/
├── app/
│   ├── api/                # API route handlers
│   │   └── routes/
│   ├── core/               # Configuration and logging setup
│   ├── db/                 # Database connection and session management
│   ├── models/             # Data models
│   │   ├── domain/         # Internal domain models (SQLAlchemy)
│   │   └── schemas/        # API request/response models (Pydantic)
│   ├── services/           # Business logic
│   └── main.py             # Application entry point
├── docs/                   # Documentation
├── tests/                  # Automated tests
├── .env                    # Environment variables
├── .gitignore
└── requirements.txt
```

## Data Models

### User
- `id`: UUID (Primary Key)
- `username`: String (Unique)
- `email`: String (Unique)
- `hashed_password`: String
- `balance`: Decimal
- `is_active`: Boolean

### Bet
- `id`: UUID (Primary Key)
- `user_id`: UUID (Foreign Key)
- `event_id`: UUID (Foreign Key)
- `amount`: Decimal
- `odds`: Decimal
- `selection`: String (e.g., "Home", "Away", "Draw")
- `status`: Enum (Pending, Won, Lost, Cancelled)
- `created_at`: DateTime

### Event
- `id`: UUID (Primary Key)
- `name`: String
- `sport`: String
- `start_time`: DateTime
- `status`: Enum (Upcoming, Live, Finished, Cancelled)
- `result`: String (Optional)
