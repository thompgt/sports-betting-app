from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import String, Float, DateTime, Boolean, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

class Base(DeclarativeBase):
    pass

class DetectedEdge(Base):
    __tablename__ = "detected_edges"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    canonical_game_id: Mapped[str] = mapped_column(String, index=True)
    sport: Mapped[str] = mapped_column(String)
    market_type: Mapped[str] = mapped_column(String)
    bookmaker_name: Mapped[str] = mapped_column(String)
    outcome_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    odds_offered: Mapped[float] = mapped_column(Float)
    fair_odds: Mapped[float] = mapped_column(Float)
    calculated_ev: Mapped[float] = mapped_column(Float)
    timestamp_detected: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_game_market_book", "canonical_game_id", "market_type", "bookmaker_name"),
    )

class CanonicalTeam(Base):
    __tablename__ = "canonical_teams"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    aliases_json: Mapped[str] = mapped_column(String, default="[]")

class CanonicalGame(Base):
    __tablename__ = "canonical_games"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    home_team_id: Mapped[str] = mapped_column(String)
    away_team_id: Mapped[str] = mapped_column(String)
    sport: Mapped[str] = mapped_column(String)
    start_time: Mapped[datetime] = mapped_column(DateTime)
