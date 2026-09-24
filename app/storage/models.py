from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Float, DateTime, Boolean, Integer, Text, Index, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass

class DetectedEdge(Base):
    __tablename__ = "detected_edges"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    canonical_game_id: Mapped[str] = mapped_column(String(128), index=True)
    sport: Mapped[str] = mapped_column(String(64))
    market_type: Mapped[str] = mapped_column(String(64))
    bookmaker_name: Mapped[str] = mapped_column(String(128))
    outcome_name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    odds_offered: Mapped[float] = mapped_column(Float)
    fair_odds: Mapped[float] = mapped_column(Float)
    calculated_ev: Mapped[float] = mapped_column(Float)
    timestamp_detected: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    closing_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clv_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    __table_args__ = (
        Index("ix_game_market_book", "canonical_game_id", "market_type", "bookmaker_name"),
    )

class CanonicalTeam(Base):
    __tablename__ = "canonical_teams"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    aliases_json: Mapped[str] = mapped_column(Text, default="[]")

class CanonicalGame(Base):
    __tablename__ = "canonical_games"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    home_team_id: Mapped[str] = mapped_column(String(128))
    away_team_id: Mapped[str] = mapped_column(String(128))
    sport: Mapped[str] = mapped_column(String(64))
    start_time: Mapped[datetime] = mapped_column(DateTime)

# ---------------------------------------------------------------------------
# Kalshi-specific ingestion & mispricing tables (MySQL & SQLite compatible)
# ---------------------------------------------------------------------------

class KalshiMarket(Base):
    __tablename__ = "kalshi_markets"

    ticker: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_ticker: Mapped[str] = mapped_column(String(128), index=True)
    title: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), index=True)
    open_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    close_time: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    tick_size_cents: Mapped[int] = mapped_column(Integer, default=1)
    last_price_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    volume: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)

class KalshiOrderbookSnapshot(Base):
    __tablename__ = "kalshi_orderbook_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    best_bid_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    best_ask_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    spread_cents: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mid_price_cents: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    micro_price_cents: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_crossed: Mapped[bool] = mapped_column(Boolean, default=False)
    bid_depth_total: Mapped[int] = mapped_column(Integer, default=0)
    ask_depth_total: Mapped[int] = mapped_column(Integer, default=0)

class KalshiOrderbookLevel(Base):
    __tablename__ = "kalshi_orderbook_levels"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    snapshot_id: Mapped[int] = mapped_column(Integer, index=True)
    ticker: Mapped[str] = mapped_column(String(128), index=True)
    side: Mapped[str] = mapped_column(String(8))  # 'bid' or 'ask'
    level_index: Mapped[int] = mapped_column(Integer)
    price_cents: Mapped[int] = mapped_column(Integer)
    quantity: Mapped[int] = mapped_column(Integer)

    __table_args__ = (
        Index("ix_snapshot_side_level", "snapshot_id", "side", "level_index"),
    )

class KalshiMispricing(Base):
    __tablename__ = "kalshi_mispricings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=utc_now, index=True)
    mispricing_type: Mapped[str] = mapped_column(String(64))  # 'single_contract_depth', 'basket_dutch_book', 'basket_overpriced'
    fair_prob: Mapped[float] = mapped_column(Float)
    depth_evaluated: Mapped[int] = mapped_column(Integer)
    effective_price_cents: Mapped[float] = mapped_column(Float)
    fee_cents: Mapped[float] = mapped_column(Float)
    gross_edge_cents: Mapped[float] = mapped_column(Float)
    net_edge_cents: Mapped[float] = mapped_column(Float)
    ev_pct: Mapped[float] = mapped_column(Float)
    details_json: Mapped[str] = mapped_column(Text, default="{}")
