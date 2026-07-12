from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LINEEDGE_", extra="ignore")

    db_url: str = "sqlite:///./lineedge.db"
    poll_interval_seconds: float = 30.0
    max_backoff_seconds: float = 300.0
    ev_threshold: float = 0.02
    edge_cache_ttl_minutes: int = 15
    edge_cache_spike_threshold: float = 0.02
    log_level: str = "INFO"
    log_dir: str = "logs"
    mock_fixture_path: str = "tests/fixtures/sample_odds_payload.json"
    canonical_seed_path: str = "app/ingestion/seed_data/canonical_entities.json"

def get_settings() -> Settings:
    return Settings()
