from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.storage.models import Base

class DatabaseManager:
    def __init__(self, db_url: str = "sqlite:///./sports_betting.db"):
        self.db_url = db_url
        if db_url.startswith("sqlite"):
            self.engine = create_engine(db_url, connect_args={"check_same_thread": False})
        else:
            # MySQL or Postgres connection with pooling and liveness verification
            self.engine = create_engine(
                db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20
            )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()

    def create_tables(self):
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        return self.SessionLocal()

