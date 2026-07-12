import os
import logging
from logging.handlers import RotatingFileHandler
from app.core.config import Settings

def setup_logging(settings: Settings) -> None:
    os.makedirs(settings.log_dir, exist_ok=True)
    handlers = [
        logging.StreamHandler(),
        RotatingFileHandler(
            os.path.join(settings.log_dir, "service.log"),
            maxBytes=5_000_000,
            backupCount=3
        ),
    ]
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
        force=True
    )
