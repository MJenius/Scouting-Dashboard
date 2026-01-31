"""
database.py - SQLAlchemy database configuration for Scouting Dashboard.

Provides:
- SQLite connection with multi-threading support
- Session factory for database operations
- Dependency injection for FastAPI endpoints
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database URL - SQLite at project root
# Using check_same_thread=False for multi-threaded API access (required for FastAPI)
SQLALCHEMY_DATABASE_URL = "sqlite:///./scouting.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

# Session factory - creates new database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for ORM models
Base = declarative_base()


def get_db():
    """
    Dependency generator for FastAPI endpoint injection.
    
    Yields a database session and ensures cleanup after request completion.
    
    Usage in FastAPI:
        @app.get("/players/")
        def get_players(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
