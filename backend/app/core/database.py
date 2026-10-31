import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings

db_url = settings.DATABASE_URL
# Automatically normalize legacy postgres:// URLs to postgresql:// for SQLAlchemy 2.0+
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

# SQLite-specific or PostgreSQL-specific engine configuration
if db_url.startswith("sqlite"):
    engine = create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=settings.DEBUG,
    )
else:
    engine = create_engine(
        db_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        echo=settings.DEBUG,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """FastAPI dependency yielding an active SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initializes tables in database and applies lightweight schema updates."""
    # Import all models to register them in Base.metadata
    import app.models  # noqa: F401
    Base.metadata.create_all(bind=engine)

    # Ensure schema migration for is_guest column on existing databases
    from sqlalchemy import inspect, text
    try:
        with engine.connect() as conn:
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            if "trip_sessions" in tables:
                columns = [c["name"] for c in inspector.get_columns("trip_sessions")]
                if "is_guest" not in columns:
                    conn.execute(text("ALTER TABLE trip_sessions ADD COLUMN is_guest BOOLEAN DEFAULT TRUE"))
                    conn.commit()
    except Exception:
        pass

