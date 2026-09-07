"""
Database models and connection for fraud detection system.
Uses SQLAlchemy ORM for easy migration from SQLite to PostgreSQL.
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Database URL (SQLite for local, easily swappable to PostgreSQL)
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fraud_detection.db")

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


class Transaction(Base):
    """Stores transaction data and predictions."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True, nullable=False)

    # Transaction details
    amount = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Prediction results
    prediction = Column(String, nullable=False)  # "FRAUD" or "LEGITIMATE"
    fraud_probability = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    model_version = Column(String, nullable=False)

    # Performance tracking
    latency_ms = Column(Float, nullable=False)

    # Optional: actual label (for monitoring/retraining)
    actual_label = Column(String, nullable=True)  # Set later if fraud is confirmed/denied

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db():
    """Initialize the database (create tables if they don't exist)."""
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print(f"✓ Database initialized at: {DATABASE_URL}")


def get_db():
    """Get a database session (use as a context manager)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Test the database setup
    init_db()
    print("\n✅ Database schema created successfully!")
    print("\nTables created:")
    print("  - transactions (stores all predictions)")
