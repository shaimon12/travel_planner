import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base

class DestinationAnalytics(Base):
    """
    DestinationAnalytics stores tourism metrics and trends generated for a destination.
    Replaces Streamlit's st.session_state.analytics_data.
    """
    __tablename__ = "destination_analytics"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    trip_id = Column(String(36), ForeignKey("trip_sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    destination = Column(String(255), nullable=False)
    visitors = Column(String(50), nullable=True)          # e.g., "4.2M"
    growth = Column(String(50), nullable=True)            # e.g., "+8%"
    satisfaction = Column(String(50), nullable=True)      # e.g., "4.7/5"
    top_demographic = Column(String(255), nullable=True)   # e.g., "Couples & Backpackers"
    best_month = Column(String(50), nullable=True)        # e.g., "September"
    raw_data = Column(JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    trip = relationship("TripSession", back_populates="analytics")

    def __repr__(self) -> str:
        return f"<DestinationAnalytics id={self.id} destination={self.destination}>"

