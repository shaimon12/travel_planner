import uuid
from datetime import datetime, date
from sqlalchemy import Column, String, Integer, Date, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship

from app.core.database import Base

class TripSession(Base):
    """
    TripSession represents a complete trip planning session, replacing Streamlit's
    ephemeral session_state (st.session_state.chat_sessions, destinations, etc.).
    """
    __tablename__ = "trip_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    is_guest = Column(Boolean, default=True, nullable=False)
    
    # Trip metadata & parameters
    title = Column(String(255), nullable=False, default="My Trip")
    destination = Column(String(255), nullable=True, index=True)
    raw_destination = Column(String(255), nullable=True)
    traveler_name = Column(String(255), nullable=True)
    start_date = Column(Date, nullable=True, default=date.today)
    duration = Column(Integer, default=3, nullable=False)
    budget = Column(String(50), default="Moderate", nullable=False)  # Shoestring, Moderate, Luxury
    travel_type = Column(String(50), default="Solo", nullable=False)  # Solo, Couple, Family, Friends
    group_size = Column(Integer, default=1, nullable=False)
    interests = Column(JSON, default=list, nullable=False)  # e.g., ["History", "Food"]
    dietary_restrictions = Column(JSON, default=list, nullable=False)  # e.g., ["Vegetarian"]
    status = Column(String(50), default="active", nullable=False)  # active, completed, archived

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="trips")
    messages = relationship(
        "Message",
        back_populates="trip",
        cascade="all, delete-orphan",
        order_by="Message.created_at.asc()"
    )
    analytics = relationship(
        "DestinationAnalytics",
        back_populates="trip",
        uselist=False,
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<TripSession id={self.id} title={self.title} destination={self.destination}>"

