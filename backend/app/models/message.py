import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base

class Message(Base):
    """
    Message model representing individual chat messages within a trip session.
    Replaces Streamlit's chat history arrays in st.session_state.chat_sessions.
    """
    __tablename__ = "messages"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    trip_id = Column(String(36), ForeignKey("trip_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    # Using 'meta' as python attribute name with 'metadata' as column name to avoid collision with Base.metadata
    meta = Column("metadata", JSON, nullable=True, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    trip = relationship("TripSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} trip_id={self.trip_id} role={self.role}>"

