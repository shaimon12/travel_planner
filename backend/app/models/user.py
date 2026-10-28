import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import relationship

from app.core.database import Base

class User(Base):
    """Represents an application user or travel planner client."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    email = Column(String(255), unique=True, index=True, nullable=True)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    trips = relationship(
        "TripSession",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="TripSession.created_at.desc()"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} name={self.name} email={self.email}>"

