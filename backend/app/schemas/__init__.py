"""Pydantic schemas package."""
from app.schemas.user import UserBase, UserCreate, UserResponse
from app.schemas.analytics import (
    DestinationAnalyticsBase,
    DestinationAnalyticsCreate,
    DestinationAnalyticsResponse,
)
from app.schemas.message import (
    MessageBase,
    MessageCreate,
    MessageResponse,
    ChatRequest,
    ChatResponse,
)
from app.schemas.trip import (
    TripBase,
    TripCreate,
    TripUpdate,
    TripSessionResponse,
    TripSessionDetailResponse,
)

__all__ = [
    "UserBase",
    "UserCreate",
    "UserResponse",
    "DestinationAnalyticsBase",
    "DestinationAnalyticsCreate",
    "DestinationAnalyticsResponse",
    "MessageBase",
    "MessageCreate",
    "MessageResponse",
    "ChatRequest",
    "ChatResponse",
    "TripBase",
    "TripCreate",
    "TripUpdate",
    "TripSessionResponse",
    "TripSessionDetailResponse",
]
