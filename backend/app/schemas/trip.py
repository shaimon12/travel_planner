from datetime import date, datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.message import MessageResponse
from app.schemas.analytics import DestinationAnalyticsResponse

class TripBase(BaseModel):
    title: Optional[str] = "My Trip"
    traveler_name: Optional[str] = None
    destination: Optional[str] = None
    raw_destination: Optional[str] = None
    start_date: Optional[date] = None
    duration: int = Field(default=3, ge=1, le=30)
    budget: str = Field(default="Moderate")  # Shoestring, Moderate, Luxury
    travel_type: str = Field(default="Solo")  # Solo, Couple, Family, Friends
    group_size: int = Field(default=1, ge=1, le=20)
    interests: List[str] = Field(default_factory=list)
    dietary_restrictions: List[str] = Field(default_factory=list)

class TripCreate(TripBase):
    raw_destination: str
    traveler_name: str

class TripUpdate(BaseModel):
    title: Optional[str] = None
    destination: Optional[str] = None
    start_date: Optional[date] = None
    duration: Optional[int] = None
    budget: Optional[str] = None
    travel_type: Optional[str] = None
    group_size: Optional[int] = None
    interests: Optional[List[str]] = None
    dietary_restrictions: Optional[List[str]] = None
    status: Optional[str] = None

class TripSessionResponse(TripBase):
    id: str
    user_id: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)

class TripSessionDetailResponse(TripSessionResponse):
    messages: List[MessageResponse] = Field(default_factory=list)
    analytics: Optional[DestinationAnalyticsResponse] = None

    model_config = ConfigDict(from_attributes=True)
