from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict

class DestinationAnalyticsBase(BaseModel):
    destination: str
    visitors: Optional[str] = None
    growth: Optional[str] = None
    satisfaction: Optional[str] = None
    top_demographic: Optional[str] = None
    best_month: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None

class DestinationAnalyticsCreate(DestinationAnalyticsBase):
    trip_id: str

class DestinationAnalyticsResponse(DestinationAnalyticsBase):
    id: str
    trip_id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
