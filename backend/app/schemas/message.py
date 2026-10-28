from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, ConfigDict

class MessageBase(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    meta: Optional[Dict[str, Any]] = None

class MessageCreate(BaseModel):
    content: str
    role: Optional[str] = "user"
    meta: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    id: str
    trip_id: str
    role: str
    content: str
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    message: MessageResponse
    analytics: Optional[Any] = None

