from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.trip import TripSession
from app.models.message import Message
from app.models.analytics import DestinationAnalytics
from app.schemas.trip import (
    TripCreate,
    TripSessionResponse,
    TripSessionDetailResponse,
)
from app.schemas.message import ChatRequest, ChatResponse, MessageResponse
from app.schemas.analytics import DestinationAnalyticsResponse
from app.services.agent import plan_new_trip, handle_trip_chat

router = APIRouter(prefix="/trips", tags=["Trips"])

@router.post("", response_model=TripSessionDetailResponse, status_code=status.HTTP_201_CREATED)
def create_trip(
    trip_in: TripCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Creates a new trip session tied to the authenticated tenant user.
    Routes parameters through LangGraph workflow:
      Node 1: Destination Validator & Weather Fetcher
      Node 2: Itinerary Generator (strictly formatted markdown)
      Node 3: Analytics Generator (strictly structured JSON)
    Persists trip session, initial itinerary message, and analytics data.
    """
    if not trip_in.traveler_name or not trip_in.raw_destination:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Traveler name and destination are required."
        )

    # Auto-adjust group size logic matching prototype
    corrected_size = trip_in.group_size
    if trip_in.travel_type == "Couple" and corrected_size < 2:
        corrected_size = 2
    elif trip_in.travel_type in ["Family", "Friends"] and corrected_size < 3:
        corrected_size = 3

    # Execute LangGraph Multi-Agent Workflow
    agent_output = plan_new_trip({
        "traveler_name": trip_in.traveler_name,
        "raw_destination": trip_in.raw_destination,
        "start_date": trip_in.start_date,
        "duration": trip_in.duration,
        "budget": trip_in.budget,
        "travel_type": trip_in.travel_type,
        "group_size": corrected_size,
        "interests": trip_in.interests or ["General Sightseeing", "Local Culture"],
        "dietary_restrictions": trip_in.dietary_restrictions or [],
    })

    if agent_output.get("error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=agent_output["error"]
        )

    destination_name = agent_output.get("validated_destination") or trip_in.raw_destination
    title = trip_in.title if trip_in.title and trip_in.title != "My Trip" else f"Trip to {destination_name}"

    # 1. Create Trip Session entity strictly linked to current authenticated user
    trip = TripSession(
        user_id=current_user.id,
        title=title,
        traveler_name=trip_in.traveler_name,
        raw_destination=trip_in.raw_destination,
        destination=destination_name,
        start_date=trip_in.start_date,
        duration=trip_in.duration,
        budget=trip_in.budget,
        travel_type=trip_in.travel_type,
        group_size=corrected_size,
        interests=trip_in.interests or ["General Sightseeing", "Local Culture"],
        dietary_restrictions=trip_in.dietary_restrictions or [],
        status="active"
    )
    db.add(trip)
    db.commit()
    db.refresh(trip)

    # 2. Persist initial generated itinerary as assistant message
    itinerary_content = agent_output.get("itinerary", "")
    initial_message = Message(
        trip_id=trip.id,
        role="assistant",
        content=itinerary_content,
        meta={
            "type": "initial_itinerary",
            "weather_report": agent_output.get("weather_report")
        }
    )
    db.add(initial_message)

    # 3. Persist destination analytics
    analytics_dict = agent_output.get("analytics") or {}
    analytics = DestinationAnalytics(
        trip_id=trip.id,
        destination=destination_name,
        visitors=analytics_dict.get("visitors", "N/A"),
        growth=analytics_dict.get("growth", "N/A"),
        satisfaction=analytics_dict.get("satisfaction", "N/A"),
        top_demographic=analytics_dict.get("top_demographic", "N/A"),
        best_month=analytics_dict.get("best_month", "N/A"),
        raw_data=analytics_dict.get("raw_data") or analytics_dict
    )
    db.add(analytics)

    db.commit()
    db.refresh(trip)

    return trip

@router.get("", response_model=List[TripSessionResponse])
def list_trips(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Lists all trip sessions belonging exclusively to the authenticated user.
    """
    return (
        db.query(TripSession)
        .filter(TripSession.user_id == current_user.id)
        .order_by(TripSession.created_at.desc())
        .all()
    )

@router.get("/{trip_id}", response_model=TripSessionDetailResponse)
def get_trip(
    trip_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetches full trip session details including chat history and destination analytics.
    Enforces user ownership.
    """
    trip = (
        db.query(TripSession)
        .filter(TripSession.id == trip_id, TripSession.user_id == current_user.id)
        .first()
    )
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    return trip

@router.post("/{trip_id}/chat", response_model=ChatResponse)
def chat_with_trip(
    trip_id: str,
    chat_in: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Accepts user follow-up questions, routes through LangGraph workflow with conversation
    context, persists user message and AI response, and returns response payload.
    Enforces user ownership.
    """
    trip = (
        db.query(TripSession)
        .filter(TripSession.id == trip_id, TripSession.user_id == current_user.id)
        .first()
    )
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )

    clean_user_message = chat_in.message.strip()
    if not clean_user_message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message content cannot be empty."
        )

    # 1. Persist user message to database
    user_msg = Message(
        trip_id=trip.id,
        role="user",
        content=clean_user_message
    )
    db.add(user_msg)
    db.commit()

    # 2. Extract conversation history for agent context
    past_messages = [
        {"role": m.role, "content": m.content}
        for m in trip.messages
        if m.role in ["user", "assistant"]
    ]

    # 3. Route through LangGraph conversational workflow
    chat_result = handle_trip_chat(
        trip_params={
            "id": trip.id,
            "traveler_name": trip.traveler_name,
            "destination": trip.destination,
            "raw_destination": trip.raw_destination,
            "start_date": str(trip.start_date),
            "duration": trip.duration,
            "budget": trip.budget,
            "travel_type": trip.travel_type,
            "group_size": trip.group_size,
            "interests": trip.interests,
            "dietary_restrictions": trip.dietary_restrictions,
        },
        chat_history=past_messages,
        user_message=clean_user_message
    )

    ai_reply = chat_result.get("response") or "I apologize, I could not generate a response. Please try again."

    # 4. Persist AI response message to database
    assistant_msg = Message(
        trip_id=trip.id,
        role="assistant",
        content=ai_reply,
        meta={"type": "chat_followup"}
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        response=ai_reply,
        message=MessageResponse.model_validate(assistant_msg),
        analytics=DestinationAnalyticsResponse.model_validate(trip.analytics) if trip.analytics else None
    )

@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trip(
    trip_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Deletes a trip session, cascading removal of associated messages and analytics.
    Enforces user ownership.
    """
    trip = (
        db.query(TripSession)
        .filter(TripSession.id == trip_id, TripSession.user_id == current_user.id)
        .first()
    )
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    db.delete(trip)
    db.commit()
    return None

@router.get("/{trip_id}/analytics", response_model=DestinationAnalyticsResponse)
def get_trip_analytics(
    trip_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Fetches destination analytics metrics for a specific trip session.
    Enforces user ownership.
    """
    trip = (
        db.query(TripSession)
        .filter(TripSession.id == trip_id, TripSession.user_id == current_user.id)
        .first()
    )
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    if not trip.analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analytics data generated for trip '{trip_id}'."
        )
    return trip.analytics
