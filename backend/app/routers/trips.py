from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
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
from app.services.validator import validate_destination
from app.services.agent import generate_initial_itinerary_and_analytics

router = APIRouter(prefix="/trips", tags=["Trips"])

@router.post("", response_model=TripSessionDetailResponse, status_code=status.HTTP_201_CREATED)
def create_trip(trip_in: TripCreate, db: Session = Depends(get_db)):
    """
    Creates a new trip session.
    Validates the destination, persists parameters, generates initial itinerary and analytics.
    """
    cleaned_destination = validate_destination(trip_in.raw_destination)
    if not cleaned_destination:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not validate destination: '{trip_in.raw_destination}'. Please check spelling."
        )

    # Title defaults to destination or provided title
    title = trip_in.title or f"Trip to {cleaned_destination}"

    # Auto-adjust group size logic matching prototype
    corrected_size = trip_in.group_size
    if trip_in.travel_type == "Couple" and corrected_size < 2:
        corrected_size = 2
    elif trip_in.travel_type in ["Family", "Friends"] and corrected_size < 3:
        corrected_size = 3

    trip = TripSession(
        title=title,
        traveler_name=trip_in.traveler_name,
        raw_destination=trip_in.raw_destination,
        destination=cleaned_destination,
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

    # Generate initial itinerary and analytics via agent service
    agent_output = generate_initial_itinerary_and_analytics({
        "destination": trip.destination,
        "duration": trip.duration,
        "budget": trip.budget,
        "traveler_name": trip.traveler_name,
        "travel_type": trip.travel_type,
        "group_size": trip.group_size,
        "interests": trip.interests,
        "dietary_restrictions": trip.dietary_restrictions,
    })

    # Save initial assistant message (itinerary)
    initial_message = Message(
        trip_id=trip.id,
        role="assistant",
        content=agent_output["itinerary"],
        meta={"type": "initial_itinerary"}
    )
    db.add(initial_message)

    # Save analytics
    analytics_data = agent_output.get("analytics", {})
    analytics = DestinationAnalytics(
        trip_id=trip.id,
        destination=trip.destination,
        visitors=analytics_data.get("visitors"),
        growth=analytics_data.get("growth"),
        satisfaction=analytics_data.get("satisfaction"),
        top_demographic=analytics_data.get("top_demographic"),
        best_month=analytics_data.get("best_month"),
        raw_data=analytics_data.get("raw_data")
    )
    db.add(analytics)

    db.commit()
    db.refresh(trip)

    return trip

@router.get("", response_model=List[TripSessionResponse])
def list_trips(db: Session = Depends(get_db)):
    """Lists all trip sessions for persistent sidebar history."""
    return db.query(TripSession).order_by(TripSession.created_at.desc()).all()

@router.get("/{trip_id}", response_model=TripSessionDetailResponse)
def get_trip(trip_id: str, db: Session = Depends(get_db)):
    """Retrieves full trip session details including chat history and analytics."""
    trip = db.query(TripSession).filter(TripSession.id == trip_id).first()
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    return trip

@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trip(trip_id: str, db: Session = Depends(get_db)):
    """Deletes a trip session and cascades to all associated messages and analytics."""
    trip = db.query(TripSession).filter(TripSession.id == trip_id).first()
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    db.delete(trip)
    db.commit()
    return None

@router.post("/{trip_id}/chat", response_model=ChatResponse)
def chat_with_trip(trip_id: str, chat_in: ChatRequest, db: Session = Depends(get_db)):
    """
    Handles user follow-up questions for a trip session.
    Routes to agent orchestrator and persists conversation history.
    """
    trip = db.query(TripSession).filter(TripSession.id == trip_id).first()
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )

    # Persist user message
    user_msg = Message(
        trip_id=trip.id,
        role="user",
        content=chat_in.message
    )
    db.add(user_msg)
    db.commit()

    # In Phase 2, this calls LangGraph runner with full conversational state
    # Provide structured assistant response
    dest = trip.destination or "your destination"
    assistant_content = (
        f"Regarding your question about **{dest}**: '{chat_in.message}'\n\n"
        f"Here are my tailored recommendations for your {trip.budget.lower()} trip! "
        f"You can explore [{dest} City Center](https://www.google.com/maps/search/?api=1&query={dest.replace(' ', '+')}+City+Center) "
        f"which aligns with your interests in {', '.join(trip.interests) if trip.interests else 'local culture'}."
    )

    assistant_msg = Message(
        trip_id=trip.id,
        role="assistant",
        content=assistant_content
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        response=assistant_content,
        message=MessageResponse.model_validate(assistant_msg),
        analytics=DestinationAnalyticsResponse.model_validate(trip.analytics) if trip.analytics else None
    )

@router.get("/{trip_id}/analytics", response_model=DestinationAnalyticsResponse)
def get_trip_analytics(trip_id: str, db: Session = Depends(get_db)):
    """Retrieves destination tourism analytics for a specific trip session."""
    trip = db.query(TripSession).filter(TripSession.id == trip_id).first()
    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    if not trip.analytics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No analytics data generated yet for trip '{trip_id}'."
        )
    return trip.analytics
