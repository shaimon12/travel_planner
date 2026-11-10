from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request, Header
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.core.security import get_current_user, get_current_user_optional
from app.core.limiter import limiter
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
@limiter.limit("10/minute")
def create_trip(
    request: Request,
    trip_in: TripCreate,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
    x_api_key: Optional[str] = Header(None),
    x_base_url: Optional[str] = Header(None),
    x_model_name: Optional[str] = Header(None),
):
    """
    Creates a new trip session.
    If authenticated, links strictly to current user.
    If unauthenticated, creates as an onboarding guest trip (is_guest=True).
    Routes parameters through LangGraph workflow:
      Node 1: Destination Validator & Weather Fetcher
      Node 2: Itinerary Generator (strictly formatted markdown)
      Node 3: Analytics Generator (strictly structured JSON)
    Persists trip session, initial itinerary message, and analytics data.
    Supports BYOK (Bring Your Own Key) header overrides for custom LLM models.
    """
    # Header resolution with fallback to raw request headers
    x_api_key = x_api_key or request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    x_base_url = x_base_url or request.headers.get("x-base-url") or request.headers.get("X-Base-URL")
    x_model_name = x_model_name or request.headers.get("x-model-name") or request.headers.get("X-Model-Name")

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

    # =========================================================================
    # 0. PostgreSQL / SQLite Semantic Cache Check (Last 48 Hours)
    # Exact match: destination, duration, travel_type, group_size, budget
    # Bypassed if BYOK custom credentials are provided so users get fresh model output
    # =========================================================================
    if not x_api_key:
        cutoff_time = datetime.utcnow() - timedelta(hours=48)
        clean_dest = trip_in.raw_destination.strip()

        cached_trip = (
            db.query(TripSession)
            .filter(
                (func.lower(TripSession.raw_destination) == clean_dest.lower()) |
                (func.lower(TripSession.destination) == clean_dest.lower()),
                TripSession.duration == trip_in.duration,
                TripSession.travel_type == trip_in.travel_type,
                TripSession.group_size == corrected_size,
                TripSession.budget == trip_in.budget,
                TripSession.created_at >= cutoff_time,
            )
            .order_by(TripSession.created_at.desc())
            .first()
        )

        # Check if cached trip contains an initial generated itinerary message
        cached_itinerary_msg = None
        if cached_trip and cached_trip.messages:
            for m in cached_trip.messages:
                if m.role == "assistant" and m.content and len(m.content) > 50:
                    cached_itinerary_msg = m
                    break

        if cached_trip and cached_itinerary_msg:
            # Clone cached trip directly, bypassing LLM completely
            destination_name = cached_trip.destination or clean_dest
            title = trip_in.title if trip_in.title and trip_in.title != "My Trip" else f"Trip to {destination_name}"
            is_guest = current_user is None
            user_id = current_user.id if current_user else None

            new_trip = TripSession(
                user_id=user_id,
                is_guest=is_guest,
                title=title,
                traveler_name=trip_in.traveler_name,
                origin_city=trip_in.origin_city or cached_trip.origin_city or "Sydney, Australia",
                raw_destination=trip_in.raw_destination,
                destination=destination_name,
                start_date=trip_in.start_date or cached_trip.start_date,
                duration=trip_in.duration,
                budget=trip_in.budget,
                travel_type=trip_in.travel_type,
                group_size=corrected_size,
                children_ages=trip_in.children_ages,
                interests=trip_in.interests or cached_trip.interests or ["General Sightseeing"],
                dietary_restrictions=trip_in.dietary_restrictions or cached_trip.dietary_restrictions or [],
                status="active"
            )
            db.add(new_trip)
            db.commit()
            db.refresh(new_trip)

            # Clone assistant message with itinerary & flight_data
            cached_meta = dict(cached_itinerary_msg.meta or {})
            cached_meta["cached"] = True
            cached_meta["cached_from"] = cached_trip.id

            new_message = Message(
                trip_id=new_trip.id,
                role="assistant",
                content=cached_itinerary_msg.content,
                meta=cached_meta
            )
            db.add(new_message)

            # Clone destination analytics if available
            if cached_trip.analytics:
                ca = cached_trip.analytics
                new_analytics = DestinationAnalytics(
                    trip_id=new_trip.id,
                    destination=destination_name,
                    visitors=ca.visitors,
                    growth=ca.growth,
                    satisfaction=ca.satisfaction,
                    top_demographic=ca.top_demographic,
                    best_month=ca.best_month,
                    raw_data=ca.raw_data
                )
                db.add(new_analytics)

            db.commit()
            db.refresh(new_trip)
            return new_trip

    # Execute LangGraph Multi-Agent Workflow
    agent_output = plan_new_trip({
        "traveler_name": trip_in.traveler_name,
        "origin_city": trip_in.origin_city or "Sydney, Australia",
        "raw_destination": trip_in.raw_destination,
        "start_date": trip_in.start_date,
        "duration": trip_in.duration,
        "budget": trip_in.budget,
        "travel_type": trip_in.travel_type,
        "group_size": corrected_size,
        "children_ages": trip_in.children_ages,
        "interests": trip_in.interests or ["General Sightseeing", "Local Culture"],
        "dietary_restrictions": trip_in.dietary_restrictions or [],
        "x_api_key": x_api_key,
        "x_base_url": x_base_url,
        "x_model_name": x_model_name,
    })

    if agent_output.get("error"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=agent_output["error"]
        )

    destination_name = agent_output.get("validated_destination") or trip_in.raw_destination
    title = trip_in.title if trip_in.title and trip_in.title != "My Trip" else f"Trip to {destination_name}"

    is_guest = current_user is None
    user_id = current_user.id if current_user else None

    # 1. Create Trip Session entity
    trip = TripSession(
        user_id=user_id,
        is_guest=is_guest,
        title=title,
        traveler_name=trip_in.traveler_name,
        origin_city=trip_in.origin_city or "Sydney, Australia",
        raw_destination=trip_in.raw_destination,
        destination=destination_name,
        start_date=trip_in.start_date,
        duration=trip_in.duration,
        budget=trip_in.budget,
        travel_type=trip_in.travel_type,
        group_size=corrected_size,
        children_ages=trip_in.children_ages,
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
            "weather_report": agent_output.get("weather_report"),
            "flight_data": agent_output.get("flight_data"),
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
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Lists all trip sessions belonging exclusively to the authenticated user.
    If unauthenticated, returns an empty list (guests track their single trip session via localStorage).
    """
    if not current_user:
        return []

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
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Fetches full trip session details including chat history and destination analytics.
    Allows access if the trip is a guest trip, or if it belongs to the authenticated user.
    """
    if current_user:
        trip = (
            db.query(TripSession)
            .filter(
                TripSession.id == trip_id,
                (TripSession.user_id == current_user.id) | (TripSession.is_guest == True)
            )
            .first()
        )
    else:
        trip = (
            db.query(TripSession)
            .filter(TripSession.id == trip_id, TripSession.is_guest == True)
            .first()
        )

    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )
    return trip

@router.post("/{trip_id}/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
def chat_with_trip(
    request: Request,
    trip_id: str,
    chat_in: ChatRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
    x_api_key: Optional[str] = Header(None),
    x_base_url: Optional[str] = Header(None),
    x_model_name: Optional[str] = Header(None),
):
    """
    Accepts user follow-up questions, routes through LangGraph workflow with conversation
    context, persists user message and AI response, and returns response payload.
    Supports continuous free chatting for both unauthenticated guest travelers and registered users.
    Supports BYOK (Bring Your Own Key) header overrides for custom LLM models.
    """
    x_api_key = x_api_key or request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    x_base_url = x_base_url or request.headers.get("x-base-url") or request.headers.get("X-Base-URL")
    x_model_name = x_model_name or request.headers.get("x-model-name") or request.headers.get("X-Model-Name")

    if current_user:
        trip = (
            db.query(TripSession)
            .filter(
                TripSession.id == trip_id,
                (TripSession.user_id == current_user.id) | (TripSession.is_guest == True)
            )
            .first()
        )
    else:
        trip = (
            db.query(TripSession)
            .filter(TripSession.id == trip_id)
            .first()
        )

    if not trip:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trip session '{trip_id}' not found."
        )

    # Automatically claim guest trip if owned by guest and user logs in
    if current_user and trip.is_guest and not trip.user_id:
        trip.user_id = current_user.id
        trip.is_guest = False
        db.commit()
        db.refresh(trip)

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
            "origin_city": trip.origin_city or "Sydney, Australia",
            "destination": trip.destination,
            "raw_destination": trip.raw_destination,
            "start_date": str(trip.start_date),
            "duration": trip.duration,
            "budget": trip.budget,
            "travel_type": trip.travel_type,
            "group_size": trip.group_size,
            "children_ages": trip.children_ages,
            "interests": trip.interests,
            "dietary_restrictions": trip.dietary_restrictions,
            "x_api_key": x_api_key,
            "x_base_url": x_base_url,
            "x_model_name": x_model_name,
        },
        chat_history=past_messages,
        user_message=clean_user_message
    )

    ai_reply = chat_result.get("response") or "I apologize, I could not generate a response. Please try again."

    flight_data = chat_result.get("flight_data")

    # 4. Persist AI response message to database
    assistant_msg = Message(
        trip_id=trip.id,
        role="assistant",
        content=ai_reply,
        meta={
            "type": "chat_followup",
            "flight_data": flight_data,
        }
    )
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    return ChatResponse(
        response=ai_reply,
        content=ai_reply,
        flight_data=flight_data,
        message=MessageResponse.model_validate(assistant_msg),
        analytics=DestinationAnalyticsResponse.model_validate(trip.analytics) if trip.analytics else None
    )

@router.delete("/{trip_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_trip(
    trip_id: str,
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Deletes a trip session, cascading removal of associated messages and analytics.
    Enforces user ownership or guest ownership.
    """
    if current_user:
        trip = (
            db.query(TripSession)
            .filter(
                TripSession.id == trip_id,
                (TripSession.user_id == current_user.id) | (TripSession.is_guest == True)
            )
            .first()
        )
    else:
        trip = (
            db.query(TripSession)
            .filter(TripSession.id == trip_id, TripSession.is_guest == True)
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
    current_user: Optional[User] = Depends(get_current_user_optional),
):
    """
    Fetches destination analytics metrics for a specific trip session.
    """
    if current_user:
        trip = (
            db.query(TripSession)
            .filter(
                TripSession.id == trip_id,
                (TripSession.user_id == current_user.id) | (TripSession.is_guest == True)
            )
            .first()
        )
    else:
        trip = (
            db.query(TripSession)
            .filter(TripSession.id == trip_id, TripSession.is_guest == True)
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
