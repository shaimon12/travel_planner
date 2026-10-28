"""
Multi-Agent Orchestration Service for AI Travel Planner Pro.
Provides high-level APIs powered by the LangGraph 3-node agent workflow.
"""
from typing import Dict, Any, List, Optional
from app.services.graph import run_agent_workflow, AgentState

def plan_new_trip(trip_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes initial trip planning across:
    1. Node 1: Destination Validator & Weather Fetcher
    2. Node 2: Itinerary Generator (strict markdown)
    3. Node 3: Analytics Generator (strict JSON)
    """
    initial_state: AgentState = {
        "mode": "initial_plan",
        "trip_id": trip_params.get("id"),
        "traveler_name": trip_params.get("traveler_name", "Traveler"),
        "raw_destination": trip_params.get("raw_destination", ""),
        "destination": trip_params.get("destination", ""),
        "start_date": str(trip_params.get("start_date", "")),
        "duration": trip_params.get("duration", 3),
        "budget": trip_params.get("budget", "Moderate"),
        "travel_type": trip_params.get("travel_type", "Solo"),
        "group_size": trip_params.get("group_size", 1),
        "interests": trip_params.get("interests", []),
        "dietary_restrictions": trip_params.get("dietary_restrictions", []),
        "messages": [],
        "user_input": "",
    }

    result = run_agent_workflow(initial_state)

    return {
        "validated_destination": result.get("destination"),
        "weather_report": result.get("weather_report"),
        "itinerary": result.get("itinerary_output") or result.get("final_response"),
        "analytics": result.get("analytics_output"),
        "error": result.get("error")
    }

def handle_trip_chat(
    trip_params: Dict[str, Any],
    chat_history: List[Dict[str, str]],
    user_message: str
) -> Dict[str, Any]:
    """
    Handles conversational follow-ups for an existing trip:
    Routes through Node 2 (Itinerary Generator in Q&A mode) with persistent context and Google Maps rules.
    """
    state: AgentState = {
        "mode": "chat_followup",
        "trip_id": trip_params.get("id"),
        "traveler_name": trip_params.get("traveler_name", "Traveler"),
        "destination": trip_params.get("destination", ""),
        "raw_destination": trip_params.get("raw_destination", ""),
        "start_date": str(trip_params.get("start_date", "")),
        "duration": trip_params.get("duration", 3),
        "budget": trip_params.get("budget", "Moderate"),
        "travel_type": trip_params.get("travel_type", "Solo"),
        "group_size": trip_params.get("group_size", 1),
        "interests": trip_params.get("interests", []),
        "dietary_restrictions": trip_params.get("dietary_restrictions", []),
        "weather_report": trip_params.get("weather_report", ""),
        "messages": chat_history,
        "user_input": user_message,
    }

    result = run_agent_workflow(state)

    return {
        "response": result.get("final_response", ""),
        "weather_report": result.get("weather_report"),
        "error": result.get("error")
    }
