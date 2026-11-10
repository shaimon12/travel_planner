"""Services package for external APIs and AI agent workflows."""
from app.services.weather import get_weather_data, get_structured_weather
from app.services.validator import validate_destination
from app.services.llm import (
    invoke_llm,
    get_destination_analytics,
    invoke_gemini,
    invoke_gemini_json,
)
from app.services.flight_search import search_flights
from app.services.graph import (
    AgentState,
    validator_weather_node,
    flight_search_node,
    itinerary_generator_node,
    analytics_generator_node,
    run_agent_workflow,
)
from app.services.agent import plan_new_trip, handle_trip_chat

__all__ = [
    "get_weather_data",
    "get_structured_weather",
    "validate_destination",
    "invoke_llm",
    "get_destination_analytics",
    "invoke_gemini",
    "invoke_gemini_json",
    "search_flights",
    "AgentState",
    "validator_weather_node",
    "flight_search_node",
    "itinerary_generator_node",
    "analytics_generator_node",
    "run_agent_workflow",
    "plan_new_trip",
    "handle_trip_chat",
]
