import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import date

from app.services.llm import invoke_gemini, invoke_gemini_json
from app.services.weather import get_weather_data
from app.services.validator import validate_destination
from app.services.flight_search import search_flights

logger = logging.getLogger("agent.workflow")

class AgentState(TypedDict, total=False):
    """LangGraph multi-agent shared state definition."""
    # Identification & Mode
    trip_id: Optional[str]
    mode: str  # "initial_plan" or "chat_followup"
    
    # Trip Context
    traveler_name: str
    origin_city: Optional[str]
    raw_destination: str
    destination: str
    start_date: str
    duration: int
    budget: str
    travel_type: str
    group_size: int
    children_ages: Optional[str]
    interests: List[str]
    dietary_restrictions: List[str]
    
    # Live External Enrichment
    weather_report: str
    flight_data: Optional[Dict[str, Any]]
    flight_markdown: Optional[str]
    
    # Chat & Interaction History
    messages: List[Dict[str, str]]
    user_input: str
    
    # Agent Outputs & Errors
    itinerary_output: Optional[str]
    analytics_output: Optional[Dict[str, Any]]
    final_response: str
    error: Optional[str]

# ==============================================================================
# Node 1: Destination Validator & Weather Fetcher
# ==============================================================================
def validator_weather_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates geographic destination and enriches state with real-time Open-Meteo weather.
    Runs during initial trip planning or when follow-up queries require weather updates.
    """
    raw_dest = state.get("raw_destination") or state.get("destination", "")
    logger.info(f"[Node 1: Validator & Weather] Processing: {raw_dest}")

    validated = validate_destination(raw_dest)
    if not validated:
        return {
            "error": f"Invalid destination: '{raw_dest}'. Please verify the spelling.",
            "destination": raw_dest,
            "weather_report": "Weather data unavailable."
        }

    # Fetch real-time weather from Open-Meteo
    weather_info = get_weather_data(validated)

    return {
        "destination": validated,
        "weather_report": weather_info,
        "error": None
    }

# ==============================================================================
# Node 2: Live Flight Booking Agent
# ==============================================================================
def flight_search_node(state: AgentState) -> Dict[str, Any]:
    """
    Simulates querying real-time flight fares, schedules, and carrier routes.
    Produces structured options and formatted Markdown booking blocks with deep-links.
    """
    origin = state.get("origin_city") or "Sydney, Australia"
    destination = state.get("destination") or state.get("raw_destination", "Destination")
    start_date = state.get("start_date", str(date.today()))
    budget = state.get("budget", "Moderate")

    logger.info(f"[Flight Search Node] Querying flights: {origin} -> {destination} on {start_date}")

    flight_result = search_flights(
        origin_city=origin,
        destination=destination,
        start_date=start_date,
        budget=budget,
    )

    return {
        "flight_data": flight_result,
        "flight_markdown": flight_result.get("markdown_block", "")
    }

# ==============================================================================
# Node 3: Itinerary Generator (Strict Markdown & Flight Booking Block)
# ==============================================================================
def itinerary_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates structured markdown itinerary for new trips (Mode 1),
    or answers conversational follow-up questions adhering to strict Google Maps & Flight link rules (Mode 2).
    """
    mode = state.get("mode", "initial_plan")
    destination = state.get("destination", "Destination")
    origin_city = state.get("origin_city") or "Sydney, Australia"
    traveler = state.get("traveler_name", "Traveler")
    duration = state.get("duration", 3)
    start_date = state.get("start_date", str(date.today()))
    budget = state.get("budget", "Moderate")
    travel_type = state.get("travel_type", "Solo")
    group_size = state.get("group_size", 1)
    children_ages = state.get("children_ages")
    interests = state.get("interests", ["General Sightseeing"])
    dietary = state.get("dietary_restrictions", [])
    weather_report = state.get("weather_report", "Seasonal conditions.")
    flight_markdown = state.get("flight_markdown") or ""

    # If flight_markdown is not yet present, run flight search if origin exists
    if not flight_markdown and origin_city:
        flt_res = search_flights(origin_city, destination, start_date, budget)
        flight_markdown = flt_res.get("markdown_block", "")

    group_info = f"Traveling {travel_type} ({group_size} people)"
    if travel_type == "Family" and children_ages:
        group_info += f" with children (Ages: {children_ages})"

    dietary_str = ", ".join(dietary) if dietary else "None"
    interests_str = ", ".join(interests) if interests else "General Sightseeing, Local Culture"

    system_prompt = f"""
You are an elite, award-winning Travel Concierge and Booking Agent.

**⛔ STRICT OPERATING RULES:**
1. Only discuss travel, itineraries, flights, and local recommendations. Refuse anything else politely.
2. Never break character. Never state you are an AI or an LLM.
3. If the Group includes "Family" and children ages are provided ({children_ages or 'unspecified'}), you MUST default to kid-friendly and accessible activities tailored specifically for those age groups.

**🚨 GLOBAL RULE 1: STRICT GOOGLE MAPS LINKS (NON-NEGOTIABLE) 🚨**
ANY time you mention a specific physical place (restaurant, hotel, museum, shop, park, cafe, boba shop, etc.), you MUST hyperlink it using EXACTLY this format:
[Place Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Place+Name)
*Failure to use this exact link structure will break the application architecture.*

**✈️ GLOBAL RULE 2: LIVE FLIGHT BOOKING BLOCKS ✈️**
Whenever presenting flight options (either on the initial plan or when asked about flights/airfare/airlines in chat), you MUST format the available flight options as a clear, actionable booking block with departure times, airline names, and the exact provided [Book Flight 🎫](url) booking hyperlinks.

**USER & TRIP CONTEXT:**
- Traveler: {traveler}
- Origin: {origin_city}
- Destination: {destination}
- Duration: {duration} Days (Starting {start_date})
- Budget: {budget}
- Group: {group_info}
- Dietary Restrictions: {dietary_str}
- Core Interests: {interests_str}
- Live Weather Forecast: {weather_report}
"""

    if mode == "initial_plan":
        flight_section_instruction = f"""
Include this exact live flight search block right above Day 1:
{flight_markdown}
""" if flight_markdown else ""

        user_prompt = f"""
Create a highly logistical, geo-optimized {duration}-day itinerary. Group activities by neighborhood to minimize travel time. Use the live weather to prioritize indoor vs. outdoor activities.

{flight_section_instruction}

You MUST output EXACTLY in this markdown format:

## 📅 {duration}-Day Itinerary: {destination}

{flight_markdown}

### 🎒 Smart Packing List
* [Essential 1 based on live weather]
* [Essential 2]
* [Essential 3]

---
### Day 1: [Catchy Theme Name]
* **Morning:** [Activity Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Description.
* **Lunch:** [Restaurant Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Restaurant+Name) - Cuisine & Price. (Must respect {dietary_str})
* **Afternoon:** [Activity Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Description.
* **Dinner:** [Restaurant Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Restaurant+Name) - Ambience.
* **🌙 Evening:** [Activity/Bar](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Wind down.
* **💡 Local Secret:** A specific local tip.
*(Repeat structure for all {duration} days)*
"""
        response = invoke_gemini(user_prompt, system_instruction=system_prompt)
        return {
            "itinerary_output": response,
            "final_response": response,
        }

    else:
        # Conversational Q&A follow-up mode
        chat_instruction = f"""
{system_prompt}

**CONVERSATIONAL Q&A MODE:**
The user is asking a follow-up question regarding their trip to {destination}.
1. DO NOT rewrite the entire itinerary unless explicitly asked to regenerate it.
2. Provide a brief, friendly, and highly specific answer based on {destination}.
3. If the user asks about flights, airlines, airfare, or booking travel from {origin_city} to {destination}, you MUST include this live booking block in your response:
{flight_markdown}
4. Apply the **GLOBAL RULE: STRICT GOOGLE MAPS LINKS** to every physical location you recommend:
   [Place Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Place+Name)
"""
        user_message = state.get("user_input", "")
        history = state.get("messages", [])

        response = invoke_gemini(
            prompt=user_message,
            system_instruction=chat_instruction,
            history=history
        )
        return {
            "final_response": response
        }

# ==============================================================================
# Node 4: Analytics Generator (Strict JSON Output)
# ==============================================================================
def analytics_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates estimated tourism trends and visitor demographics as structured JSON.
    """
    destination = state.get("destination", "Destination")
    logger.info(f"[Node 4: Analytics Generator] Generating metrics for: {destination}")

    analytics_prompt = f"""
    Act as a tourism data analyst. Provide estimated travel analytics for '{destination}'.
    Return ONLY a raw JSON object (no markdown formatting, no backticks).
    Use this exact schema:
    {{
        "visitors": "Number with M or K (e.g., 4.2M)",
        "growth": "Percentage with sign (e.g., +8%)",
        "satisfaction": "Rating out of 5 (e.g., 4.7/5)",
        "top_demographic": "Short string (e.g., Couples & Backpackers)",
        "best_month": "Month name"
    }}
    """
    analytics_data = invoke_gemini_json(analytics_prompt)

    # Attach raw metadata
    analytics_data["raw_data"] = {
        "destination": destination,
        "source": "AI Travel Market Intelligence (Gemini 2.5 Flash)",
    }

    return {
        "analytics_output": analytics_data
    }

# ==============================================================================
# Edge Routing & LangGraph Assembly
# ==============================================================================
def route_entry_point(state: AgentState) -> str:
    """Routes to validator, flight search, or directly to itinerary generator based on query."""
    if state.get("mode") == "initial_plan":
        return "validator_weather"

    query = (state.get("user_input") or "").lower()
    flight_keywords = ["flight", "flights", "fly", "airline", "airfare", "plane", "ticket", "book flight", "airports"]
    if any(k in query for k in flight_keywords):
        return "flight_search"

    weather_keywords = ["weather", "forecast", "rain", "temperature", "climate", "new destination", "change city"]
    if any(k in query for k in weather_keywords):
        return "validator_weather"

    return "itinerary_generator"

def route_after_validator(state: AgentState) -> str:
    """Aborts if validation encountered an error; otherwise proceeds to flight search or itinerary."""
    if state.get("error"):
        return "end"
    if state.get("mode") == "initial_plan" and state.get("origin_city"):
        return "flight_search"
    return "itinerary_generator"

def route_after_itinerary(state: AgentState) -> str:
    """Triggers analytics generator only on initial trip setup."""
    if state.get("mode") == "initial_plan":
        return "analytics_generator"
    return "end"

# Build LangGraph graph if langgraph is available, else provide native state runner
def build_langgraph_workflow():
    """Builds and compiles the LangGraph StateGraph."""
    try:
        from langgraph.graph import StateGraph, END

        workflow = StateGraph(AgentState)

        workflow.add_node("validator_weather", validator_weather_node)
        workflow.add_node("flight_search", flight_search_node)
        workflow.add_node("itinerary_generator", itinerary_generator_node)
        workflow.add_node("analytics_generator", analytics_generator_node)

        workflow.set_conditional_entry_point(
            route_entry_point,
            {
                "validator_weather": "validator_weather",
                "flight_search": "flight_search",
                "itinerary_generator": "itinerary_generator",
            }
        )

        workflow.add_conditional_edges(
            "validator_weather",
            route_after_validator,
            {
                "flight_search": "flight_search",
                "itinerary_generator": "itinerary_generator",
                "end": END,
            }
        )

        workflow.add_edge("flight_search", "itinerary_generator")

        workflow.add_conditional_edges(
            "itinerary_generator",
            route_after_itinerary,
            {
                "analytics_generator": "analytics_generator",
                "end": END,
            }
        )

        workflow.add_edge("analytics_generator", END)

        return workflow.compile()
    except Exception as e:
        logger.warning(f"Native LangGraph compile fallback: {e}")
        return None

# Compiled LangGraph instance (or None if fallback is active)
compiled_graph = build_langgraph_workflow()

def run_agent_workflow(initial_state: AgentState) -> AgentState:
    """
    Executes the multi-agent workflow.
    Uses compiled LangGraph StateGraph if present, or robust sequential state machine.
    """
    # 1. Attempt LangGraph native execution
    if compiled_graph is not None:
        try:
            result = compiled_graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"LangGraph execution exception, falling back to sequential runner: {e}")

    # 2. Sequential State Machine Runner (Identical routing semantics)
    state = dict(initial_state)

    next_node = route_entry_point(state)

    if next_node == "validator_weather":
        n1_update = validator_weather_node(state)
        state.update(n1_update)
        if route_after_validator(state) == "end":
            state["final_response"] = state.get("error", "Destination validation error.")
            return state
        if route_after_validator(state) == "flight_search":
            flt_update = flight_search_node(state)
            state.update(flt_update)

    elif next_node == "flight_search":
        flt_update = flight_search_node(state)
        state.update(flt_update)

    n2_update = itinerary_generator_node(state)
    state.update(n2_update)

    if route_after_itinerary(state) == "analytics_generator":
        n3_update = analytics_generator_node(state)
        state.update(n3_update)

    return state
