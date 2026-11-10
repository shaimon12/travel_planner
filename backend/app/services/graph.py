import json
import logging
from typing import TypedDict, List, Dict, Any, Optional
from datetime import date

from app.services.llm import invoke_llm, get_destination_analytics
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
    flight_data: Optional[List[Dict[str, Any]]]
    
    # Chat & Interaction History
    messages: List[Dict[str, str]]
    user_input: str
    
    # Agent Outputs & Errors
    itinerary_output: Optional[str]
    analytics_output: Optional[Dict[str, Any]]
    final_response: str
    error: Optional[str]

    # BYOK Custom Model Credentials
    x_api_key: Optional[str]
    x_base_url: Optional[str]
    x_model_name: Optional[str]

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
    Produces structured options list for Generative UI rendering.
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
    }

# ==============================================================================
# Node 3: Itinerary Generator (Strict Markdown & Generative UI Flight Handoff)
# ==============================================================================
def itinerary_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates structured markdown itinerary for new trips (Mode 1),
    or answers conversational follow-up questions adhering to strict Google Maps rules (Mode 2).
    Delegates flight options rendering to the Generative UI frontend components.
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
    flight_data = state.get("flight_data")

    # BYOK Overrides
    x_api_key = state.get("x_api_key")
    x_base_url = state.get("x_base_url")
    x_model_name = state.get("x_model_name")

    # If flight_data is not yet present and origin_city exists, run flight search
    if flight_data is None and origin_city:
        flight_data = search_flights(origin_city, destination, start_date, budget)

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

**✈️ GLOBAL RULE 2: GENERATIVE UI FLIGHTS (DO NOT FORMAT TABLES OR PRICES) ✈️**
Do NOT format markdown tables, lists of flights, airline prices, or flight booking blocks.
The system will display flight options automatically via the UI.
If the user asks about flights, or on the initial trip generation, simply let them know that available live flight options from {origin_city} to {destination} are presented below in the interactive flight cards.

**🌟 GOLDEN RULE FEW-SHOT EXEMPLAR (STRICT PREMIUM MARKDOWN FORMATTING):**
You MUST match this exact level of detail, emoji placement, precise bolding, Google Maps linking, and structural polish for every day:

---
### Day 1: Historic Alleys & Sunset Gastronomy
* **Morning:** Immerse yourself in cultural heritage at [Fushimi Inari Shrine](https://www.google.com/maps/search/?api=1&query=Kyoto+Fushimi+Inari+Shrine). Hike through the iconic thousand vermilion torii gates under cool morning shade.
* **Lunch:** Savor seasonal handmade soba noodles at [Honke Owariya](https://www.google.com/maps/search/?api=1&query=Kyoto+Honke+Owariya) - Established in 1465, famous for 8-layer soba bowls (~$18).
* **Afternoon:** Stroll the peaceful bamboo groves and tranquil Zen rock gardens of [Tenryu-ji Temple](https://www.google.com/maps/search/?api=1&query=Kyoto+Tenryu-ji+Temple) in the scenic Arashiyama district.
* **Dinner:** Enjoy authentic charcoal-grilled multi-course dining at [Gion Karyo](https://www.google.com/maps/search/?api=1&query=Kyoto+Gion+Karyo) - Traditional machiya wooden townhouse ambience.
* **🌙 Evening:** Wander through the lantern-lit stone alleys of [Pontocho Alley](https://www.google.com/maps/search/?api=1&query=Kyoto+Pontocho+Alley), stopping for matcha cocktails alongside the Kamogawa River.
* **💡 Local Secret:** Skip the crowded main trail after the first 200m; take the quiet forested path on the right for unobstructed panoramic views and zero crowds.
---

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
        user_prompt = f"""
Create a highly logistical, geo-optimized {duration}-day itinerary. Group activities by neighborhood to minimize travel time. Use the live weather to prioritize indoor vs. outdoor activities.
Remember: Follow the GOLDEN RULE FEW-SHOT EXEMPLAR strictly. DO NOT create markdown flight tables or list flight prices. The system will display flight options automatically via the UI.

You MUST output EXACTLY in this markdown format:

## 📅 {duration}-Day Itinerary: {destination}

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
        response = invoke_llm(
            prompt=user_prompt,
            system_instruction=system_prompt,
            x_api_key=x_api_key,
            x_base_url=x_base_url,
            x_model_name=x_model_name,
        )
        return {
            "itinerary_output": response,
            "final_response": response,
            "flight_data": flight_data,
        }

    else:
        # Conversational Q&A follow-up mode
        chat_instruction = f"""
{system_prompt}

**CONVERSATIONAL Q&A MODE:**
The user is asking a follow-up question regarding their trip to {destination}.
1. DO NOT rewrite the entire itinerary unless explicitly asked to regenerate it.
2. Provide a brief, friendly, and highly specific answer based on {destination}.
3. If the user asks about flights, airlines, airfare, or booking travel from {origin_city} to {destination}: Do NOT format markdown flight tables or list flight prices. The system will display flight options automatically via the UI.
4. Apply the **GLOBAL RULE: STRICT GOOGLE MAPS LINKS** to every physical location you recommend:
   [Place Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Place+Name)
"""
        user_message = state.get("user_input", "")
        history = state.get("messages", [])

        response = invoke_llm(
            prompt=user_message,
            system_instruction=chat_instruction,
            history=history,
            x_api_key=x_api_key,
            x_base_url=x_base_url,
            x_model_name=x_model_name,
        )
        return {
            "final_response": response,
            "flight_data": flight_data if state.get("flight_data") else None,
        }

# ==============================================================================
# Node 4: Analytics Generator (Strict JSON Output)
# ==============================================================================
def analytics_generator_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates estimated tourism trends and visitor demographics as structured JSON.
    """
    destination = state.get("destination", "Destination")
    x_api_key = state.get("x_api_key")
    x_base_url = state.get("x_base_url")
    x_model_name = state.get("x_model_name")
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
    analytics_data = get_destination_analytics(
        prompt=analytics_prompt,
        x_api_key=x_api_key,
        x_base_url=x_base_url,
        x_model_name=x_model_name,
    )

    # Attach raw metadata
    analytics_data["raw_data"] = {
        "destination": destination,
        "source": "AI Travel Market Intelligence (OpenRouter Free Auto-Router)",
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
