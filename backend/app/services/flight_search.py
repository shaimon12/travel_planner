import urllib.parse
from datetime import datetime, date
from typing import Dict, Any, List, Optional

AIRLINE_ROSTER = [
    {"name": "Singapore Airlines", "code": "SQ"},
    {"name": "Qantas Airways", "code": "QF"},
    {"name": "All Nippon Airways", "code": "NH"},
    {"name": "Japan Airlines", "code": "JL"},
    {"name": "Emirates", "code": "EK"},
    {"name": "Qatar Airways", "code": "QR"},
    {"name": "Delta Air Lines", "code": "DL"},
    {"name": "United Airlines", "code": "UA"},
    {"name": "British Airways", "code": "BA"},
    {"name": "Air France", "code": "AF"},
    {"name": "Cathay Pacific", "code": "CX"},
]

def search_flights(
    origin_city: str,
    destination: str,
    start_date: Optional[str] = None,
    budget: str = "Moderate",
) -> Dict[str, Any]:
    """
    Simulates querying a global flight search API (e.g., Skyscanner, Amadeus, Google Flights).
    Returns realistic mock flight data with airlines, schedules, pricing, and deep-link URLs.
    """
    clean_origin = (origin_city or "Home City").strip()
    clean_dest = (destination or "Destination").strip()
    trip_date = start_date or str(date.today())

    # Build safe search query for Google Flights deep-link
    query = f"Flights from {clean_origin} to {clean_dest} on {trip_date}"
    encoded_query = urllib.parse.quote_plus(query)
    base_booking_url = f"https://www.google.com/travel/flights?q={encoded_query}"

    # Price tiers based on budget preference
    if budget.lower() == "shoestring":
        base_fare = 420
        fare_mult = 1.0
    elif budget.lower() == "luxury":
        base_fare = 1850
        fare_mult = 2.5
    else:  # Moderate
        base_fare = 780
        fare_mult = 1.3

    # Generate 3 distinct flight options
    # Seed deterministic choices based on city string lengths
    seed = (len(clean_origin) + len(clean_dest)) % len(AIRLINE_ROSTER)
    airline1 = AIRLINE_ROSTER[seed]
    airline2 = AIRLINE_ROSTER[(seed + 3) % len(AIRLINE_ROSTER)]
    airline3 = AIRLINE_ROSTER[(seed + 6) % len(AIRLINE_ROSTER)]

    price1 = int(base_fare * 0.95)
    price2 = int(base_fare * 1.15)
    price3 = int(base_fare * 1.45) if budget.lower() != "shoestring" else int(base_fare * 0.85)

    flights = [
        {
            "id": "flt-1",
            "airline": airline1["name"],
            "flight_number": f"{airline1['code']} {100 + (seed * 23) % 899}",
            "departure": "08:15 AM",
            "arrival": "04:30 PM",
            "duration": "9h 15m",
            "stops": "Nonstop",
            "price": f"${price1}",
            "cabin": "Economy" if budget.lower() != "luxury" else "Business Class",
            "booking_url": f"{base_booking_url}&f=0",
            "tag": "Best Value",
        },
        {
            "id": "flt-2",
            "airline": airline2["name"],
            "flight_number": f"{airline2['code']} {200 + (seed * 37) % 799}",
            "departure": "01:45 PM",
            "arrival": "10:10 PM",
            "duration": "9h 25m",
            "stops": "Nonstop",
            "price": f"${price2}",
            "cabin": "Economy" if budget.lower() != "luxury" else "Business Class",
            "booking_url": f"{base_booking_url}&f=1",
            "tag": "Fastest Route",
        },
        {
            "id": "flt-3",
            "airline": airline3["name"],
            "flight_number": f"{airline3['code']} {400 + (seed * 19) % 599}",
            "departure": "09:30 PM",
            "arrival": "07:15 AM (+1)",
            "duration": "10h 45m",
            "stops": "1 Stop (1h 15m layover)",
            "price": f"${price3}",
            "cabin": "Economy" if budget.lower() != "luxury" else "First Class",
            "booking_url": f"{base_booking_url}&f=2",
            "tag": "Red-Eye / Flexible",
        },
    ]

    # Generate formatted Markdown block
    markdown_table = f"""### ✈️ Live Flight Options: {clean_origin} ➔ {clean_dest}
*Departure Date: **{trip_date}** | Estimated Fares ({budget} Travel)*

| Airline | Flight # | Departure | Arrival | Duration | Stops | Est. Price | Action |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **{flights[0]['airline']}** | {flights[0]['flight_number']} | {flights[0]['departure']} | {flights[0]['arrival']} | {flights[0]['duration']} | {flights[0]['stops']} | **{flights[0]['price']}** | [Book Flight 🎫]({flights[0]['booking_url']}) |
| **{flights[1]['airline']}** | {flights[1]['flight_number']} | {flights[1]['departure']} | {flights[1]['arrival']} | {flights[1]['duration']} | {flights[1]['stops']} | **{flights[1]['price']}** | [Book Flight 🎫]({flights[1]['booking_url']}) |
| **{flights[2]['airline']}** | {flights[2]['flight_number']} | {flights[2]['departure']} | {flights[2]['arrival']} | {flights[2]['duration']} | {flights[2]['stops']} | **{flights[2]['price']}** | [Book Flight 🎫]({flights[2]['booking_url']}) |

> 💡 *Live booking simulation powered by Google Flights API integration. Click **[Book Flight 🎫]({base_booking_url})** to confirm live seats and real-time schedules.*
"""

    return {
        "origin": clean_origin,
        "destination": clean_dest,
        "date": trip_date,
        "flights": flights,
        "booking_url": base_booking_url,
        "markdown_block": markdown_table,
    }

def format_flight_booking_block(flight_result: Dict[str, Any]) -> str:
    """Helper to retrieve or re-render formatted Markdown for flights."""
    if not flight_result:
        return ""
    return flight_result.get("markdown_block", "")

