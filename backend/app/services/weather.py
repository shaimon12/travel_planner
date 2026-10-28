import requests
from typing import Dict, Any, Optional

def get_weather_data(city: str) -> str:
    """
    Fetches real-time weather averages and 3-day forecast from Open-Meteo API.
    Handles geocoding and forecast queries with robust fallback error handling.
    """
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()

        if not geo_res.get("results"):
            return "Weather data unavailable for this location."

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_probability_max&forecast_days=3"
        )
        weather_res = requests.get(weather_url, timeout=5).json()

        daily = weather_res.get("daily", {})
        temps = daily.get("temperature_2m_max")
        rain_probs = daily.get("precipitation_probability_max")

        if not daily or not temps or not rain_probs:
            return "Weather forecast data currently unavailable."

        avg_temp = sum(temps) / len(temps)
        avg_rain = sum(rain_probs) / len(rain_probs)

        summary = f"Typical weather for {city} currently:\n"
        summary += f"- Average High Temp: {avg_temp:.1f}°C\n"
        summary += f"- Rain Probability: {avg_rain:.1f}%\n"

        if avg_rain > 40:
            summary += "Warning: High chance of rain. Suggest indoor activities and packing waterproof gear."
        else:
            summary += "Conditions: Likely dry and good for outdoor sightseeing."

        return summary
    except Exception as e:
        return f"Weather service unavailable: {str(e)}"

def get_structured_weather(city: str) -> Dict[str, Any]:
    """Returns structured weather parameters for analytics or specialized agent nodes."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()

        if not geo_res.get("results"):
            return {"available": False, "summary": "Location not found."}

        first_match = geo_res["results"][0]
        lat = first_match["latitude"]
        lon = first_match["longitude"]
        resolved_name = f"{first_match.get('name', city)}, {first_match.get('country', '')}".strip(", ")

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_probability_max&forecast_days=3"
        )
        weather_res = requests.get(weather_url, timeout=5).json()
        daily = weather_res.get("daily", {})

        temps = daily.get("temperature_2m_max", [])
        rain_probs = daily.get("precipitation_probability_max", [])

        avg_temp = sum(temps) / len(temps) if temps else 20.0
        avg_rain = sum(rain_probs) / len(rain_probs) if rain_probs else 10.0

        return {
            "available": True,
            "resolved_name": resolved_name,
            "latitude": lat,
            "longitude": lon,
            "avg_temp": round(avg_temp, 1),
            "rain_probability": round(avg_rain, 1),
            "summary": get_weather_data(city)
        }
    except Exception:
        return {"available": False, "summary": "Weather service error."}
