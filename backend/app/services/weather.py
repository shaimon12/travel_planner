import requests
from typing import Dict, Any, Optional

def get_weather_data(city: str) -> str:
    """
    Fetches real-time weather averages or 3-day forecast from Open-Meteo.
    Handles geocoding and forecast queries with fallback error handling.
    """
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()

        if not geo_res.get("results"):
            return "Weather data unavailable."

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_probability_max&forecast_days=3"
        )
        weather_res = requests.get(weather_url, timeout=5).json()

        daily = weather_res.get("daily", {})
        if not daily or not daily.get("temperature_2m_max") or not daily.get("precipitation_probability_max"):
            return "Weather data unavailable."

        avg_temp = sum(daily["temperature_2m_max"]) / len(daily["temperature_2m_max"])
        avg_rain = sum(daily["precipitation_probability_max"]) / len(daily["precipitation_probability_max"])

        summary = f"Typical weather for {city} currently:\n"
        summary += f"- Average High Temp: {avg_temp:.1f}°C\n"
        summary += f"- Rain Probability: {avg_rain:.1f}%\n"

        if avg_rain > 40:
            summary += "Warning: High chance of rain. Suggest indoor activities."
        else:
            summary += "Conditions: Likely dry and good for outdoors."

        return summary
    except Exception as e:
        return f"Weather service error: {str(e)}"
