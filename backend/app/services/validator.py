import os
from typing import Optional
from app.core.config import settings

def validate_destination(user_input: str) -> Optional[str]:
    """
    Cleans and standardizes geographic destination input.
    Uses Gemini API if configured; otherwise returns sanitized input or None for gibberish.
    """
    if not user_input or not user_input.strip():
        return None

    cleaned_input = user_input.strip()

    # If Gemini API key is configured, use LLM validation
    api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(settings.GEMINI_MODEL)
            prompt = f"""
            You are a geographic data cleaner.
            User Input: "{cleaned_input}"

            Task:
            1. Correct any spelling errors.
            2. Return the standardized format "City, Country" or "Country".
            3. If the input is invalid/gibberish, return "INVALID".

            Output ONLY the corrected name. No other text.
            """
            response = model.generate_content(prompt)
            result = response.text.strip()
            if "INVALID" in result.upper():
                return None
            return result
        except Exception:
            # Fall back to title case formatted input if LLM call fails
            pass

    # Basic local sanitization fallback
    return cleaned_input.title()
