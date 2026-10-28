from typing import Optional
from app.services.llm import invoke_gemini

def validate_destination(user_input: str) -> Optional[str]:
    """
    Cleans, spells-corrects, and standardizes geographic destination input.
    Returns standardized format 'City, Country' or 'Country', or None if gibberish/invalid.
    """
    if not user_input or not user_input.strip():
        return None

    cleaned_input = user_input.strip()

    prompt = f"""
    You are a geographic data cleaner.
    User Input: "{cleaned_input}"

    Task:
    1. Correct any spelling errors.
    2. Return the standardized format "City, Country" or "Country".
    3. If the input is invalid/gibberish, return "INVALID".

    Output ONLY the corrected name. No other text.
    """

    response = invoke_gemini(prompt, temperature=0.0)

    if not response or "⚠️" in response or "error" in response.lower():
        # Fall back to capitalized input if LLM call unavailable or encounters error
        return cleaned_input.title()

    cleaned_result = response.strip().replace('"', '').replace("'", "")
    if "INVALID" in cleaned_result.upper():
        return None

    return cleaned_result
