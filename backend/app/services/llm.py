import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("services.llm")

def get_api_key() -> str:
    """Retrieves OpenAI-compatible / OpenRouter API key from settings or environment."""
    return (
        getattr(settings, "OPENAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or getattr(settings, "GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    )

def get_base_url() -> str:
    """Retrieves OpenAI-compatible API base URL."""
    return (
        getattr(settings, "OPENAI_BASE_URL", "")
        or os.getenv("OPENAI_BASE_URL", "")
        or "https://openrouter.ai/api/v1"
    )

def get_model_name(override: Optional[str] = None) -> str:
    """Resolves model name, defaulting to Qwen 2.5 on OpenRouter."""
    if override:
        return override
    return (
        getattr(settings, "LLM_MODEL", "")
        or os.getenv("LLM_MODEL", "")
        or getattr(settings, "GEMINI_MODEL", "")
        or "qwen/qwen-2.5-7b-instruct:free"
    )

def get_openai_client():
    """Initializes and returns an OpenAI client configured for OpenRouter / Qwen / OpenAI."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        base_url = get_base_url()
        # Include OpenRouter attribution headers when targeting OpenRouter
        default_headers = {}
        if "openrouter.ai" in base_url:
            default_headers = {
                "HTTP-Referer": "https://github.com/shaimon12/travel_planner",
                "X-Title": "AI Travel Planner Pro",
            }
        return OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers if default_headers else None,
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return None

def invoke_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> str:
    """
    Invokes OpenAI-compatible API (e.g., Qwen on OpenRouter, Ollama, or OpenAI)
    with optional chat history and system instructions.
    """
    api_key = get_api_key()
    model_name = get_model_name(model)

    if not api_key:
        return (
            "⚠️ **API Key Missing:** Please provide a valid `OPENAI_API_KEY` in `backend/.env` "
            "to generate live itineraries and chat responses with Qwen/OpenRouter."
        )

    try:
        from openai import OpenAI
    except ImportError:
        return "⚠️ **Dependency Missing:** Please install `openai` via `pip install openai` to use the LLM service."

    client = get_openai_client()
    if not client:
        return "⚠️ **Client Error:** Unable to initialize OpenAI-compatible client. Please verify your configuration."

    # Format messages array for Chat Completions
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})

    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ["assistant", "ai", "model"]:
                messages.append({"role": "assistant", "content": content})
            elif role == "system":
                messages.append({"role": "system", "content": content})
            else:
                messages.append({"role": "user", "content": content})

    messages.append({"role": "user", "content": prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content or ""
            return content.strip()
        return ""
    except Exception as e:
        err_msg = str(e)
        logger.error(f"OpenAI/OpenRouter call failed: {err_msg}")
        if "429" in err_msg or "rate" in err_msg.lower() or "quota" in err_msg.lower():
            return (
                "⚠️ **Rate Limit Exceeded:** The open-weight model quota or rate limit was reached. "
                "Please retry in a moment or provide an OpenRouter API key."
            )
        elif "401" in err_msg or "auth" in err_msg.lower():
            return "⚠️ **Authentication Error:** The provided `OPENAI_API_KEY` was rejected by the provider."
        return f"⚠️ **AI Service Error:** {err_msg}"

def get_destination_analytics(prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Invokes LLM expecting a structured JSON destination analytics object.
    Requests response_format={"type": "json_object"} when supported,
    with automatic fallback and robust regex/markdown stripping.
    """
    fallback_data = {
        "visitors": "2.8M",
        "growth": "+5%",
        "satisfaction": "4.6/5",
        "top_demographic": "Families & Solo Explorers",
        "best_month": "October",
    }

    api_key = get_api_key()
    if not api_key:
        return fallback_data

    client = get_openai_client()
    if not client:
        return fallback_data

    model_name = get_model_name(model)
    messages = [
        {
            "role": "system",
            "content": "You are a travel analytics engine. You MUST respond with ONLY a valid, parseable JSON object matching the requested schema."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    raw_response = ""
    # 1. Attempt with json_object response_format
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        if response.choices:
            raw_response = response.choices[0].message.content or ""
    except Exception as e:
        # If response_format={"type": "json_object"} is unsupported by the specific open-weight model, retry without it
        logger.warning(f"JSON mode request failed ({e}), retrying with standard completion...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
            )
            if response.choices:
                raw_response = response.choices[0].message.content or ""
        except Exception as retry_err:
            logger.error(f"Fallback completion call failed: {retry_err}")
            return fallback_data

    if not raw_response:
        return fallback_data

    # 2. Robust Markdown and code-fence cleaning
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_response.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE).strip()

    # Extract first '{' through matching or last '}'
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        parsed = json.loads(cleaned)
        # Ensure minimum required keys exist
        for key in ["visitors", "growth", "satisfaction", "top_demographic", "best_month"]:
            if key not in parsed:
                parsed[key] = fallback_data[key]
        return parsed
    except Exception as parse_err:
        logger.warning(f"Failed to parse LLM JSON output ({parse_err}). Cleaned string: '{cleaned}'")
        return fallback_data

# Backwards compatibility aliases matching exact function signatures
invoke_gemini = invoke_llm
invoke_gemini_json = get_destination_analytics
invoke_llm_json = get_destination_analytics
