import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from app.core.config import settings

logger = logging.getLogger("services.llm")

def get_api_key() -> str:
    """Retrieves OpenRouter / OpenAI SDK API key from settings or environment."""
    return (
        getattr(settings, "OPENROUTER_API_KEY", "")
        or os.getenv("OPENROUTER_API_KEY", "")
        or getattr(settings, "OPENAI_API_KEY", "")
        or os.getenv("OPENAI_API_KEY", "")
        or getattr(settings, "GEMINI_API_KEY", "")
        or os.getenv("GEMINI_API_KEY", "")
    )

def get_base_url() -> str:
    """Retrieves OpenRouter / OpenAI-compatible API base URL."""
    return (
        getattr(settings, "OPENROUTER_BASE_URL", "")
        or os.getenv("OPENROUTER_BASE_URL", "")
        or getattr(settings, "OPENAI_BASE_URL", "")
        or os.getenv("OPENAI_BASE_URL", "")
        or "https://openrouter.ai/api/v1"
    )

def get_model_name(override: Optional[str] = None) -> str:
    """Resolves model name, defaulting to 'openrouter/free' for auto-routing zero-cost inference."""
    if override and override.strip():
        return override.strip()
    return (
        getattr(settings, "PRIMARY_MODEL", "")
        or os.getenv("PRIMARY_MODEL", "")
        or getattr(settings, "LLM_MODEL", "")
        or os.getenv("LLM_MODEL", "")
        or "openrouter/free"
    )

def get_openai_client():
    """Initializes and returns the shared global server OpenAI client configured for OpenRouter auto-routing."""
    api_key = get_api_key()
    if not api_key:
        return None
    try:
        from openai import OpenAI
        base_url = get_base_url()
        # OpenRouter attribution headers
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
        logger.error(f"Failed to initialize server OpenAI/OpenRouter client: {e}")
        return None

def invoke_llm(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
    x_api_key: Optional[str] = None,
    x_base_url: Optional[str] = None,
    x_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Invokes LLM with optional multi-turn conversation history and system instructions.
    Supports BYOK (Bring Your Own Key) overrides via localized temporary client instantiation
    scoped strictly to this call (never mutating or overwriting the global server client).
    Enforces strict zero-trust logging (custom keys are never logged or persisted).
    """
    custom_key = (x_api_key or api_key or "").strip()
    effective_api_key = custom_key if custom_key else get_api_key()
    custom_base_url = (x_base_url or base_url or "").strip()
    effective_base_url = custom_base_url if custom_base_url else get_base_url()
    model_name = get_model_name(x_model_name or model)

    # Zero-trust logging: Never log custom_key or secret tokens
    if custom_key:
        logger.info(f"Invoking LLM with temporary BYOK client (model='{model_name}', base_url='{effective_base_url}')")
    else:
        logger.info(f"Invoking LLM with default server client (model='{model_name}')")

    try:
        from openai import OpenAI, AsyncClient
    except ImportError:
        return "⚠️ **Dependency Missing:** Please install `openai` via `pip install openai` to use the LLM service."

    # Client Resolution: Localized temporary client vs shared server client
    if custom_key:
        # Instantiate a localized, temporary client scoped strictly to this specific function call.
        # Do NOT overwrite the global server client.
        custom_headers = {}
        if "openrouter.ai" in effective_base_url:
            custom_headers = {
                "HTTP-Referer": "https://github.com/shaimon12/travel_planner",
                "X-Title": "AI Travel Planner Pro (BYOK)",
            }
        try:
            # Instantiate temporary scoped client for thread-safe execution
            client = OpenAI(
                api_key=custom_key,
                base_url=effective_base_url,
                default_headers=custom_headers if custom_headers else None,
            )
            # Reference AsyncClient class definition for localized scoped async pipelines
            _ = AsyncClient
        except Exception as init_err:
            logger.error(f"Failed to instantiate localized temporary BYOK client: {init_err}")
            return f"⚠️ **BYOK Client Error:** Failed to initialize client with provided base URL: {init_err}"
    else:
        if not effective_api_key:
            return (
                "⚠️ **API Key Missing:** Please provide a valid `OPENROUTER_API_KEY` in `backend/.env` "
                "or supply a custom API key via the Advanced Engine Settings (BYOK)."
            )
        # Default to the global server client (the free OpenRouter tier)
        client = get_openai_client()
        if not client:
            return "⚠️ **Client Error:** Unable to initialize OpenAI-compatible server client. Please verify your configuration."

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
        logger.error(f"LLM completion call failed: {err_msg}")
        if "429" in err_msg or "rate" in err_msg.lower() or "quota" in err_msg.lower():
            return (
                "⚠️ **Rate Limit Exceeded:** The model provider rate limit was reached. "
                "Please retry in a moment or provide your own API key via Settings."
            )
        elif "401" in err_msg or "auth" in err_msg.lower():
            return "⚠️ **Authentication Error:** The provided API key was rejected by the provider."
        elif "404" in err_msg or "not found" in err_msg.lower():
            return f"⚠️ **Model Not Found (404):** The model '{model_name}' was not found on '{effective_base_url}'."
        return f"⚠️ **AI Service Error:** {err_msg}"

async def invoke_llm_async(
    prompt: str,
    system_instruction: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
    x_api_key: Optional[str] = None,
    x_base_url: Optional[str] = None,
    x_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """
    Asynchronous version of invoke_llm using a localized openai.AsyncClient.
    Instantiates a temporary AsyncClient strictly scoped to this coroutine if BYOK key is provided.
    Enforces strict zero-trust logging.
    """
    custom_key = (x_api_key or api_key or "").strip()
    effective_api_key = custom_key if custom_key else get_api_key()
    custom_base_url = (x_base_url or base_url or "").strip()
    effective_base_url = custom_base_url if custom_base_url else get_base_url()
    model_name = get_model_name(x_model_name or model)

    if custom_key:
        logger.info(f"Invoking LLM (async) with temporary BYOK client (model='{model_name}', base_url='{effective_base_url}')")
    else:
        logger.info(f"Invoking LLM (async) with default server client (model='{model_name}')")

    try:
        from openai import AsyncClient
    except ImportError:
        return "⚠️ **Dependency Missing:** Please install `openai` via `pip install openai` to use the LLM service."

    if not effective_api_key:
        return "⚠️ **API Key Missing:** Please provide a valid API key."

    custom_headers = {}
    if "openrouter.ai" in effective_base_url:
        custom_headers = {
            "HTTP-Referer": "https://github.com/shaimon12/travel_planner",
            "X-Title": "AI Travel Planner Pro (BYOK)",
        }

    # Format messages
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
        async with AsyncClient(
            api_key=effective_api_key,
            base_url=effective_base_url,
            default_headers=custom_headers if custom_headers else None,
        ) as async_client:
            response = await async_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
            )
            if response.choices and len(response.choices) > 0:
                return (response.choices[0].message.content or "").strip()
            return ""
    except Exception as e:
        err_msg = str(e)
        logger.error(f"Async LLM completion call failed: {err_msg}")
        return f"⚠️ **AI Service Error:** {err_msg}"

def get_destination_analytics(
    prompt: str,
    model: Optional[str] = None,
    x_api_key: Optional[str] = None,
    x_base_url: Optional[str] = None,
    x_model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Invokes LLM expecting a structured JSON destination analytics object.
    Requests strict JSON output with robust markdown code-fence parsing.
    Supports localized temporary client instantiation if BYOK credentials are provided.
    Enforces strict zero-trust logging.
    """
    fallback_data = {
        "visitors": "2.8M",
        "growth": "+5%",
        "satisfaction": "4.6/5",
        "top_demographic": "Families & Solo Explorers",
        "best_month": "October",
    }

    custom_key = (x_api_key or api_key or "").strip()
    effective_api_key = custom_key if custom_key else get_api_key()
    custom_base_url = (x_base_url or base_url or "").strip()
    effective_base_url = custom_base_url if custom_base_url else get_base_url()
    model_name = get_model_name(x_model_name or model)

    if not effective_api_key:
        return fallback_data

    try:
        from openai import OpenAI
    except ImportError:
        return fallback_data

    if custom_key:
        custom_headers = {}
        if "openrouter.ai" in effective_base_url:
            custom_headers = {
                "HTTP-Referer": "https://github.com/shaimon12/travel_planner",
                "X-Title": "AI Travel Planner Pro (BYOK)",
            }
        try:
            client = OpenAI(
                api_key=custom_key,
                base_url=effective_base_url,
                default_headers=custom_headers if custom_headers else None,
            )
        except Exception:
            return fallback_data
    else:
        client = get_openai_client()
        if not client:
            return fallback_data

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert tourism data analyst. You MUST return ONLY a raw JSON object. "
                "Do NOT include markdown formatting, code block ticks, or any introductory/explanatory text."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    raw_response = ""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
        )
        if response.choices:
            raw_response = response.choices[0].message.content or ""
    except Exception as e:
        logger.error(f"Analytics completion call failed: {e}")
        return fallback_data

    if not raw_response:
        return fallback_data

    # Markdown and code-fence regex cleaning
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        cleaned = cleaned[start_idx : end_idx + 1]

    try:
        parsed = json.loads(cleaned)
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
