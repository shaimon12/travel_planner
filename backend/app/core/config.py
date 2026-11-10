import os
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Travel Planner Pro"
    API_V1_STR: str = "/api"
    DEBUG: bool = True
    
    # Database URL: default to SQLite for seamless local dev, easily overridden by PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./travel_planner.db")

    # JWT Authentication & Security - strictly loaded from environment variable
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # LLM Settings (OpenAI-compatible: OpenRouter, Qwen, Ollama, vLLM, OpenAI)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen/qwen-2.5-7b-instruct:free")

    # Backwards compatibility alias for Gemini settings
    GEMINI_API_KEY: str = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("LLM_MODEL") or os.getenv("GEMINI_MODEL", "qwen/qwen-2.5-7b-instruct:free")
    
    # CORS Origins
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ]
    
    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
