# AI Travel Planner Pro — Backend API

Modern, scalable FastAPI backend for the AI Travel Planner SaaS platform.

## Architecture
- **Framework**: FastAPI (Python 3.11+)
- **ORM / Persistence**: SQLAlchemy 2.0 with PostgreSQL (and zero-config SQLite local development fallback)
- **AI Orchestration**: LangGraph & Google Gemini (`gemini-2.5-flash`)
- **External Services**: Open-Meteo Weather API & Geocoding

## Project Structure
```
backend/
├── app/
│   ├── main.py              # Application entrypoint & CORS configuration
│   ├── core/                # Settings & database engine/session lifecycle
│   │   ├── config.py
│   │   └── database.py
│   ├── models/              # SQLAlchemy database entities
│   │   ├── user.py          # User management
│   │   ├── trip.py          # TripSession (replaces Streamlit session_state)
│   │   ├── message.py       # Chat message history
│   │   └── analytics.py     # Destination tourism analytics
│   ├── schemas/             # Pydantic validation & response schemas
│   │   ├── user.py
│   │   ├── trip.py
│   │   ├── message.py
│   │   └── analytics.py
│   ├── routers/             # FastAPI REST endpoints
│   │   ├── health.py        # System health & DB check
│   │   └── trips.py         # Trip session & chat endpoints
│   └── services/            # External integrations & Agent workflows
│       ├── weather.py       # Open-Meteo weather integration
│       ├── validator.py     # Destination cleaning & validation
│       └── agent.py         # Multi-agent workflow orchestrator
├── requirements.txt         # Production & development dependencies
└── .env.example             # Environment variable template
```

## Setup & Running Locally

1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and supply GEMINI_API_KEY
```

4. Run the server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive Swagger docs available at `http://localhost:8000/docs`.
