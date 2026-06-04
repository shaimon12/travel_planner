# 🌍 AI Travel Planner Pro

### Overview
A production-grade, personalized travel concierge application built with Streamlit and powered by Google Gemini 2.5 Flash. Designed with a product-first mindset, this tool generates highly logical, budget-aware, and geo-optimized travel itineraries while managing multiple concurrent user sessions and rendering live destination analytics.

### Key Features
* **Multi-Trip Memory:** Robust session state management allows users to plan, navigate, and save multiple concurrent trip itineraries without losing chat history.
* **Destination Analytics:** Leverages a structured JSON pipeline via the LLM to generate a real-time dashboard of tourism trends, peak seasons, and demographic data.
* **Intelligent Routing & Pacing:** Uses advanced prompt engineering to create day-by-day logistical plans, grouping activities by neighborhood to minimize travel time.
* **Live Weather Integration:** Fetches real-time climate data via the Open-Meteo API to dynamically adjust packing lists and prioritize indoor vs. outdoor activities based on rain probability.
* **Smart Maps:** Automatically generates precise, deep-linked Google Maps URLs for every suggested restaurant, hotel, and attraction.
* **Responsive "SaaS" UI:** Features a dynamic view-state controller that seamlessly transitions from a clean, centered landing page form into a fully featured chat and analytics dashboard.
* **Exportable:** Users can instantly download their complete itinerary as a formatted Markdown file for offline use.

### Tech Stack
* **Frontend:** Streamlit (with custom HTML/JS injections for responsive mobile auto-scrolling)
* **AI Model:** Google Gemini 2.5 Flash (google-generativeai)
* **External APIs:** Open-Meteo (Geocoding & Live Weather)
* **Deployment:** AWS EC2 (Linux/Ubuntu)
* **Language:** Python 3.11+

### Installation
1. **Clone the repo:**
    git clone https://github.com/shaimon12/travel_planner.git
    cd travel_planner

2. **Install dependencies:**
    pip install -r requirements.txt

3. **Set up your API Key:**
    Create a .streamlit/secrets.toml file or set an environment variable:
    GEMINI_API_KEY = "your_api_key_here"

4. **Run the application:**
    streamlit run app.py

---
**Built by Shaimon Rahman** [![LinkedIn Connect](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/shaimonrahman)
