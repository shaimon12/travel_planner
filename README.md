# AI Travel Planner Pro

### Overview
A personalized travel concierge application powered by Google Gemini 2.5 Flash and Streamlit. This tool generates logical, budget-aware, and interest-based travel itineraries with real-time Google Maps integration.

### Key Features
* **Intelligent Planning:** Uses Chain-of-Thought prompting to create logistical day-by-day plans.
* **Smart Maps:** Automatically generates deep links to Google Maps for every suggested location.
* **Context Aware:** Suggests packing lists based on the destination's current weather.
* **Exportable:** Users can download their complete itinerary as a Markdown file.
* **Robust Error Handling:** Includes AI-powered destination validation to fix typos automatically.

### Tech Stack
* **Frontend:** Streamlit
* **AI Model:** Google Gemini 2.5 Flash (via google-generativeai)
* **Deployment:** AWS EC2 (Linux/Ubuntu)
* **Language:** Python 3.11+

### Installation
1. Clone the repo:
   git clone https://github.com/shaimon12/travel_planner.git

2. Install dependencies:
   pip install -r requirements.txt

3. Set up your API Key (in .streamlit/secrets.toml or env vars):
   GEMINI_API_KEY = "your_api_key_here"

4. Run the app:
   streamlit run app.py

---
Built by Shaimon Rahman (https://www.linkedin.com/in/shaimonrahman)
