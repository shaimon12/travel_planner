import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os
import requests
import json
from datetime import datetime, date
import time
import streamlit.components.v1 as components

# === JavaScript Auto-Scroll Injection (Community Fix) ===
def scroll_to_bottom():
    """Scrolls to a hidden anchor element at the bottom of the chat."""
    timestamp = time.time()
    js = f"""
    <script>
        // The Community Fix: A raw JS variable that changes every run
        var dummy_counter = {timestamp};
        
        function forceScroll() {{
            const parent = window.parent;
            const bottomElement = parent.document.getElementById('chat-bottom');
            
            if (bottomElement) {{
                bottomElement.scrollIntoView({{behavior: 'smooth', block: 'end'}});
            }} else {{
                parent.window.scrollTo({{top: parent.document.body.scrollHeight, behavior: 'smooth'}});
            }}
        }}
        
        // Multi-tap to ensure it fires after the LLM finishes rendering the markdown
        setTimeout(forceScroll, 100);
        setTimeout(forceScroll, 400);
        setTimeout(forceScroll, 800);
    </script>
    """
    components.html(js, height=0, width=0)

# === Configuration ===
st.set_page_config(
    page_title="AI Travel Planner | Smart Trip Itineraries",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto"
)

# === Secure API Key Handling ===
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except FileNotFoundError:
        st.error("⚠️ API Key not found! Please set the GEMINI_API_KEY environment variable.")
        st.stop()

genai.configure(api_key=api_key)

# === Initialize Multi-Session State ===
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Trip 1": []}
if "analytics_data" not in st.session_state:
    st.session_state.analytics_data = {"Trip 1": None}
if "destinations" not in st.session_state:
    st.session_state.destinations = {"Trip 1": "My_Trip"}
    
if "current_session" not in st.session_state:
    st.session_state.current_session = "Trip 1"
if "session_counter" not in st.session_state:
    st.session_state.session_counter = 1

# === Function 1: Smart Destination Validator ===
def validate_destination(user_input):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
        You are a geographic data cleaner.
        User Input: "{user_input}"
        
        Task:
        1. Correct any spelling errors.
        2. Return the standardized format "City, Country" or "Country".
        3. If the input is invalid/gibberish, return "INVALID".
        
        Output ONLY the corrected name. No other text.
        """
        response = model.generate_content(prompt)
        cleaned = response.text.strip()
        return None if "INVALID" in cleaned else cleaned
    except Exception:
        return user_input

# === Function 2: Live Weather Fetcher ===
def get_weather_data(city):
    """Fetches weather averages or forecast from Open-Meteo (Free API)."""
    try:
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if not geo_res.get("results"):
            return "Weather data unavailable."

        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_probability_max&forecast_days=3"
        weather_res = requests.get(weather_url, timeout=5).json()
        
        daily = weather_res.get("daily", {})
        if not daily:
            return "Weather data unavailable."
            
        summary = f"Typical weather for {city} currently:\n"
        avg_temp = sum(daily['temperature_2m_max']) / len(daily['temperature_2m_max'])
        avg_rain = sum(daily['precipitation_probability_max']) / len(daily['precipitation_probability_max'])
        
        summary += f"- Average High Temp: {avg_temp:.1f}°C\n"
        summary += f"- Rain Probability: {avg_rain:.1f}%\n"
        
        if avg_rain > 40:
            summary += "Warning: High chance of rain. Suggest indoor activities."
        else:
            summary += "Conditions: Likely dry and good for outdoors."
            
        return summary
    except Exception as e:
        return f"Weather service error: {str(e)}"

# === Function 3: Main Itinerary Generator ===
def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        current_history = st.session_state.chat_sessions[st.session_state.current_session]
        
        gemini_history = [
            {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
            for msg in current_history
        ]
        chat = model.start_chat(history=gemini_history)
        response = chat.send_message(prompt)
        return response.text
    except ResourceExhausted:
        return "⚠️ **Demo Limit Reached:** This AI agent is currently running on Google's Free Tier and has exhausted its daily quota. Please try again tomorrow!"
    except Exception as e:
        return f"⚠️ **System Error:** {str(e)}"

# === Function 4: AI Destination Analytics (JSON) ===
def get_destination_analytics(destination):
    """Uses LLM to simulate or fetch destination tourism metrics as structured JSON."""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        prompt = f"""
        Act as a tourism data analyst. Provide estimated travel analytics for '{destination}'.
        Return ONLY a raw JSON object (no markdown formatting, no backticks).
        Use this exact schema:
        {{
            "visitors": "Number with M or K (e.g., 4.2M)",
            "growth": "Percentage with sign (e.g., +8%)",
            "satisfaction": "Rating out of 5 (e.g., 4.7/5)",
            "top_demographic": "Short string (e.g., Couples & Backpackers)",
            "best_month": "Month name"
        }}
        """
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        # Fallback dictionary if parsing fails
        return {
            "visitors": "2.1M", "growth": "+5%", "satisfaction": "4.5/5",
            "top_demographic": "Families & Solo Travelers", "best_month": "September"
        }

# === Sidebar: Advanced Trip Settings & Sessions ===
with st.sidebar:
    # --- Multi-Session UI ---
    if st.button("➕ New Trip Session", use_container_width=True, type="primary"):
        st.session_state.session_counter += 1
        new_session_name = f"Trip {st.session_state.session_counter}"
        
        # Initialize state for new session
        st.session_state.chat_sessions[new_session_name] = []
        st.session_state.analytics_data[new_session_name] = None
        st.session_state.destinations[new_session_name] = "My_Trip"
        
        st.session_state.current_session = new_session_name
        st.rerun()

    st.markdown("### 🗂️ Chat History")
    
    for session_name in list(st.session_state.chat_sessions.keys()):
        col1, col2 = st.columns([0.85, 0.15])
        
        is_active = session_name == st.session_state.current_session
        button_type = "primary" if is_active else "secondary"
        
        with col1:
            if st.button(f"💬 {session_name}", key=f"btn_{session_name}", use_container_width=True, type=button_type):
                st.session_state.current_session = session_name
                st.rerun()
                
        with col2:
            if st.button("🗑️", key=f"del_{session_name}", use_container_width=True, help="Delete this trip"):
                # Clean up all state related to this session
                del st.session_state.chat_sessions[session_name]
                if session_name in st.session_state.analytics_data:
                    del st.session_state.analytics_data[session_name]
                if session_name in st.session_state.destinations:
                    del st.session_state.destinations[session_name]
                
                # Switch to a valid session
                if st.session_state.current_session == session_name:
                    remaining_sessions = list(st.session_state.chat_sessions.keys())
                    if remaining_sessions:
                        st.session_state.current_session = remaining_sessions[-1]
                    else:
                        st.session_state.session_counter += 1
                        new_name = f"Trip {st.session_state.session_counter}"
                        st.session_state.chat_sessions[new_name] = []
                        st.session_state.analytics_data[new_name] = None
                        st.session_state.destinations[new_name] = "My_Trip"
                        st.session_state.current_session = new_name
                        
                st.rerun()
            
    st.markdown("---")
    st.title("🌍 AI Travel Planner")
    st.caption("Your personalized travel concierge.")
    
    st.header("Trip Details")
    
    with st.form("trip_settings_form"):
        name = st.text_input("Traveler Name", placeholder="e.g. Alex")
        raw_destination = st.text_input("Destination", placeholder="e.g. Kyoto, Japan")
        
        default_date = date.today().strftime("%Y-%m-%d")
        start_date_str = st.text_input("Start Date (YYYY-MM-DD)", value=default_date)
        
        col1, col2 = st.columns(2)
        duration = col1.number_input("Days", 1, 30, 3)
        budget = col2.selectbox("Budget", ["Shoestring", "Moderate", "Luxury"])
        
        travel_type = st.selectbox("Travel Style", ["Solo", "Couple", "Family", "Friends"])
        group_size = st.number_input("Total Travelers", min_value=1, max_value=20, value=1)
        
        # --- THE FIX: Wrap the large multiselects in a collapsible expander ---
        with st.expander("⚙️ Advanced Preferences (Optional)", expanded=False):
            interests = st.multiselect(
                "Interests",
                ["History 🏛️", "Nature 🌳", "Food 🍜", "Adventure 🧗", "Shopping 🛍️", "Art 🎨", "Nightlife 🍹"]
            )
            
            dietary_list = st.multiselect(
                "Dietary Restrictions", 
                ["Vegetarian", "Vegan", "Halal", "Gluten-Free", "Kosher", "Seafood Allergy"]
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        submit = st.form_submit_button("Plan My Trip", type="primary", use_container_width=True)

    # Download Button 
    st.markdown("---")
    current_history = st.session_state.chat_sessions[st.session_state.current_session]
    if len(current_history) > 1:
        st.subheader("📥 Save Your Plan")
        last_msg = current_history[-1]["content"]
        
        if isinstance(last_msg, str):
            current_dest = st.session_state.destinations.get(st.session_state.current_session, "My_Trip")
            st.download_button(
                label="📄 Download Itinerary",
                data=last_msg,
                file_name=f"Itinerary_{current_dest}.md",
                mime="text/markdown"
            )
    
    # Developer Branding
    st.markdown("### 👨‍💻 Developer")
    st.caption("Built by **Shaimon Rahman**")
    st.markdown(
        """
        <div style='display: flex; gap: 10px;'>
            <a href='https://www.linkedin.com/in/shaimonrahman' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin' alt='LinkedIn'>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Main Interface ===
st.title("🌍 AI Travel Planner")

current_history = st.session_state.chat_sessions[st.session_state.current_session]

# === Main Logic (Form Submission) ===
if submit:
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        if start_date < date.today():
            st.error("⚠️ The start date cannot be in the past. Please enter a valid date.")
            st.stop()
    except ValueError:
        st.error("⚠️ Please enter a valid date in the format YYYY-MM-DD.")
        st.stop()

    group_info = f"Traveling {travel_type} ({group_size} people)"
    dietary_requirements = ", ".join(dietary_list) if dietary_list else "None"

    if not name or not raw_destination:
        st.error("⚠️ Please enter both your Name and a Destination to proceed.")
        st.stop()
    
    with st.status("🔍 Verifying destination...", expanded=True) as status:
        destination = validate_destination(raw_destination)
        
        if not destination:
            status.update(label="❌ Invalid Destination!", state="error")
            st.error(f"Could not find a place called '{raw_destination}'. Please check the spelling.")
            st.stop()
        
        # Save destination for this session
        st.session_state.destinations[st.session_state.current_session] = destination.replace(" ", "_")
        status.update(label=f"✅ Destination confirmed: {destination}", state="complete")

    if not interests:
        interests = ["General Sightseeing", "Local Culture"]

    with st.spinner("☁️ Checking live weather conditions..."):
        weather_report = get_weather_data(destination)

    with st.spinner("📊 Compiling travel analytics..."):
        st.session_state.analytics_data[st.session_state.current_session] = get_destination_analytics(destination)

    month_name = start_date.strftime("%B")
    system_prompt = f"""
    You are an elite, award-winning Travel Concierge. 
    
    **⛔ STRICT OPERATING RULES:**
    1. Only discuss travel, itineraries, and local recommendations. Refuse anything else politely.
    2. Never break character. Never state you are an AI or an LLM.
    3. If the Group includes "Family", you MUST default to kid-friendly and accessible activities. In your first response, politely ask the user if they would like to share the ages of the children so you can tailor the recommendations even further.

    **🚨 GLOBAL RULE: STRICT GOOGLE MAPS LINKS (NON-NEGOTIABLE) 🚨**
    ANY time you mention a specific physical place (restaurant, hotel, museum, shop, park, cafe, boba shop, etc.), you MUST hyperlink it using EXACTLY this format:
    [Place Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Place+Name)
    *Failure to use this exact link structure will break the application architecture.*

    **USER & TRIP CONTEXT:**
    - Traveler: {name}
    - Destination: {destination}
    - Duration: {duration} Days (Starting {start_date})
    - Budget: {budget}
    - Group: {group_info}
    - Dietary Restrictions: {dietary_requirements}
    - Core Interests: {', '.join(interests)}
    - Live Weather Forecast: {weather_report}

    **INSTRUCTION SET:**
    Analyze the user's input and determine which mode to use:

    ---

    **MODE 1: INITIAL ITINERARY GENERATION (If planning the whole trip)**
    Create a highly logistical, geo-optimized {duration}-day itinerary. Group activities by neighborhood to minimize travel time. Use the live weather to prioritize indoor vs. outdoor activities.
    
    You MUST output EXACTLY in this markdown format:
    
    ## 📅 {duration}-Day Itinerary: {destination}
    
    ### 🎒 Smart Packing List
    * [Essential 1 based on live weather]
    * [Essential 2]
    * [Essential 3]

    ---
    ### Day 1: [Catchy Theme Name]
    * **Morning:** [Activity Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Description.
    * **Lunch:** [Restaurant Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Restaurant+Name) - Cuisine & Price. (Must respect {dietary_requirements})
    * **Afternoon:** [Activity Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Description.
    * **Dinner:** [Restaurant Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Restaurant+Name) - Ambience.
    * **🌙 Evening:** [Activity/Bar](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Activity+Name) - Wind down.
    * **💡 Local Secret:** A specific local tip.
    *(Repeat structure for all days)*

    ---

    **MODE 2: CONVERSATIONAL Q&A (If the user asks a follow-up question)**
    (Examples: "Where can I get boba tea?", "Find me a halal burger place", "Make Day 2 more relaxing")
    1. DO NOT rewrite the entire itinerary unless explicitly asked to regenerate it.
    2. Provide a brief, friendly, and highly specific answer based on {destination}.
    3. You MUST STILL apply the **GLOBAL LINK RULE** to every single new place you recommend in your chat response.
    """

    st.session_state.chat_sessions[st.session_state.current_session] = [{"role": "user", "content": system_prompt}]

    with st.spinner(f"Designing the perfect {budget} trip to {destination}..."):
        response_text = get_gemini_response(system_prompt)
        st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response_text})
        
        # THE FIX: Force Streamlit to rebuild the UI instantly with the new data
        st.rerun()

# === UI Tabs Rendering ===
tab1, tab2 = st.tabs(["🗺️ Interactive Itinerary", "📊 Destination Analytics"])

with tab1:
    if len(current_history) == 0:
        st.info("👈 Fill out your trip details in the sidebar and click **Plan My Trip** to get started!")
    else:
        # Render all past messages
        for message in current_history:
            if message == current_history[0]: 
                continue 
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # INJECT THE INVISIBLE ANCHOR FOR SCROLLING
        st.markdown("<div id='chat-bottom'></div>", unsafe_allow_html=True)

with tab2:
    analytics = st.session_state.analytics_data.get(st.session_state.current_session)
    if analytics:
        current_dest = st.session_state.destinations.get(st.session_state.current_session, "My_Trip")
        dest_display = current_dest.replace("_", " ")
        
        st.header(f"📈 Tourism Trends: {dest_display}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Est. Annual Visitors", analytics.get("visitors", "N/A"), analytics.get("growth", "N/A"))
        col2.metric("Satisfaction Rating", analytics.get("satisfaction", "N/A"), "⭐️⭐️⭐️⭐️⭐️")
        col3.metric("Peak Season", analytics.get("best_month", "N/A"))
        
        st.markdown("---")
        st.subheader("👥 Primary Demographics")
        st.write(f"Most popular among: **{analytics.get('top_demographic', 'N/A')}**")
        
        st.markdown("---")
        st.subheader("🎥 Featured Video Guides")
        yt_query = dest_display.replace(" ", "+")
        st.info(f"👉 [Click here to watch the top 10 travel vlogs for {dest_display} on YouTube](https://www.youtube.com/results?search_query=top+10+travel+vlog+{yt_query})")
    else:
        st.info("👈 Generate an itinerary to unlock destination analytics!")

# === Chat Input (Outside Tabs to Pin to Bottom) ===
if prompt := st.chat_input("Ask for changes (e.g., 'Make Day 2 more relaxing')"):
    st.session_state.chat_sessions[st.session_state.current_session].append({"role": "user", "content": prompt})
    
    with st.spinner("Updating itinerary..."):
        response = get_gemini_response(prompt)
        st.session_state.chat_sessions[st.session_state.current_session].append({"role": "assistant", "content": response})
        st.rerun()

# === Copyright Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        <p>Copyright © 2026 Shaimon Rahman. All rights reserved.</p>
        <p>Powered by Google Gemini 2.5 Flash & Open-Meteo API</p>
    </div>
    """, 
    unsafe_allow_html=True
)

# === Trigger the Scroll ===
if len(current_history) > 1:
    scroll_to_bottom()