import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime

# === Configuration ===
st.set_page_config(
    page_title="AI Travel Planner Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
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

# === Initialize Session State ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_destination" not in st.session_state:
    st.session_state.current_destination = "My_Trip"

# === Function 1: Smart Destination Validator ===
def validate_destination(user_input):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
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

# === Function 2: Main Itinerary Generator ===
def get_gemini_response(prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    gemini_history = [
        {"role": "user" if msg["role"] == "user" else "model", "parts": [msg["content"]]}
        for msg in st.session_state.messages
    ]
    chat = model.start_chat(history=gemini_history)
    response = chat.send_message(prompt)
    return response.text

# === Sidebar: Advanced Trip Settings ===
with st.sidebar:
    st.title("🌍 AI Travel Planner")
    st.caption("Your personalized travel concierge.")
    
    st.header("📝 Trip Details")
    
    name = st.text_input("Traveler Name", placeholder="e.g. Alex")
    raw_destination = st.text_input("Destination", placeholder="e.g. Kyoto, Japan")
    
    col1, col2 = st.columns(2)
    duration = col1.number_input("Days", 1, 30, 3)
    budget = col2.selectbox("Budget", ["Shoestring", "Moderate", "Luxury"])
    
    travel_type = st.selectbox("Travel Style", ["Solo", "Couple", "Family", "Friends"])
    
    group_info = f"Traveling {travel_type}"
    if travel_type == "Family":
        adults = st.slider("Adults", 1, 5, 2)
        kids = st.slider("Kids", 0, 5, 1)
        group_info += f" with {adults} adults and {kids} children"
    elif travel_type == "Friends":
        people = st.slider("Group Size", 2, 10, 4)
        group_info += f" ({people} people)"

    st.subheader("🎨 Preferences")
    interests = st.multiselect(
        "Interests",
        ["History 🏛️", "Nature 🌳", "Food 🍜", "Adventure 🧗", "Shopping 🛍️", "Art 🎨", "Nightlife 🍹"]
    )
    
    dietary_requirements = "None"
    if "Food 🍜" in interests:
        st.info("🍽️ Since you like food, let's refine the menu:")
        dietary_list = st.multiselect("Dietary Restrictions", 
                                      ["Vegetarian", "Vegan", "Halal", "Gluten-Free", "Kosher", "Seafood Allergy"])
        if dietary_list:
            dietary_requirements = ", ".join(dietary_list)

    submit = st.button("🚀 Plan My Trip", type="primary")
    
    # === Sidebar: Download & Branding ===
    st.markdown("---")
    
    # Download Button (Visible only after generation)
    if len(st.session_state.messages) > 1:
        st.subheader("📥 Save Your Plan")
        last_msg = st.session_state.messages[-1]["content"]
        safe_filename = f"Itinerary_{st.session_state.current_destination}.md"
        
        st.download_button(
            label="📄 Download Itinerary",
            data=last_msg,
            file_name=safe_filename,
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
            <a href='https://github.com/shaimon12' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-View_Code-black?style=flat&logo=github' alt='GitHub'>
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# === Main Logic ===
if submit:
    if not name or not raw_destination:
        st.error("⚠️ Please enter both your Name and a Destination to proceed.")
    else:
        # Step 1: Validate Destination
        with st.status("🔍 Verifying destination...", expanded=True) as status:
            destination = validate_destination(raw_destination)
            
            if not destination:
                status.update(label="❌ Invalid Destination!", state="error")
                st.error(f"Could not find a place called '{raw_destination}'. Please check the spelling.")
                st.stop()
            else:
                st.session_state.current_destination = destination.replace(" ", "_")
                status.update(label=f"✅ Destination confirmed: {destination}", state="complete")

        # Step 2: Set Defaults
        if not interests:
            interests = ["General Sightseeing", "Local Culture"]

        # Step 3: Advanced Prompt Engineering
        current_month = datetime.now().strftime("%B")
        system_prompt = f"""
        **⛔ SCOPE & CHARACTER (CRITICAL):**
        1. You are a **Travel Concierge**, NOT a general AI assistant.
        2. If the user asks for code, creative writing, math, or general life advice, you MUST politely refuse.
           *Example Refusal:* "I specialize only in travel planning. Let's focus on your trip to {destination}!"
        3. STAY IN CHARACTER. Do not mention that you are an AI model.
        **TASK:**
        Act as an award-winning travel designer.
        Create a **highly logical, logistical, and personalized** {duration}-day itinerary for {name} visiting {destination} in {current_month}.
        
        **🚦 CORE CONSTRAINTS (MUST FOLLOW):**
        1. **Budget:** {budget} (Be strictly realistic with restaurant/activity choices).
        2. **Travel Group:** {group_info} (Adjust pace and activity types accordingly).
        3. **Dietary:** {dietary_requirements} (CRITICAL: Only suggest compatible places. If none, suggest safe alternatives).
        4. **Geography:** Group activities by neighborhood to minimize travel time. Don't zigzag across the city.
        
        **✨ THE EXPERIENCE:**
        - **Interests:** Focus heavily on {', '.join(interests)}.
        - **Pacing:** Morning activity -> Lunch -> Afternoon activity -> Dinner -> Evening vibe.
        
        **🔗 LINK FORMATTING (STRICT):**
        For EVERY specific place (Restaurant, Hotel, Park, Museum), you MUST provide a Google Maps Search link in this EXACT format:
        [Place Name](https://www.google.com/maps/search/?api=1&query={destination.replace(' ', '+')}+Place+Name)
        *Example: [Tokyo Tower](https://www.google.com/maps/search/?api=1&query=Tokyo+Japan+Tokyo+Tower)*

        **📝 OUTPUT STRUCTURE:**
        
        ## 📅 {duration}-Day Itinerary: {destination}
        
        ### 🎒 Smart Packing List ({current_month})
        * [Essential 1]
        * [Essential 2]
        * [Essential 3]

        ---
        
        ### Day 1: [Catchy Theme Name]
        * **Morning:** [Activity Name](Link) - 2 sentence description focusing on why it fits the interest.
        * **Lunch:** [Restaurant Name](Link) - Cuisine type & price range. Why it fits the budget.
        * **Afternoon:** [Activity Name](Link) - 2 sentence description.
        * **Dinner:** [Restaurant Name](Link) - Ambience check (Romantic? Lively? Family-friendly?).
        * **🌙 Evening:** [Activity/Bar/Walk](Link) - A way to wind down.
        * **💡 Local Secret:** A specific tip (e.g., "Ask for the off-menu sauce" or "Best view is from the 3rd floor").

        (Repeat for all days)
        """

        st.session_state.messages = [{"role": "user", "content": system_prompt}]

        with st.spinner(f"Designing the perfect {budget} trip to {destination}..."):
            try:
                response_text = get_gemini_response(system_prompt)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# === Chat & Results Interface ===
st.title("🌍 AI Travel Planner Pro")

for message in st.session_state.messages:
    if message == st.session_state.messages[0]:
        continue 
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask for changes (e.g., 'Make Day 2 more relaxing')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Updating itinerary..."):
            response = get_gemini_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# === Copyright Footer ===
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey;'>
        <p>Copyright © 2026 Shaimon Rahman. All rights reserved.</p>
        <p>Powered by Google Gemini 2.5 Flash</p>
    </div>
    """, 
    unsafe_allow_html=True
)