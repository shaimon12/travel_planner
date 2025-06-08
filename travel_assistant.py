# import streamlit as st
# import google.generativeai as genai
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import pycountry
# import geonamescache
# from difflib import get_close_matches
# import torch
# import time

import streamlit as st
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import pycountry
import geonamescache
from difflib import get_close_matches
import torch
import time
import html

# === Configure Gemini Securely ===
genai.configure(api_key="YOUR_API_KEY")
gemini_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# === Load TinyLLaMA Once ===
if "llama_model" not in st.session_state:
    with st.spinner("Loading TinyLLaMA model..."):
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        llama_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        ).to("mps" if torch.backends.mps.is_available() else "cpu")
        st.session_state.llama_model = llama_model
        st.session_state.tokenizer = tokenizer

# === Utility Functions ===
def get_all_country_names():
    return [country.name for country in pycountry.countries]

def get_all_city_names():
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()
    return [city_data['name'] for city_data in cities.values()]

def validate_destination(user_input):
    countries = get_all_country_names()
    cities = get_all_city_names()
    match_country = get_close_matches(user_input, countries, n=1, cutoff=0.7)
    match_city = get_close_matches(user_input, cities, n=1, cutoff=0.8)
    if match_country:
        return match_country[0], "country"
    elif match_city:
        return match_city[0], "city"
    else:
        return None, None

def build_prompt(name, age, travel_type, group_info, destination, interests, budget, duration_days, dietary):
    group_details = f", {group_info}" if group_info else ""
    dietary_info = f" They have the following dietary restrictions: {', '.join(dietary)}." if dietary else ""
    return f"""
    You are a travel assistant. Create a detailed {duration_days}-day itinerary for a {travel_type} traveler named {name}, aged {age}{group_details}, visiting {destination}.
    They are interested in {', '.join(interests)}, with a {budget} budget.{dietary_info}
    Provide morning, afternoon, and evening activities for each day. Include food suggestions and local cultural highlights.
    """

def render_chat_box(chat_history):
    with st.container():
        for user_input, reply in chat_history:
            with st.container():
                st.markdown("**👤 You:**")
                st.markdown(f"> {user_input}")
                st.markdown("**🤖 Gemini:**")
                with st.expander("View response", expanded=True):
                    st.markdown(reply)



# === Page Title ===
st.title("🌍 AI Travel Assistant")
st.caption("Plan trips with Gemini (Online) and TinyLLaMA (Offline, Private)")

# === Travel Type Input ===
travel_type = st.selectbox("Travel Style", ["", "solo", "with family", "with friends"])
group_info = ""
if travel_type == "with family":
    st.markdown("👨‍👩‍👧 Family Details")
    num_adults = st.number_input("Number of Adults", min_value=1, max_value=10, value=2)
    num_kids = st.number_input("Number of Children", min_value=0, max_value=10, value=1)
    group_info = f"traveling with family ({num_adults} adults and {num_kids} children)"
elif travel_type == "with friends":
    st.markdown("👫 Friends Group Details")
    num_friends = st.number_input("Number of Friends (excluding you)", min_value=1, max_value=10, value=3)
    group_info = f"traveling with {num_friends} friends"

# === Input Form ===
with st.form("trip_form"):
    st.header("✈️ Trip Preferences")
    name = st.text_input("Your Name")
    age = st.number_input("Age", 1, 100)
    interests = st.multiselect("Travel Interests", ["museums", "local food", "architecture", "nature", "beach", "adventure"])
    destination = st.text_input("Destination")
    budget = st.selectbox("Budget", ["", "low", "moderate", "high"])
    dietary = st.multiselect("Any Dietary Restrictions?", ["No dietary restrictions", "vegetarian", "vegan", "gluten-free", "halal", "kosher", "lactose intolerant"])
    if "No dietary restrictions" in dietary:
        dietary = []
    duration_days = st.slider("Duration (Days)", 1, 14)

    col1, col2 = st.columns(2)
    gemini_submit = col1.form_submit_button("🌐 Generate with Gemini")
    llama_submit = col2.form_submit_button("🦙 Generate with TinyLLaMA")

# === Prompt Processing ===
def process_prompt_submission():
    corrected_destination, destination_type = validate_destination(destination)
    if not name or not travel_type or not interests or not destination or not budget:
        st.warning("Please complete all fields.")
        return None, None
    elif not corrected_destination:
        st.warning("⚠️ Destination not recognized. Please check the spelling.")
        return None, None
    st.session_state.destination_type = destination_type
    st.info(f"Using corrected destination: **{corrected_destination}** (recognized as a {destination_type})")
    prompt = build_prompt(name, age, travel_type, group_info, corrected_destination, interests, budget, duration_days, dietary)
    return prompt, corrected_destination

# === Gemini Output ===
if gemini_submit:
    prompt, destination = process_prompt_submission()
    if prompt:
        with st.spinner("Generating with Gemini..."):
            start = time.time()
            gemini_response = gemini_model.generate_content(prompt)
            end = time.time()
            st.session_state.gemini_result = gemini_response.text
            st.session_state.gemini_time = round(end - start, 2)
            st.session_state.chat_history = [("Initial request", gemini_response.text)]

# === TinyLLaMA Output ===
if llama_submit:
    prompt, destination = process_prompt_submission()
    if prompt:
        with st.spinner("Generating with TinyLLaMA..."):
            tokenizer = st.session_state.tokenizer
            model = st.session_state.llama_model
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            start = time.time()
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=500, temperature=0.8, top_p=0.9, do_sample=True)
            end = time.time()

            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            st.session_state.llama_result = output_text
            st.session_state.llama_time = round(end - start, 2)

# === Display Results ===
if "gemini_result" in st.session_state or "llama_result" in st.session_state:
    st.header("🧳 Itinerary Comparison")
    tabs = st.tabs(["🌐 Gemini", "🦙 TinyLLaMA"])

    with tabs[0]:
        if "gemini_result" in st.session_state:
            render_chat_box(st.session_state.chat_history)

            st.markdown(f"⏱️ **Time Taken**: {st.session_state.gemini_time} seconds")
            st.download_button("📥 Download Gemini Itinerary", st.session_state.gemini_result, file_name="gemini_itinerary.txt")

            # === Modify Itinerary Chat ===
            st.subheader("💬 Modify Your Itinerary")
            chat_input = st.text_input("Ask a question or request a change")

            if st.button("Submit Modification"):
                if chat_input:
                    full_context = f"""
                    Current itinerary:
                    {st.session_state.chat_history[-1][1]}

                    User request:
                    {chat_input}
                    """
                    with st.spinner("Processing your request..."):
                        response = gemini_model.generate_content(full_context)
                        st.session_state.chat_history.append((chat_input, response.text))

            render_chat_box(st.session_state.chat_history)

    with tabs[1]:
        if "llama_result" in st.session_state:
            st.markdown("### TinyLLaMA Result (Offline)")
            st.code(st.session_state.llama_result)
            st.markdown(f"⏱️ **Time Taken**: {st.session_state.llama_time} seconds")
            st.info("🛡️ Runs entirely offline. No data sent externally.")
            st.download_button("📥 Download TinyLLaMA Itinerary", st.session_state.llama_result, file_name="tinyllama_itinerary.txt")

# === Summary ===
if "gemini_time" in st.session_state and "llama_time" in st.session_state:
    st.markdown("---")
    st.subheader("📊 Model Comparison Summary")
    st.markdown(f"""
    - **Gemini**: {st.session_state.gemini_time} seconds  
    - **TinyLLaMA**: {st.session_state.llama_time} seconds  
    - 📝 *Gemini is fluent & fast online. TinyLLaMA runs offline for privacy-focused use.*
    """)
