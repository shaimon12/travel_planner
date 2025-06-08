# app.py

import streamlit as st
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import pycountry
from difflib import get_close_matches
import torch
import time

# === Configure Gemini ===
genai.configure(api_key="AIzaSyCLool3B5af9FgqBl4twisrgPwbYprW3F8")
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

def validate_destination(user_input):
    countries = get_all_country_names()
    match = get_close_matches(user_input, countries, n=1, cutoff=0.7)
    return match[0] if match else None

def build_prompt(name, age, travel_type, destination, interests, budget, duration_days):
    return f"""
    You are a travel assistant. Create a detailed {duration_days}-day itinerary for a {travel_type} traveler named {name}, aged {age}, visiting {destination}.
    They are interested in {', '.join(interests)}, with a {budget} budget.
    Provide morning, afternoon, and evening activities for each day. Include food suggestions and local cultural highlights.
    """

# === Page Title ===
st.title("🌍 AI Travel Assistant")
st.caption("Plan trips with Gemini (Online) and TinyLLaMA (Offline, Private)")

# === User Input Form ===
with st.form("trip_form"):
    st.header("✈️ Trip Preferences")
    name = st.text_input("Your Name")
    age = st.number_input("Age", 1, 100)
    travel_type = st.selectbox("Travel Style", ["", "solo", "with family", "with friends"])
    interests = st.multiselect("Travel Interests", ["museums", "local food", "architecture", "nature", "beach", "adventure"])
    destination = st.text_input("Destination")
    budget = st.selectbox("Budget", ["", "low", "moderate", "high"])
    duration_days = st.slider("Duration (Days)", 1, 14)

    col1, col2 = st.columns(2)
    with col1:
        gemini_submit = st.form_submit_button("🌐 Generate with Gemini")
    with col2:
        llama_submit = st.form_submit_button("🦙 Generate with TinyLLaMA")

# === Validate and Build Prompt ===
def process_prompt_submission(generate_with):
    corrected_destination = validate_destination(destination)
    if not name or not travel_type or not interests or not destination or not budget:
        st.warning("Please complete all fields.")
        return None, None
    elif not corrected_destination:
        st.warning("⚠️ Destination not recognized. Please check the spelling.")
        return None, None
    else:
        st.info(f"Using corrected destination: **{corrected_destination}**")
        prompt = build_prompt(name, age, travel_type, corrected_destination, interests, budget, duration_days)
        return prompt, corrected_destination

# === GEMINI Output ===
if gemini_submit:
    prompt, destination = process_prompt_submission("gemini")
    if prompt:
        with st.spinner("Generating with Gemini..."):
            start = time.time()
            gemini_response = gemini_model.generate_content(prompt)
            end = time.time()
            st.session_state.gemini_result = gemini_response.text
            st.session_state.gemini_time = round(end - start, 2)

# === TinyLLaMA Output ===
if llama_submit:
    prompt, destination = process_prompt_submission("llama")
    if prompt:
        with st.spinner("Generating with TinyLLaMA (offline)..."):
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

# === Results Display ===
if "gemini_result" in st.session_state or "llama_result" in st.session_state:
    st.header("🧳 Itinerary Comparison")

    tabs = st.tabs(["🌐 Gemini", "🦙 TinyLLaMA"])

    with tabs[0]:
        if "gemini_result" in st.session_state:
            st.markdown(st.session_state.gemini_result)
            st.markdown(f"⏱️ **Time Taken**: {st.session_state.gemini_time} seconds")

            # Translation
            lang = st.selectbox("🌍 Translate Gemini Result", ["None", "Spanish", "French", "German", "Arabic", "Hindi"])
            if lang != "None":
                with st.spinner(f"Translating to {lang}..."):
                    trans_prompt = f"Translate this travel itinerary into {lang}:\n\n{st.session_state.gemini_result}"
                    translated = gemini_model.generate_content(trans_prompt)
                    st.markdown(f"### 📄 Gemini Translation ({lang})")
                    st.markdown(translated.text)

            st.download_button("📥 Download Gemini Itinerary", st.session_state.gemini_result, file_name="gemini_itinerary.txt")

    with tabs[1]:
        if "llama_result" in st.session_state:
            st.markdown(st.session_state.llama_result)
            st.markdown(f"⏱️ **Time Taken**: {st.session_state.llama_time} seconds")
            st.info("🛡️ Runs entirely offline on your device. No data is sent to any external server.")
            st.download_button("📥 Download TinyLLaMA Itinerary", st.session_state.llama_result, file_name="tinyllama_itinerary.txt")

# === Comparison Message ===
if "gemini_time" in st.session_state and "llama_time" in st.session_state:
    st.markdown("---")
    st.subheader("📊 Model Comparison Summary")
    st.markdown(f"""
    - **Gemini**: {st.session_state.gemini_time} seconds  
    - **TinyLLaMA**: {st.session_state.llama_time} seconds  
    - 📝 *Gemini is fast and fluent. TinyLLaMA works offline for enhanced privacy but may be slower or less detailed.*
    """)
