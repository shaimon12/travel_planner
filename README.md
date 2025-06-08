# Personal Travel Assistant

A multi-day travel itinerary generator powered by Large Language Models (LLMs). This tool supports both cloud-based (Gemini 1.5 Pro) and offline (TinyLLaMA 1.1B) modes, allowing for flexibility across connected and private environments. Built using Python and Streamlit, it dynamically tailors travel plans to user preferences such as destination, group type, and dietary needs.

## Features

- Dual LLM Mode: Switch between Gemini (via API) and TinyLLaMA (offline model)
- Country & City Name Matching using fuzzy logic and geonamescache
- Intelligent prompt generation based on user inputs
- Interactive UI built with Streamlit
- Evaluation across multiple user profiles and travel types

## Technologies Used

- Frontend: Streamlit  
- LLMs: Gemini 1.5 Pro (Google Generative AI), TinyLLaMA 1.1B (Hugging Face Transformers)  
- Location Data: pycountry, geonamescache  
- Backend: Python, Transformers, Torch

## Environment Setup and API Key

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Add your Gemini API key:

    - Option 1: Set it as an environment variable:
    
      ```bash
      export GEMINI_API_KEY="your_key_here"
      ```

    - Option 2: Create a `.env` file in the root directory:
    
      ```
      GEMINI_API_KEY=your_key_here
      ```

4. Run the Streamlit app:

    ```bash
    streamlit run travel_assistant.py
    ```
## Project Structure

    ├── travel_assistant.ipynb         # Jupyter notebook with full project analysis
    ├── travel_assistant.py            # Streamlit-based web app for itinerary generation
    ├── README.md                      # Project overview and usage instructions
    ├── .env                           # Environment file for API keys (excluded from Git)
    ├── .gitignore                     # Specifies files/folders to ignore in Git
    └── requirements.txt               # Python dependencies list



## Author

Shaimon Rahman  
Master of IT in Artificial Intelligence — Macquarie University  
[LinkedIn](www.linkedin.com/in/shaimonrahman)

## License

This project is licensed under the MIT License.

