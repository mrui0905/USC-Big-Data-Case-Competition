import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import google.generativeai as genai
import json
import re

from regimen_rec import regimen_reccomend

# Set up Gemini API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

def extract_data_from_text(text):
    """Extract patient information using Gemini"""
    prompt = f"""Extract the following information from this medical text:
    - Patient gender
    - Patient ethnicity
    - Patient VL levels
    - Patient CD4 levels
    - Patient CD4% levels
    
    Return ONLY a JSON object with these keys: gender, ethnicity, vl_levels, cd4_levels, cd4_percent. Do not include units, percent symbols, etc.
    If any information is missing, use null.
    
    Text: {text}"""
    
    try:
        response = model.generate_content(prompt)
        json_str = response.text.strip()
        json_match = re.search(r"\{.*\}", json_str, re.DOTALL)
        return json.loads(json_match.group(0))
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return None

def create_dataframe(data):
    """Create and return pandas DataFrame"""
    gender, ethnicity, vl_levels, cd4_levels, cd4_percent = data.get('gender'), data.get('ethnicity'), data.get('vl_levels'), data.get('cd4_levels'), data.get('cd4_percent')
    return regimen_reccomend(gender, ethnicity, vl_levels, cd4_levels, cd4_percent).reset_index(drop=True)

# Streamlit UI
st.title("Treatment Regimen Efficacy Predictor")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = pd.DataFrame()
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""
if 'extracted_values' not in st.session_state:
    st.session_state.extracted_values = {}

# Chat Interface
with st.expander("Chat Interface", expanded=True):
    user_input = st.chat_input("Enter patient information:")
    
    if user_input:
        with st.spinner("Analyzing text..."):
            # Store input text in session state
            st.session_state.input_text = user_input
            
            # Extract data from input text
            data = extract_data_from_text(user_input)
            if data:
                # Store extracted values in session state
                st.session_state.extracted_values = data
                
                # Generate recommendations and store in session state
                st.session_state.results = create_dataframe(data)

# Display Input Text and Extracted Values
if st.session_state.input_text:
    st.subheader("Input Text")
    st.write(st.session_state.input_text)

if st.session_state.extracted_values:
    st.subheader("Extracted Patient Metrics")
    extracted_df = pd.DataFrame([st.session_state.extracted_values])
    st.table(extracted_df)


# Display Recommendations
if not st.session_state.results.empty:
    st.subheader("Recommendations")
    st.dataframe(st.session_state.results)
else:
    st.info("Enter patient summary in the chat to begin")