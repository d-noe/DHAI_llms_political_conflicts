import streamlit as st
import pandas as pd
import requests
import time
import re

# -----------------------
# Part 1: Title
# -----------------------
st.title("DHAI Political bias in LLM")


# -----------------------
# Part 2: Select Data Loading Method
# -----------------------
st.header("1. Load Data")
load_method = st.radio("Choose a method to load data:", ("Load from Google Drive", "Upload a file locally"))

data = None
generated_prompts = []

if load_method == "Load from Google Drive":
    drive_url = st.text_input("Enter Google Drive file link:")
    if st.button("Load Data"):
        try:
            # Extract file ID using regex
            match = re.search(r"(?<=/d/|id=)[^/?&]+", drive_url)
            if not match:
                raise ValueError("Invalid Google Drive URL format.")
            file_id = match.group()
            download_url = f"https://drive.google.com/uc?id={file_id}"
            data = pd.read_csv(download_url)
            st.success("Data loaded successfully!")
            st.write("Preview of the data:", data.head())
        except Exception as e:
            st.error(f"Failed to load data: {e}")

elif load_method == "Upload a file locally":
    uploaded_file = st.file_uploader("Upload a CSV file:", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.write("Preview of the data:", data.head())
        except Exception as e:
            st.error(f"Failed to load data: {e}")

if data is not None:
    column_options = list(data.columns)
    selected_column = st.selectbox("Select a column to save as a list variable:", column_options)
    list_variable = data[selected_column].dropna().tolist()
    st.session_state.list_variable = list_variable


# -----------------------
# Part 3: Prompt Generation
# -----------------------
st.header("2. Prompt Generation")
if "list_variable" not in st.session_state:
    st.session_state.list_variable = ["resolution_1", "resolution_2", "resolution_3"]

if st.session_state.list_variable:
    st.write("Available items in the list:")
    st.write(st.session_state.list_variable)

user_input = st.text_input("Enter text to concatenate with the list items:", value="")
if st.button("Generate Prompts"):
    st.session_state.generated_prompts = [f"...{item}...{user_input}" for item in st.session_state.list_variable]
    st.success("Prompts have been successfully generated!")
    st.write("Generated Prompts:")
    st.write(st.session_state.generated_prompts)

if "generated_prompts" in st.session_state:
    generated_prompts = st.session_state.generated_prompts

# -----------------------
# Part 4: Hugging Face API Integration
# -----------------------
st.header("3. Hugging Face API Integration")
if generated_prompts:
    st.subheader("API Configuration")
    api_key = st.text_input("Enter your HuggingFace API key:", type="password")
    predefined_models = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "google/gemma-7b",
        "bigcode/starcoder",
        "openai-community/gpt2",
        "HuggingFaceH4/zephyr-7b-beta"
    ]
    model_name = st.selectbox("Select a Hugging Face model:", options=predefined_models + ["Custom"])
    if model_name == "Custom":
        model_name = st.text_input("Enter custom model name:")

    send_interval = st.number_input("Interval between requests (seconds):", 0.0, 10.0, 0.5, 0.1)
    max_tokens = st.number_input("Maximum tokens to return:", 1, 1024, 100)
    temperature = st.slider("Temperature (Randomness):", 0.0, 1.0, 0.7, 0.1)
    top_p = st.slider("Top-p (Nucleus Sampling):", 0.0, 1.0, 0.9, 0.1)

    if api_key and model_name and st.button("Send Prompts to Model"):
        results = []
        try:
            huggingface_api_url = f"https://api-inference.huggingface.co/models/{model_name}"
            headers = {"Authorization": f"Bearer {api_key}"}
            for idx, prompt in enumerate(generated_prompts):
                st.write(f"Processing prompt {idx + 1}/{len(generated_prompts)}: {prompt}")
                payload = {"inputs": prompt, "parameters": {"max_length": max_tokens, "temperature": temperature, "top_p": top_p}}
                response = requests.post(huggingface_api_url, json=payload, headers=headers)
                time.sleep(send_interval)
                if response.status_code == 200:
                    try:
                        result = response.json()
                        results.append(result)
                        st.success(f"Response for Prompt {idx + 1}:")
                        st.write(result)
                    except ValueError:
                        st.error(f"Error decoding JSON response for Prompt {idx + 1}")
                else:
                    error_message = response.json().get("error", f"Failed with status code {response.status_code}")
                    st.error(f"API request failed for Prompt {idx + 1}: {error_message}")
                    results.append({"error": error_message})
            st.success("All prompts processed!")
            st.write("Final Results:", results)
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")

# -----------------------
# Part 5: Statistical Charts
# -----------------------
st.header("4. Statistical Charts (Future Extension)")
st.write("This section can be expanded to include statistical charts based on the model results.")
