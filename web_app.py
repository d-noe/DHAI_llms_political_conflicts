import streamlit as st
import pandas as pd
import requests

# -----------------------
# Part 1: Title
# -----------------------
st.title("Streamlit App: Data Loading and HuggingFace Integration")

# -----------------------
# Part 2: Select Data Loading Method
# -----------------------
st.header("1. Load Data")
load_method = st.radio("Choose a method to load data:", ("Load from Google Drive", "Upload a file locally"))

data = None
if load_method == "Load from Google Drive":
    drive_url = st.text_input("Enter Google Drive file link:")
    if st.button("Load Data"):
        try:
            file_id = drive_url.split("/")[-2]
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

# Select a column to save as a list variable
if data is not None:
    column_options = list(data.columns)
    selected_column = st.selectbox("Select a column to save as a list variable:", column_options)
    list_variable = data[selected_column].dropna().tolist()
    st.write(f"Saved list variable: {list_variable}")

# -----------------------
# Part 3: Prompt Generation
# -----------------------
st.header("2. Prompt Generation")
if data is not None:
    user_input = st.text_input("Enter text to concatenate with the list items:", value="")
    if st.button("Generate Prompts"):
        prompts = [f"...{item}...{user_input}" for item in list_variable]
        st.write("Generated Prompts:")
        st.write(prompts)

# -----------------------
# Part 4: HuggingFace Integration
# -----------------------
st.header("3. HuggingFace API Integration")
huggingface_api_url = st.text_input("Enter the HuggingFace model API URL:")
if huggingface_api_url:
    prompt_to_send = st.text_area("Enter the prompt to send to the HuggingFace model:", "")
    if st.button("Send Prompt"):
        try:
            headers = {"Authorization": "Bearer YOUR_HUGGINGFACE_API_TOKEN"}
            payload = {"inputs": prompt_to_send}
            response = requests.post(huggingface_api_url, json=payload, headers=headers)
            if response.status_code == 200:
                result = response.json()
                st.success("Model response:")
                st.write(result)
            else:
                st.error(f"API request failed, status code: {response.status_code}")
        except Exception as e:
            st.error(f"Error calling the HuggingFace model: {e}")

# -----------------------
# Part 5: Statistical Charts (Expandable)
# -----------------------
st.header("4. Statistical Charts (Future Extension)")
st.write("This section can be expanded to include statistical charts based on the model results, using libraries such as Matplotlib or Plotly.")
