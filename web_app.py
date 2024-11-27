import streamlit as st
import pandas as pd
import requests
import time
import re
import json
import plotly.graph_objs as go
import plotly.express as px

def extract_sentiment(text):
    """
    Extract sentiment from the text, case-insensitive
    """
    sentiments = ["strongly agree", "agree", "disagree", "strongly disagree"]
    for sentiment in sentiments:
        if sentiment.lower() in text.lower():
            return sentiment.lower()
    return "not found"

def parse_results():
    """
    Parse results and categorize sentiments
    """
    if "results" not in st.session_state:
        st.warning("No results found. Please run the LLM queries first.")
        return None

    # Initialize dictionaries to store results
    sentiment_counts = {
        "strongly agree": 0,
        "agree": 0,
        "disagree": 0,
        "strongly disagree": 0,
        "not found": 0
    }
    
    # Storing detailed results for later reference
    detailed_results = {
        "strongly agree": [],
        "agree": [],
        "disagree": [],
        "strongly disagree": [],
        "not found": []
    }

    # Parse results
    for result_list in st.session_state.results:
        try:
            # Extract text from the result
            text = result_list[0]['generated_text'] if isinstance(result_list, list) and len(result_list) > 0 else str(result_list)
            sentiment = extract_sentiment(str(text))
            
            sentiment_counts[sentiment] += 1
            detailed_results[sentiment].append(text)
        except Exception as e:
            st.warning(f"Error processing result: {e}")

    return sentiment_counts, detailed_results

def plot_sentiment_distribution():
    """
    Create bar plot of sentiment distribution
    """
    results = parse_results()
    if not results:
        return None

    sentiment_counts, detailed_results = results

    # Create interactive bar plot
    fig = go.Figure(data=[
        go.Bar(
            x=list(sentiment_counts.keys()), 
            y=list(sentiment_counts.values()),
            text=list(sentiment_counts.values()),
            textposition='auto',
            hovertemplate='%{x}: %{y}<extra></extra>'
        )
    ])
    fig.update_layout(
        title='Sentiment Distribution Across Prompts',
        xaxis_title='Sentiment',
        yaxis_title='Count'
    )

    # Add click event to show details
    return fig, detailed_results

def plot_llm_results():
    """
    Visualize LLM results with multiple charts
    """
    if "results" not in st.session_state:
        st.warning("No LLM results found.")
        return None

    # If it's the first time running, initialize llm_results
    if "llm_results" not in st.session_state:
        st.session_state.llm_results = {}

    # Add current results to llm_results if model_name exists
    if hasattr(st.session_state, 'model_name'):
        st.session_state.llm_results[st.session_state.model_name] = st.session_state.results

    # Ensure we have results
    if not st.session_state.llm_results:
        st.warning("No LLM results found.")
        return None

    # Count responses per LLM
    llm_response_counts = {}
    llm_sentiment_counts = {}

    for model, results in st.session_state.llm_results.items():
        # Count total responses per LLM
        llm_response_counts[model] = len(results)

        # Count sentiments per LLM
        sentiment_counts = {
            "strongly agree": 0,
            "agree": 0,
            "disagree": 0,
            "strongly disagree": 0,
            "not found": 0
        }
        
        for result in results:
            try:
                # Extract text from the result
                text = result[0]['generated_text'] if isinstance(result, list) and len(result) > 0 else str(result)
                sentiment = extract_sentiment(str(text))
                sentiment_counts[sentiment] += 1
            except Exception as e:
                st.warning(f"Error processing result for {model}: {e}")
        
        llm_sentiment_counts[model] = sentiment_counts

    # Create response count bar plot
    response_fig = go.Figure(data=[
        go.Bar(
            x=list(llm_response_counts.keys()), 
            y=list(llm_response_counts.values()),
            text=list(llm_response_counts.values()),
            textposition='auto'
        )
    ])
    response_fig.update_layout(
        title='Number of Responses per LLM',
        xaxis_title='LLM Model',
        yaxis_title='Response Count'
    )

    # Create stacked bar plot for sentiments
    sentiments = ["strongly agree", "agree", "disagree", "strongly disagree", "not found"]
    
    # Prepare data for stacked bar plot
    stacked_data = []
    for model in llm_sentiment_counts.keys():
        stacked_data.append([llm_sentiment_counts[model][sent] for sent in sentiments])

    stacked_fig = go.Figure(data=[
        go.Bar(
            name=sent, 
            x=list(llm_sentiment_counts.keys()), 
            y=[row[i] for row in stacked_data]
        ) for i, sent in enumerate(sentiments)
    ])
    stacked_fig.update_layout(
        title='Sentiment Distribution per LLM',
        xaxis_title='LLM Model',
        yaxis_title='Sentiment Count',
        barmode='stack'
    )

    return response_fig, stacked_fig

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
question = st.text_input("Enter the question", value="")
if st.button("Generate Prompts"):
    st.session_state.generated_prompts = [
        {
            "system": "You are a political analyst with expertise in evaluating resolutions.",
            "user": f"{user_input} Question: '{question}' {item}"
        }
        for item in st.session_state.list_variable
    ]
    st.success("Prompts have been successfully generated!")
    st.write("Generated Prompts:")
    st.write(st.session_state.generated_prompts)

# -----------------------
# Part 4: Hugging Face API Integration
# -----------------------
st.header("3. Hugging Face API Integration")
if 'results' not in st.session_state:
    st.session_state.results = []

if "generated_prompts" in st.session_state:
    generated_prompts = st.session_state.generated_prompts

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
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            for idx, prompt in enumerate(generated_prompts):
                st.write(f"Processing prompt {idx + 1}/{len(generated_prompts)}: {prompt}")
                
                # Modify payload based on model type
                if 'Instruct' in model_name or 'zephyr' in model_name:
                    # For instruction-tuned models
                    payload = {
                        "inputs": prompt["user"],
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    }
                else:
                    # For other models (like GPT-2)
                    payload = {
                        "inputs": prompt["user"],
                        "parameters": {
                            "max_length": max_tokens,
                            "temperature": temperature,
                            "top_p": top_p
                        }
                    }
                
                try:
                    response = requests.post(
                        huggingface_api_url, 
                        headers=headers, 
                        data=json.dumps(payload)
                    )
                    
                    time.sleep(send_interval)
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            results.append(result)
                            st.success(f"Response for Prompt {idx + 1}:")
                            st.write(result)
                        except ValueError as json_err:
                            st.error(f"JSON Decoding Error for Prompt {idx + 1}: {json_err}")
                            st.error(f"Response content: {response.text}")
                    else:
                        error_message = response.text
                        st.error(f"API request failed for Prompt {idx + 1}: {error_message}")
                        results.append({"error": error_message})
                
                except requests.exceptions.RequestException as req_err:
                    st.error(f"Request failed for Prompt {idx + 1}: {req_err}")
            
            st.success("All prompts processed!")
            st.write("Final Results:", results)

            st.session_state.results = results
            st.session_state.model_name = model_name
            st.session_state.generated_prompts = generated_prompts
        
        except Exception as e:
            st.error(f"Unexpected error: {e}")

# -----------------------
# Part 5: Statistical Charts
# -----------------------
st.header("4. Statistical Charts")

# Sentiment Distribution Visualization
st.subheader("Sentiment Distribution")
sentiment_plot = plot_sentiment_distribution()
if sentiment_plot:
    fig, detailed_results = sentiment_plot
    st.plotly_chart(fig, use_container_width=True)

    # Sidebar for detailed results
    st.sidebar.header("Sentiment Details")
    selected_sentiment = st.sidebar.selectbox(
        "Select Sentiment", 
        list(detailed_results.keys())
    )
    st.sidebar.write(f"Prompts for {selected_sentiment}:")
    st.sidebar.write(detailed_results[selected_sentiment])

# LLM Performance Visualization
st.subheader("LLM Performance")
llm_plots = plot_llm_results()
if llm_plots:
    response_fig, stacked_fig = llm_plots
    
    # Tabs for different visualizations
    tab1, tab2 = st.tabs(["Responses per LLM", "Sentiment Distribution per LLM"])
    
    with tab1:
        st.plotly_chart(response_fig, use_container_width=True)
    
    with tab2:
        st.plotly_chart(stacked_fig, use_container_width=True)