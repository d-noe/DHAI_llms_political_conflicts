import plotly.graph_objs as go
import pandas as pd


def extract_sentiment(text):
    """
    Extract sentiment from the text, case-insensitive
    
    Args:
        text (str): Input text to analyze sentiment
    
    Returns:
        str: Detected sentiment or 'not found'
    """
    sentiments = ["stronglyagree", "stronglydisagree", "agree", "disagree"]
    
    # Remove spaces and convert to lowercase for case-insensitive comparison
    text = text.replace(" ", "").lower()

    # Check for each sentiment and return the first match
    for sentiment in sentiments:
        if sentiment in text:
            return sentiment.lower() 
    
    return "not found"  # If no match, return "not found"


def parse_results(results):
    """
    Parse results and categorize sentiments
    
    Args:
        results (dataframe): a dataframe as the variable "output" shown at the end of notbooks/prompter.ipynb"
    
    Returns:
        tuple: Sentiment counts and detailed results
    """
    # Initialize dictionaries to store results
    sentiment_counts = {
        "strongly agree": 0,
        "agree": 0,
        "disagree": 0,
        "strongly disagree": 0,
        "not found": 0
    }
    
    # Storing detailed results for later reference
    correspond_text = {
        "strongly agree": [],
        "agree": [],
        "disagree": [],
        "strongly disagree": [],
        "not found": []
    }

    # Parse results
    results_list = results['answersLLM']
    for i in range(len(results_list)):
        try:
            # Extract text from the result
            sentiment = extract_sentiment(str(results_list[i]))
            
            sentiment_counts[sentiment] += 1
            correspond_text[sentiment].append(results['resolution'][i])
        except Exception as e:
            print(f"Error processing result: {e}")

    return sentiment_counts, correspond_text

def plot_sentiment_distribution(results):
    """
    Create bar plot of sentiment distribution
    
    Args:
        results (dataframe): a dataframe as the variable "output" shown at the end of notbooks/prompter.ipynb"
    
    Returns:
        tuple: Plotly figure and detailed results, or None if no results
    """
    parsed_results = parse_results(results)
    if not parsed_results:
        return None

    sentiment_counts, correspond_text = parsed_results

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

    return fig, correspond_text

def plot_sentiment_comparison(prompt_results, parliament_data):
    """
    Create a line graph comparing sentiment distributions
    
    Args:
        original_results (dict): Original results DataFrame with sentiment data
        comparison_data (dict): Comparison sentiment data dictionary
    
    Returns:
        plotly.graph_objs.Figure: Interactive line graph of sentiment comparisons
    """
    # Parse original results to get sentiment counts
    prompt_counts, _ = parse_results(prompt_results)
    
    # Validate comparison data keys match original sentiment keys
    expected_keys = ["stronglyagree", "agree", "disagree", "stronglydisagree", "not found"]
    
    # Check if comparison data has the correct keys
    if not all(key in parliament_data for key in expected_keys):
        raise ValueError("Comparison data must have keys matching: " + ", ".join(expected_keys))
    
    # Create line graph
    fig = go.Figure()
    
    # Add original results as first line
    fig.add_trace(go.Scatter(
        x=list(prompt_counts.keys()),
        y=list(prompt_counts.values()),
        mode='lines+markers',
        name='Original Results',
        line=dict(color='blue')
    ))
    
    # Add comparison data as second line
    fig.add_trace(go.Scatter(
        x=list(parliament_data.keys()),
        y=list(parliament_data.values()),
        mode='lines+markers',
        name='Comparison Data',
        line=dict(color='red')
    ))
    
    # Update layout for clarity
    fig.update_layout(
        title='Sentiment Distribution Comparison',
        xaxis_title='Sentiment',
        yaxis_title='Count',
        hovermode='closest'
    )
    
    return fig


def plot_prompt_results(prompt_results):
    """
    Visualize prompt results with multiple charts
    
    Args:
        prompt_results(dictionary) : a list of dictionaries in which the key is the prompt and the value is the dataframe as the variable "output" shown at the end of notbooks/prompter.ipynb"
        ex: [{prompt:dataframe},{prompt:dataframe},..]
    
    Returns:
        tuple: Response count figure and sentiment distribution figure, or None if no results
    """
    # Ensure we have results
    if not prompt_results:
        print("No prompt results found.")
        return None

    # Count responses per LLM
    prompt_response_counts = {}
    prompt_sentiment_counts = {}
    for prompts in prompt_results :
     for prompt, results in prompts.items():
        # Count total responses per prompt
        prompt_response_counts[prompt] = len(results['answersLLM'])

        # Count sentiments per LLM
        sentiment_counts = {
            "stronglyagree": 0,
            "agree": 0,
            "disagree": 0,
            "stronglydisagree": 0,
            "not found": 0
        }
        
        for i in range(len(results['answersLLM'])):
          try:
            # Extract text from the result
            sentiment = extract_sentiment(str(results['answersLLM'][i]))
            
            sentiment_counts[sentiment] += 1
          except Exception as e:
            print(f"Error processing result: {e}")
        
        prompt_sentiment_counts[prompt] = sentiment_counts

    # Create response count bar plot
    response_fig = go.Figure(data=[
        go.Bar(
            x=list(prompt_response_counts.keys()), 
            y=list(prompt_response_counts.values()),
            text=list(prompt_response_counts.values()),
            textposition='auto'
        )
    ])
    response_fig.update_layout(
        title='Number of Responses per LLM',
        xaxis_title='LLM Model',
        yaxis_title='Response Count'
    )

    # Create stacked bar plot for sentiments
    sentiments = ["stronglyagree", "agree", "disagree", "stronglydisagree", "not found"]
    
    # Prepare data for stacked bar plot
    stacked_data = []
    for prompt in prompt_sentiment_counts.keys():
        stacked_data.append([prompt_sentiment_counts[prompt][sent] for sent in sentiments])

    stacked_fig = go.Figure(data=[
        go.Bar(
            name=sent, 
            x=list(prompt_sentiment_counts.keys()), 
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
