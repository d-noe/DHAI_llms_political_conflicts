import pandas as pd
from fuzzywuzzy import fuzz
import numpy as np
import os
import requests
from docx import Document
import time
import re

def read_and_concatenate_csv(folder_path):
    """
    Reads all CSV files in the specified folder and concatenates them into a single Pandas DataFrame.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.

    Returns:
    pd.DataFrame: DataFrame containing concatenated contents of all CSV files in the folder.
    """
    # List to store individual DataFrames
    dataframes = []
    
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a CSV
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file and append it to the list
                df = pd.read_csv(file_path)
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    # Concatenate all DataFrames in the list
    if dataframes:
        return pd.concat(dataframes, ignore_index=True)
    else:
        print("No CSV files found in the specified folder.")
        return pd.DataFrame()
    

def check_similar_words(df, column, keywords, threshold=80):
    """
    Checks if strings in a pandas column contain words similar to specified keywords.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to check.
        column (str): The name of the column to search for similar words.
        keywords (list of str): A list of target keywords to match against.
        threshold (int): The minimum similarity score to consider a match (default is 80).

    Returns:
        pd.Series: A boolean Series indicating rows with similar matches.
    """
    def has_similar_word(text):
        if pd.isna(text):  # Handle missing values
            return False
        for word in text.split():
            for keyword in keywords:
                if fuzz.ratio(word.lower(), keyword.lower()) >= threshold:
                    return True
        return False

    return df[column].apply(has_similar_word)

def process_link(link, output_folder, idx):
    """
    Downloads a Word document from a link, converts it to text, and saves it.
    Deletes the original .docx file after processing.

    Args:
        link (str): The URL to the Word document.
        output_folder (str): Path to the folder where files will be saved.
        idx (int): Index for naming the output files.

    Returns:
        str: A success or error message.
    """
    try:
        
        t = np.random.uniform(0.2, 1)
        time.sleep(t)  # Add some latency for stealth web crawling, and avoid error 403.

        # Download the Word document
        response = requests.get(link)
        response.raise_for_status()  # Raise exception for HTTP errors
        word_file_path = os.path.join(output_folder, f"document_{idx + 1}.docx")
        
        # Save the Word file locally
        with open(word_file_path, 'wb') as word_file:
            word_file.write(response.content)
        
        # Convert Word document to text
        doc = Document(word_file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        # Save the text to a file
        text_file_path = os.path.join(output_folder, f"document_{idx + 1}.txt")
        with open(text_file_path, 'w', encoding='utf-8') as text_file:
            text_file.write(text)
        
        # Delete the .docx file after processing
        os.remove(word_file_path)
        
        return f"Successfully processed: {link}"
    
    except Exception as e:
        return f"Error processing {link}: {e}"

def download_and_convert(links_series, output_folder):
    """
    Downloads Word documents from a pandas Series of links, converts them to text, 
    and saves the text files in the specified folder using `apply`.
    Deletes the original .docx files after processing.

    Args:
        links_series (pd.Series): A pandas Series containing URLs to Word documents.
        output_folder (str): Path to the folder where files will be saved.

    Returns:
        pd.Series: A Series of success/error messages for each link.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Use apply with a lambda to pass arguments
    return links_series.apply(lambda link: process_link(link, output_folder, links_series[links_series == link].index[0]))

def extract_resolutions(folder_path):
    resolutions = []
    current_main_decision = None  # Store the current main decision for context

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):  # Process only .txt files
            with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
                for line in file:
                    stripped_line = line.strip()

                    # Check for main decisions (e.g., "1.")
                    if stripped_line.startswith(tuple(f"{i}." for i in range(1, 100))):
                        # Extract main decision and clean up unnecessary characters
                        current_main_decision = re.sub(r"\.\s*", "", stripped_line[2:].strip())
                        resolutions.append(f"The European Parliament {current_main_decision}")

                    # Check for sub-decisions (e.g., "(a)")
                    elif re.match(r"^\([a-z]+\)", stripped_line):  # Matches patterns like "(a)", "(b)", etc.
                        if current_main_decision:
                            # Clean up and format sub-decision
                            sub_decision = re.sub(r"\.\s*", "", stripped_line)
                            full_sub_decision = f"The European Parliament {current_main_decision}: {sub_decision}"
                            resolutions.append(full_sub_decision)

                    # Continuation lines for the main decision
                    elif stripped_line and current_main_decision and not re.match(r"^\([a-z]+\)", stripped_line):
                        # Append additional details to the last added resolution
                        resolutions[-1] += " " + stripped_line

    # Convert resolutions to a pandas Series and clean extra whitespace
    resolutions_series = pd.Series([res.strip() for res in resolutions if res.strip()])
    return resolutions_series
