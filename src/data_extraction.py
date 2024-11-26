import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fuzzywuzzy import fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import requests
from docx import Document

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

def download_and_convert_docs_with_apply(links_series, output_folder):
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