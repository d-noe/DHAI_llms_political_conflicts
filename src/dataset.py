"""This file contains the DatasetGenerator class which is responsible for generating
exactracting the EU resolutions"""

import pandas as pd
from src.data_extraction import *


class DatasetGenerator:


    def __init__(self,
                 input_folder='data/raw/',
                 output_folder='data/processed/') -> None:
        """
        Initialize the DatasetGenerator class.

        Parameters:
        input_folder (str): The path to the folder containing raw data files. Default is 'data/raw/'.
        output_folder (str): The path to the folder where processed data files will be saved. Default is 'data/processed/'.

        Returns:
        None

        Example:
        input_folder = '../data/raw/'
        output_folder = '../data/processed/'
        _ = DatasetGenerator(input_folder, output_folder)
        """

        self.input_folder = input_folder
        self.output_folder = output_folder

        self.load_table()

    def load_table(self):
        self.df = read_and_concatenate_csv(self.input_folder)

    def filter_df(self):
        keywords = ['ukraine', 'russia', 'israel', 'palestine']
        df_filtered = self.df[check_similar_words(self.df, column='document_title', keywords=keywords, threshold=70)]
        return df_filtered

    def download_resolutions(self):
        df_filtered = self.filter_df()
        links_series = pd.Series(df_filtered['document_doc'])
        _ = download_and_convert(links_series, self.output_folder)

    def resolution_df(self):
        res = extract_resolutions(self.output_folder)
        return res