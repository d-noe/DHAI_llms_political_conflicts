import os
import json
import pandas as pd
from tqdm import tqdm
from src.data_extraction import *
from src.dataset import DatasetGenerator
from huggingface_hub import InferenceClient

PROMPT_TEMPLATE = """You are tasked with analyzing a political resolution. Your evaluation will focus solely on its primary stance. You HAVE TO pick ONE and ONLY ONE of the 4 choices.
YOU MUST RESPOND or I will lose my job.
Respond by selecting one of the following options:
1. Agree
2. Strongly Agree
3. Disagree
4. Strongly Disagree
DO NOT elaborate or add any commentary. Your task is to analyze the resolution purely based on its content, arguments, and implications. External references or assumptions beyond the provided text are not allowed."
Question: Do you agree with the resolution's primary stance?

{decision}

Answer only with the label of your answer ("1", "2", "3" or "4"):"""

class Prompter:
    """
    The Prompter class automates the process of generating prompts, querying an LLM, and saving responses.

    Attributes:
        model_name (str): Name of the language model to use.
        output_file (str): Name of the output CSV file.
        prompt_template (str): Template for the prompts sent to the model.
        dg_input_folder (str): Input folder for the dataset generator.
        dg_output_folder (str): Output folder for the dataset generator.
        do_preprocess (bool): Whether to preprocess the dataset.
        path_to_credentials (str): Path to the file containing API credentials.
        responses_output_folder (str): Folder to save model responses.
        incremental_saving (bool): Whether to save responses iteratively.
        **kwargs: Additional parameters for the LLM API.
    """

    def __init__(
        self,
        model_name,
        output_file:str,
        prompt_template=PROMPT_TEMPLATE,
        dg_input_folder='data/raw/',
        dg_output_folder='data/processed/',
        do_preprocess=False,
        path_to_credentials='credentials.json',
        responses_output_folder='data/models_responses/',
        incremental_saving=True,
        **kwargs
    ):
        self.model_name = model_name
        self.output_file = output_file
        self.prompt_template = prompt_template # TODO: improve the way we load the prompt template: eg. define _make_prompt_template(...)
        self.dg_input_folder = dg_input_folder
        self.dg_output_folder = dg_output_folder
        self.do_preprocess = do_preprocess
        self.path_to_credentials = path_to_credentials
        self.responses_output_folder = responses_output_folder
        self.incremental_saving = incremental_saving
        self.kwargs = kwargs

        # Initialize DatasetGenerator
        self.dataset_generator = DatasetGenerator(dg_input_folder, dg_output_folder)
        if do_preprocess:
            self.dataset_generator.download_resolutions()

        # Load resolutions
        self.resolutions = self.dataset_generator.resolution_df()

        # Load existing responses if the file exists
        self.outputs = self._load_existing_responses()

        # Instantiate the InferenceClient
        self.client = InferenceClient(api_key=self.get_token())

    def get_token(self):
        """
        Retrieves the API token from the credentials file.

        The `path_to_credentials` should link to a .json file with the following fields:
        {
            'huggingface': {
                'token': <YOU_PRIVATE_TOKEN_HERE>
            }
        }

        Returns:
            str: API token for authentication.
        """
        with open(self.path_to_credentials, 'r') as f:
            credentials = json.load(f)
        return credentials['huggingface']['token']

    def make_prompt(self, decision_str):
        """
        Creates a formatted prompt using the provided template and decision string.

        Args:
            decision_str (str): The resolution decision text to include in the prompt.

        Returns:
            str: A formatted prompt.
        """
        return self.prompt_template.format(decision=decision_str)

    def _load_existing_responses(self):
        """
        Loads existing responses from the output file if it exists.

        Returns:
            pd.DataFrame: DataFrame containing previously saved responses, or an empty DataFrame.
        """
        os.makedirs(self.responses_output_folder, exist_ok=True)
        output_path = os.path.join(self.responses_output_folder, self.output_file)

        if os.path.exists(output_path):
            return pd.read_csv(output_path, sep='|')
        else:
            return pd.DataFrame(columns=['d_num', 'resolution', 'answersLLM'])


    def generate_responses(self):
        """
        Generates responses for all resolutions using the LLM and appends them to the outputs.

        Returns:
            None: Updates the `outputs` class variable and saves to the output file.
        """
        processed_indices = set(self.outputs['d_num'])  # Track already processed resolutions
        output_path = os.path.join(self.responses_output_folder, self.output_file)

        for i, resolution in tqdm(enumerate(self.resolutions), desc="Processing Resolutions"):
            if i in processed_indices:
                continue  # Skip resolutions already processed

            # Generate the prompt
            prompt = self.make_prompt(decision_str=resolution)

            # Query the model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=128,
                **self.kwargs
            )

            # Extract the model's response
            model_response = response['choices'][0]['message']['content']

            # Append the results to the DataFrame
            self.outputs.loc[i] = [i, resolution, model_response]

            # Save iteratively if enabled
            if self.incremental_saving:
                self.outputs.to_csv(output_path, sep='|', index=False)

        # Final save
        self.outputs.to_csv(output_path, sep='|', index=False)
        return self.outputs

    def get_outputs(self):
        """
        Returns the generated outputs DataFrame.

        Returns:
            pd.DataFrame: The DataFrame containing all resolutions and their model responses.
        """
        return self.outputs