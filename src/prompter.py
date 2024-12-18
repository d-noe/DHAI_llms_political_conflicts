import os
import json
import pandas as pd
from tqdm import tqdm
import yaml
from src.data_extraction import *
from src.dataset import DatasetGenerator
from huggingface_hub import InferenceClient
from multiprocessing import Pool, Manager
from tqdm import tqdm


class Prompter:
    """
    The Prompter class automates the process of generating prompts, querying an LLM, and saving responses.

    Attributes:
        model_name (str): Name of the language model to use.
        output_file (str): Name of the output CSV file.
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
        dg_input_folder='data/raw/',
        dg_output_folder='data/processed/',
        do_preprocess=False,
        path_to_credentials='credentials.json',
        responses_output_folder='data/models_responses/',
        path_to_prompt='prompts/base_template.yaml',
        incremental_saving=True,
        **kwargs
    ):
        self.model_name = model_name
        self.path_to_prompt = path_to_prompt 
        self.output_file = output_file
        #self.prompt_template = prompt_template # TODO: improve the way we load the prompt template: eg. define _make_prompt_template(...)
        self.prompt_template = self.load_prompt()
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

    def load_prompt(self):
        # Load the YAML file
        with open(self.path_to_prompt, "r") as file:
            template = yaml.safe_load(file)

        # Access the components
        instructions = template["instructions"]
        question = template["question"]
        endmessage = template["endmessage"]

        # Define a Python string with the desired format
        formatted_string = instructions + """\nHere is the decision taken by the EU :\n{decision}\n"""
        formatted_string += f"""
            {question}
            
            {endmessage}"""
        
        return formatted_string


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
    def generate_responses_parallel(self):
        """
        Generates responses for all resolutions using the LLM in parallel
        and appends them to the outputs.
        """
        processed_indices = set(self.outputs['d_num'])  # Track already processed resolutions
        output_path = os.path.join(self.responses_output_folder, self.output_file)

        # Prepare inputs for multiprocessing
        tasks = [
            (i, resolution, self.make_prompt, self.client, self.model_name, self.kwargs)
            for i, resolution in enumerate(self.resolutions)
            if i not in processed_indices
        ]

        # Use Manager for shared DataFrame
        with Manager() as manager:
            outputs_dict = manager.dict()  # Shared dictionary to collect results

            # Process in parallel
            with Pool(processes=min(len(tasks), os.cpu_count())) as pool:
                for result in tqdm(pool.imap_unordered(process_resolution, tasks, chunksize=16), total=len(tasks), desc="Processing Resolutions"):
                    i, resolution, model_response = result
                    outputs_dict[i] = [i, resolution, model_response]

                    # Save incrementally if enabled
                    if self.incremental_saving:
                        temp_df = pd.DataFrame.from_dict(outputs_dict, orient="index", columns=["d_num", "resolution", "answersLLM"])
                        temp_df.to_csv(output_path, sep='|', index=False)

            # Merge results into the original outputs DataFrame
            results_df = pd.DataFrame.from_dict(outputs_dict, orient="index", columns=["d_num", "resolution", "answersLLM"])
            self.outputs = pd.concat([self.outputs, results_df]).reset_index(drop=True)

            # Final save
            self.outputs.to_csv(output_path, sep='|', index=False)

        return self.outputs


def process_resolution(args):
    """
    Worker function for processing a single resolution.
    """
    i, resolution, make_prompt, client, model_name, kwargs = args

    # Generate the prompt
    prompt = make_prompt(decision_str=resolution)

    # Query the model
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=128,
        **kwargs
    )

    # Extract the model's response
    model_response = response['choices'][0]['message']['content']
    return i, resolution, model_response

