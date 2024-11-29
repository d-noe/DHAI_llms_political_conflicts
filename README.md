# Investigating LLM Bias on Geopolitical Conflicts

## Project Overview
This project investigates whether large language models (LLMs) take implicit or explicit positions on geopolitical conflicts, including the Israeli-Palestinian and the Russian-Ukrainian conflicts. The primary data source for this project is the **ParlaMint dataset**, which contains European Parliament resolutions and speeches.

## Project Structure
The repository is organized as follows:

```
data/
  cleaned/
  models_responses/
  processed/
  raw/
  results/
notebooks/
  dataprep.ipynb
  prompter.ipynb
prompts/
  base_template.yaml
src/
  __init__.py
  _pycache_/
  data_extraction.py
  dataset.py
  get_votes.py
  prompter.py
  _init__.py
  .gitignore
  credentials.json
  extraction_visualisation_func.py
  pipeline.ipynb
  README.md
  setup.py
```

- `data/`: Contains the various stages of data processing, including raw, cleaned, processed, and model responses.
- `notebooks/`: Houses the Jupyter Notebooks used for data preparation and prompting.
- `prompts/`: Stores the YAML file containing the base template for the prompts.
- `src/`: Holds the Python source code files for data extraction, dataset management, prompting, and other utility functions.
- `README.md`: The project's main documentation file.
- `setup.py`: The Python package configuration file.

## Getting Started
To set up the project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/llm-bias-analysis.git`
2. Navigate to the project directory: `cd llm-bias-analysis`
3. (Optional) Create a virtual environment and activate it.
4. Install the required dependencies: `pip install -r requirements.txt`
5. Prepare the data by running the `dataprep.ipynb` Jupyter Notebook.
6. Generate the model responses by running the `prompter.ipynb` Jupyter Notebook.
7. Analyze the results and findings in the respective notebooks.

## Usage
The project's main functionalities are:

1. **Data Extraction and Preprocessing**: The `data_extraction.py` and `dataset.py` scripts handle the extraction and preprocessing of the ParlaMint dataset.
2. **Prompting and Response Generation**: The `prompter.py` script generates prompts and collects the model's responses.
3. **Result Analysis and Visualization**: The `extraction_visualisation_func.py` script provides functions for analyzing and visualizing the results.