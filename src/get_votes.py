import requests
import xmltodict
import numpy as np

# Constants for constructing URLs and defining categories of votes
BASE_URL = "https://www.europarl.europa.eu/doceo/document/{}.xml"
BASE_DOCUMENT_ID = "PV-{K}-{YYYY}-{MM}-{DD}-RCV_EN"
VOTE_CATEGORIES = ["For", "Against", "Abstention"]

# =================================================================================
def ta2votes(
    date: str,
    document_adopts: str,
    k:str = '9',
    vote_categories: list[str] = VOTE_CATEGORIES
):
    """
    Main pipeline function to retrieve and process voting data.

    Args:
        date (str): Date of the document in YYYY-MM-DD format.
        document_adopts (str): Document adoption identifiers.
        vote_categories (list[str]): Categories of votes (e.g., For, Against, Abstention).

    Returns:
        dict: A dictionary containing vote outcomes, both overall and per political group.
    """
    # Step 1: Get vote data by constructing the URL and parsing the XML data
    url = make_url(date=date, k=k)
    data = parse_xml(url)

    # Step 2: Retrieve the specific vote identifier and its corresponding vote results
    vote_identifier = get_rcv_identifier(dict_data=data, document_adopts=document_adopts)
    vote_results_id = get_identifier_votes(
        vote_results=data['PV.RollCallVoteResults']['RollCallVote.Result'], 
        vote_identifier=vote_identifier
    )

    # Step 3: Extract results and organize them into the specified categories
    vote_outcomes = {}

    # Step 3.1: Overall results for each vote category
    for cat in vote_categories:
        if f"Result.{cat}" in vote_results_id.keys():
            vote_outcomes[cat] = vote_results_id[f"Result.{cat}"]["@Number"]
        else:
            vote_outcomes[cat] = 0

    # Step 3.2: Results per political group for each vote category
    vote_outcomes_pp = {}
    for cat in vote_categories:
        for results in vote_results_id[f"Result.{cat}"]['Result.PoliticalGroup.List']:
            if not results['@Identifier'] in vote_outcomes_pp.keys():
                vote_outcomes_pp[results['@Identifier']] = {}
            vote_outcomes_pp[results['@Identifier']][cat] = len(results['PoliticalGroup.Member.Name'])

    # Add political group data to the overall outcomes
    vote_outcomes['PoliticalGroup'] = vote_outcomes_pp

    return vote_outcomes

# =================================================================================

def make_url(
    date: str,
    k: str,
    base_url: str = BASE_URL,
    base_document_id: str = BASE_DOCUMENT_ID
):
    """
    Constructs the URL for the XML document based on the given date.

    Args:
        date (str): Date of the document in YYYY-MM-DD format.
        base_url (str): Base URL for the document.
        base_document_id (str): Base document identifier template.

    Returns:
        str: The formatted URL for the XML document.
    """
    year, month, day = date[:4], date[5:7], date[-2:]
    return base_url.format(
        base_document_id.format(
            YYYY=year, MM=month, DD=day, K=k
        )
    )

def parse_xml(
    xml_url: str,
):
    """
    Fetches and parses XML data from the given URL.

    Args:
        xml_url (str): URL of the XML document.

    Returns:
        dict: Parsed XML data as a dictionary.
    """
    response = requests.get(xml_url)
    return xmltodict.parse(response.content)

def get_rcv_identifier(
    dict_data: dict,
    document_adopts: str,
):
    """
    Retrieves the vote identifier for the specified document adoption.

    Args:
        dict_data (dict): Parsed XML data.
        document_adopts (str): Document adoption identifiers.

    Returns:
        str: The identifier for the relevant roll call vote.
    """
    # Step 1: Parse and reformat document adoption strings
    document_adopts = document_adopts.split(";")
    document_adopts = [
        d.split("/")[-1]
        for d in document_adopts
    ]
    document_adopts = [
        d.split("-")
        for d in document_adopts
    ]
    document_adopts = [
        f"{d[0]}{d[1]}-{d[-1]}/{d[2]}"
        for d in document_adopts
    ]

    # Step 2: Extract vote descriptions from the data
    descriptions = [
        v['RollCallVote.Description.Text']
        for v in dict_data['PV.RollCallVoteResults']['RollCallVote.Result']
    ]
    descriptions = [
        f"{d['a']['#text']} {d['#text']}" if type(d) == dict else d
        for d in descriptions 
    ]

    # Step 3: Identify potential matches for the document adoption identifiers
    potential_ids = []
    for i, d in enumerate(descriptions):
        if np.any([da in d for da in document_adopts]):
            potential_ids += [i]

    # Step 4: Select the vote identifier corresponding to the last matching description
    vote_identifier = dict_data['PV.RollCallVoteResults']['RollCallVote.Result'][max(potential_ids)]['@Identifier']
    return vote_identifier

def get_identifier_votes(
    vote_results: list[dict],
    vote_identifier: str,
):
    """
    Retrieves vote details for the specified vote identifier.

    Args:
        vote_results (list[dict]): List of vote result dictionaries.
        vote_identifier (str): The identifier of the desired vote.

    Returns:
        dict or None: Vote results dictionary for the identifier, or None if not found.
    """
    ids = np.array([v['@Identifier'] for v in vote_results])
    try:
        return vote_results[np.argmax(ids == vote_identifier)]
    except:
        return None
