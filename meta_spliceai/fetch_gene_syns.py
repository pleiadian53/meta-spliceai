from Bio import Entrez  # pip install biopython
import os
import csv
import time
import urllib
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# NCBI Entrez credentials (loaded from environment variables)
Entrez.email = os.getenv('NCBI_EMAIL', '')  # Set NCBI_EMAIL environment variable
Entrez.api_key = os.getenv('NCBI_API_KEY', '')  # Empty string if not set
# NOTE: NCBI API key? See https://support.nlm.nih.gov/knowledgebase/article/KA-05317/en-us

def fetch_with_backoff(url, max_attempts=5):
    # import time
    # import urllib.error
    for attempt in range(max_attempts):
        try:
            # Attempt to fetch data from the URL
            response = urllib.request.urlopen(url)
            return response
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Calculate sleep time using exponential backoff
                sleep_time = 2 ** attempt
                print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                # Re-raise the error for all other HTTP errors
                raise
    raise Exception("Max retries exceeded")

def fetch_gene_synonyms_v0(gene_name, verbose=1):
    """
    Fetches synonyms for a specified gene name from the NCBI Gene database.

    Parameters:
    - gene_name (str): The name of the gene for which synonyms are to be fetched.
    - verbose (int, optional): A flag to control the verbosity of the function's output. 
                               If set to 1 (default), the function prints messages about its progress and results.
                               If set to 0, the function will be silent.

    Returns:
    - list: A list of synonyms for the given gene name. If the gene name is not found in the database,
            or if there are no synonyms available, an empty list is returned.
            If the gene name cannot be found at all, the function returns None.

    Note:
    - The function uses the Entrez.esearch method to search the NCBI Gene database for the given gene name,
      and selects the first gene ID from the search results.
    - It then uses the Entrez.esummary method to fetch summary information for the gene using its ID.
    - If the 'OtherAliases' field is present in the gene's summary information, it is parsed to extract the synonyms.
    - The function requires a valid email to be set for the Entrez API (not shown in this function).

    Example:
    - synonyms = fetch_gene_synonyms("BRCA1")
      This will fetch synonyms for the gene named "BRCA1".
    """

    # Use the Entrez.esearch function to search for the given gene_name in the NCBI Gene database.
    # handle = Entrez.esearch(db="gene", term=gene_name)
    # record = Entrez.read(handle)
    # handle.close()

    # The following code snippet demonstrates how to handle rate limits and retries when fetching data from the Entrez API.
    attempts = 0
    max_attempts = 5
    while attempts < max_attempts:
        try:
            handle = Entrez.esearch(db="gene", term=gene_name)
            record = Entrez.read(handle)
            handle.close()
            # Process the record as before
            # If successful, break out of the loop
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if verbose: print(f"Rate limit exceeded. Retrying...")
                time.sleep(2 ** attempts)  # Exponential backoff
                attempts += 1
            else:
                raise e  # Re-raise for other HTTP errors
    else:
        if verbose: print("Max retries exceeded.")
        return None
    
    # If the record['IdList'] is empty, meaning no gene ID was found for the given gene_name ...
    if not record['IdList']:
        # the function returns None
        if verbose: print(f"[fetch] No gene ID found for {gene_name}")
        return None
    
    # If a gene ID is found, the first gene ID in the list is selected.
    gene_id = record['IdList'][0]  
    # NOTE: The reason for taking the first element of record['IdList'][0] and 
    #       assigning it to gene_id is to select a single, most relevant gene entry 
    #       when there are multiple entries for a gene name. 
    #       The Entrez search can return multiple gene IDs if there are multiple entries 
    #       associated with the search term.
    
    # Use this gene ID to fetch the summary information for the gene.
    handle = Entrez.esummary(db="gene", id=gene_id)
    # NOTE: An Entrez.esummary request is made using this gene ID to fetch the summary information for the gene.

    # The summary information is read into a summary object.
    summary = Entrez.read(handle)
    handle.close()
    
    if 'OtherAliases' in summary['DocumentSummarySet']['DocumentSummary'][0]:

        # If OtherAliases is present, it is split by commas into a list of synonyms.
        synonyms = summary['DocumentSummarySet']['DocumentSummary'][0]['OtherAliases'].split(", ")

        # Filter out empty strings
        synonyms = [syn.strip() for syn in synonyms if syn.strip()]
        return synonyms
    return []


def fetch_gene_synonyms_v2(gene_name):
    handle = Entrez.esearch(db="gene", term=gene_name)
    record = Entrez.read(handle)
    handle.close()
    
    if not record['IdList']:
        return None
    
    # Iterate over all gene IDs and select the appropriate one
    for gene_id in record['IdList']:
        handle = Entrez.esummary(db="gene", id=gene_id)
        summary = Entrez.read(handle)
        handle.close()
        
        if 'OtherAliases' in summary['DocumentSummarySet']['DocumentSummary'][0]:
            synonyms = summary['DocumentSummarySet']['DocumentSummary'][0]['OtherAliases'].split(", ")
            # Filter out empty strings
            synonyms = [syn.strip() for syn in synonyms if syn.strip()]
            if synonyms:  # If valid synonyms are found
                return synonyms
            
    return []


def fetch_all_genes():
    """
    Fetches all known gene IDs from the Entrez database.

    Returns:
    list: A list of all gene IDs.
    """
    # Use a single call to esearch to fetch up to 100,000 gene IDs
    handle = Entrez.esearch(db="gene", term="human[orgn]", retmax=100000)
    record = Entrez.read(handle)
    handle.close()
    return record['IdList']


def create_gene_syns_file_v0(gene_names, filename="gene_syns.tsv"):
    """
    Creates a TSV file containing gene synonyms for a list of genes.
    """
    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for gene in gene_names:
            synonyms = fetch_gene_synonyms(gene)
            # print(f"[test] Fetched synonyms for {gene} => {synonyms}")
            if synonyms:
                for synonym in synonyms:
                    if synonym.strip():  # Ensure synonym is not empty
                        writer.writerow([synonym, gene])
            else:
                print(f"No synonyms found for {gene}")


def create_gene_syns_file_v0(gene_names, filename="gene_syns.tsv", batch_size=10, sleep_time=1):
    """
    Creates a TSV file containing gene synonyms for a list of genes.
    
    Parameters:
    gene_names (list): List of gene names to fetch synonyms for.
    filename (str): Name of the output TSV file.
    batch_size (int): Number of genes to process before pausing to avoid API limits.
    sleep_time (int): Time to sleep (in seconds) between batches to avoid API limits.
    """
    synonyms_dict = {}
    with open(filename, 'w') as fh:
        writer = csv.writer(fh, delimiter='\t')
        count = 0
        for gene in gene_names:
            synonyms = fetch_gene_synonyms(gene)

            # Store synonyms in a dictionary for easy lookup
            print("[info] Synonyms for gene:", gene, "=>", synonyms)
            synonyms_dict[gene] = synonyms

            # print(f"[test] Fetched synonyms for {gene} => {synonyms}")
            if synonyms:
                for synonym in synonyms:
                    if synonym.strip():  # Ensure synonym is not empty
                        writer.writerow([synonym, gene])
            else:
                print(f"No synonyms found for {gene}")
            
            count += 1
            if count % batch_size == 0:
                print(f"Processed {count} genes, sleeping for {sleep_time} seconds to avoid hitting API limits...")
                time.sleep(sleep_time)

    return synonyms_dict

########################################################################################

def fetch_gene_synonyms_basic(gene_name):
    """
    Fetch gene synonyms for a given gene name using the Ensembl REST API.

    Args:
        gene_name (str): Gene name to query.

    Returns:
        list: List of gene synonyms or an empty list if none found.
    """
    response = requests.get(f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json")
    if response.status_code == 200:
        data = response.json()
        return [entry['display_id'] for entry in data] if data else []
    return []

def fetch_gene_synonyms(gene_name):
    """
    Fetch gene synonyms for a given gene name using the Ensembl REST API, NCBI Gene database, HGNC, and UniProt.

    Args:
        gene_name (str): Gene name to query.

    Returns:
        list: List of gene synonyms or an empty list if none found.
    """
    synonyms = []

    # Fetch from Ensembl
    response = requests.get(f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json")
    if response.status_code == 200:
        try:
            data = response.json()
            if data:
                synonyms.extend([entry['display_id'] for entry in data if 'display_id' in entry])
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from Ensembl for gene {gene_name}")

    # Fetch from NCBI Gene if no synonyms found in Ensembl
    if not synonyms:
        try:
            handle = Entrez.esearch(db="gene", term=gene_name)
            record = Entrez.read(handle)
            handle.close()
            if record['IdList']:
                gene_id = record['IdList'][0]
                try:
                    handle = Entrez.efetch(db="gene", id=gene_id, retmode="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    if records:
                        gene_info = records[0]
                        if 'OtherAliases' in gene_info:
                            synonyms.extend(gene_info['OtherAliases'].split(', '))
                        if 'OtherDesignations' in gene_info:
                            synonyms.extend(gene_info['OtherDesignations'].split(', '))
                except HTTPError as e:
                    print(f"HTTP error occurred while fetching gene info for {gene_name}: {e}")
        except HTTPError as e:
            print(f"HTTP error occurred while searching for gene {gene_name}: {e}")

    # Fetch from HGNC if no synonyms found in NCBI Gene
    if not synonyms:
        response = requests.get(f"https://rest.genenames.org/fetch/symbol/{gene_name}")
        if response.status_code == 200:
            try:
                data = response.json()
                if 'response' in data and 'docs' in data['response']:
                    for doc in data['response']['docs']:
                        if 'alias_symbol' in doc:
                            synonyms.extend(doc['alias_symbol'])
                        if 'prev_symbol' in doc:
                            synonyms.extend(doc['prev_symbol'])
            except requests.exceptions.JSONDecodeError:
                print(f"Error decoding JSON response from HGNC for gene {gene_name}")

    # Fetch from UniProt if no synonyms found in HGNC
    if not synonyms:
        response = requests.get(f"https://www.uniprot.org/uniprot/?query={gene_name}&format=tab&columns=id,genes(PREFERRED),genes(ALTERNATIVE)")
        if response.status_code == 200:
            lines = response.text.strip().split('\n')
            for line in lines[1:]:  # Skip header line
                parts = line.split('\t')
                if len(parts) > 2:
                    synonyms.extend(parts[2].split(' '))  # ALTERNATIVE gene names

    return synonyms


def fetch_ensembl_id(gene_name):
    """
    Fetch Ensembl ID for a given gene name using the Ensembl REST API.

    Args:
        gene_name (str): Gene name to query.

    Returns:
        str: Ensembl ID or 'Not found' if not found.
    """
    response = requests.get(f"https://rest.ensembl.org/xrefs/symbol/homo_sapiens/{gene_name}?content-type=application/json")
    if response.status_code == 200:
        try:
            data = response.json()
            if data:
                for entry in data:
                    if 'id' in entry:
                        return entry['id']
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from Ensembl for gene {gene_name}")
    return 'Not found'


def load_additional_mappings_v0(hgnc_mapping_file, uniprot_mapping_file):
    """
    Load additional mappings from HGNC and UniProt to Ensembl IDs.

    Args:
        hgnc_mapping_file (str): Path to HGNC mapping file.
        uniprot_mapping_file (str): Path to UniProt mapping file.

    Returns:
        dict: Combined mapping dictionary.

    Memo: 
    - HGNC mappings can be downloaded via wget from 
      - ftp://ftp.ebi.ac.uk/pub/databases/genenames/new/tsv/hgnc_complete_set.txt
    
    - UniProt mappings can be downloaded via wget from
      - ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping_selected.tab.gz
    """
    combined_mappings = {}

    # Load HGNC mappings
    # e.g. 
    if hgnc_mapping_file and os.path.exists(hgnc_mapping_file):
        hgnc_df = pd.read_csv(hgnc_mapping_file, sep='\t')
        for index, row in hgnc_df.iterrows():
            if 'hgnc_id' in row and 'ensembl_gene_id' in row:
                combined_mappings[row['hgnc_id']] = row['ensembl_gene_id']

    # Load UniProt mappings
    if uniprot_mapping_file and os.path.exists(uniprot_mapping_file):
        uniprot_df = pd.read_csv(uniprot_mapping_file, sep='\t')
        for index, row in uniprot_df.iterrows():
            if 'uniprot_id' in row and 'ensembl_gene_id' in row:
                combined_mappings[row['uniprot_id']] = row['ensembl_gene_id']

    return combined_mappings


def load_hgnc_mapping_v0(file_path):
    hgnc_df = pd.read_csv(file_path, sep='\t')
    hgnc_mapping = dict(zip(hgnc_df['symbol'], hgnc_df['ensembl_gene_id']))
    return hgnc_mapping


def load_hgnc_mapping(file_path):
    """
    Load HGNC to Ensembl mappings from a file.

    Args:
        file_path (str): Path to the HGNC mapping file.

    Returns:
        dict: Dictionary mapping HGNC symbols to Ensembl IDs.
    """
    # Load the file with appropriate handling of mixed data types
    hgnc_df = pd.read_csv(file_path, sep='\t', low_memory=False)
    
    # Ensure that the relevant columns have consistent data types
    hgnc_df['symbol'] = hgnc_df['symbol'].astype(str)
    hgnc_df['ensembl_gene_id'] = hgnc_df['ensembl_gene_id'].astype(str)
    
    # Create the mapping
    hgnc_mapping = dict(zip(hgnc_df['symbol'], hgnc_df['ensembl_gene_id']))
    
    return hgnc_mapping


def load_uniprot_mapping(file_path):
    """
    Load UniProt to Ensembl mappings from a file.

    Args:
        file_path (str): Path to the UniProt mapping file.

    Returns:
        dict: Dictionary mapping UniProt IDs to Ensembl IDs.
    """
    # Load the file with appropriate handling of mixed data types
    uniprot_df = pd.read_csv(file_path, sep='\t', header=None, low_memory=False)
    
    # Example columns based on the provided UniProt file structure
    # Adjust these indices if your file structure differs
    uniprot_df.columns = [str(i) for i in range(len(uniprot_df.columns))]
    
    # Extract relevant columns
    uniprot_mapping = {}
    for _, row in uniprot_df.iterrows():
        if pd.notna(row['21']):  # Column 21 seems to have Ensembl Gene IDs
            ensembl_ids = row['21'].split(';')
            for ensembl_id in ensembl_ids:
                ensembl_id_clean = ensembl_id.split()[0]  # Remove any version number or additional data
                uniprot_mapping[row['0']] = ensembl_id_clean
    
    return uniprot_mapping


def load_additional_mappings(hgnc_mapping_file, uniprot_mapping_file):
    """
    Load additional mappings from HGNC and UniProt to Ensembl IDs.

    Args:
        hgnc_mapping_file (str): Path to HGNC mapping file.
        uniprot_mapping_file (str): Path to UniProt mapping file.

    Returns:
        dict: Combined mapping dictionary.
    """
    combined_mappings = {}

    # Load HGNC mappings
    if hgnc_mapping_file and os.path.exists(hgnc_mapping_file):
        hgnc_mapping = load_hgnc_mapping(hgnc_mapping_file)
        combined_mappings.update(hgnc_mapping)

    # Load UniProt mappings
    if uniprot_mapping_file and os.path.exists(uniprot_mapping_file):
        uniprot_mapping = load_uniprot_mapping(uniprot_mapping_file)
        combined_mappings.update(uniprot_mapping)

    return combined_mappings


def find_ensembl_ids(gene_names, hgnc_mapping_file=None, uniprot_mapping_file=None):
    """
    Find Ensembl IDs for a list of gene names by fetching synonyms and querying Ensembl.

    Args:
        gene_names (list): List of gene names to query.
        hgnc_mapping_file (str): Path to HGNC mapping file.
        uniprot_mapping_file (str): Path to UniProt mapping file.

    Returns:
        dict: Dictionary mapping gene names to their Ensembl IDs.
    """
    gene_to_ensembl = {}
    gene_synonyms_lookup = {}

    # Load additional mappings
    additional_mappings = load_additional_mappings(hgnc_mapping_file, uniprot_mapping_file)

    for gene in gene_names:
        synonyms = fetch_gene_synonyms(gene)
        gene_synonyms_lookup[gene] = synonyms

        # Try to find Ensembl ID for the gene or its synonyms
        ensembl_id = fetch_ensembl_id(gene)
        if ensembl_id == 'Not found':
            for synonym in synonyms:
                ensembl_id = fetch_ensembl_id(synonym)
                if ensembl_id != 'Not found':
                    break
        if ensembl_id == 'Not found' and gene in additional_mappings:
            ensembl_id = additional_mappings[gene]

        gene_to_ensembl[gene] = ensembl_id

        # Print progress and sleep to avoid hitting API rate limits
        print(f"Gene: {gene}, Ensembl ID: {ensembl_id}, Synonyms: {synonyms}")
        time.sleep(1)

    return gene_to_ensembl


def create_gene_syns_file(gene_names, output_tsv="gene_syns.tsv", batch_size=10, sleep_time=1):
    """
    Creates a TSV file containing gene synonyms for a list of genes and attempts to find Ensembl IDs.

    Parameters:
    gene_names (list): List of gene names to fetch synonyms for.
    output_tsv (str): Name of the output TSV file.
    batch_size (int): Number of genes to process before pausing to avoid API limits.
    sleep_time (int): Time to sleep (in seconds) between batches to avoid API limits.
    """
    gene_synonyms_lookup = {}
    ensembl_ids_dict = {}
    gene_to_synonyms = {}
    count = 0

    with open(output_tsv, 'w', newline='') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        writer.writerow(['gene_name', 'synonym'])

        for gene in gene_names:
            synonyms = fetch_gene_synonyms(gene)

            # Ensure synonyms is a list
            if synonyms is None:
                synonyms = []

            # Store synonyms in a dictionary for easy lookup
            print("[info] Synonyms for gene:", gene, "=>", synonyms)
            gene_synonyms_lookup[gene] = synonyms

            # Attempt to find Ensembl ID for the gene or its synonyms
            ensembl_id = fetch_ensembl_id(gene)
            if ensembl_id == 'Not found':
                for synonym in synonyms:
                    ensembl_id = fetch_ensembl_id(synonym)
                    if ensembl_id != 'Not found':
                        break

            ensembl_ids_dict[gene] = ensembl_id
            print(f"[info] Ensembl ID for {gene} => {ensembl_id}")

            # Store synonyms in the desired format
            if synonyms:
                for synonym in synonyms:
                    if synonym.strip():  # Ensure synonym is not empty
                        writer.writerow([synonym, gene])
            else:
                print(f"No synonyms found for {gene}")

            count += 1
            if count % batch_size == 0:
                print(f"Processed {count} genes, sleeping for {sleep_time} seconds to avoid hitting API limits...")
                time.sleep(sleep_time)

    return gene_synonyms_lookup, ensembl_ids_dict


def test_fetch_gene_synonyms():
    # Example usage:
    # gene_names = ["BRCA1", "BRCA2", "TP53"]
    gene_names = ['AC016747.1', 'LINC01137', 'AC016831.1', 'AC073050.1', 'AC026304.1', 'GS1-124K5.4']

    # Fetch all known genes
    # gene_names = fetch_all_genes()

    # Fetch synonyms for all genes
    filepath= "/path/to/meta-spliceai/data/Han1/gene_syn-ncbi.tsv"
    # create_gene_syns_file(gene_names, output_tsv=filepath)

    # gene_synonyms_lookup, ensembl_ids_dict = create_gene_syns_file(gene_names, output_tsv=filepath)
    # print("Gene Synonyms Lookup:", gene_synonyms_lookup)
    # print("Ensembl IDs Dictionary:", ensembl_ids_dict)

    hgnc_mapping_file = '/path/to/meta-spliceai/data/HGNG/hgnc_complete_set.txt'
    uniprot_mapping_file = '/path/to/meta-spliceai/data/UniProt/HUMAN_9606_idmapping_selected.tab'
    # hgnc_mapping = load_hgnc_mapping(hgnc_mapping_file)
    # uniprot_mapping = load_uniprot_mapping(uniprot_mapping_file)

    ensembl_ids_dict = find_ensembl_ids(gene_names, hgnc_mapping_file, uniprot_mapping_file)
    print(ensembl_ids_dict)
    print("Ensembl IDs Dictionary:", ensembl_ids_dict)


def test(): 
    test_fetch_gene_synonyms()


if __name__ == "__main__":
    test()