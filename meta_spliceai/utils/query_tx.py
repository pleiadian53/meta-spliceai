import requests
import pandas as pd
from meta_spliceai.sphere_pipeline.access_sphere import (
    submit_query, 
    get_db_connection, 
    download_from_blob)


def get_transcript_coordinates(transcript_id):
    """

    Usage Example: 

    transcript_id = "ENST00000644676.1"
    get_transcript_coordinates(transcript_id)
    """

    url = f"https://rest.ensembl.org/lookup/id/{transcript_id}"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        
        if 'start' in data and 'end' in data:
            start = data['start']
            end = data['end']
            print(f"Transcript {transcript_id} starts at {start} and ends at {end}.")
        else:
            print("Start or end position not found in the response.")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response content: {response.content}")
    except Exception as err:
        print(f"Other error occurred: {err}")



def demo_query_spliceprep_metadata(): 
    # from synapse.query_templates import query_templates
    # import utils_data as ud
    # import sphere_pipeline.utils_algo as ua

    # Count disintct values of sample names 
    sql = \
"""
SELECT sample_name, COUNT(*) AS count
FROM spliceprep_metadata
GROUP BY sample_name
ORDER BY count DESC, sample_name;  -- Optional: Order by count in descending order, then by sample_name
"""
    # A. Using synapse_db_conn
    # query = sqa.text(sql)  # sqlalchemy
    # df_sn_cnt = pd.read_sql(query, db_conn) # connection obtained from Synapse, synapse_sql.connect(engine = "sqlalchemy")
    df_sn_cnt = submit_query(sql)

    print(df_sn_cnt.head())


def test(): 
    
    demo_query_spliceprep_metadata()


if __name__ == "__main__": 
    test()