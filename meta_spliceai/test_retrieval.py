import sys, os, re
from pathlib import Path 

# current_dir = Path.cwd() # os.path.dirname(os.path.realpath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir) 
# os.environ['RUNTIME_ENV'] = 'LOCAL'

import pandas as pd

from .system import sys_config as config
from envisapycore.connectors import create_sql_engine
from envisapycore.auth.sql_connections import retry_on_failure

from .sphere_pipeline import constants as const
from .sphere_pipeline.process_trpts_from_synapse import (
    map_gene_source_id, map_gene_source_id_given_gene_set,
    map_tx_source_id, map_tx_source_id_given_tx_set)

from .sphere_pipeline.data_model import (Concept, MetaIO,  
                                       TranscriptIO, )

from .sphere_pipeline.access_sphere import submit_query

# Synapse SQL (Outdated)
# synapse_pyodbc_engine = create_sql_engine(
#     sql_service_type="Synapse",
#     sql_engine_type= "pyodbc"  # "sqlalchemy", "pyodbc"
# 	# NOTE: "sqlalchemy" does not work yet
# )

# Fabric SQL
# fabric_pyodbc_engine = create_sql_engine(
#     sql_service_type="fabric-base_lakehouse", sql_engine_type="pyodbc"
# )


###########################################################################
# 
# Q: RUNTIME_ENV not set. Defaulting to LOCAL? 
#    os.environ['RUNTIME_ENV'] = 'LOCAL'
# 
#    echo "export RUNTIME_ENV=LOCAL" >> ~/.bashrc
#    source ~/.bashrc
# 
###########################################################################

def read_file_into_dataframe(filepath):
    if os.path.isdir(filepath):
        df = pd.read_parquet(filepath) # assume it's a directory containing parquet files
    else:
        _, file_extension = os.path.splitext(filepath)
        if file_extension in ['.csv', '.tsv']:
            sep = ',' if file_extension == '.csv' else '\t'
            df = pd.read_csv(filepath, sep=sep)
        elif file_extension == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")
    return df


@retry_on_failure(max_retries=3, delay=5)
def get_transcripts():
    sql = "select top 5 * from transcripts"
    with synapse_pyodbc_engine as conn:
        df = pd.read_sql_query(sql, conn)

    print(df)


def test_map_gene_source_id():
    import meta_spliceai.system.sys_config as config

    target_biotype = "generic"
    target_suffix = "test"
    base_concept = "exon"

    # Constants
    col_tid = const.col_tid
    col_gid = const.col_gid     

    # Get system project directory (which uses external storage)
    # proj_dir = config.get_proj_dir()
    # print("> System project directory:\n{}\n".format(proj_dir))

    # Use local project dirctory
    proj_dir =  config.get_proj_dir() # Config.PROJ_DIR
    tx_path = os.path.join(proj_dir, "examples/crc_toy_data.csv")

    # concept = Concept(concept=base_concept, biotype=target_biotype, suffix=target_suffix)
    # print(f"[info] Mapping internal gene IDs to their source IDs ({concept_target.id_str}) ...")
    txio_testset = TranscriptIO(biotype=target_biotype, suffix=target_suffix)

    if tx_path is not None:
        # If tx_path is not a file path, prepend txio_testset.data_dir
        if not os.path.dirname(tx_path):
            print(f"... using default data path (ID={txio_testset.ID}):\n{txio_testset.tx_dir}\n")
            txio_testset.create_tx_dir()

            tx_path = os.path.join(txio_testset.tx_dir, tx_path)

    # Check if the file exists
    df_tx = None
    if os.path.exists(tx_path):
        # sep = detect_separator(tx_path)
        print(f"[test] Reading from file:\n{tx_path}\n")

        # Read the file into a DataFrame
        df = read_file_into_dataframe(tx_path)
        print(f"[main] Found the following columns in the transcript file:\n{list(df.columns)}")

        # Keep only the 'tx_id' and 'gene_id' columns, if they exist
        if col_tid in df.columns and col_gid in df.columns:
            df = df[[col_tid, col_gid]]

            num_samples = min(10, len(df[col_gid]))
            sample_values = df[col_gid].sample(num_samples)
            print(f"[test] Example genes:\n{sample_values}\n")
        elif col_tid in df.columns:
            df = df[[col_tid]]
        else:
            raise ValueError("The file must contain a 'tx_id' column")   

        df_tx = df 
        n_genes = df_tx[col_gid].nunique()
        n_trpts = df_tx[col_tid].nunique()
        print(f"... Given n={n_genes} genes and {n_trpts} transcripts in the file")

    gene_set = df_tx[col_gid].unique()

    tx_io, gene_path = map_gene_source_id_given_gene_set(gene_set, tx_io=txio_testset)
    # tx_io, gene_path = map_gene_source_id(biotype=target_biotype, suffix=target_suffix)  

    print("> Gene source ID mapping:") 
    gene_src_id = tx_io.load_gene_src_id() 
    print("... path: {}\n".format(gene_path))
    print("... columns: {}\n".format(list(gene_src_id.columns)))
    print("... n(genes): nrows={} >=? nunique={}\n".format(gene_src_id.shape[0], gene_src_id[col_gid].nunique()))

    return

def test_sequence_retrieval(): 
    from envisapycore.connectors import BlobConnector
    # NOTE: connectors package already defines relative imports like from .storage_connectors.blob_connector import BlobConnector

    target_biotype = "generic"
    target_suffix = "test"

    tx_io = TranscriptIO(biotype='generic', suffix='test')

    output_dir = tx_io.data_dir
    if not os.path.exists(output_dir):
        print(f"[info] Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    hg38 = 'hg38.p14.2bit'
    path_to_hg38 = os.path.join(output_dir, hg38)

    container_client = BlobConnector(
                storage_prefix = 'SPLICEPREP:STORAGEHOT',
                container = 'misc-resources'
            )     
    container_client.download_blob(
        blob_path = hg38, output_file_path = path_to_hg38)

    print("> Saved transcript sequences to: {}".format(path_to_hg38))
    assert os.path.exists(path_to_hg38)

    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(path_to_hg38)
    file_size_megabytes = file_size_bytes / 1024 / 1024  # Convert the file size to megabytes
    print(f"The size of the downloaded dataset is {file_size_megabytes} Mb or {file_size_bytes} bytes")

    return

def test_basic_connection(): 
    # import sqlalchemy as sqa
    # print("> Create a basic SQL pyodbc engine that connects to Synapse using your CLI credentials")
    
    # sql = "select top 5 * from txdb.transcripts"
    # # sql = sqa.text(sql)
    # with synapse_pyodbc_engine as conn:
    #     df = pd.read_sql_query(sql, conn)
    # print(df.head())

    # print("> Use a more robust method using a decorator so that it can manage retries on failure")

    # get_transcripts()

    # Fabric SQL
    print("\n\n")
    print("> Using Fabric connection to connect to the Fabric SQL database")
    # fabric_pyodbc_engine = create_sql_engine(
    #     sql_service_type="fabric-base_lakehouse", sql_engine_type="pyodbc"
    # )

    sql = "SELECT TOP 10 * FROM transcripts"   # Notice that the prefix is changed to dbo, or no prefix is needed
    # with fabric_pyodbc_engine as conn:
    #     df = pd.read_sql_query(sql, conn)
    df = submit_query(sql)
    print(df.head())

    return 

def test_query_transcripts():
    from .analyze_pub_results import prepare_input_data_for_uorfinder
    # from sphere_pipeline.synapse.query_templates import query_templates, format_sql_query
    from .sphere_pipeline import (query_templates, format_sql_query, 
                                    remove_constraints_from_sql_query, match_tx_source_ids, truncate_sql_query)

    col_tid = const.col_tid
    col_gid = const.col_gid

    # Fabric SQL
    print("\n\n")
    print("> Using Fabric connection to connect to the Fabric SQL database")

    sql = "SELECT TOP 10 * FROM transcripts"   # Notice that the prefix is changed to dbo, or no prefix is needed
    # with fabric_pyodbc_engine as conn:
    #     df = pd.read_sql_query(sql, conn)
    df = submit_query(sql)
    print(df.head())

    print("> Columns: {}".format(list(df.columns)))
    # NOTE: ['tx_id', 'pt_id', 'gene_id', 'junc_id', 
    #        'tx_length', 'start', 'end', 'chromosome', 'strand', 'cds_start', 'cds_end', 'tx_type_ref', 
    #        'tx_type_pred', 'is_pt_pred', 'is_known', 'spark_pipeline_id', 'spark_creation_date', 'table_creation_date']

    print("> Given SeqSphere data, prepare input data for uORFinder ...")
    data_dir = config.get_proj_dir()
    input_dir = os.path.join(data_dir, "data")  
    dfs = prepare_input_data_for_uorfinder(input_dir=input_dir)

    df = dfs['S2. uORF-connected transcripts']
    print(df.head())
    print("> Columns: {}".format(list(df.columns)))
    
    reference_ids = df['reference_id'].unique()
    print("> Number of reference IDs: {}".format(len(reference_ids)))

    # Example usage
    constraints = {
        'tx_alias_base': reference_ids, # ['ENST00000644676.1', 'ENST00000361544.11', 'ENST00000602296.6'],
    }
    table_name = 'tx_ann'
    # query = format_sql_query(table_name, constraints) # template_id='get_reference_info_from_tx_ann', 

    print("> Match transcript source IDs (minus versions)...")
    query = match_tx_source_ids(table_name, constraints, match_version=False)
    print("> Query: {}".format(query[:500]))
    df = submit_query(query)
    
    n_trpts = df[col_tid].nunique()
    print(f"> shape(df): {df.shape}")
    print(f"> n(trpts): {n_trpts}")
    print(df.head())
    # NOTE: 
    #                                  tx_id    tx_alias_base tx_alias_version reference_base                        reference_version
    # 0  TX.dc99bd20c38e3ca53cbd50a070cea34b  ENST00000339624                9        gencode  gencode.v26.primary_assembly.annotation
    # 1  TX.244b6ed34e86d840bf16a59358830df3  ENST00000564644                5        gencode  gencode.v26.primary_assembly.annotation
    # 2  TX.bc64f2be2872d7ad4b20131acbf50b58  ENST00000682901                1        gencode  gencode.v43.primary_assembly.annotation
    # 3  TX.c6b236b8640a546962cf94f31c4f3d77  ENST00000345306               10        gencode  gencode.v26.primary_assembly.annotation
    # 4  TX.0f0353e891a676e4dac73d0337377e40  ENST00000545620                5        gencode  gencode.v43.primary_assembly.annotation

    if df.empty or df is None: 
        query = remove_constraints_from_sql_query(query, top_n=1000)
        print("> Query minus constraints: {}".format(query[:500]))
        df = submit_query(query)
        print(f"> shape(df): {df.shape}")
        print(f"> n(trpts): {n_trpts}")
        print(df.head())
    
    print()
    print("> Match transcript source IDs (including versions)...")
    query = match_tx_source_ids(table_name, constraints, match_version=True)

    trucated_query_str = truncate_sql_query(query, 10)
    print("> Query: {}".format(trucated_query_str))
    
    # df = submit_query(query)

    # n_trpts = df[col_tid].nunique()
    # print(f"> shape(df): {df.shape}") 
    # print(f"> n(trpts): {n_trpts}")
    # print(df.head())

    # NOTE: If you're querying the database without specifying a version number, 
    #       you're likely getting the latest version of each transcript ID in your results. 
    #       This is why the number of matches is the same whether or not you include the version number in your query.

    return

def test(): 
    # Test basic SQL server connection
    # test_basic_connection()

    # Query tables from SeqSphere via Fabric
    test_query_transcripts()

    # ----------------------------------------------------

    # Sequence retrieval
    # test_sequence_retrieval()  # ok

    # ----------------------------------------------------

    # Test mapping gene source IDs
    # test_map_gene_source_id()
    

    return

def demo(): 

    return

if __name__ == '__main__':
	# demo()

    test()