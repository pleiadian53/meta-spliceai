import os
import time
from datetime import timedelta

import pandas as pd
import polars as pl

import logging
import warnings
import subprocess


def extract_gene_features_from_gtf_by_awk_command(gtf_file, output_file):
    """
    Execute the awk command to extract only gene information from a GTF file.

    Parameters:
    - gtf_file (str): Path to the input GTF file.
    - output_file (str): Path to the output file to save the extracted gene information.
    """
    cmd = f'''awk '$3 == "gene" && $0 ~ /gene_id "ENSG/' {gtf_file} > {output_file}'''
    subprocess.run(cmd, shell=True, check=True)


def setup_logger(log_file_path):
    """
    Set up a logger that prints to the console and writes to a log file.

    Parameters:
    - log_file_path (str): The file path to log warnings.
    """
    logger = logging.getLogger('gene_id_logger')
    logger.setLevel(logging.DEBUG)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    # Create formatters and add them to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def pandas_to_polars(pandas_df):
    """
    Convert a Pandas DataFrame to a Polars DataFrame.
    
    Parameters:
    - pandas_df (pd.DataFrame): The Pandas DataFrame to convert.
    
    Returns:
    - polars_df (pl.DataFrame): The converted Polars DataFrame.
    """
    return pl.from_pandas(pandas_df)

def polars_to_pandas(polars_df):
    """
    Convert a Polars DataFrame to a Pandas DataFrame.
    
    Parameters:
    - polars_df (pl.DataFrame): The Polars DataFrame to convert.
    
    Returns:
    - pandas_df (pd.DataFrame): The converted Pandas DataFrame.
    """
    return polars_df.to_pandas()


def convert_csv_to_tsv(input_csv_path, output_tsv_path=None, schema_overrides=None):
    """
    Read a .csv file into a Polars DataFrame and save it to a .tsv file.

    Parameters:
    - input_csv_path (str): Path to the input .csv file.
    - output_tsv_path (str, optional): Path to the output .tsv file. If None, use the same file name with .tsv extension.
    - schema_overrides (dict, optional): Dictionary to specify column data types. Default is None.
    """
    # Determine the output path if not provided
    if output_tsv_path is None:
        output_tsv_path = os.path.splitext(input_csv_path)[0] + '.tsv'

    # Read the .csv file into a Polars DataFrame with schema overrides if provided
    df = pl.read_csv(input_csv_path, schema_overrides=schema_overrides)
    
    # Save the DataFrame to a .tsv file
    df.write_csv(output_tsv_path, separator='\t')


def calculate_and_format_duration(start_time):
    """
    Calculate the duration from the start time and return it in hours, minutes, and seconds.

    Parameters:
    - start_time (float): The start time in seconds since the epoch.

    Returns:
    - str: The formatted duration string.
    """
    # Debugging: Print the type and value of start_time
    print(f"Debug: start_time type: {type(start_time)}, value: {start_time}")

    # Ensure start_time is a float
    if not isinstance(start_time, (float, int)):
        raise ValueError("start_time must be a float or int representing seconds since the epoch")

    duration_so_far = time.time() - start_time

    # Debugging: Print the value of duration_so_far
    print(f"Debug: duration_so_far: {duration_so_far}")

    hours, remainder = divmod(duration_so_far, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"


def calculate_duration(start_time):
    """
    Calculate the duration from the start time and return it in hours, minutes, and seconds.
    """
    return calculate_and_format_duration(start_time)


def format_duration_as_string_or_tuple(duration, return_tuple=False):
    """
    Format a duration in seconds as a string in hours, minutes, and seconds,
    or return the hours, minutes, and seconds as a tuple.

    Parameters:
    - duration (float): The duration in seconds to format.
    - return_tuple (bool): If True, return a tuple (hours, minutes, seconds).
                           If False, return a formatted string.

    Returns:
    - str or tuple: The formatted duration string or a tuple (hours, minutes, seconds).
    """
    # Convert duration to timedelta if it's not already
    if not isinstance(duration, timedelta):
        duration = timedelta(seconds=duration)
    
    # Extract hours, minutes, and seconds
    total_seconds = duration.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if return_tuple:
        return int(hours), int(minutes), int(seconds)
    else:
        return f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"

def format_time(duration, return_tuple=False):
    """
    Format a duration in seconds as a string in hours, minutes, and seconds,
    or return the hours, minutes, and seconds as a tuple.
    """
    return format_duration_as_string_or_tuple(duration, return_tuple=return_tuple)


def print_emphasized(text, style='bold', edge_effect=True, symbol='='):
    styles = {
        'bold': '\033[1m',
        'underline': '\033[4m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m'
    }
    end_style = '\033[0m'  # Reset to default style

    if edge_effect:
        edge_line = symbol * len(text)
        print(edge_line)
        print(f"{styles.get(style, '')}{text}{end_style}")
        print(edge_line)
    else:
        print(f"{styles.get(style, '')}{text}{end_style}")


def demo_calculate_and_format_duration():

    # Example usage
    start_time = time.time()
    time.sleep(2)  # Simulate some processing time
    print(calculate_and_format_duration(start_time))

def demo_file_conversion():

    input_csv_path = "/path/to/meta-spliceai/data/ensembl/spliceai_analysis/exon_df_from_gtf.csv"
    convert_csv_to_tsv(input_csv_path, schema_overrides={'seqname': pl.Utf8})


def test(): 

    # demo_calculate_and_format_duration()

    demo_file_conversion()


if __name__ == "__main__":
    test()