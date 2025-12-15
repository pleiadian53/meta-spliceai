import os, sys
import pandas as pd


def get_sheet_names(file_path, verbose=1):
    """

    Memo
    ----
    1. Dependencies:
        pip install openpyxl
    """

    # Load the Excel file
    excel_file = pd.ExcelFile(file_path)  

    # Get the sheet names
    sheet_names = excel_file.sheet_names

    if verbose:
        print(f"(get_sheet_names) excel_file.sheet_names:\n{sheet_names}\n")

    return sheet_names


def convert_excel(file_path, output_dir=None, output_format='tsv', verbose=1):
    """
    Convert each sheet in an Excel file to a separate TSV or CSV file.

    Parameters:
        file_path (str): The path to the Excel file to convert.
        output_dir (str, optional): The directory to output the TSV or CSV files to. 
                                    If None, the directory of the input file is used. 
                                    Defaults to None.
        output_format (str, optional): The format to output the files in. 
                                    Can be either 'tsv' or 'csv'. Defaults to 'tsv'.

    Returns:
        str: A message indicating whether the conversion was successful or the error message.

    Memo
    ----
    1. Dependencies:
        pip install openpyxl
    """
    file_names = {}
    try:
        # If no output directory is specified, use the directory of the input file
        if output_dir is None:
            output_dir = os.path.dirname(file_path)

        # Load the Excel file
        excel_file = pd.ExcelFile(file_path)

        if verbose:
            print(f"(convert_excel) excel_file.sheet_names:\n{excel_file.sheet_names}\n")
        
        # Determine the separator based on the output format
        sep = '\t' if output_format == 'tsv' else ','

        # Iterate over each sheet and save as TSV or CSV
        for sheet_name in excel_file.sheet_names:
            # Read the sheet into a DataFrame
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Define the output file name
            output_file_name = os.path.join(output_dir, f"{sheet_name}.{output_format}")
            file_names[sheet_name] = output_file_name
            
            # Save the DataFrame as a TSV or CSV file
            df.to_csv(output_file_name, sep=sep, index=False)
            print(f"Converted {sheet_name} to {output_file_name}")
            
        return "Conversion successful"
    except Exception as e:
        print(str(e)) 

    return file_names


def convert_excel_to_tsv_v0(file_path):
    try:
        # Load the Excel file
        excel_file = pd.ExcelFile(file_path)
        
        # Iterate over each sheet and save as TSV
        for sheet_name in excel_file.sheet_names:
            # Read the sheet into a DataFrame
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Define the TSV file name
            tsv_file_name = f"/mnt/data/{sheet_name}.tsv"
            
            # Save the DataFrame as a TSV file
            df.to_csv(tsv_file_name, sep='\t', index=False)
            print(f"Converted {sheet_name} to {tsv_file_name}")
            
        return "Conversion successful"
    except Exception as e:
        return str(e)


def read_files_into_dataframes(file_names, input_dir, file_format='tsv'):
    """
    Read a list of TSV or CSV files into dataframes.

    Parameters:
    file_names (list or str): A list of file names (without the extension) or a single file name.
    input_dir (str): The directory where the files are located.
    file_format (str, optional): The format of the files. Can be either 'tsv' or 'csv'. Defaults to 'tsv'.

    Returns:
    dict: A dictionary mapping the file names to their corresponding dataframes.
    """
    # If file_names is a string, convert it to a list
    if isinstance(file_names, str):
        file_names = [file_names, ]

    # Determine the separator based on the file format
    sep = '\t' if file_format == 'tsv' else ','

    # Dictionary to hold dataframes
    dfs = {}

    # Iterate over the file names
    for file_name in file_names:
        # Construct the file path
        file_path = os.path.join(input_dir, f"{file_name}.{file_format}")
        
        # Read the file into a dataframe
        df = pd.read_csv(file_path, sep=sep)
        
        # Store the dataframe in the dictionary
        dfs[file_name] = df

    return dfs

def demo_read_files_into_dataframes():

    # Get the home directory
    home_dir = os.path.expanduser("~")
    proj_dirname = 'uORF_explorer'

    # Define the path to the newly uploaded Excel file
    input_dir = os.path.join(home_dir, f"work/{proj_dirname}/data")
    file_path = os.path.join(input_dir, 'supplementary-tables.xlsx')
    assert os.path.exists(file_path), f"File not found: {file_path}"
    # file_path = '/mnt/data/supplementary-tables.xlsx'

    print(f"> Converting Excel file to TSV files: {file_path}")

    # Convert the provided Excel file to TSV files
    sheet_names_to_paths = convert_excel(file_path)

    # Define the names of the TSV files to read
    sheet_names = list(sheet_names_to_paths.keys())
    
    # Example Sheet Names
    # sheet_names = ['S1. Novel protein-coding exons', 
    #             'S2. uORF-connected transcripts', 
    #             'S3. Select transcripts', 
    #             'S4. uORF conservation scores', 
    #             'S5. Number of conserved uORFs', 
    #             'S6. Missing gene names', 
    #             'S7. CDSs with non-AUG starts', 
    #             'S8. CDSs encoding same proteins']
                
    dfs = read_files_into_dataframes(sheet_names, input_dir, file_format='tsv')

    return dfs 


def load_excel_sheets_into_dataframes(file_path, file_format='tsv', to_tsv=False, **kargs): 
    
    verbose = kargs.get('verbose', 1)
    return_paths = kargs.get('return_paths', False)
    
    data_dir = os.path.dirname(file_path)

    if verbose: print(f"(load_excel_data) Reading Excel file: {file_path}")
    sheet_names = get_sheet_names(file_path, verbose=1)

    if verbose: print("(load_excel_data) sheet_names:\n", sheet_names)

    sheet_names_to_paths = {}
    if to_tsv:  
        sheet_names_to_paths = \
            convert_excel(file_path,
                          output_dir=data_dir,
                          output_format=file_format, verbose=1)
    else: 
        for sheet_name in sheet_names:
            # Define the output file name
            output_file_name = os.path.join(data_dir, f"{sheet_name}.{file_format}")
            sheet_names_to_paths[sheet_name] = output_file_name
            
            print(f"Converted {sheet_name} to {output_file_name}")

    # Read the TSV files into dataframes
    dfs = read_files_into_dataframes(sheet_names, input_dir=data_dir, file_format=file_format) 
    # print(dfs.keys())
    
    # pprint.pprint(dfs)  
    if verbose: 
        example_sheet = 'S2. PHASE I Ribo-seq ORFs'
        df = dfs[example_sheet]
        print(df.head())
        print(f"(load_excel_sheets_into_dataframes) Columns:\n{list(df.columns)}\n")

    if return_paths: 
        return dfs, sheet_names_to_paths

    return dfs


def demo(): 

    input_dir = "/path/to/meta-spliceai/data/ORF"
    file_path = os.path.join(input_dir, "NIHMS1854551-supplement-supplementary_tables.xlsx")
    # convert_excel(file_path, output_dir=None, output_format='tsv', verbose=1)
    dfs = load_excel_sheets_into_dataframes(file_path)
  
    # demo_read_files_into_dataframes()


    return


if __name__ == "__main__":
    demo()
