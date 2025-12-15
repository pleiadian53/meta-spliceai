
import os
import re

# Define the lakehouse path where the config file is stored
workspace = "metaspliceai"
lakehouse_name = 'metaspliceai_lakehouse'
lakehouse_path = 'abfss://metaspliceai@onelake.dfs.fabric.microsoft.com/metaspliceai_lakehouse.Lakehouse'
lakehouse_api_path_prefix = "/lakehouse/default/Files"
use_spark = False


class BaseLakehouse: 
    workspace = "seqsphere_ingestion"  # "txdb"
    lakehouse = "base_lakehouse"
    # table_name = "transcripts"

    # table_path = f"abfss://{workspace}@onelake.dfs.fabric.microsoft.com/{lakehouse}.Lakehouse/Tables/{table_name}"
    table_path = "abfss://{workspace}@onelake.dfs.fabric.microsoft.com/{lakehouse}.Lakehouse/Tables/{table_name}"

    # Let table_name = "transcripts"
    #     table_path = table_path.format(workspace=workspace, lakehouse=lakehouse, table_name=table_name)
    # gives 
    #     abfss://txdb@onelake.dfs.fabric.microsoft.com/base_lakehouse.Lakehouse/Tables/transcripts


def get_full_path(relative_path=None, **kargs):
    workspace = kargs.get("workspace", "metaspliceai")
    lakehouse_name = kargs.get("lakehouse_name", "metaspliceai_lakehouse")
    prefix = f"abfss://{workspace}@onelake.dfs.fabric.microsoft.com/{lakehouse_name}.Lakehouse/Files"

    if relative_path is None or relative_path == "":
        return prefix
    else: 
        if relative_path.startswith("Files/"):
            relative_path = relative_path[len("Files/"):]

        if relative_path.startswith("/"):
            relative_path = relative_path[1:]

        full_path = f"{prefix}/{relative_path}"
        return full_path


def get_abfs_path(relative_path=None, **kargs):
    return get_full_path(relative_path=relative_path, **kargs)


def get_api_path(relative_path=None, **kargs):
    prefix = kargs.get("prefix", lakehouse_api_path_prefix)

    if relative_path is None or relative_path == "":
        return prefix
    else:
        if relative_path.startswith("Files/"):
            relative_path = relative_path[len("Files/"):]

        if relative_path.startswith("/"):
            relative_path = relative_path[1:]

        full_path = f"{prefix}/{relative_path}"
        return full_path


def is_valid_abfs_path(path):
    """
    Checks if the given path is a valid ABFS path.

    Parameters:
    ----------
    path : str
        The path to check.

    Returns:
    -------
    bool
        True if the path is a valid ABFS path, False otherwise.
    """
    # abfs_pattern = re.compile(r"^abfss://[^/]+@[^/]+\.dfs\.fabric\.microsoft\.com/[^/]+\.Lakehouse/Files/.*$")
    abfs_pattern = re.compile(r"^abfss://[^/]+@[^/]+\.dfs\.fabric\.microsoft\.com/[^/]+\.Lakehouse/Files/?$")
    return bool(abfs_pattern.match(path))


def is_valid_api_path(path):
    """
    Checks if the given path is a valid API path.

    Parameters:
    ----------
    path : str
        The path to check.

    Returns:
    -------
    bool
        True if the path is a valid API path, False otherwise.
    """
    return path.rstrip('/').startswith("/lakehouse/default/Files")


def normalize_path(file_path, target_format='abfs', **kargs):
    """
    Normalizes the file path to the target format (ABFS or API).

    Parameters:
    ----------
    file_path : str
        The file path to normalize.
    target_format : str, default 'abfs'
        The target format for the file path ('abfs' or 'api').
    **kargs : dict
        Additional keyword arguments.

    Returns:
    -------
    str
        The normalized file path.
    """
    prefix = kargs.get("prefix", lakehouse_api_path_prefix)  # "/lakehouse/default/Files/"

    # Check if the path is already in the target format
    if target_format.lower() in ['abfs', 'abfss', ]:
        if file_path.startswith("abfss://"):
            return file_path
        else:

            # Convert API path to ABFS path
            relative_path = file_path.replace(prefix, "")  # prefix="/lakehouse/default/Files/"
            return get_abfs_path(relative_path, **kargs)
    elif target_format.lower() == 'api':
        if file_path.startswith(prefix):  # "/lakehouse/default/Files/"
            return file_path
        else:
            # Convert ABFS path to API path
            relative_path = file_path.split(".Lakehouse/Files/")[-1]
            return get_api_path(relative_path, **kargs)
    else:
        raise ValueError(f"Unsupported target format: {target_format}")


# def is_fabric_environment():
#     # Check for environment-specific markers like env variables or directories
#     return 'FABRIC_ENV' in os.environ

def get_lakehouse_path(lakehouse_name):
    return f"abfss://{lakehouse_name}@onelake.dfs.fabric.microsoft.com/{lakehouse_name}.Lakehouse"


def evaluate_fabric_environment(): 
    return 'AZURE_SERVICE' in os.environ and os.environ['AZURE_SERVICE'] == 'Microsoft.ProjectArcadia' or \
           'MMLSPARK_PLATFORM_INFO' in os.environ and os.environ['MMLSPARK_PLATFORM_INFO'] == 'synapse'


def is_fabric_environment():
    """
    Check if the current environment is a Fabric environment.

    Returns:
    bool: True if the environment is a Fabric environment, False otherwise.
    """
    return 'AZURE_SERVICE' in os.environ and os.environ['AZURE_SERVICE'] == 'Microsoft.ProjectArcadia' or \
           'MMLSPARK_PLATFORM_INFO' in os.environ and os.environ['MMLSPARK_PLATFORM_INFO'] == 'synapse'


####################################################################################################

lakehouse_config_path = get_full_path("config/config.ini")
# NOTE: Can't use the API path => Py4JJavaError: An error occurred while calling o8145.text.
# get_full_path("config/config.ini") if use_spark else get_api_path("config/config.ini")

def read_config_file_spark(spark, lakehouse_path):
    """
    Read the config file from the lakehouse.

    Parameters:
    lakehouse_path (str): The path to the config file in the lakehouse.

    Returns:
    str: The content of the config file as a single string.
    """
    # Read the config file from the lakehouse
    config_data = spark.read.text(lakehouse_path).collect()
    return "\n".join([row.value for row in config_data])