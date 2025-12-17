# encoding: utf-8

import sys, os
from pathlib import Path
from os.path import expanduser
import re

import configparser
from configparser import ConfigParser
import threading

# PySpark is optional - only needed for Fabric/Lakehouse environments
try:
    import pyspark
    PYSPARK_AVAILABLE = True
except ImportError:
    pyspark = None
    PYSPARK_AVAILABLE = False

from . import lakehouse_config
from .lakehouse_config import (
    evaluate_fabric_environment,
    get_full_path, 
    get_api_path, 
    lakehouse_config_path, 
    read_config_file_spark
)

from .utils_doc import (
    print_emphasized, 
    print_with_indent, 
    print_section_separator 
)


# Global variable to store the result of is_fabric_environment()
_is_fabric = None

def is_fabric_environment():
    global _is_fabric
    if _is_fabric is None:
        # Evaluate the environment only once
        _is_fabric = evaluate_fabric_environment()  # Replace with actual evaluation logic
    return _is_fabric


# Initialize a global variable to store the configuration
# _config_loaded = False
# _config_lock = threading.Lock()
# _config_values = {}


class CommentStripperConfigParser(configparser.ConfigParser):

    def read(self, filenames, encoding=None):
        if isinstance(filenames, (str, bytes, os.PathLike)):
            filenames = [filenames]
        read_ok = []
        for filename in filenames:
            try:
                if is_fabric_environment():
                    data = self.read_from_fabric(filename)
                else:
                    with open(filename, encoding=encoding) as fp:
                        data = fp.read()

                # Remove inline comments
                data = re.sub(r'\s+#.*', '', data)
                self.read_string(data)

                # Log the filename for debugging
                print_emphasized(f"Parsed configuration from: {filename}")
                read_ok.append(filename)
            except Exception as e:
                print(f"Error reading configuration from {filename}: {e}")
                continue
        return read_ok

    def read_from_fabric(self, filename):
        from pyspark.sql import SparkSession
        try:
            spark = SparkSession.builder.getOrCreate()
            print(f"Attempting to read configuration file from: {filename}")
            file_data = spark.read.text(filename).collect()
            print(f"Successfully read configuration file from: {filename}")
            return "\n".join([row.value for row in file_data])
        except Exception as e:
            print(f"Error reading file from Fabric using Spark: {e}")
            raise

    def read_string(self, string):
        # Remove inline comments
        data = re.sub(r'\s+#.*', '', string)
        super().read_string(data)


def get_config_path():
    if is_fabric_environment():
        return lakehouse_config_path
        # NOTE: Can't use the API path => Py4JJavaError: An error occurred while calling o8145.text.
    else:
        return os.path.join(os.path.dirname(__file__), 'config.ini')
        # return os.path.join(os.getcwd(), 'config.ini')  # Default path is the current working directory


################################### #####


class Config:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Config, cls).__new__(cls)
                    cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        self.config = CommentStripperConfigParser()   # configparser.ConfigParser()
        try:
            config_path = get_config_path()
            self.config.read(config_path)
            # Optionally print the sections for debugging
            # print(f"Config sections: {self.config.sections()}")
        except Exception as e:
            # Handle exceptions or set default values
            print(f"Error reading config file: {e}")
            self.config = None

    def get(self, section, option, fallback=None):
        if self.config is not None:
            return self.config.get(section, option, fallback=fallback)
        else:
            return fallback


# Initialize the singleton instance on the driver node
config = Config()

# Test 
# source = config.get('DataSource', 'source')
# version = annotation_version = config.get('DataSource', 'version')
# print(f"[info] Source: {source}, Version: {version} for DataSource")

################################### #####


class Txdb(object): 
    genes = "genes"  # txdb.genes
    gn_ann = "gn_ann"  # gene annotation, txdb.gn_ann
    pt_ann = "pt_ann"  # protein annotation, txdb.pt_ann
    transcripts = "transcripts" 
    exons = "exons"
    tx_ann = "tx_ann"
    sequences = 'hg38.p14.2bit'
    
    tpm = "consensus_tpm"  # or "tpm"
    # NOTE: If you are using the consensus_tx_id column, use this column for the TPM
    #  'tpm': TPM calculated during stringtie
    #  'consensus_tpm': The summed TPM for when the same consensus_tx_id, reference, and sample has 2 TPMS. 
    
    spliceprep_metadata = "samples"  # "samples", "spliceprep_metadata"
    col_sample_id = 'sample_id'
    col_sample_name = 'sample_id'  # 'sample_name' doesn't exist in the metadata table anymore 


class DataSource: 
    source = config.get('DataSource', 'source', fallback='synapse')
    version = annotation_version = config.get('DataSource', 'version', fallback='hg38.p14.2bit')  

    matrix_version = tpm_matrix_version = config.get('DataSource', 'matrix_version', fallback='gtex-nmd_eff-trainset.tpm')  
    data_prefix = config.get('DataSource', 'data_prefix', fallback='/mnt/nfs1/splice-mediator')

    # data_dir = os.path.join(data_prefix, f"{source}/{version}")
    # NOTE: Defining data_dir like this (as a class attribute) will not reflect changes in data_prefix, source, or version 
    # after the class is defined. This is because class attributes in Python are evaluated when the class is defined,
    # not when an attribute is updated. Therefore, if data_prefix, source, or version are updated after 
    # the class is defined, data_dir will still use the old values. To have data_dir update when 
    # data_prefix, source, or version are changed, consider using a class method or property instead. 

    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


class EnsemblSource(DataSource):
    source = config.get('EnsemblSource', 'source', fallback='ensembl')
    version = annotation_version = config.get('EnsemblSource', 'version', fallback='GRCh38.106')
    data_prefix = config.get('EnsemblSource', 'data_prefix', fallback='/mnt/nfs1/splice-mediator')
    
    # data_dir = os.path.join(data_prefix, f"{source}/{version}")

    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


class SynapseSource(DataSource):
    source = config.get('SynapseSource', 'source', fallback='synapse')
    version = annotation_version = config.get('SynapseSource', 'version', fallback='hg38.p14.2bit')
    matrix_version = config.get('SynapseSource', 'matrix_version', fallback='gtex-nmd_eff-trainset.tpm')
    data_prefix = config.get('SynapseSource', 'data_prefix', fallback='/mnt/nfs1/splice-mediator')

    # data_dir = os.path.join(data_prefix, f"{source}/{version}")
    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


class FabricSource(DataSource):
    source = config.get('FabricSource', 'source', fallback='synapse')
    version = annotation_version = config.get('FabricSource', 'version', fallback='hg38.p14.2bit')
    matrix_version = config.get('FabricSource', 'matrix_version', fallback='gtex-nmd_eff-trainset.tpm')
    data_prefix = config.get('FabricSource', 'data_prefix', fallback='/mnt/nfs1/splice-mediator')

    # data_dir = os.path.join(data_prefix, f"{source}/{version}")
    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


class LakehouseSource(DataSource): 
    source = config.get('LakehouseSource', 'source', fallback='synapse')
    version = annotation_version = config.get('LakehouseSource', 'version', fallback='hg38.p14.2bit')
    matrix_version = config.get('LakehouseSource', 'matrix_version', fallback='gtex-nmd_eff-trainset.tpm')
    data_prefix = config.get('LakehouseSource', 'data_prefix', fallback='/lakehouse/default/Files')

    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


class SparkLakehouseSource(DataSource):
    source = config.get('SparkLakehouseSource', 'source', fallback='synapse')
    version = annotation_version = config.get('SparkLakehouseSource', 'version', fallback='hg38.p14.2bit')
    matrix_version = config.get('SparkLakehouseSource', 'matrix_version', fallback='gtex-nmd_eff-trainset.tpm')
    data_prefix = \
        config.get('SparkLakehouseSource', 'data_prefix', 
                    fallback='abfss://splicemediator@onelake.dfs.fabric.microsoft.com/splicemediator_lakehouse.Lakehouse/Files')

    @classmethod
    def data_dir(cls):
        return os.path.join(cls.data_prefix, f"{cls.source}/{cls.version}")


################################### #####

class SpliceIO(object):  # SpliceIO is a singleton class
    _instance = None  # This class-level attribute will hold the singleton instance.

    def __new__(cls, *args, **kwargs):
        """
        The __new__ method is overridden to ensure that only one instance of the class is created.
        If _instance is None, a new instance is created using super(SpliceIO, cls).__new__(cls).
        Otherwise, the existing instance is returned.
        """
        if cls._instance is None:
            cls._instance = super(SpliceIO, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, data_source='fabric'):
        """
        Initialize the SpliceIO instance with the data source attributes.

        Memo
        ----
        * The __init__ method checks if the instance has already been initialized using 
          the _initialized attribute. If _initialized is not set, the instance variables are computed 
          and _initialized is set to True. This ensures that the __init__ method is only executed once.
        """
        if not hasattr(self, '_initialized'):
            self.source = self.get_data_attribute(data_source, 'source')
            self.version = self.get_data_attribute(data_source, 'version')
            self.proj_dir = self.get_data_attribute(data_source, 'data_prefix')
            self.data_dir = self.get_data_attribute(data_source, 'data_dir')

            # SQL server configurations
            self.sql_service_type = "fabric-base_lakehouse"
            self.sql_engine_type = "pyodbc"

            # Blob storage configurations
            self.storage_prefix = 'SPLICEPREP:STORAGEHOT'  # old system: 'SPLICEPREP:STORAGELEGACY'
            self.container = 'misc-resources'

            # Set the _initialized attribute to True
            self._initialized = True

    @property
    def data_prefix(self): 
        return self.proj_dir

    @staticmethod
    def get_data_attribute(data_source, attribute):
        
        if data_source is None: 
            data_source = 'fabric'

        source = '?'
        if is_fabric_environment() or data_source.find('lakehouse') > 0: 
            if lakehouse_config.use_spark: 
                source = SparkLakehouseSource
            else:
                source = LakehouseSource
        else: 
            if data_source.lower().startswith('fabric'):  
                source = FabricSource
            elif data_source.lower().startswith(('synapse', 'sequence')):  # Synapse
                source = SynapseSource
            elif data_source.lower().startswith('ensem'):
                source = EnsemblSource
            else: 
                source = DataSource  # local
        
        # If the attribute is 'data_dir', call it as a method
        if attribute == 'data_dir':
            return source.data_dir()  # Call the class method to get the data directory ...
            # ... because this inferred, not defined in config.ini
        else:
            return getattr(source, attribute)

    @staticmethod
    def compute_proj_dir(data_source='fabric', proj_name=None):
        proj_dir = SpliceIO.get_data_attribute(data_source, 'data_prefix')
        if proj_name:
            return os.path.join(proj_dir, proj_name)
        return proj_dir
    
    @staticmethod
    def compute_data_dir(data_source='fabric'):
        return SpliceIO.get_data_attribute(data_source, 'data_dir')

    @staticmethod
    def get_proj_dir(data_source=None):
        if data_source is None:
            if is_fabric_environment() or data_source.find('lakehouse') > 0: 
                data_source = lakehouse_config.lakehouse_name  # 'splicemediator_lakehouse'
        else: 
            data_source = 'fabric'
        return SpliceIO.get_data_attribute(data_source, 'data_prefix')

    @staticmethod 
    def get_data_dir(data_source='fabric'): 
        return SpliceIO.get_data_attribute(data_source, 'data_dir')


def create_spliceio_class(name):
    """
    Factory function to create a class with a given name. The created class is used to handle data IO operations for
    SpliceX packages, where X is a specific instance of the package such as Mediator.  

    Parameters:
    name (str): The name of the class to be created.

    Returns:
    class: A class with the given name and predefined methods and variables.

    The created class has the following methods and variables:

    Variables:
    source: The data source attribute.
    version: The version attribute of the data source.
    proj_dir: The project directory attribute of the data source.
    data_dir: The data directory attribute of the data source.
    sql_service_type: The type of SQL service. Default is "fabric-base_lakehouse".
    sql_engine_type: The type of SQL engine. Default is "pyodbc".

    Methods:
    data_prefix: Property that returns the project directory.
    get_data_attribute(data_source, attribute): Static method to get a specific attribute from a data source.
    get_proj_dir(data_source='fabric'): Static method to get the project directory from a data source.
    get_data_dir(data_source='fabric'): Static method to get the data directory from a data source.
    """
    class _SpliceIO(object): 

        def __init__(self, data_source='fabric'): 
            self.source = self.get_data_attribute(data_source, 'source')
            self.version = self.get_data_attribute(data_source, 'version')
            self.proj_dir = self.get_data_attribute(data_source, 'data_prefix')
            self.data_dir = self.get_data_attribute(data_source, 'data_dir')

            # SQL server configurations
            self.sql_service_type = "fabric-base_lakehouse"
            self.sql_engine_type="pyodbc"

            # Blob storage configurations
            self.storage_prefix = 'SPLICEPREP:STORAGEHOT' # old system: 'SPLICEPREP:STORAGELEGACY'
            self.container = 'misc-resources'

        @property
        def data_prefix(self): 
            return self.proj_dir

        @staticmethod
        def get_data_attribute(data_source, attribute):
            source = None

            if data_source.lower().find("lakehouse") > 0: 
                if lakehouse_config.use_spark: 
                    source = SparkLakehouseSource
                else:
                    source = LakehouseSource
            if data_source.lower().startswith('fabric'):  
                source = FabricSource
            elif data_source.lower().startswith(('synapse', 'sequence')):  # Synapse
                source = SynapseSource
            elif data_source.lower().startswith('ensem'):
                source = EnsemblSource
            else: 
                source = DataSource  # local
            
            if attribute == 'data_dir':
                return source.data_dir()
            else:
                return getattr(source, attribute)

        @staticmethod
        def get_proj_dir(data_source=None):
            if data_source is None:
                if is_fabric_environment(): 
                    data_source = lakehouse_config.lakehouse_name
                else: 
                    data_source = 'fabric'
            return _SpliceIO.get_data_attribute(data_source, 'data_prefix')

        @staticmethod 
        def get_data_dir(data_source=None): 
            if data_source is None:
                if is_fabric_environment(): 
                    data_source = lakehouse_config.lakehouse_name
                else: 
                    data_source = 'fabric'

            return _SpliceIO.get_data_attribute(data_source, 'data_dir')

    _SpliceIO.__name__ = name
    return _SpliceIO


################################### #####
splice_io = None

def initialize_splice_io():
    """
    Initialize the splice_io singleton based on the environment.
    """
    global splice_io

    if is_fabric_environment(): 
        from pyspark.sql import SparkSession

        lakehouse_name = lakehouse_config.lakehouse_name  # "splicemediator_lakehouse"

        spark = SparkSession.builder.getOrCreate()
        spark.catalog.setCurrentDatabase(lakehouse_name)
        # Confirm the current lakehouse is set
        current_db = spark.catalog.currentDatabase()
        print(f"[info] Current database (lakehouse): {current_db}")

        # Accessing the parsed configuration
        print('Data Source: ', config.config['LakehouseSource']['source'])  # Output: splicemediator_lakehouse
        print('Version: ', config.config['LakehouseSource']['version'])  # Output: hg38.p14.2bit
        print('TPM Version: ', config.config['LakehouseSource']['matrix_version'])  # Output: gtex-nmd_eff-trainset.tpm
        print('Data Prefix: ', config.config['LakehouseSource']['data_prefix'])
        # To use Spark: abfss://splicemediator@onelake.dfs.fabric.microsoft.com/splicemediator_lakehouse.Lakehouse
        # To use API path: /lakehouse/default/Files/

        # Instantiate SpliceIO with data_source set to "splicemediator_lakehouse"
        splice_io = SpliceIO(data_source=lakehouse_name)
    else:
        # Instantiate SpliceIO with data_source set to "fabric" by default
        splice_io = SpliceIO(data_source='fabric')


# Ensure splice_io is initialized
print_emphasized("[info] Initializing SpliceIO singleton ...")
initialize_splice_io()

################################### #####
# Convenience wrappers for SpliceMediator method calls
################################### #####

def get_local_dir(**kargs):
    # Use the static method to compute the project directory for the 'local' data source
    proj_name = kargs.get("proj_name", None)
    return SpliceIO.compute_proj_dir(data_source='local', proj_name=proj_name)


def get_fabric_dir(**kargs):
    proj_name = kargs.get("proj_name", None)
    return SpliceIO.compute_proj_dir(data_source='fabric', proj_name=proj_name)


def get_ensembl_dir(**kargs):
    proj_name = kargs.get("proj_name", None)
    return SpliceIO.compute_proj_dir(data_source='ensembl', proj_name=proj_name)


def get_source(): 
    # Use the already created singleton instance
    if splice_io is None: 
        initialize_splice_io()
    return splice_io.source

def get_version(): 
    # Use the already created singleton instance
    if splice_io is None: 
        initialize_splice_io()
    return splice_io.version

def get_prefix(): 
    # Use the already created singleton instance
    if splice_io is None: 
        initialize_splice_io()
    return splice_io.data_prefix
    
def get_lakehouse_dir(**kargs):
    # Use the static method to compute the project directory for the 'fabric' data source
    proj_name = kargs.get("proj_name", None)
    return SpliceIO.compute_proj_dir(data_source=lakehouse_config.lakehouse_name, proj_name=proj_name)


def get_proj_dir(source=None, **kargs):
    # curdir = os.getcwd()
    proj_name = kargs.get("proj_name", None)

    if splice_io is None: 
        initialize_splice_io()

    if source is None: 
        # Use the already created singleton instance
        io = splice_io

        if kargs.get('verbose', 0): 
            print(f'[test] source={io.source}, io={io.data_prefix}')

        if proj_name:
            return os.path.join(io.proj_dir, proj_name)
        return io.proj_dir
    else: 
        return SpliceIO.compute_proj_dir(data_source=source, proj_name=proj_name)


def get_connection_conifg(source=None):
    if splice_io is None: 
        initialize_splice_io()
    return splice_io.sql_service_type, splice_io.sql_engine_type


def get_blob_config():
    if splice_io is None: 
        initialize_splice_io()
    return (splice_io.storage_prefix, splice_io.container)
    

def get_data_dir(source=None, **kargs):
    if splice_io is None: 
        initialize_splice_io()

    if source is None: 
        return splice_io.data_dir
    else: 
        return SpliceIO.compute_data_dir(data_source=source)

def get_proj_env(source=None, **kargs): 
    import glob
    from pathlib import Path

    verbose = kargs.get('verbose', 1)
    # Getting the name of the directory where the this file is present.
    res = {}

    res['proj_dir'] = proj_dir = get_proj_dir(source=source, **kargs)
    res['data_dir'] = data_dir = get_data_dir(source=source, **kargs)

    if is_fabric_environment(): 
        res['home_dir'] = lakehouse_config.lakehouse_api_path_prefix
    else: 
        res['home_dir'] = home_dir = Path.home()
    
    # NOTE: When navigating a lakehouse on Microsoft's Fabric platform, the typical concept of a "current working directory" 
    # from a local file system (like you'd manage with os.getcwd() on a Linux or Windows system) 
    # doesn't exist in the same sense. Instead, you're working within a structured cloud storage system, 
    # where access to files is achieved using fully qualified paths or specific APIs designed for cloud data management.
    res['cur_dir'] = cur_dir = Path.cwd() 
    if verbose: 
        print(f"> Current directory:\n{Path.cwd()}\n") 

    # Getting the parent directory name where the current directory is present.
    res['parent_dir'] = parent_dir = os.path.dirname(cur_dir)
    if verbose: 
        print(f"> Parent directory:\n{parent_dir}\n")

    return res


def demo_search_dir(): 
    from pathlib import Path
    import fnmatch
    import glob

    path = "/home/bchiu/work/nmd/data/ensembl/GRCh38.106/sequence"
    root = "/home/bchiu/work/nmd"

    for f in glob.glob(os.path.join(root, '*/ensembl/*.106/seq*'), recursive=True):
        print(f)

    # matches = []
    # for path in Path(root).rglob('*.106/seq*'):
    #     r = path.root
    #     name = path.name 
    #     matches.append(os.path.join(r, name))
    # print(matches)

    # matches = []
    # for root, dirnames, filenames in os.walk(root):
    #     for dirname in fnmatch.filter(dirnames, 'seq*'):
    #         print(dirname) # dirname is the "leaf" of a path 
    #         matches.append(os.path.join(r, dirname))
    # print(matches)

    return

def demo_basic_config(): 

    print_emphasized("[info] Default project and data directories")
    print(f"> data dir:\n{get_data_dir()}\n")
    print(f"> proj dir:\n{get_proj_dir()}\n")
    print(f"> source: {get_source()}\n")
    print(f"> version: {get_version()}\n")

    print_section_separator()
    
    print_emphasized("[info] Using Factory Function to Create SpliceIO Class")
    SpliceIO = create_spliceio_class("MetaSpliceAIIO")

    for source in ['local', 'fabric', 'ensembl', ]:   
        print_emphasized(f"> source: {source}")
        data_dir = get_data_dir(source)
        print(f"> data dir:\n{data_dir}\n")
        proj_dir = get_proj_dir(source)
        print(f"> proj dir:\n{proj_dir}\n")
        print('-' * 50) 

        data_dir = SpliceIO.get_data_dir(source)
        print(f"> SpliceIO: data dir:\n{data_dir}\n")
        proj_dir = SpliceIO.get_proj_dir(source)
        print(f"> SpliceIO: proj_dir:\n{proj_dir}\n")
        print('-' * 50) 

        res = get_proj_env(source=source, verbose=1)
        print(res)
        print('-' * 50) 

    return 

def demo_user_specified_prefix(): 

    print("> Sometimes we want to customize the data_dir and proj_dir arbitrarily without following local rules ...")
    print("... useful for working with another system such as splice-prep")

    data_prefix = "/mnt/nfs1/splice-mapper"
    DataSource.data_prefix = data_prefix  

    print_emphasized("[info] Original data_dir and proj_dir")
    data_dir = get_data_dir()
    print(f"> data dir:\n{data_dir}\n")
    proj_dir = get_proj_dir()
    print(f"> proj dir:\n{proj_dir}\n")

    print_emphasized("[info] User-specified data_dir and proj_dir")
    FabricSource.data_prefix = data_prefix
    data_dir = get_data_dir('fabric')
    print(f"[FabricSource] data dir:\n{data_dir}\n")
    proj_dir = get_proj_dir(source='fabric')
    print(f"[FabricSource] proj dir:\n{proj_dir}\n")

    return 



def test(): 

    # Default project and data directories
    demo_basic_config()

    # User-defined project and data directories via specifying respective prefixes
    demo_user_specified_prefix()

    # SpliceprepIO is essentially an application of demo_user_specified_prefix()
    # demo_spliceprep_dir()

    # Path utilities 
    # demo_search_dir()

    print(config.config['LakehouseSource']['source'])  # Output: splicemediator_lakehouse
    print(config.config['LakehouseSource']['version'])  # Output: hg38.p14.2bit
    print(config.config['LakehouseSource']['matrix_version'])  # Output: gtex-nmd_eff-trainset.tpm
    print(config.config['LakehouseSource']['data_prefix'])  

    return

if __name__ == "__main__": 
    test() 
