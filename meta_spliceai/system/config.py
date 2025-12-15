# system/config.py

import os

class Config:
    HOME_DIR = os.path.expanduser("~")
    PROJ_DIRNAME = 'meta-spliceai'
    PROJ_DIR = os.path.join(HOME_DIR, f"work/{PROJ_DIRNAME}")
    INPUT_DIR = DATA_DIR = os.path.join(HOME_DIR, f"work/{PROJ_DIRNAME}/data")
    FOUNDATION_MODEL = 'spliceai'  # Change to 'mmsplice' or other models as needed



MODEL_REGISTRY = {

    # External packages
    'spliceai': {
        'module_path': 'spliceai',                      # external package
        'class_name': 'SpliceAI',
        'utils_module': 'spliceai.utils'
    },

    # Internal packages
    # 'mmsplice': {
    #     'module_path': 'meta_spliceai.foundation_models.mmsplice.model',  # internal
    #     'class_name': 'MMSplice',
    #     'utils_module': 'meta_spliceai.foundation_models.mmsplice.utils'
    # },
    
    # Add additional models here
}


# Find project root directory and set up cache path near annotation files
def find_project_root(current_path='./'):
    """Find the project root directory by looking for common project markers.
    
    Args:
        current_path: The path to start searching from (defaults to current directory)
        
    Returns:
        The absolute path to the project root directory
        
    Notes:
        This function looks for common project markers (.git, setup.py, etc.) to identify
        the project root. If no markers are found, it falls back to a reasonable default.
    """
    # Convert to absolute path if not already
    path = os.path.abspath(current_path)
    
    # Project root markers in order of preference
    root_markers = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt', 'meta_spliceai']
    
    # Start from the directory and move up until we find project markers
    while True:
        # Check if we've reached the filesystem root
        if path == os.path.dirname(path):  # Works on both Unix and Windows
            break
            
        # Check for project markers
        for marker in root_markers:
            if os.path.exists(os.path.join(path, marker)):
                return path
                
        # Move up one directory
        path = os.path.dirname(path)
    
    # If we reach here, we couldn't find a project root
    # For backward compatibility, fall back to 5 directories up from original path
    fallback_path = current_path
    for _ in range(5):
        fallback_path = os.path.dirname(fallback_path)
    
    return fallback_path