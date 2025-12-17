# system/config.py

import os
from pathlib import Path


def _detect_project_root() -> str:
    """
    Detect the project root directory dynamically.
    
    Priority order:
    1. META_SPLICEAI_ROOT environment variable (explicit override)
    2. Walk up from this file to find project markers
    3. Check common locations (/workspace/meta-spliceai for RunPods)
    4. Fall back to ~/work/meta-spliceai (legacy default)
    
    Returns
    -------
    str
        Absolute path to the project root
    """
    # 1. Environment variable override (highest priority)
    env_root = os.environ.get('META_SPLICEAI_ROOT')
    if env_root and os.path.isdir(env_root):
        return env_root
    
    # 2. Walk up from this file to find project markers
    root_markers = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt']
    current = Path(__file__).resolve().parent  # meta_spliceai/system/
    
    # Walk up to find project root (max 5 levels)
    for _ in range(5):
        for marker in root_markers:
            if (current / marker).exists():
                return str(current)
        if current.parent == current:
            break
        current = current.parent
    
    # 3. Check common locations
    common_locations = [
        Path('/workspace/meta-spliceai'),  # RunPods
        Path.home() / 'work' / 'meta-spliceai',  # Local development
        Path('/home') / os.environ.get('USER', 'root') / 'work' / 'meta-spliceai',  # Linux
    ]
    
    for loc in common_locations:
        if loc.exists() and (loc / 'meta_spliceai').exists():
            return str(loc)
    
    # 4. Fall back to legacy default
    return str(Path.home() / 'work' / 'meta-spliceai')


class Config:
    HOME_DIR = os.path.expanduser("~")
    PROJ_DIRNAME = 'meta-spliceai'
    
    # Dynamic project root detection (works on local, RunPods, Azure, etc.)
    PROJ_DIR = _detect_project_root()
    
    # Data directory is always under the project root
    INPUT_DIR = DATA_DIR = os.path.join(PROJ_DIR, "data")
    
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
        the project root. If no markers are found, it checks common locations and falls 
        back to a reasonable default.
        
    Priority:
        1. META_SPLICEAI_ROOT environment variable
        2. Walk up from current_path to find project markers  
        3. Check common locations (/workspace/meta-spliceai for RunPods)
        4. Fall back to ~/work/meta-spliceai
    """
    # 1. Environment variable override (highest priority)
    env_root = os.environ.get('META_SPLICEAI_ROOT')
    if env_root and os.path.isdir(env_root):
        return env_root
    
    # 2. Walk up from current_path to find project markers
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
    
    # 3. Check common locations (useful when called from unexpected directories)
    common_locations = [
        '/workspace/meta-spliceai',  # RunPods
        os.path.expanduser('~/work/meta-spliceai'),  # Local development
        '/home/{}/work/meta-spliceai'.format(os.environ.get('USER', 'root')),  # Linux
    ]
    
    for loc in common_locations:
        if os.path.isdir(loc) and os.path.isdir(os.path.join(loc, 'meta_spliceai')):
            return loc
    
    # 4. Fall back to legacy default
    return os.path.expanduser('~/work/meta-spliceai')