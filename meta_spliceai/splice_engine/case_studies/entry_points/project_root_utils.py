"""
Project Root Detection Utilities

This module provides utilities for detecting the project root directory
in a systematic way, avoiding hardcoded directory level counting.
"""

import os
import sys
from pathlib import Path
from typing import Optional


def find_project_root(current_path: str = './') -> str:
    """
    Find the project root directory by looking for common project markers.
    
    This function searches upward from the given path until it finds
    common project markers like .git, setup.py, pyproject.toml, etc.
    
    Parameters
    ----------
    current_path : str
        The path to start searching from (defaults to current directory)
        
    Returns
    -------
    str
        The absolute path to the project root directory
        
    Notes
    -----
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


def setup_project_imports(script_path: str) -> None:
    """
    Set up project imports by adding the project root to sys.path.
    
    This is a convenience function that combines project root detection
    with sys.path modification for easy use in entry point scripts.
    
    Parameters
    ----------
    script_path : str
        Path to the current script (usually __file__)
        
    Examples
    --------
    >>> # In an entry point script
    >>> setup_project_imports(__file__)
    >>> from meta_spliceai.some_module import SomeClass
    """
    project_root = find_project_root(script_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def get_project_root(script_path: str) -> Path:
    """
    Get the project root as a Path object.
    
    Parameters
    ----------
    script_path : str
        Path to the current script (usually __file__)
        
    Returns
    -------
    Path
        Project root directory as a Path object
    """
    return Path(find_project_root(script_path))


def validate_project_structure(project_root: Optional[str] = None) -> bool:
    """
    Validate that the project structure looks correct.
    
    Parameters
    ----------
    project_root : str, optional
        Project root path to validate (auto-detected if None)
        
    Returns
    -------
    bool
        True if project structure appears valid
    """
    if project_root is None:
        project_root = find_project_root()
    
    # Check for key project directories
    required_dirs = ['meta_spliceai', 'meta_spliceai/splice_engine']
    required_files = ['meta_spliceai/__init__.py']
    
    for dir_path in required_dirs:
        if not os.path.exists(os.path.join(project_root, dir_path)):
            return False
    
    for file_path in required_files:
        if not os.path.exists(os.path.join(project_root, file_path)):
            return False
    
    return True


# Convenience function for common use case
def setup_entry_point_imports(script_path: str) -> None:
    """
    Alias for setup_project_imports for better semantic clarity in entry points.
    
    Parameters
    ----------
    script_path : str
        Path to the current script (usually __file__)
    """
    setup_project_imports(script_path)
