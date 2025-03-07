"""
Utility functions for the Whisper GUI application
"""

import os

def find_dll_file(directory, possible_names):
    """
    Find a DLL file from a list of possible names in the given directory.
    
    Args:
        directory (str): Directory to search in
        possible_names (list): List of possible filenames
        
    Returns:
        str or None: Path to the found DLL file, or None if not found
    """
    for name in possible_names:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None

def check_required_dlls(directory, required_dlls):
    """
    Check if required DLL files exist in the directory.
    
    Args:
        directory (str): Directory to check
        required_dlls (list): List of required DLL filenames
        
    Returns:
        tuple: (missing_dlls, status) where missing_dlls is a list of missing DLL files
               and status is True if all required DLLs are found, False otherwise
    """
    missing_dlls = []
    
    for dll in required_dlls:
        if not os.path.exists(os.path.join(directory, dll)):
            missing_dlls.append(dll)
    
    return missing_dlls, len(missing_dlls) == 0

def get_file_size_str(file_path):
    """
    Get file size in human-readable format.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: File size in human-readable format (e.g. "15.2 MB")
    """
    size_bytes = os.path.getsize(file_path)
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024