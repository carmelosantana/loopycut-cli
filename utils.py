"""
LoopyCut - Utility Functions

This module contains utility functions for common operations like file validation,
logging, and configuration management.
"""

import os
import sys
import json
import re
from typing import Optional, Any, Dict, Union
from pathlib import Path


def parse_time_string(time_str: Union[str, float, int]) -> float:
    """
    Parse a time string or number into seconds.
    
    Supports formats:
    - Float/int: 14.5 (seconds)
    - MM:SS: "14:30" (14 minutes 30 seconds)
    - HH:MM:SS: "01:14:30" (1 hour 14 minutes 30 seconds)
    - SS: "90" (90 seconds)
    
    Args:
        time_str: Time string, float, or int to parse
        
    Returns:
        Time in seconds as float
        
    Raises:
        ValueError: If time format is invalid
    """
    if isinstance(time_str, (int, float)):
        return float(time_str)
    
    if not isinstance(time_str, str):
        raise ValueError(f"Invalid time format: {time_str}")
    
    time_str = time_str.strip()
    
    # Check for HH:MM:SS format
    if re.match(r'^\d{1,2}:\d{2}:\d{2}(\.\d+)?$', time_str):
        parts = time_str.split(':')
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    
    # Check for MM:SS format
    elif re.match(r'^\d{1,2}:\d{2}(\.\d+)?$', time_str):
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    
    # Check for plain seconds (string)
    elif re.match(r'^\d+(\.\d+)?$', time_str):
        return float(time_str)
    
    else:
        raise ValueError(f"Invalid time format: {time_str}. Use HH:MM:SS, MM:SS, or seconds")


def validate_video_file(file_path: str) -> bool:
    """
    Validate that a file exists and appears to be a video file.
    
    Args:
        file_path: Path to the video file
        
    Returns:
        True if file exists and has a video extension, False otherwise
    """
    if not os.path.exists(file_path):
        return False
    
    # Check file extension
    video_extensions = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v'}
    file_ext = Path(file_path).suffix.lower()
    
    return file_ext in video_extensions


def ensure_output_directory(output_path: str) -> bool:
    """
    Ensure the output directory exists, creating it if necessary.
    
    Args:
        output_path: Path to the output file
        
    Returns:
        True if directory exists or was created successfully, False otherwise
    """
    try:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return False


def format_time(seconds: float) -> str:
    """
    Format seconds into a human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1:23.45")
    """
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    return f"{minutes}:{remaining_seconds:06.2f}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to a readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def parse_time_string(time_str: str) -> float:
    """
    Parse a time string into seconds.
    
    Supports formats:
    - "123.45" (seconds)
    - "2:03.45" (minutes:seconds)
    - "1:23:45.67" (hours:minutes:seconds)
    
    Args:
        time_str: Time string to parse
        
    Returns:
        Time in seconds
        
    Raises:
        ValueError: If time string format is invalid
    """
    try:
        # Try parsing as plain seconds first
        return float(time_str)
    except ValueError:
        pass
    
    # Parse as time format
    parts = time_str.split(':')
    
    if len(parts) == 2:  # MM:SS.ss
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:  # HH:MM:SS.ss
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid time format: {time_str}")


def save_loop_metadata(output_path: str, loop_info: Dict[str, Any]) -> bool:
    """
    Save loop metadata to a JSON file alongside the output video.
    
    Args:
        output_path: Path to the output video file
        loop_info: Dictionary containing loop information
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        # Create metadata file path
        metadata_path = Path(output_path).with_suffix('.json')
        
        # Add timestamp and version info
        import time
        metadata = {
            'loopycut_version': '1.0.0',
            'created_at': time.time(),
            'created_at_readable': time.strftime('%Y-%m-%d %H:%M:%S'),
            'loop_info': loop_info
        }
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Loop metadata saved: {metadata_path}")
        return True
        
    except Exception as e:
        print(f"Error saving metadata: {e}")
        return False


def load_loop_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    """
    Load loop metadata from JSON file if it exists.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Metadata dictionary or None if not found
    """
    try:
        metadata_path = Path(video_path).with_suffix('.json')
        
        if not metadata_path.exists():
            return None
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return None


def get_safe_filename(filename: str) -> str:
    """
    Convert a filename to a safe version by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Characters to remove or replace
    invalid_chars = '<>:"/\\|?*'
    safe_filename = filename
    
    for char in invalid_chars:
        safe_filename = safe_filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    safe_filename = safe_filename.strip('. ')
    
    # Ensure filename is not empty
    if not safe_filename:
        safe_filename = "output"
    
    return safe_filename


def check_dependencies() -> Dict[str, bool]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Dictionary mapping dependency names to availability status
    """
    dependencies = {}
    
    # Check OpenCV
    try:
        import cv2
        dependencies['opencv'] = True
    except ImportError:
        dependencies['opencv'] = False
    
    # Check FFmpeg Python
    try:
        import ffmpeg
        dependencies['ffmpeg-python'] = True
    except ImportError:
        dependencies['ffmpeg-python'] = False
    
    # Check NumPy
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        dependencies['numpy'] = False
    
    # Check Click
    try:
        import click
        dependencies['click'] = True
    except ImportError:
        dependencies['click'] = False
    
    # Check tqdm
    try:
        import tqdm
        dependencies['tqdm'] = True
    except ImportError:
        dependencies['tqdm'] = False
    
    return dependencies


def print_system_info():
    """Print system and dependency information."""
    print("LoopyCut System Information")
    print("=" * 40)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    print("Dependency Status:")
    print("-" * 20)
    deps = check_dependencies()
    
    for dep, available in deps.items():
        status = "✓ Available" if available else "✗ Missing"
        print(f"{dep:15} {status}")
    
    if not all(deps.values()):
        print("\nSome dependencies are missing. Install with:")
        print("pip install -r requirements.txt")


def calculate_file_size(file_path: str) -> Optional[int]:
    """
    Get the size of a file in bytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in bytes or None if file doesn't exist
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
