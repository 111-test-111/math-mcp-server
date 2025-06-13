#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Utility Module: Provides file name processing, path construction, and file existence checking
"""

import os
import uuid
import datetime
from typing import Optional, Tuple


def get_output_path() -> str:
    """Get the path for saving output files, using environment variables first, then system temporary directory"""
    # First try using the OUTPUT_PATH environment variable
    output_path = os.environ.get("OUTPUT_PATH")
    if output_path and os.path.isdir(output_path):
        return output_path

    # Then try using the TMPFILE_PATH environment variable (for existing signal processing)
    tmp_path = os.environ.get("TMPFILE_PATH")
    if tmp_path and os.path.isdir(tmp_path):
        return tmp_path

    # Finally use the system temporary directory
    import tempfile

    return tempfile.gettempdir()


def ensure_file_directory(filepath: str) -> None:
    """Ensure the directory for the file exists, creating it if it doesn't"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def generate_unique_filename(
    prefix: str, extension: str, custom_name: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate a unique filename to avoid overwriting existing files

    Args:
        prefix: File name prefix
        extension: File extension (without dot)
        custom_name: User-provided custom file name (optional)

    Returns:
        Tuple[str, str]: (full file path, only filename)
    """
    output_path = get_output_path()

    # If a custom name is provided, use it as the base
    if custom_name:
        # Ensure the file name has the correct extension
        if not custom_name.lower().endswith(f".{extension.lower()}"):
            filename = f"{custom_name}.{extension}"
        else:
            filename = custom_name

        # Build the full path
        filepath = os.path.join(output_path, filename)
        if os.path.exists(filepath):
            name_part, ext_part = os.path.splitext(filename)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name_part}_{timestamp}{ext_part}"
            filepath = os.path.join(output_path, filename)

            # If it still exists (very rare case), add a random UUID
            if os.path.exists(filepath):
                unique_id = str(uuid.uuid4())[:8]
                filename = f"{name_part}_{timestamp}_{unique_id}{ext_part}"
                filepath = os.path.join(output_path, filename)
    else:
        # Use default naming rules
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        filename = f"{prefix}_{timestamp}_{random_id}.{extension}"
        filepath = os.path.join(output_path, filename)

        # Just in case there's still a conflict
        while os.path.exists(filepath):
            random_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}_{timestamp}_{random_id}.{extension}"
            filepath = os.path.join(output_path, filename)

    # Ensure the directory exists
    ensure_file_directory(filepath)

    return filepath, filename


def resolve_signal_file_path(signal_file: str) -> str:
    """
    Resolve signal file path, supporting both relative and absolute paths

    Args:
        signal_file: Signal file name or path

    Returns:
        str: Full signal file path
    """
    # If it's already an absolute path, return it
    if os.path.isabs(signal_file) or signal_file.startswith(get_output_path()):
        return signal_file

    # If it doesn't end with .json, add the extension
    if not signal_file.lower().endswith(".json"):
        signal_file = f"{signal_file}.json"

    # Join to the output directory
    return os.path.join(get_output_path(), signal_file)
