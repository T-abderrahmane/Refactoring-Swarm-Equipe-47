"""
Refactoring tools for reading, writing, and testing Python files.
These tools are used by the agents to interact with the codebase.
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.tools import tool


# Sandbox directory for safe file operations
SANDBOX_DIR = Path("./sandbox")


def ensure_in_sandbox(file_path: str) -> Path:
    """
    Ensure file path is within the sandbox directory for security.
    
    Args:
        file_path: Relative or absolute file path
        
    Returns:
        Absolute Path within sandbox
        
    Raises:
        ValueError: If path is outside sandbox
    """
    # Convert to absolute path within sandbox
    if os.path.isabs(file_path):
        path = Path(file_path)
    else:
        path = SANDBOX_DIR / file_path
    
    # Resolve to absolute path
    abs_path = path.resolve()
    abs_sandbox = SANDBOX_DIR.resolve()
    
    # Ensure path is within sandbox
    try:
        abs_path.relative_to(abs_sandbox)
    except ValueError:
        raise ValueError(f"Path {file_path} is outside sandbox directory")
    
    return abs_path


@tool
def read_python_file(file_path: str) -> str:
    """
    Read the contents of a Python file from the sandbox directory.
    
    Args:
        file_path: Path to the Python file (relative to sandbox)
        
    Returns:
        The contents of the file as a string
    """
    try:
        safe_path = ensure_in_sandbox(file_path)
        
        if not safe_path.exists():
            return f"Error: File {file_path} does not exist"
        
        with open(safe_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"Read file: {file_path}")
        return content
    
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_python_file(file_path: str, content: str) -> str:
    """
    Write content to a Python file in the sandbox directory.
    Creates parent directories if they don't exist.
    
    Args:
        file_path: Path to the Python file (relative to sandbox)
        content: The content to write to the file
        
    Returns:
        Success or error message
    """
    try:
        safe_path = ensure_in_sandbox(file_path)
        
        # Create parent directories if they don't exist
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(safe_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Wrote file: {file_path}")
        return f"Successfully wrote to {file_path}"
    
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_python_files(directory: str = ".") -> str:
    """
    List all Python files in a directory within the sandbox.
    
    Args:
        directory: Directory path (relative to sandbox, default is root)
        
    Returns:
        List of Python files as a string
    """
    try:
        safe_dir = ensure_in_sandbox(directory)
        
        if not safe_dir.exists():
            return f"Error: Directory {directory} does not exist"
        
        if not safe_dir.is_dir():
            return f"Error: {directory} is not a directory"
        
        # Find all Python files recursively
        python_files = []
        for py_file in safe_dir.rglob("*.py"):
            # Get path relative to sandbox
            rel_path = py_file.relative_to(SANDBOX_DIR.resolve())
            python_files.append(str(rel_path))
        
        if not python_files:
            return "No Python files found in directory"
        
        print(f"Listed {len(python_files)} Python files in {directory}")
        return "\n".join(sorted(python_files))
    
    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool
def run_pylint(file_path: str) -> str:
    """
    Run pylint static analysis on a Python file.
    
    Args:
        file_path: Path to the Python file (relative to sandbox)
        
    Returns:
        Pylint output and score
    """
    try:
        safe_path = ensure_in_sandbox(file_path)
        
        if not safe_path.exists():
            return f"Error: File {file_path} does not exist"
        
        # Run pylint
        result = subprocess.run(
            ['pylint', str(safe_path), '--output-format=text'],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        output = result.stdout + result.stderr
        
        print(f"Run pylint on: {file_path}")
        return output if output else "Pylint completed with no output"
    
    except subprocess.TimeoutExpired:
        return "Error: Pylint execution timed out"
    except FileNotFoundError:
        return "Error: Pylint is not installed"
    except Exception as e:
        return f"Error running pylint: {str(e)}"


@tool
def run_pytest(test_path: str = ".") -> str:
    """
    Run pytest on a test file or directory.
    
    Args:
        test_path: Path to test file or directory (relative to sandbox)
        
    Returns:
        Pytest output showing test results
    """
    try:
        safe_path = ensure_in_sandbox(test_path)
        
        if not safe_path.exists():
            return f"Error: Path {test_path} does not exist"
        
        # Run pytest with verbose output
        result = subprocess.run(
            ['pytest', str(safe_path), '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(SANDBOX_DIR.resolve())
        )
        
        output = result.stdout + result.stderr
        
        # Determine if tests passed
        tests_passed = result.returncode == 0
        status = "PASSED" if tests_passed else "❌ FAILED"
        
        print(f"Run pytest on: {test_path} - {status}")
        return output if output else "Pytest completed with no output"
    
    except subprocess.TimeoutExpired:
        return "Error: Pytest execution timed out"
    except FileNotFoundError:
        return "Error: Pytest is not installed"
    except Exception as e:
        return f"Error running pytest: {str(e)}"


@tool
def get_file_info(file_path: str) -> str:
    """
    Get information about a file (size, lines, etc.).
    
    Args:
        file_path: Path to the file (relative to sandbox)
        
    Returns:
        File information as a string
    """
    try:
        safe_path = ensure_in_sandbox(file_path)
        
        if not safe_path.exists():
            return f"Error: File {file_path} does not exist"
        
        # Get file stats
        size = safe_path.stat().st_size
        
        # Count lines
        with open(safe_path, 'r', encoding='utf-8') as f:
            lines = len(f.readlines())
        
        info = f"File: {file_path}\n"
        info += f"Size: {size} bytes\n"
        info += f"Lines: {lines}\n"
        info += f"Exists: Yes"
        
        print(f"Got info for: {file_path}")
        return info
    
    except Exception as e:
        return f"Error getting file info: {str(e)}"


# Export all tools as a list for easy access
ALL_TOOLS = [
    read_python_file,
    write_python_file,
    list_python_files,
    run_pylint,
    run_pytest,
    get_file_info
]
