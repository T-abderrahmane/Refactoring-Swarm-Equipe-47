"""
Pytest configuration and fixtures for integration tests.
"""

import os
import sys
import tempfile
import shutil
import pytest
from pathlib import Path

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="function")
def temp_workspace():
    """Create a temporary workspace for tests."""
    temp_dir = tempfile.mkdtemp(prefix="refactoring_test_")
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def sample_buggy_code(temp_workspace, test_data_dir):
    """Create sample buggy code in temporary workspace."""
    target_dir = os.path.join(temp_workspace, "target")
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy sample buggy code
    buggy_code_dir = test_data_dir / "buggy_code"
    for file_path in buggy_code_dir.glob("*.py"):
        shutil.copy2(file_path, target_dir)
    
    return target_dir


@pytest.fixture(scope="function")
def mock_llm_response():
    """Provide mock LLM response for testing."""
    return """
def calculate_area(length: float, width: float) -> float:
    \"\"\"Calculate the area of a rectangle.
    
    Args:
        length: Length of the rectangle
        width: Width of the rectangle
        
    Returns:
        Area of the rectangle
    \"\"\"
    result = length * width
    return result
"""


@pytest.fixture(autouse=True)
def cleanup_logs():
    """Clean up log files after each test."""
    yield
    # Clean up experiment logs
    log_file = "logs/experiment_data.json"
    if os.path.exists(log_file):
        try:
            os.remove(log_file)
        except:
            pass  # Ignore cleanup errors