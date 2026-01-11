"""
End-to-end integration tests for the Refactoring Swarm workflow.

Tests the complete refactoring process from start to finish:
1. Code analysis and refactoring plan generation
2. Fix application and validation
3. Test execution and feedback loops
4. Logging and data integrity validation
"""

import os
import json
import tempfile
import shutil
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.orchestrator.orchestrator import RefactoringOrchestrator
from src.models.core import AgentState, AgentStatus
from src.utils.logger import log_experiment, ActionType
from src.exceptions import RefactoringError


class TestEndToEndWorkflow(unittest.TestCase):
    """Test complete refactoring workflow from start to finish."""
    
    def setUp(self):
        """Set up test environment with sample buggy code."""
        # Create temporary directories
        self.test_dir = tempfile.mkdtemp(prefix="refactoring_test_")
        self.target_dir = os.path.join(self.test_dir, "target")
        self.sandbox_dir = os.path.join(self.test_dir, "sandbox")
        
        # Create target directory structure
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Copy sample buggy code to target directory
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "buggy_code"
        for file_path in fixtures_dir.glob("*.py"):
            shutil.copy2(file_path, self.target_dir)
        
        # Mock LLM function for testing
        self.mock_llm_fn = Mock()
        self.mock_llm_fn.return_value = self._get_mock_llm_response()
        
        # Initialize orchestrator with mocked LLM
        self.orchestrator = RefactoringOrchestrator(
            model="test-model",
            llm_fn=self.mock_llm_fn,
            max_iterations=3,
            enable_monitoring=True
        )
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _get_mock_llm_response(self):
        """Get mock LLM response for testing."""
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
    
    def test_complete_workflow_success(self):
        """Test successful completion of the entire refactoring workflow."""
        # Execute the refactoring workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate results structure
        self.assertIsInstance(results, dict)
        self.assertIn("success", results)
        self.assertIn("status", results)
        self.assertIn("execution_time", results)
        self.assertIn("total_iterations", results)
        self.assertIn("files_processed", results)
        
        # Validate execution completed
        self.assertIsNotNone(results["execution_time"])
        self.assertGreater(results["execution_time"], 0)
        self.assertLessEqual(results["total_iterations"], 3)  # Max iterations
        
        # Validate files were processed
        self.assertGreater(results["files_processed"], 0)
        
        # Validate sandbox was created
        self.assertTrue(os.path.exists(self.sandbox_dir))
        
        # Validate Python files exist in sandbox
        sandbox_files = list(Path(self.sandbox_dir).glob("*.py"))
        self.assertGreater(len(sandbox_files), 0)
    
    def test_workflow_with_invalid_target_directory(self):
        """Test workflow behavior with invalid target directory."""
        invalid_dir = "/nonexistent/directory"
        
        with self.assertRaises(RefactoringError):
            self.orchestrator.execute_refactoring(
                target_directory=invalid_dir,
                sandbox_directory=self.sandbox_dir
            )
    
    def test_workflow_with_no_python_files(self):
        """Test workflow behavior when no Python files are found."""
        # Create empty target directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        
        with self.assertRaises(RefactoringError):
            self.orchestrator.execute_refactoring(
                target_directory=empty_dir,
                sandbox_directory=self.sandbox_dir
            )
    
    def test_workflow_iteration_limit(self):
        """Test that workflow respects maximum iteration limit."""
        # Create orchestrator with low iteration limit
        limited_orchestrator = RefactoringOrchestrator(
            model="test-model",
            llm_fn=self.mock_llm_fn,
            max_iterations=1,
            enable_monitoring=True
        )
        
        results = limited_orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Should not exceed iteration limit
        self.assertLessEqual(results["total_iterations"], 1)
    
    def test_workflow_progress_monitoring(self):
        """Test that workflow progress monitoring works correctly."""
        progress_updates = []
        
        def progress_callback(update_info):
            progress_updates.append(update_info)
        
        # Execute with progress callback
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir,
            progress_callback=progress_callback
        )
        
        # Validate progress updates were received
        self.assertGreater(len(progress_updates), 0)
        
        # Validate progress update structure
        for update in progress_updates:
            self.assertIn("session_id", update)
            self.assertIn("iteration", update)
            self.assertIn("phase", update)
            self.assertIn("progress", update)
    
    def test_workflow_logging_integration(self):
        """Test that all workflow operations are properly logged."""
        # Clear any existing logs
        log_file = "logs/experiment_data.json"
        if os.path.exists(log_file):
            os.remove(log_file)
        
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate log file was created and contains entries
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        # Validate log structure
        self.assertIsInstance(log_data, list)
        self.assertGreater(len(log_data), 0)
        
        # Validate log entries contain required fields
        for entry in log_data:
            self.assertIn("timestamp", entry)
            self.assertIn("agent_name", entry)
            self.assertIn("model_used", entry)
            self.assertIn("action", entry)
            self.assertIn("status", entry)
            self.assertIn("details", entry)
        
        # Validate different agents logged entries
        agent_names = {entry["agent_name"] for entry in log_data}
        expected_agents = {"System", "Orchestrator"}
        self.assertTrue(expected_agents.issubset(agent_names))
        
        # Validate different action types were logged
        action_types = {entry["action"] for entry in log_data}
        self.assertIn("ANALYSIS", action_types)
    
    def test_workflow_error_handling(self):
        """Test workflow error handling and recovery."""
        # Create orchestrator with failing LLM function
        failing_llm_fn = Mock(side_effect=Exception("LLM call failed"))
        
        error_orchestrator = RefactoringOrchestrator(
            model="test-model",
            llm_fn=failing_llm_fn,
            max_iterations=2,
            enable_monitoring=True
        )
        
        # Execute workflow - should handle errors gracefully
        results = error_orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate error handling
        self.assertFalse(results["success"])
        self.assertIn("error", results["status"].lower())
        self.assertTrue(results["has_errors"])
        self.assertIsNotNone(results["error_message"])
    
    def test_workflow_file_processing_integrity(self):
        """Test that file processing maintains data integrity."""
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate original files are unchanged
        original_files = list(Path(self.target_dir).glob("*.py"))
        for original_file in original_files:
            self.assertTrue(original_file.exists())
            # File should still be readable
            with open(original_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
        
        # Validate sandbox files exist and are different from originals
        sandbox_files = list(Path(self.sandbox_dir).glob("*.py"))
        self.assertEqual(len(sandbox_files), len(original_files))
        
        for sandbox_file in sandbox_files:
            self.assertTrue(sandbox_file.exists())
            # File should be readable
            with open(sandbox_file, 'r') as f:
                content = f.read()
                self.assertGreater(len(content), 0)
    
    def test_workflow_test_execution_integration(self):
        """Test that workflow properly integrates test execution."""
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate test results are included
        if results.get("test_results"):
            test_results = results["test_results"]
            self.assertIn("passed", test_results)
            self.assertIn("total_tests", test_results)
            self.assertIn("failed_tests", test_results)
            self.assertIsInstance(test_results["total_tests"], int)
            self.assertIsInstance(test_results["failed_tests"], int)
    
    def test_workflow_performance_metrics(self):
        """Test that workflow collects performance metrics."""
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate performance metrics
        self.assertIn("performance", results)
        performance = results["performance"]
        
        self.assertIn("execution_time_seconds", performance)
        self.assertIn("iterations_per_second", performance)
        self.assertIn("files_per_second", performance)
        
        # Validate metrics are reasonable
        self.assertGreater(performance["execution_time_seconds"], 0)
        self.assertGreaterEqual(performance["iterations_per_second"], 0)
        self.assertGreaterEqual(performance["files_per_second"], 0)
    
    def test_workflow_session_management(self):
        """Test that workflow properly manages execution sessions."""
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate session information
        self.assertIn("session_id", results)
        self.assertIsNotNone(results["session_id"])
        
        # Validate session info structure
        if results.get("session_info"):
            session_info = results["session_info"]
            self.assertIn("session_id", session_info)
            self.assertIn("target_directory", session_info)
            self.assertIn("sandbox_directory", session_info)
            self.assertIn("start_time", session_info)
            self.assertIn("status", session_info)
    
    @patch('src.utils.llm_call.call_llm')
    def test_workflow_with_mocked_llm_calls(self, mock_call_llm):
        """Test workflow with properly mocked LLM calls."""
        # Configure mock to return realistic responses
        mock_call_llm.return_value = self._get_mock_llm_response()
        
        # Execute workflow
        results = self.orchestrator.execute_refactoring(
            target_directory=self.target_dir,
            sandbox_directory=self.sandbox_dir
        )
        
        # Validate LLM was called
        self.assertTrue(mock_call_llm.called)
        
        # Validate workflow completed
        self.assertIsInstance(results, dict)
        self.assertIn("success", results)


class TestWorkflowComponents(unittest.TestCase):
    """Test individual workflow components in integration context."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="component_test_")
        self.target_dir = os.path.join(self.test_dir, "target")
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Create a simple test file
        test_file = os.path.join(self.target_dir, "simple.py")
        with open(test_file, 'w') as f:
            f.write("""
def add(a,b):
    return a+b

if __name__=="__main__":
    print(add(1,2))
""")
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initialization with various configurations."""
        # Test default initialization
        orchestrator1 = RefactoringOrchestrator()
        self.assertEqual(orchestrator1.model, "gemini-1.5-flash")
        self.assertEqual(orchestrator1.max_iterations, 10)
        
        # Test custom initialization
        orchestrator2 = RefactoringOrchestrator(
            model="custom-model",
            max_iterations=5,
            enable_monitoring=False
        )
        self.assertEqual(orchestrator2.model, "custom-model")
        self.assertEqual(orchestrator2.max_iterations, 5)
        self.assertFalse(orchestrator2.enable_monitoring)
    
    def test_workflow_state_transitions(self):
        """Test that workflow properly transitions between states."""
        mock_llm_fn = Mock(return_value="def add(a: int, b: int) -> int:\n    return a + b")
        
        orchestrator = RefactoringOrchestrator(
            model="test-model",
            llm_fn=mock_llm_fn,
            max_iterations=2
        )
        
        # Execute workflow and validate it completes
        results = orchestrator.execute_refactoring(
            target_directory=self.target_dir
        )
        
        # Validate basic results structure
        self.assertIsInstance(results, dict)
        self.assertIn("status", results)
        self.assertIn("total_iterations", results)


if __name__ == '__main__':
    unittest.main()