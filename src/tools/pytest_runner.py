"""
Pytest integration wrapper for test execution functionality.
Provides test result parsing, failure analysis, timeout handling and resource management.
"""

import subprocess
import json
import os
import time
import signal
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import TestingError


@dataclass
class TestFailure:
    """Represents a single test failure."""
    test_name: str
    test_file: str
    line_number: Optional[int]
    failure_message: str
    traceback: str
    test_type: str  # "FAILED", "ERROR", "SKIPPED"


@dataclass
class TestResult:
    """Structured pytest execution result."""
    passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    execution_time: float
    failures: List[TestFailure]
    success: bool
    raw_output: str
    exit_code: int


class PytestRunner:
    """Wrapper for running pytest tests with comprehensive result analysis."""
    
    def __init__(self, timeout: int = 300, max_workers: Optional[int] = None):
        """
        Initialize pytest runner.
        
        Args:
            timeout: Maximum execution time in seconds (default: 5 minutes)
            max_workers: Maximum number of parallel test workers
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self._verify_pytest_available()
    
    def _verify_pytest_available(self) -> None:
        """Verify that pytest is available in the system."""
        try:
            result = subprocess.run(
                ["pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise TestingError("Pytest is not properly installed or accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise TestingError(f"Failed to verify pytest installation: {e}")
    
    def run_tests(self, test_path: str, **kwargs) -> TestResult:
        """
        Run tests in the specified path.
        
        Args:
            test_path: Path to test file or directory
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If test execution fails
        """
        if not os.path.exists(test_path):
            raise TestingError(f"Test path not found: {test_path}")
        
        return self._run_pytest([test_path], **kwargs)
    
    def run_test_file(self, test_file: str, **kwargs) -> TestResult:
        """
        Run tests in a specific test file.
        
        Args:
            test_file: Path to the test file
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If test execution fails
        """
        if not os.path.exists(test_file):
            raise TestingError(f"Test file not found: {test_file}")
        
        if not test_file.endswith('.py'):
            raise TestingError(f"File is not a Python test file: {test_file}")
        
        return self._run_pytest([test_file], **kwargs)
    
    def run_test_directory(self, test_dir: str, **kwargs) -> TestResult:
        """
        Run all tests in a directory.
        
        Args:
            test_dir: Path to the test directory
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If test execution fails
        """
        if not os.path.exists(test_dir):
            raise TestingError(f"Test directory not found: {test_dir}")
        
        if not os.path.isdir(test_dir):
            raise TestingError(f"Path is not a directory: {test_dir}")
        
        return self._run_pytest([test_dir], **kwargs)
    
    def run_specific_test(self, test_file: str, test_name: str, **kwargs) -> TestResult:
        """
        Run a specific test by name.
        
        Args:
            test_file: Path to the test file
            test_name: Name of the specific test to run
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If test execution fails
        """
        test_target = f"{test_file}::{test_name}"
        return self._run_pytest([test_target], **kwargs)
    
    def _run_pytest(self, targets: List[str], **kwargs) -> TestResult:
        """
        Run pytest on the specified targets.
        
        Args:
            targets: List of test targets (files, directories, or specific tests)
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If pytest execution fails
        """
        # Build pytest command
        cmd = ["pytest", "--tb=short", "--json-report", "--json-report-file=/tmp/pytest_report.json"]
        
        # Add parallel execution if specified
        if self.max_workers:
            cmd.extend(["-n", str(self.max_workers)])
        
        # Add verbose output for better failure analysis
        cmd.append("-v")
        
        # Add custom arguments
        for key, value in kwargs.items():
            if key.startswith('--'):
                cmd.append(key)
                if value is not None and value is not True:
                    cmd.append(str(value))
            elif value is True:
                cmd.append(f"--{key}")
            elif value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        cmd.extend(targets)
        
        start_time = time.time()
        process = None
        
        try:
            # Run pytest with timeout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid  # Create new process group for clean termination
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout)
                execution_time = time.time() - start_time
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    process.communicate()
                
                raise TestingError(f"Pytest execution timed out after {self.timeout} seconds")
            
            return self._parse_pytest_output(stdout, stderr, return_code, time.time() - start_time)
            
        except Exception as e:
            if process and process.poll() is None:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.communicate(timeout=5)
                except:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.communicate()
                    except:
                        pass
            
            if isinstance(e, TestingError):
                raise
            raise TestingError(f"Failed to execute pytest: {e}")
    
    def _parse_pytest_output(self, stdout: str, stderr: str, return_code: int, execution_time: float) -> TestResult:
        """
        Parse pytest output into structured result.
        
        Args:
            stdout: Pytest standard output
            stderr: Pytest standard error
            return_code: Pytest process return code
            execution_time: Test execution time in seconds
            
        Returns:
            TestResult containing parsed results
            
        Raises:
            TestingError: If output parsing fails
        """
        try:
            # Try to parse JSON report first
            json_result = self._parse_json_report()
            if json_result:
                return json_result
            
            # Fallback to parsing text output
            return self._parse_text_output(stdout, stderr, return_code, execution_time)
            
        except Exception as e:
            # If all parsing fails, create a basic result from available info
            return TestResult(
                passed=return_code == 0,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=0,
                execution_time=execution_time,
                failures=[],
                success=return_code == 0,
                raw_output=stdout + stderr,
                exit_code=return_code
            )
    
    def _parse_json_report(self) -> Optional[TestResult]:
        """
        Parse pytest JSON report if available.
        
        Returns:
            TestResult if JSON report exists and is valid, None otherwise
        """
        json_file = "/tmp/pytest_report.json"
        try:
            if not os.path.exists(json_file):
                return None
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Clean up the temporary file
            os.remove(json_file)
            
            # Extract test results
            summary = data.get('summary', {})
            tests = data.get('tests', [])
            
            total_tests = summary.get('total', 0)
            passed_tests = summary.get('passed', 0)
            failed_tests = summary.get('failed', 0)
            skipped_tests = summary.get('skipped', 0)
            error_tests = summary.get('error', 0)
            
            # Parse failures
            failures = []
            for test in tests:
                if test.get('outcome') in ['failed', 'error']:
                    failure = TestFailure(
                        test_name=test.get('nodeid', ''),
                        test_file=test.get('file', ''),
                        line_number=test.get('lineno'),
                        failure_message=test.get('call', {}).get('longrepr', ''),
                        traceback=test.get('call', {}).get('longrepr', ''),
                        test_type=test.get('outcome', '').upper()
                    )
                    failures.append(failure)
            
            return TestResult(
                passed=failed_tests == 0 and error_tests == 0,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                error_tests=error_tests,
                execution_time=data.get('duration', 0.0),
                failures=failures,
                success=failed_tests == 0 and error_tests == 0,
                raw_output=json.dumps(data, indent=2),
                exit_code=0 if failed_tests == 0 and error_tests == 0 else 1
            )
            
        except Exception:
            # Clean up file if it exists
            try:
                if os.path.exists(json_file):
                    os.remove(json_file)
            except:
                pass
            return None
    
    def _parse_text_output(self, stdout: str, stderr: str, return_code: int, execution_time: float) -> TestResult:
        """
        Parse pytest text output as fallback.
        
        Args:
            stdout: Standard output
            stderr: Standard error
            return_code: Process return code
            execution_time: Execution time
            
        Returns:
            TestResult parsed from text output
        """
        output = stdout + stderr
        lines = output.split('\n')
        
        # Initialize counters
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        error_tests = 0
        failures = []
        
        # Parse summary line (e.g., "5 failed, 3 passed, 1 skipped in 2.34s")
        for line in lines:
            if ' in ' in line and ('passed' in line or 'failed' in line):
                parts = line.split(' in ')[0].split(', ')
                for part in parts:
                    part = part.strip()
                    if 'failed' in part:
                        failed_tests = int(part.split()[0])
                    elif 'passed' in part:
                        passed_tests = int(part.split()[0])
                    elif 'skipped' in part:
                        skipped_tests = int(part.split()[0])
                    elif 'error' in part:
                        error_tests = int(part.split()[0])
                break
        
        total_tests = passed_tests + failed_tests + skipped_tests + error_tests
        
        # Parse failures (basic parsing)
        current_failure = None
        in_failure = False
        
        for line in lines:
            if line.startswith('FAILED ') or line.startswith('ERROR '):
                if current_failure:
                    failures.append(current_failure)
                
                test_name = line.split(' ', 1)[1] if ' ' in line else line
                current_failure = TestFailure(
                    test_name=test_name,
                    test_file='',
                    line_number=None,
                    failure_message=line,
                    traceback='',
                    test_type='FAILED' if line.startswith('FAILED') else 'ERROR'
                )
                in_failure = True
            elif in_failure and line.strip():
                if current_failure:
                    current_failure.traceback += line + '\n'
        
        if current_failure:
            failures.append(current_failure)
        
        return TestResult(
            passed=return_code == 0,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            execution_time=execution_time,
            failures=failures,
            success=return_code == 0,
            raw_output=output,
            exit_code=return_code
        )
    
    def get_failed_tests(self, result: TestResult) -> List[TestFailure]:
        """
        Get all failed tests from the result.
        
        Args:
            result: TestResult to filter
            
        Returns:
            List of failed tests
        """
        return [f for f in result.failures if f.test_type in ['FAILED', 'ERROR']]
    
    def get_failure_summary(self, result: TestResult) -> str:
        """
        Generate a summary of test failures.
        
        Args:
            result: TestResult to summarize
            
        Returns:
            Formatted failure summary
        """
        if result.success:
            return f"All {result.total_tests} tests passed in {result.execution_time:.2f}s"
        
        summary = f"Test Results Summary:\n"
        summary += f"Total Tests: {result.total_tests}\n"
        summary += f"Passed: {result.passed_tests}\n"
        summary += f"Failed: {result.failed_tests}\n"
        summary += f"Errors: {result.error_tests}\n"
        summary += f"Skipped: {result.skipped_tests}\n"
        summary += f"Execution Time: {result.execution_time:.2f}s\n"
        
        if result.failures:
            summary += f"\nFailure Details:\n"
            for i, failure in enumerate(result.failures[:5], 1):  # Show first 5 failures
                summary += f"{i}. {failure.test_name}\n"
                summary += f"   {failure.failure_message}\n"
        
        return summary
    
    def analyze_failures(self, result: TestResult) -> Dict[str, Any]:
        """
        Analyze test failures to provide insights for fixing.
        
        Args:
            result: TestResult to analyze
            
        Returns:
            Dictionary containing failure analysis
        """
        if result.success:
            return {"status": "success", "insights": []}
        
        analysis = {
            "status": "failed",
            "total_failures": len(result.failures),
            "failure_types": {},
            "common_patterns": [],
            "insights": []
        }
        
        # Categorize failures by type
        for failure in result.failures:
            failure_type = failure.test_type
            if failure_type not in analysis["failure_types"]:
                analysis["failure_types"][failure_type] = 0
            analysis["failure_types"][failure_type] += 1
        
        # Look for common patterns in failure messages
        messages = [f.failure_message for f in result.failures]
        
        # Common error patterns
        patterns = {
            "assertion": ["AssertionError", "assert"],
            "import": ["ImportError", "ModuleNotFoundError"],
            "attribute": ["AttributeError"],
            "type": ["TypeError"],
            "value": ["ValueError"],
            "syntax": ["SyntaxError"],
            "indentation": ["IndentationError"]
        }
        
        for pattern_name, keywords in patterns.items():
            count = sum(1 for msg in messages if any(kw in msg for kw in keywords))
            if count > 0:
                analysis["common_patterns"].append({
                    "pattern": pattern_name,
                    "count": count,
                    "percentage": (count / len(messages)) * 100
                })
        
        # Generate insights
        if analysis["failure_types"].get("ERROR", 0) > 0:
            analysis["insights"].append("Code has syntax or import errors that prevent test execution")
        
        if analysis["failure_types"].get("FAILED", 0) > 0:
            analysis["insights"].append("Tests are running but assertions are failing")
        
        for pattern in analysis["common_patterns"]:
            if pattern["percentage"] > 50:
                analysis["insights"].append(f"Most failures are {pattern['pattern']}-related")
        
        return analysis