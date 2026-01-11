"""
Pylint integration wrapper for code analysis functionality.
Provides structured output generation and error handling for pylint execution.
"""

import subprocess
import json
import tempfile
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import AnalysisError


@dataclass
class PylintIssue:
    """Represents a single pylint issue."""
    line: int
    column: int
    message: str
    message_id: str
    symbol: str
    type: str  # error, warning, refactor, convention, info
    path: str


@dataclass
class PylintResult:
    """Structured pylint analysis result."""
    score: float
    issues: List[PylintIssue]
    total_issues: int
    error_count: int
    warning_count: int
    refactor_count: int
    convention_count: int
    info_count: int
    success: bool
    raw_output: str


class PylintRunner:
    """Wrapper for running pylint analysis on Python code."""
    
    def __init__(self, pylint_config: Optional[str] = None):
        """
        Initialize pylint runner.
        
        Args:
            pylint_config: Optional path to pylint configuration file
        """
        self.pylint_config = pylint_config
        self._verify_pylint_available()
    
    def _verify_pylint_available(self) -> None:
        """Verify that pylint is available in the system."""
        try:
            result = subprocess.run(
                ["pylint", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                raise AnalysisError("Pylint is not properly installed or accessible")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise AnalysisError(f"Failed to verify pylint installation: {e}")
    
    def analyze_file(self, file_path: str) -> PylintResult:
        """
        Analyze a single Python file with pylint.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            PylintResult containing analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not os.path.exists(file_path):
            raise AnalysisError(f"File not found: {file_path}")
        
        if not file_path.endswith('.py'):
            raise AnalysisError(f"File is not a Python file: {file_path}")
        
        return self._run_pylint([file_path])
    
    def analyze_directory(self, directory_path: str) -> PylintResult:
        """
        Analyze all Python files in a directory with pylint.
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            PylintResult containing analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not os.path.exists(directory_path):
            raise AnalysisError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise AnalysisError(f"Path is not a directory: {directory_path}")
        
        # Find all Python files in the directory
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        if not python_files:
            raise AnalysisError(f"No Python files found in directory: {directory_path}")
        
        return self._run_pylint(python_files)
    
    def _run_pylint(self, targets: List[str]) -> PylintResult:
        """
        Run pylint on the specified targets.
        
        Args:
            targets: List of file or directory paths to analyze
            
        Returns:
            PylintResult containing analysis results
            
        Raises:
            AnalysisError: If pylint execution fails
        """
        # Build pylint command
        cmd = ["pylint", "--output-format=json", "--score=yes"]
        
        if self.pylint_config:
            cmd.extend(["--rcfile", self.pylint_config])
        
        cmd.extend(targets)
        
        try:
            # Run pylint
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            return self._parse_pylint_output(result.stdout, result.stderr, result.returncode)
            
        except subprocess.TimeoutExpired:
            raise AnalysisError("Pylint execution timed out after 5 minutes")
        except Exception as e:
            raise AnalysisError(f"Failed to execute pylint: {e}")
    
    def _parse_pylint_output(self, stdout: str, stderr: str, return_code: int) -> PylintResult:
        """
        Parse pylint JSON output into structured result.
        
        Args:
            stdout: Pylint standard output
            stderr: Pylint standard error
            return_code: Pylint process return code
            
        Returns:
            PylintResult containing parsed results
            
        Raises:
            AnalysisError: If output parsing fails
        """
        try:
            # Parse JSON output
            issues_data = []
            score = 0.0
            
            if stdout.strip():
                # Split output to separate JSON from score
                lines = stdout.strip().split('\n')
                json_lines = []
                score_line = None
                
                for line in lines:
                    if line.startswith('Your code has been rated at'):
                        score_line = line
                    elif line.strip() and not line.startswith('*'):
                        json_lines.append(line)
                
                # Parse JSON issues
                if json_lines:
                    json_content = '\n'.join(json_lines)
                    if json_content.strip():
                        issues_data = json.loads(json_content)
                
                # Parse score
                if score_line:
                    score = self._extract_score(score_line)
            
            # Convert to structured issues
            issues = []
            for issue_data in issues_data:
                issue = PylintIssue(
                    line=issue_data.get('line', 0),
                    column=issue_data.get('column', 0),
                    message=issue_data.get('message', ''),
                    message_id=issue_data.get('message-id', ''),
                    symbol=issue_data.get('symbol', ''),
                    type=issue_data.get('type', 'unknown'),
                    path=issue_data.get('path', '')
                )
                issues.append(issue)
            
            # Count issues by type
            error_count = sum(1 for issue in issues if issue.type == 'error')
            warning_count = sum(1 for issue in issues if issue.type == 'warning')
            refactor_count = sum(1 for issue in issues if issue.type == 'refactor')
            convention_count = sum(1 for issue in issues if issue.type == 'convention')
            info_count = sum(1 for issue in issues if issue.type == 'info')
            
            # Determine success (pylint returns 0 for no issues, non-zero for issues found)
            success = return_code == 0 or (return_code > 0 and error_count == 0)
            
            return PylintResult(
                score=score,
                issues=issues,
                total_issues=len(issues),
                error_count=error_count,
                warning_count=warning_count,
                refactor_count=refactor_count,
                convention_count=convention_count,
                info_count=info_count,
                success=success,
                raw_output=stdout + stderr
            )
            
        except json.JSONDecodeError as e:
            raise AnalysisError(f"Failed to parse pylint JSON output: {e}")
        except Exception as e:
            raise AnalysisError(f"Failed to parse pylint output: {e}")
    
    def _extract_score(self, score_line: str) -> float:
        """
        Extract numerical score from pylint score line.
        
        Args:
            score_line: Line containing the score information
            
        Returns:
            Numerical score value
        """
        try:
            # Example: "Your code has been rated at 8.50/10"
            parts = score_line.split()
            for i, part in enumerate(parts):
                if '/' in part and part.replace('.', '').replace('/', '').isdigit():
                    score_part = part.split('/')[0]
                    return float(score_part)
            return 0.0
        except (ValueError, IndexError):
            return 0.0
    
    def get_issues_by_type(self, result: PylintResult, issue_type: str) -> List[PylintIssue]:
        """
        Filter issues by type.
        
        Args:
            result: PylintResult to filter
            issue_type: Type of issues to return (error, warning, refactor, convention, info)
            
        Returns:
            List of issues matching the specified type
        """
        return [issue for issue in result.issues if issue.type == issue_type]
    
    def get_high_priority_issues(self, result: PylintResult) -> List[PylintIssue]:
        """
        Get high priority issues (errors and warnings).
        
        Args:
            result: PylintResult to filter
            
        Returns:
            List of high priority issues
        """
        return [issue for issue in result.issues if issue.type in ['error', 'warning']]
    
    def format_issue_summary(self, result: PylintResult) -> str:
        """
        Format a human-readable summary of pylint results.
        
        Args:
            result: PylintResult to summarize
            
        Returns:
            Formatted summary string
        """
        summary = f"Pylint Analysis Summary:\n"
        summary += f"Score: {result.score:.2f}/10\n"
        summary += f"Total Issues: {result.total_issues}\n"
        summary += f"  - Errors: {result.error_count}\n"
        summary += f"  - Warnings: {result.warning_count}\n"
        summary += f"  - Refactor: {result.refactor_count}\n"
        summary += f"  - Convention: {result.convention_count}\n"
        summary += f"  - Info: {result.info_count}\n"
        
        if result.error_count > 0:
            summary += "\nHigh Priority Issues (Errors):\n"
            errors = self.get_issues_by_type(result, 'error')
            for error in errors[:5]:  # Show first 5 errors
                summary += f"  - Line {error.line}: {error.message}\n"
        
        return summary