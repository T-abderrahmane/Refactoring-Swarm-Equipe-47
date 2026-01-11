"""
Auditor Agent for code analysis and refactoring plan generation.

The Auditor Agent is responsible for:
1. Analyzing Python code for issues using Pylint
2. Generating comprehensive refactoring plans
3. Providing LLM-based code review and issue identification
"""

import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from ..models.core import (
    CodeIssue, RefactoringPlan, IssueType, IssueSeverity
)
from ..tools.pylint_runner import PylintRunner, PylintIssue
from ..utils.llm_call import call_llm
from ..utils.logger import log_experiment, ActionType
from ..exceptions import AnalysisError


class AuditorAgent:
    """Agent responsible for code analysis and refactoring plan generation."""
    
    def __init__(self, model: str = "gemini-1.5-flash", llm_fn=None):
        """
        Initialize the Auditor Agent.
        
        Args:
            model: LLM model to use for analysis
            llm_fn: Function to call the LLM (for dependency injection)
        """
        self.model = model
        self.llm_fn = llm_fn
        self.pylint_runner = PylintRunner()
        self.agent_name = "Auditor"
    
    def analyze_file(self, file_path: str) -> RefactoringPlan:
        """
        Analyze a single Python file and generate a refactoring plan.
        
        Args:
            file_path: Path to the Python file to analyze
            
        Returns:
            RefactoringPlan containing identified issues and recommendations
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not os.path.exists(file_path):
            raise AnalysisError(f"File not found: {file_path}")
        
        if not file_path.endswith('.py'):
            raise AnalysisError(f"File is not a Python file: {file_path}")
        
        try:
            # Step 1: Run Pylint analysis
            pylint_result = self.pylint_runner.analyze_file(file_path)
            
            # Step 2: Convert Pylint issues to CodeIssue objects
            code_issues = self._convert_pylint_issues(pylint_result.issues, file_path)
            
            # Step 3: Use LLM to enhance analysis and identify additional issues
            if self.llm_fn:
                enhanced_issues = self._enhance_analysis_with_llm(file_path, code_issues)
                code_issues.extend(enhanced_issues)
            
            # Step 4: Prioritize and categorize issues
            code_issues = self._prioritize_issues(code_issues)
            
            # Step 5: Generate refactoring plan
            plan = self._generate_refactoring_plan(file_path, code_issues, pylint_result.score)
            
            return plan
            
        except AnalysisError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to analyze file {file_path}: {e}")
    
    def analyze_directory(self, directory_path: str) -> List[RefactoringPlan]:
        """
        Analyze all Python files in a directory.
        
        Args:
            directory_path: Path to the directory to analyze
            
        Returns:
            List of RefactoringPlan objects for each file
            
        Raises:
            AnalysisError: If analysis fails
        """
        if not os.path.exists(directory_path):
            raise AnalysisError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise AnalysisError(f"Path is not a directory: {directory_path}")
        
        plans = []
        python_files = self._find_python_files(directory_path)
        
        if not python_files:
            raise AnalysisError(f"No Python files found in directory: {directory_path}")
        
        for file_path in python_files:
            try:
                plan = self.analyze_file(file_path)
                plans.append(plan)
            except AnalysisError as e:
                # Log error but continue with other files
                print(f"Warning: Failed to analyze {file_path}: {e}")
                continue
        
        return plans
    
    def _find_python_files(self, directory_path: str) -> List[str]:
        """
        Find all Python files in a directory recursively.
        
        Args:
            directory_path: Path to search
            
        Returns:
            List of Python file paths
        """
        python_files = []
        for root, dirs, files in os.walk(directory_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'venv', '.pytest_cache']]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        return sorted(python_files)
    
    def _convert_pylint_issues(self, pylint_issues: List[PylintIssue], 
                               file_path: str) -> List[CodeIssue]:
        """
        Convert Pylint issues to CodeIssue objects.
        
        Args:
            pylint_issues: List of PylintIssue objects from Pylint
            file_path: Path to the analyzed file
            
        Returns:
            List of CodeIssue objects
        """
        code_issues = []
        
        for pylint_issue in pylint_issues:
            # Map Pylint issue type to CodeIssue type
            issue_type = self._map_pylint_type_to_issue_type(pylint_issue.type)
            
            # Map Pylint issue type to severity
            severity = self._map_pylint_type_to_severity(pylint_issue.type)
            
            # Generate suggested fix based on issue type
            suggested_fix = self._generate_suggested_fix(pylint_issue)
            
            code_issue = CodeIssue(
                line_number=pylint_issue.line,
                column_number=pylint_issue.column,
                issue_type=issue_type,
                description=pylint_issue.message,
                suggested_fix=suggested_fix,
                severity=severity,
                file_path=file_path
            )
            
            code_issues.append(code_issue)
        
        return code_issues
    
    def _map_pylint_type_to_issue_type(self, pylint_type: str) -> IssueType:
        """
        Map Pylint issue type to CodeIssue type.
        
        Args:
            pylint_type: Pylint issue type (error, warning, refactor, convention, info)
            
        Returns:
            Corresponding IssueType
        """
        type_mapping = {
            'error': IssueType.SYNTAX_ERROR,
            'warning': IssueType.LOGIC_ERROR,
            'refactor': IssueType.MAINTAINABILITY,
            'convention': IssueType.STYLE_VIOLATION,
            'info': IssueType.MAINTAINABILITY
        }
        
        return type_mapping.get(pylint_type, IssueType.MAINTAINABILITY)
    
    def _map_pylint_type_to_severity(self, pylint_type: str) -> IssueSeverity:
        """
        Map Pylint issue type to severity level.
        
        Args:
            pylint_type: Pylint issue type
            
        Returns:
            Corresponding IssueSeverity
        """
        severity_mapping = {
            'error': IssueSeverity.CRITICAL,
            'warning': IssueSeverity.HIGH,
            'refactor': IssueSeverity.MEDIUM,
            'convention': IssueSeverity.LOW,
            'info': IssueSeverity.INFO
        }
        
        return severity_mapping.get(pylint_type, IssueSeverity.MEDIUM)
    
    def _generate_suggested_fix(self, pylint_issue: PylintIssue) -> str:
        """
        Generate a suggested fix for a Pylint issue.
        
        Args:
            pylint_issue: The Pylint issue
            
        Returns:
            Suggested fix string
        """
        # Map common Pylint symbols to suggested fixes
        fix_suggestions = {
            'missing-docstring': 'Add a docstring to document the function/class/module',
            'line-too-long': 'Break the line into multiple lines (max 100 characters)',
            'unused-import': 'Remove the unused import statement',
            'unused-variable': 'Remove the unused variable or use it in the code',
            'invalid-name': 'Rename the variable/function to follow naming conventions',
            'trailing-whitespace': 'Remove trailing whitespace',
            'multiple-statements': 'Put each statement on a separate line',
            'bad-indentation': 'Fix the indentation to be consistent',
            'undefined-variable': 'Define the variable before using it',
            'no-member': 'Check that the object has the referenced member',
        }
        
        return fix_suggestions.get(
            pylint_issue.symbol,
            f"Fix the {pylint_issue.type}: {pylint_issue.message}"
        )
    
    def _enhance_analysis_with_llm(self, file_path: str, 
                                   code_issues: List[CodeIssue]) -> List[CodeIssue]:
        """
        Use LLM to enhance code analysis and identify additional issues.
        
        Args:
            file_path: Path to the analyzed file
            code_issues: Initial code issues from Pylint
            
        Returns:
            Additional CodeIssue objects identified by LLM
        """
        if not self.llm_fn:
            return []
        
        try:
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            # Prepare prompt for LLM
            prompt = self._prepare_llm_analysis_prompt(file_path, file_content, code_issues)
            
            # Call LLM for enhanced analysis
            response = call_llm(
                agent_name=self.agent_name,
                model=self.model,
                action=ActionType.ANALYSIS,
                prompt=prompt,
                llm_fn=self.llm_fn
            )
            
            # Parse LLM response to extract additional issues
            additional_issues = self._parse_llm_analysis_response(response, file_path)
            
            return additional_issues
            
        except Exception as e:
            # Log error but don't fail - LLM enhancement is optional
            print(f"Warning: LLM analysis enhancement failed: {e}")
            return []
    
    def _prepare_llm_analysis_prompt(self, file_path: str, file_content: str,
                                     code_issues: List[CodeIssue]) -> str:
        """
        Prepare a prompt for LLM-based code analysis.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            code_issues: Initial issues from Pylint
            
        Returns:
            Formatted prompt for LLM
        """
        existing_issues = "\n".join([
            f"- Line {issue.line_number}: {issue.description}"
            for issue in code_issues[:10]  # Limit to first 10 issues
        ])
        
        prompt = f"""Analyze the following Python code for potential issues and improvements.

File: {file_path}

Code:
```python
{file_content}
```

Already identified issues:
{existing_issues if existing_issues else "None"}

Please identify any additional issues related to:
1. Logic errors or potential bugs
2. Performance problems
3. Security vulnerabilities
4. Code maintainability issues
5. Best practice violations

For each issue, provide:
- Line number
- Issue type (logic_error, performance_issue, security_issue, maintainability, etc.)
- Description
- Suggested fix

Format your response as a JSON array with objects containing: line_number, issue_type, description, suggested_fix"""
        
        return prompt
    
    def _parse_llm_analysis_response(self, response: str, file_path: str) -> List[CodeIssue]:
        """
        Parse LLM response to extract additional code issues.
        
        Args:
            response: LLM response text
            file_path: Path to the analyzed file
            
        Returns:
            List of CodeIssue objects extracted from response
        """
        import json
        
        additional_issues = []
        
        try:
            # Try to extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                issues_data = json.loads(json_str)
                
                for issue_data in issues_data:
                    try:
                        issue_type = IssueType[issue_data.get('issue_type', 'MAINTAINABILITY').upper()]
                    except (KeyError, ValueError):
                        issue_type = IssueType.MAINTAINABILITY
                    
                    code_issue = CodeIssue(
                        line_number=issue_data.get('line_number', 0),
                        issue_type=issue_type,
                        description=issue_data.get('description', ''),
                        suggested_fix=issue_data.get('suggested_fix', ''),
                        severity=IssueSeverity.MEDIUM,
                        file_path=file_path
                    )
                    additional_issues.append(code_issue)
        
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, just return empty list
            print(f"Warning: Failed to parse LLM analysis response: {e}")
        
        return additional_issues
    
    def _prioritize_issues(self, code_issues: List[CodeIssue]) -> List[CodeIssue]:
        """
        Prioritize and sort code issues by severity and type.
        
        Args:
            code_issues: List of code issues to prioritize
            
        Returns:
            Sorted list of code issues
        """
        # Define severity order (higher number = higher priority)
        severity_order = {
            IssueSeverity.CRITICAL: 5,
            IssueSeverity.HIGH: 4,
            IssueSeverity.MEDIUM: 3,
            IssueSeverity.LOW: 2,
            IssueSeverity.INFO: 1
        }
        
        # Sort by severity (descending) then by line number (ascending)
        sorted_issues = sorted(
            code_issues,
            key=lambda x: (-severity_order.get(x.severity, 0), x.line_number)
        )
        
        return sorted_issues
    
    def _generate_refactoring_plan(self, file_path: str, code_issues: List[CodeIssue],
                                   pylint_score: float) -> RefactoringPlan:
        """
        Generate a comprehensive refactoring plan based on identified issues.
        
        Args:
            file_path: Path to the analyzed file
            code_issues: List of identified code issues
            pylint_score: Pylint quality score
            
        Returns:
            RefactoringPlan object
        """
        # Calculate priority based on issue count and severity
        critical_count = sum(1 for issue in code_issues if issue.severity == IssueSeverity.CRITICAL)
        high_count = sum(1 for issue in code_issues if issue.severity == IssueSeverity.HIGH)
        
        priority = min(10, critical_count * 3 + high_count)
        
        # Estimate effort based on issue count and types
        total_issues = len(code_issues)
        if total_issues == 0:
            estimated_effort = "minimal"
        elif total_issues <= 5:
            estimated_effort = "low"
        elif total_issues <= 15:
            estimated_effort = "medium"
        elif total_issues <= 30:
            estimated_effort = "high"
        else:
            estimated_effort = "very_high"
        
        plan = RefactoringPlan(
            file_path=file_path,
            issues=code_issues,
            priority=priority,
            estimated_effort=estimated_effort
        )
        
        return plan
    
    def generate_analysis_report(self, plans: List[RefactoringPlan]) -> str:
        """
        Generate a human-readable analysis report from refactoring plans.
        
        Args:
            plans: List of RefactoringPlan objects
            
        Returns:
            Formatted analysis report
        """
        report = "=== Code Analysis Report ===\n\n"
        
        total_issues = sum(len(plan.issues) for plan in plans)
        total_critical = sum(
            len([i for i in plan.issues if i.severity == IssueSeverity.CRITICAL])
            for plan in plans
        )
        total_high = sum(
            len([i for i in plan.issues if i.severity == IssueSeverity.HIGH])
            for plan in plans
        )
        
        report += f"Summary:\n"
        report += f"  Files analyzed: {len(plans)}\n"
        report += f"  Total issues: {total_issues}\n"
        report += f"  Critical: {total_critical}\n"
        report += f"  High: {total_high}\n\n"
        
        report += "Files:\n"
        for plan in plans:
            report += f"\n  {plan.file_path}\n"
            report += f"    Priority: {plan.priority}/10\n"
            report += f"    Effort: {plan.estimated_effort}\n"
            report += f"    Issues: {len(plan.issues)}\n"
            
            if plan.issues:
                report += "    Top issues:\n"
                for issue in plan.issues[:3]:
                    report += f"      - Line {issue.line_number}: {issue.description}\n"
        
        return report
