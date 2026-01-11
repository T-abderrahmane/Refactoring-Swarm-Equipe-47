"""
Judge Agent for test execution and validation.

The Judge Agent is responsible for:
1. Executing unit tests using pytest
2. Validating code functionality after fixes
3. Providing detailed error feedback for failed tests
4. Determining success/failure status for the self-healing loop
"""

import os
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..models.core import TestResult, AgentStatus
from ..tools.pytest_runner import PytestRunner, TestResult as PytestTestResult, TestFailure
from ..utils.llm_call import call_llm
from ..utils.logger import log_experiment, ActionType
from ..exceptions import TestingError


class JudgeAgent:
    """Agent responsible for test execution and validation."""
    
    def __init__(self, model: str = "gemini-1.5-flash", llm_fn=None, timeout: int = 300):
        """
        Initialize the Judge Agent.
        
        Args:
            model: LLM model to use for error analysis
            llm_fn: Function to call the LLM (for dependency injection)
            timeout: Maximum test execution time in seconds
        """
        self.model = model
        self.llm_fn = llm_fn
        self.agent_name = "Judge"
        self.pytest_runner = PytestRunner(timeout=timeout)
        self.test_history: List[Dict[str, Any]] = []
    
    def run_tests(self, test_directory: str, **kwargs) -> TestResult:
        """
        Run tests in the specified directory and return structured results.
        
        Args:
            test_directory: Path to directory containing tests
            **kwargs: Additional pytest arguments
            
        Returns:
            TestResult containing execution results
            
        Raises:
            TestingError: If test execution fails
        """
        if not os.path.exists(test_directory):
            raise TestingError(f"Test directory not found: {test_directory}")
        
        if not os.path.isdir(test_directory):
            raise TestingError(f"Path is not a directory: {test_directory}")
        
        try:
            # Log test execution start
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Running tests in directory: {test_directory}",
                    "output_response": "Test execution started"
                },
                status="SUCCESS"
            )
            
            # Run pytest
            pytest_result = self.pytest_runner.run_test_directory(test_directory, **kwargs)
            
            # Convert pytest result to our TestResult format
            test_result = self._convert_pytest_result(pytest_result, test_directory)
            
            # Record test execution in history
            self._record_test_execution(test_directory, test_result, pytest_result)
            
            # Log test execution completion
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Test execution completed for {test_directory}",
                    "output_response": f"Tests: {test_result.total_tests}, Passed: {test_result.total_tests - test_result.failed_tests}, Failed: {test_result.failed_tests}"
                },
                status="SUCCESS" if test_result.passed else "FAILURE"
            )
            
            return test_result
            
        except TestingError:
            raise
        except Exception as e:
            raise TestingError(f"Failed to run tests in {test_directory}: {e}")
    
    def run_specific_test_file(self, test_file: str, **kwargs) -> TestResult:
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
        
        try:
            # Log test execution start
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Running tests in file: {test_file}",
                    "output_response": "Test execution started"
                },
                status="SUCCESS"
            )
            
            # Run pytest on specific file
            pytest_result = self.pytest_runner.run_test_file(test_file, **kwargs)
            
            # Convert pytest result to our TestResult format
            test_result = self._convert_pytest_result(pytest_result, test_file)
            
            # Record test execution in history
            self._record_test_execution(test_file, test_result, pytest_result)
            
            # Log test execution completion
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Test execution completed for {test_file}",
                    "output_response": f"Tests: {test_result.total_tests}, Passed: {test_result.total_tests - test_result.failed_tests}, Failed: {test_result.failed_tests}"
                },
                status="SUCCESS" if test_result.passed else "FAILURE"
            )
            
            return test_result
            
        except TestingError:
            raise
        except Exception as e:
            raise TestingError(f"Failed to run tests in {test_file}: {e}")
    
    def analyze_test_failures(self, test_result: TestResult) -> Dict[str, Any]:
        """
        Analyze test failures and provide detailed feedback for fixing.
        
        Args:
            test_result: TestResult containing failure information
            
        Returns:
            Dictionary containing failure analysis and recommendations
            
        Raises:
            TestingError: If analysis fails
        """
        if test_result.passed:
            return {
                "status": "success",
                "message": "All tests passed successfully",
                "recommendations": []
            }
        
        try:
            # Basic failure analysis
            analysis = {
                "status": "failed",
                "total_failures": test_result.failed_tests,
                "failure_categories": {},
                "common_patterns": [],
                "recommendations": [],
                "detailed_failures": test_result.error_details
            }
            
            # Categorize failures by type
            failure_types = self._categorize_failures(test_result.error_details)
            analysis["failure_categories"] = failure_types
            
            # Generate basic recommendations
            basic_recommendations = self._generate_basic_recommendations(failure_types)
            analysis["recommendations"].extend(basic_recommendations)
            
            # Use LLM for enhanced analysis if available
            if self.llm_fn and test_result.error_details:
                enhanced_analysis = self._analyze_failures_with_llm(test_result)
                if enhanced_analysis:
                    analysis["llm_insights"] = enhanced_analysis
                    analysis["recommendations"].extend(enhanced_analysis.get("recommendations", []))
            
            # Log the analysis
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Analyzing {test_result.failed_tests} test failures",
                    "output_response": f"Generated {len(analysis['recommendations'])} recommendations"
                },
                status="SUCCESS"
            )
            
            return analysis
            
        except Exception as e:
            raise TestingError(f"Failed to analyze test failures: {e}")
    
    def generate_feedback_for_fixer(self, test_result: TestResult) -> str:
        """
        Generate detailed feedback for the Fixer Agent based on test results.
        
        Args:
            test_result: TestResult containing test execution results
            
        Returns:
            Formatted feedback string for the Fixer Agent
        """
        if test_result.passed:
            return "All tests passed successfully. No fixes needed."
        
        try:
            # Analyze failures
            analysis = self.analyze_test_failures(test_result)
            
            # Generate structured feedback
            feedback = self._format_feedback_for_fixer(analysis, test_result)
            
            # Log feedback generation
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Generating feedback for {test_result.failed_tests} failed tests",
                    "output_response": feedback[:500] + "..." if len(feedback) > 500 else feedback
                },
                status="SUCCESS"
            )
            
            return feedback
            
        except Exception as e:
            return f"Error generating feedback: {str(e)}"
    
    def validate_code_functionality(self, code_directory: str, test_directory: str) -> Tuple[bool, TestResult]:
        """
        Validate code functionality by running all tests.
        
        Args:
            code_directory: Directory containing the code to validate
            test_directory: Directory containing the tests
            
        Returns:
            Tuple of (success, TestResult)
        """
        try:
            # Run all tests
            test_result = self.run_tests(test_directory)
            
            # Determine overall success
            success = test_result.passed
            
            # Log validation result
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Validating functionality for {code_directory}",
                    "output_response": f"Validation {'passed' if success else 'failed'}: {test_result.total_tests - test_result.failed_tests}/{test_result.total_tests} tests passed"
                },
                status="SUCCESS" if success else "FAILURE"
            )
            
            return success, test_result
            
        except Exception as e:
            # Create a failed test result
            failed_result = TestResult(
                passed=False,
                total_tests=0,
                failed_tests=1,
                error_details=[f"Validation error: {str(e)}"],
                execution_time=0.0,
                test_file=test_directory
            )
            
            return False, failed_result
    
    def _convert_pytest_result(self, pytest_result: PytestTestResult, test_path: str) -> TestResult:
        """
        Convert PytestTestResult to our TestResult format.
        
        Args:
            pytest_result: Result from pytest runner
            test_path: Path to the test file or directory
            
        Returns:
            TestResult in our format
        """
        # Extract error details from failures
        error_details = []
        for failure in pytest_result.failures:
            error_msg = f"{failure.test_name}: {failure.failure_message}"
            if failure.traceback:
                error_msg += f"\n{failure.traceback}"
            error_details.append(error_msg)
        
        return TestResult(
            passed=pytest_result.passed,
            total_tests=pytest_result.total_tests,
            failed_tests=pytest_result.failed_tests,
            error_details=error_details,
            execution_time=pytest_result.execution_time,
            test_file=test_path,
            stdout=pytest_result.raw_output,
            stderr=""
        )
    
    def _record_test_execution(self, test_path: str, test_result: TestResult, 
                              pytest_result: PytestTestResult) -> None:
        """
        Record test execution in history for tracking.
        
        Args:
            test_path: Path to the test file or directory
            test_result: Our TestResult
            pytest_result: Original pytest result
        """
        execution_record = {
            "test_path": test_path,
            "timestamp": pytest_result.raw_output,  # Using raw_output as timestamp placeholder
            "result": test_result.to_dict(),
            "pytest_details": {
                "exit_code": pytest_result.exit_code,
                "passed_tests": pytest_result.passed_tests,
                "skipped_tests": pytest_result.skipped_tests,
                "error_tests": pytest_result.error_tests
            }
        }
        
        self.test_history.append(execution_record)
        
        # Keep only last 50 executions to prevent memory issues
        if len(self.test_history) > 50:
            self.test_history = self.test_history[-50:]
    
    def _categorize_failures(self, error_details: List[str]) -> Dict[str, int]:
        """
        Categorize test failures by type.
        
        Args:
            error_details: List of error messages
            
        Returns:
            Dictionary mapping failure types to counts
        """
        categories = {
            "assertion_errors": 0,
            "import_errors": 0,
            "attribute_errors": 0,
            "type_errors": 0,
            "value_errors": 0,
            "syntax_errors": 0,
            "other_errors": 0
        }
        
        for error in error_details:
            error_lower = error.lower()
            
            if "assertionerror" in error_lower or "assert" in error_lower:
                categories["assertion_errors"] += 1
            elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
                categories["import_errors"] += 1
            elif "attributeerror" in error_lower:
                categories["attribute_errors"] += 1
            elif "typeerror" in error_lower:
                categories["type_errors"] += 1
            elif "valueerror" in error_lower:
                categories["value_errors"] += 1
            elif "syntaxerror" in error_lower or "indentationerror" in error_lower:
                categories["syntax_errors"] += 1
            else:
                categories["other_errors"] += 1
        
        return categories
    
    def _generate_basic_recommendations(self, failure_types: Dict[str, int]) -> List[str]:
        """
        Generate basic recommendations based on failure types.
        
        Args:
            failure_types: Dictionary of failure type counts
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if failure_types["assertion_errors"] > 0:
            recommendations.append("Review test assertions and ensure expected values match actual implementation")
        
        if failure_types["import_errors"] > 0:
            recommendations.append("Check import statements and ensure all required modules are available")
        
        if failure_types["attribute_errors"] > 0:
            recommendations.append("Verify that objects have the expected attributes and methods")
        
        if failure_types["type_errors"] > 0:
            recommendations.append("Check function arguments and return types for compatibility")
        
        if failure_types["value_errors"] > 0:
            recommendations.append("Validate input values and handle edge cases properly")
        
        if failure_types["syntax_errors"] > 0:
            recommendations.append("Fix syntax errors and indentation issues in the code")
        
        return recommendations
    
    def _analyze_failures_with_llm(self, test_result: TestResult) -> Optional[Dict[str, Any]]:
        """
        Use LLM to analyze test failures and provide insights.
        
        Args:
            test_result: TestResult containing failure information
            
        Returns:
            Dictionary containing LLM analysis, or None if analysis fails
        """
        if not self.llm_fn or not test_result.error_details:
            return None
        
        try:
            # Prepare prompt for LLM analysis
            prompt = self._prepare_failure_analysis_prompt(test_result)
            
            # Call LLM for analysis
            response = call_llm(
                agent_name=self.agent_name,
                model=self.model,
                action=ActionType.DEBUG,
                prompt=prompt,
                llm_fn=self.llm_fn
            )
            
            # Parse LLM response
            analysis = self._parse_llm_analysis_response(response)
            
            return analysis
            
        except Exception as e:
            # Log error but don't fail - LLM analysis is optional
            print(f"Warning: LLM failure analysis failed: {e}")
            return None
    
    def _prepare_failure_analysis_prompt(self, test_result: TestResult) -> str:
        """
        Prepare a prompt for LLM-based failure analysis.
        
        Args:
            test_result: TestResult containing failure information
            
        Returns:
            Formatted prompt for LLM
        """
        error_summary = "\n".join(test_result.error_details[:5])  # Limit to first 5 errors
        
        prompt = f"""Analyze the following test failures and provide insights for fixing the code.

Test Execution Summary:
- Total Tests: {test_result.total_tests}
- Failed Tests: {test_result.failed_tests}
- Execution Time: {test_result.execution_time:.2f}s

Test Failures:
{error_summary}

Please analyze these failures and provide:
1. Root cause analysis for the most common failure patterns
2. Specific recommendations for fixing the code (not the tests)
3. Priority order for addressing the issues
4. Any potential side effects to consider when making fixes

Format your response as JSON with the following structure:
{{
    "root_causes": ["cause1", "cause2", ...],
    "recommendations": ["rec1", "rec2", ...],
    "priority_order": ["high_priority_issue", "medium_priority_issue", ...],
    "side_effects": ["potential_side_effect1", ...]
}}"""
        
        return prompt
    
    def _parse_llm_analysis_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response for failure analysis.
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary containing parsed analysis
        """
        import json
        
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                analysis = json.loads(json_str)
                
                # Validate expected keys
                expected_keys = ["root_causes", "recommendations", "priority_order", "side_effects"]
                for key in expected_keys:
                    if key not in analysis:
                        analysis[key] = []
                
                return analysis
        
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: create basic analysis from text
        return {
            "root_causes": ["Unable to parse detailed analysis"],
            "recommendations": [response[:200] + "..." if len(response) > 200 else response],
            "priority_order": [],
            "side_effects": []
        }
    
    def _format_feedback_for_fixer(self, analysis: Dict[str, Any], test_result: TestResult) -> str:
        """
        Format analysis results into feedback for the Fixer Agent.
        
        Args:
            analysis: Failure analysis results
            test_result: Original test results
            
        Returns:
            Formatted feedback string
        """
        feedback = f"=== Test Execution Feedback ===\n\n"
        
        # Summary
        feedback += f"Test Results Summary:\n"
        feedback += f"- Total Tests: {test_result.total_tests}\n"
        feedback += f"- Failed Tests: {test_result.failed_tests}\n"
        feedback += f"- Success Rate: {test_result.success_rate:.1f}%\n"
        feedback += f"- Execution Time: {test_result.execution_time:.2f}s\n\n"
        
        # Failure categories
        if analysis.get("failure_categories"):
            feedback += "Failure Categories:\n"
            for category, count in analysis["failure_categories"].items():
                if count > 0:
                    feedback += f"- {category.replace('_', ' ').title()}: {count}\n"
            feedback += "\n"
        
        # Recommendations
        if analysis.get("recommendations"):
            feedback += "Recommendations for Fixing:\n"
            for i, rec in enumerate(analysis["recommendations"], 1):
                feedback += f"{i}. {rec}\n"
            feedback += "\n"
        
        # LLM insights if available
        if analysis.get("llm_insights"):
            llm_insights = analysis["llm_insights"]
            
            if llm_insights.get("root_causes"):
                feedback += "Root Cause Analysis:\n"
                for cause in llm_insights["root_causes"]:
                    feedback += f"- {cause}\n"
                feedback += "\n"
            
            if llm_insights.get("priority_order"):
                feedback += "Priority Order for Fixes:\n"
                for i, priority in enumerate(llm_insights["priority_order"], 1):
                    feedback += f"{i}. {priority}\n"
                feedback += "\n"
        
        # Detailed errors (limited)
        if test_result.error_details:
            feedback += "Detailed Error Information:\n"
            for i, error in enumerate(test_result.error_details[:3], 1):  # Show first 3 errors
                feedback += f"{i}. {error}\n"
                feedback += "---\n"
        
        return feedback
    
    def get_test_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of test execution history.
        
        Returns:
            Dictionary containing test history statistics
        """
        if not self.test_history:
            return {"total_executions": 0, "success_rate": 0.0}
        
        total_executions = len(self.test_history)
        successful_executions = sum(1 for record in self.test_history if record["result"]["passed"])
        
        total_tests = sum(record["result"]["total_tests"] for record in self.test_history)
        total_failures = sum(record["result"]["failed_tests"] for record in self.test_history)
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "execution_success_rate": (successful_executions / total_executions) * 100,
            "total_tests_run": total_tests,
            "total_test_failures": total_failures,
            "overall_test_success_rate": ((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0.0,
            "recent_executions": self.test_history[-5:]  # Last 5 executions
        }
    
    def clear_test_history(self) -> None:
        """Clear the test execution history."""
        self.test_history.clear()
    
    # Self-healing feedback loop functionality
    
    def create_feedback_loop_state(self, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Create initial state for self-healing feedback loop.
        
        Args:
            max_iterations: Maximum number of iterations allowed
            
        Returns:
            Dictionary containing feedback loop state
        """
        return {
            "iteration_count": 0,
            "max_iterations": max_iterations,
            "status": "active",
            "test_results_history": [],
            "feedback_history": [],
            "success_achieved": False,
            "termination_reason": None
        }
    
    def process_feedback_loop_iteration(self, loop_state: Dict[str, Any], 
                                       test_result: TestResult) -> Dict[str, Any]:
        """
        Process a single iteration of the self-healing feedback loop.
        
        Args:
            loop_state: Current state of the feedback loop
            test_result: Results from the latest test execution
            
        Returns:
            Updated loop state with iteration results
        """
        try:
            # Increment iteration count
            loop_state["iteration_count"] += 1
            
            # Record test results
            loop_state["test_results_history"].append({
                "iteration": loop_state["iteration_count"],
                "result": test_result.to_dict(),
                "timestamp": f"iteration_{loop_state['iteration_count']}"
            })
            
            # Check for success
            if test_result.passed:
                loop_state["success_achieved"] = True
                loop_state["status"] = "completed"
                loop_state["termination_reason"] = "success"
                
                # Log successful completion
                log_experiment(
                    agent_name=self.agent_name,
                    model_used=self.model,
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": f"Self-healing loop completed successfully after {loop_state['iteration_count']} iterations",
                        "output_response": f"All {test_result.total_tests} tests passed"
                    },
                    status="SUCCESS"
                )
                
                return loop_state
            
            # Check for iteration limit
            if loop_state["iteration_count"] >= loop_state["max_iterations"]:
                loop_state["status"] = "terminated"
                loop_state["termination_reason"] = "max_iterations_reached"
                
                # Log termination due to iteration limit
                log_experiment(
                    agent_name=self.agent_name,
                    model_used=self.model,
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": f"Self-healing loop terminated after {loop_state['iteration_count']} iterations",
                        "output_response": f"Maximum iterations reached. {test_result.failed_tests} tests still failing"
                    },
                    status="FAILURE"
                )
                
                return loop_state
            
            # Generate feedback for next iteration
            feedback = self.generate_detailed_feedback_for_iteration(
                test_result, loop_state["iteration_count"]
            )
            
            loop_state["feedback_history"].append({
                "iteration": loop_state["iteration_count"],
                "feedback": feedback,
                "failed_tests": test_result.failed_tests,
                "error_count": len(test_result.error_details)
            })
            
            # Log iteration completion
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Self-healing loop iteration {loop_state['iteration_count']} completed",
                    "output_response": f"Generated feedback for {test_result.failed_tests} failed tests"
                },
                status="SUCCESS"
            )
            
            return loop_state
            
        except Exception as e:
            loop_state["status"] = "error"
            loop_state["termination_reason"] = f"error: {str(e)}"
            
            log_experiment(
                agent_name=self.agent_name,
                model_used=self.model,
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Self-healing loop iteration {loop_state['iteration_count']} failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            
            return loop_state
    
    def generate_detailed_feedback_for_iteration(self, test_result: TestResult, 
                                                iteration: int) -> str:
        """
        Generate detailed feedback for a specific iteration of the self-healing loop.
        
        Args:
            test_result: Test results from current iteration
            iteration: Current iteration number
            
        Returns:
            Detailed feedback string for the Fixer Agent
        """
        try:
            # Analyze failures
            analysis = self.analyze_test_failures(test_result)
            
            # Create iteration-specific feedback
            feedback = f"=== Self-Healing Loop Iteration {iteration} Feedback ===\n\n"
            
            # Add iteration context
            feedback += f"Iteration: {iteration}\n"
            feedback += f"Status: {'FAILED' if not test_result.passed else 'PASSED'}\n"
            feedback += f"Failed Tests: {test_result.failed_tests}/{test_result.total_tests}\n"
            feedback += f"Execution Time: {test_result.execution_time:.2f}s\n\n"
            
            # Add failure analysis
            if not test_result.passed:
                feedback += "=== FAILURE ANALYSIS ===\n"
                
                # Failure categories
                if analysis.get("failure_categories"):
                    feedback += "Failure Types:\n"
                    for category, count in analysis["failure_categories"].items():
                        if count > 0:
                            feedback += f"- {category.replace('_', ' ').title()}: {count}\n"
                    feedback += "\n"
                
                # Priority recommendations
                feedback += "=== PRIORITY FIXES NEEDED ===\n"
                if analysis.get("recommendations"):
                    for i, rec in enumerate(analysis["recommendations"][:3], 1):  # Top 3 recommendations
                        feedback += f"{i}. {rec}\n"
                    feedback += "\n"
                
                # LLM insights for iteration-specific guidance
                if analysis.get("llm_insights"):
                    llm_insights = analysis["llm_insights"]
                    
                    if llm_insights.get("root_causes"):
                        feedback += "Root Causes to Address:\n"
                        for cause in llm_insights["root_causes"][:2]:  # Top 2 causes
                            feedback += f"- {cause}\n"
                        feedback += "\n"
                
                # Specific error details for this iteration
                feedback += "=== SPECIFIC ERRORS TO FIX ===\n"
                for i, error in enumerate(test_result.error_details[:2], 1):  # Top 2 errors
                    feedback += f"Error {i}:\n{error}\n"
                    feedback += "---\n"
                
                # Iteration-specific guidance
                feedback += f"=== ITERATION {iteration} GUIDANCE ===\n"
                if iteration == 1:
                    feedback += "- Focus on critical syntax and import errors first\n"
                    feedback += "- Ensure basic functionality is working\n"
                elif iteration <= 3:
                    feedback += "- Address logic errors and assertion failures\n"
                    feedback += "- Check function signatures and return values\n"
                elif iteration <= 6:
                    feedback += "- Fine-tune edge cases and error handling\n"
                    feedback += "- Review test expectations vs implementation\n"
                else:
                    feedback += "- Consider if tests need adjustment (but prefer fixing code)\n"
                    feedback += "- Focus on remaining stubborn failures\n"
                    feedback += "- Check for complex interaction issues\n"
                
                feedback += "\n"
            
            return feedback
            
        except Exception as e:
            return f"Error generating iteration feedback: {str(e)}"
    
    def should_continue_loop(self, loop_state: Dict[str, Any]) -> bool:
        """
        Determine if the self-healing loop should continue.
        
        Args:
            loop_state: Current state of the feedback loop
            
        Returns:
            True if loop should continue, False otherwise
        """
        # Don't continue if already completed or terminated
        if loop_state["status"] in ["completed", "terminated", "error"]:
            return False
        
        # Don't continue if success achieved
        if loop_state["success_achieved"]:
            return False
        
        # Don't continue if max iterations reached
        if loop_state["iteration_count"] >= loop_state["max_iterations"]:
            return False
        
        return True
    
    def get_loop_termination_reason(self, loop_state: Dict[str, Any]) -> str:
        """
        Get a human-readable termination reason for the feedback loop.
        
        Args:
            loop_state: Final state of the feedback loop
            
        Returns:
            Human-readable termination reason
        """
        if loop_state["success_achieved"]:
            return f"Success: All tests passed after {loop_state['iteration_count']} iterations"
        
        if loop_state["termination_reason"] == "max_iterations_reached":
            return f"Terminated: Maximum iterations ({loop_state['max_iterations']}) reached without success"
        
        if loop_state["termination_reason"] and loop_state["termination_reason"].startswith("error"):
            return f"Error: {loop_state['termination_reason']}"
        
        return f"Unknown termination reason: {loop_state.get('termination_reason', 'undefined')}"
    
    def analyze_loop_progress(self, loop_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the progress of the self-healing loop.
        
        Args:
            loop_state: Current or final state of the feedback loop
            
        Returns:
            Dictionary containing progress analysis
        """
        if not loop_state["test_results_history"]:
            return {"status": "no_data", "message": "No test results available"}
        
        # Extract test results over iterations
        iterations = []
        for record in loop_state["test_results_history"]:
            iterations.append({
                "iteration": record["iteration"],
                "total_tests": record["result"]["total_tests"],
                "failed_tests": record["result"]["failed_tests"],
                "success_rate": record["result"]["success_rate"],
                "execution_time": record["result"]["execution_time"]
            })
        
        # Calculate progress metrics
        first_iteration = iterations[0]
        last_iteration = iterations[-1]
        
        progress_analysis = {
            "total_iterations": len(iterations),
            "initial_failures": first_iteration["failed_tests"],
            "final_failures": last_iteration["failed_tests"],
            "failures_reduced": first_iteration["failed_tests"] - last_iteration["failed_tests"],
            "improvement_rate": 0.0,
            "success_achieved": loop_state["success_achieved"],
            "termination_reason": self.get_loop_termination_reason(loop_state),
            "iterations_detail": iterations
        }
        
        # Calculate improvement rate
        if first_iteration["failed_tests"] > 0:
            progress_analysis["improvement_rate"] = (
                (first_iteration["failed_tests"] - last_iteration["failed_tests"]) / 
                first_iteration["failed_tests"] * 100
            )
        
        # Determine overall assessment
        if loop_state["success_achieved"]:
            progress_analysis["assessment"] = "successful"
        elif progress_analysis["failures_reduced"] > 0:
            progress_analysis["assessment"] = "improving"
        elif progress_analysis["failures_reduced"] == 0:
            progress_analysis["assessment"] = "stagnant"
        else:
            progress_analysis["assessment"] = "regressing"
        
        return progress_analysis
    
    def generate_loop_completion_report(self, loop_state: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report for the completed self-healing loop.
        
        Args:
            loop_state: Final state of the feedback loop
            
        Returns:
            Formatted completion report
        """
        progress = self.analyze_loop_progress(loop_state)
        
        report = "=== SELF-HEALING LOOP COMPLETION REPORT ===\n\n"
        
        # Summary
        report += f"Final Status: {loop_state['status'].upper()}\n"
        report += f"Success Achieved: {'YES' if loop_state['success_achieved'] else 'NO'}\n"
        report += f"Total Iterations: {loop_state['iteration_count']}\n"
        report += f"Termination Reason: {progress['termination_reason']}\n\n"
        
        # Progress metrics
        report += "=== PROGRESS METRICS ===\n"
        report += f"Initial Failures: {progress['initial_failures']}\n"
        report += f"Final Failures: {progress['final_failures']}\n"
        report += f"Failures Reduced: {progress['failures_reduced']}\n"
        report += f"Improvement Rate: {progress['improvement_rate']:.1f}%\n"
        report += f"Overall Assessment: {progress['assessment'].upper()}\n\n"
        
        # Iteration details
        if progress["iterations_detail"]:
            report += "=== ITERATION HISTORY ===\n"
            for iteration in progress["iterations_detail"]:
                report += f"Iteration {iteration['iteration']}: "
                report += f"{iteration['failed_tests']}/{iteration['total_tests']} failed "
                report += f"({iteration['success_rate']:.1f}% success) "
                report += f"in {iteration['execution_time']:.2f}s\n"
            report += "\n"
        
        # Recommendations for future
        report += "=== RECOMMENDATIONS ===\n"
        if loop_state["success_achieved"]:
            report += "- All tests are now passing\n"
            report += "- Consider running additional validation tests\n"
            report += "- Monitor for regressions in future changes\n"
        else:
            report += "- Review remaining test failures manually\n"
            report += "- Consider adjusting test expectations if appropriate\n"
            report += "- Analyze if additional iterations might help\n"
            if progress["assessment"] == "stagnant":
                report += "- No progress in recent iterations - may need different approach\n"
        
        return report