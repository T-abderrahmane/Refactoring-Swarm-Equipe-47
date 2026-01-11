"""
Main orchestration logic for the Refactoring Swarm system.

This module provides the high-level orchestration interface that manages
agent coordination, execution management, and workflow monitoring.
"""

import os
import time
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

from .workflow import RefactoringWorkflow, WorkflowState
from .error_handler import WorkflowErrorHandler
from ..models.core import AgentState, AgentStatus, RefactoringPlan, TestResult
from ..models.state_manager import StateManager, StateValidationError
from ..security.sandbox import SandboxManager
from ..utils.logger import log_experiment, ActionType
from ..exceptions import RefactoringError


class RefactoringOrchestrator:
    """
    Main orchestrator for coordinating the refactoring workflow.
    
    Manages agent execution, iteration counting, loop prevention,
    and workflow monitoring with progress tracking.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash", llm_fn=None, 
                 max_iterations: int = 10, enable_monitoring: bool = True):
        """
        Initialize the refactoring orchestrator.
        
        Args:
            model: LLM model to use for all agents
            llm_fn: Function to call the LLM (for dependency injection)
            max_iterations: Maximum number of iterations for self-healing loop
            enable_monitoring: Whether to enable workflow monitoring
        """
        self.model = model
        self.llm_fn = llm_fn
        self.max_iterations = max_iterations
        self.enable_monitoring = enable_monitoring
        
        # Initialize workflow
        self.workflow = RefactoringWorkflow(
            model=model, 
            llm_fn=llm_fn, 
            max_iterations=max_iterations
        )
        
        # Initialize error handler
        self.error_handler = WorkflowErrorHandler()
        
        # Execution state
        self.current_session: Optional[Dict[str, Any]] = None
        self.execution_history: List[Dict[str, Any]] = []
        self.monitoring_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Performance tracking
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def execute_refactoring(self, target_directory: str, 
                          sandbox_directory: Optional[str] = None,
                          progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Execute the complete refactoring workflow.
        
        Args:
            target_directory: Directory containing Python files to refactor
            sandbox_directory: Directory for sandbox operations (optional)
            progress_callback: Callback function for progress updates
            
        Returns:
            Dictionary containing execution results and statistics
            
        Raises:
            RefactoringError: If execution fails
        """
        if not os.path.exists(target_directory):
            raise RefactoringError(f"Target directory not found: {target_directory}")
        
        if not os.path.isdir(target_directory):
            raise RefactoringError(f"Target path is not a directory: {target_directory}")
        
        # Set up sandbox directory
        if not sandbox_directory:
            sandbox_directory = os.path.join(target_directory, ".refactoring_sandbox")
        
        try:
            # Start execution timing
            self.start_time = time.time()
            
            # Initialize session
            session_id = f"refactoring_{int(time.time())}"
            self.current_session = self._create_session(
                session_id, target_directory, sandbox_directory
            )
            
            # Add progress callback if provided
            if progress_callback:
                self.add_monitoring_callback(progress_callback)
            
            # Find Python files to process
            python_files = self._find_python_files(target_directory)
            if not python_files:
                raise RefactoringError(f"No Python files found in {target_directory}")
            
            # Create initial agent state
            initial_state = AgentState(
                current_files=python_files,
                iteration_count=0,
                refactoring_plan=None,
                test_results=None,
                status=AgentStatus.IDLE,
                target_directory=target_directory,
                sandbox_directory=sandbox_directory,
                last_error=""
            )
            
            # Create workflow state
            workflow_state = WorkflowState({
                "agent_state": initial_state,
                "target_files": python_files,
                "session_id": session_id
            })
            
            # Execute workflow
            execution_result = self._execute_workflow(workflow_state)
            
            # Finalize session
            self.end_time = time.time()
            self._finalize_session(execution_result)
            
            return execution_result
            
        except Exception as e:
            self.end_time = time.time()
            
            # Use error handler for comprehensive error handling
            context = {
                "phase": "execution",
                "target_directory": target_directory,
                "sandbox_directory": sandbox_directory,
                "critical_operation": True
            }
            
            # Handle the error
            error_state = {"should_continue": False}
            error_state = self.error_handler.handle_error(e, context, error_state)
            
            # Create error result
            error_result = self._handle_execution_error(str(e))
            
            # Perform cleanup
            self.error_handler.perform_graceful_cleanup(error_state)
            
            return error_result
    
    def _execute_workflow(self, initial_state: WorkflowState) -> Dict[str, Any]:
        """
        Execute the LangGraph workflow with monitoring.
        
        Args:
            initial_state: Initial workflow state
            
        Returns:
            Dictionary containing execution results
        """
        try:
            # Create workflow graph
            workflow_graph = self.workflow.create_workflow()
            
            # Initialize monitoring
            if self.enable_monitoring:
                self._start_monitoring()
            
            # Execute workflow
            log_experiment(
                agent_name="Orchestrator",
                model_used=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Starting workflow execution",
                    "output_response": f"Processing {len(initial_state.get('target_files', []))} files"
                },
                status="SUCCESS"
            )
            
            # Run the workflow
            final_state = None
            iteration_count = 0
            
            # Execute workflow with state updates
            for state_update in workflow_graph.stream(
                initial_state, 
                config=self.workflow.workflow_config
            ):
                iteration_count += 1
                
                # Update monitoring
                if self.enable_monitoring:
                    self._update_monitoring(state_update, iteration_count)
                
                # Store final state
                final_state = state_update
                
                # Check for termination conditions
                if self._should_terminate_execution(state_update, iteration_count):
                    break
            
            # Process final results
            execution_result = self._process_final_results(final_state, iteration_count)
            
            log_experiment(
                agent_name="Orchestrator",
                model_used=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Workflow execution completed",
                    "output_response": f"Final status: {execution_result.get('status', 'unknown')}"
                },
                status="SUCCESS" if execution_result.get("success", False) else "FAILURE"
            )
            
            return execution_result
            
        except Exception as e:
            log_experiment(
                agent_name="Orchestrator",
                model_used=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Workflow execution failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
            raise RefactoringError(f"Workflow execution failed: {str(e)}")
    
    def _create_session(self, session_id: str, target_directory: str, 
                       sandbox_directory: str) -> Dict[str, Any]:
        """
        Create a new execution session.
        
        Args:
            session_id: Unique session identifier
            target_directory: Target directory path
            sandbox_directory: Sandbox directory path
            
        Returns:
            Session dictionary
        """
        session = {
            "session_id": session_id,
            "target_directory": target_directory,
            "sandbox_directory": sandbox_directory,
            "start_time": time.time(),
            "status": "running",
            "iterations": [],
            "errors": [],
            "progress": {
                "current_phase": "initializing",
                "completion_percentage": 0.0,
                "files_processed": 0,
                "total_files": 0
            }
        }
        
        log_experiment(
            agent_name="Orchestrator",
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Created session {session_id}",
                "output_response": f"Target: {target_directory}, Sandbox: {sandbox_directory}"
            },
            status="SUCCESS"
        )
        
        return session
    
    def _find_python_files(self, directory: str) -> List[str]:
        """
        Find all Python files in the target directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of Python file paths
        """
        python_files = []
        
        for root, dirs, files in os.walk(directory):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in [
                '.git', '__pycache__', '.venv', 'venv', '.pytest_cache',
                '.refactoring_sandbox', 'node_modules', '.tox'
            ]]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    python_files.append(file_path)
        
        return sorted(python_files)
    
    def _start_monitoring(self) -> None:
        """Start workflow monitoring."""
        if self.current_session:
            self.current_session["monitoring_started"] = time.time()
            
            log_experiment(
                agent_name="Orchestrator",
                model_used=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Started workflow monitoring",
                    "output_response": f"Session: {self.current_session['session_id']}"
                },
                status="SUCCESS"
            )
    
    def _update_monitoring(self, state_update: Dict[str, Any], iteration: int) -> None:
        """
        Update monitoring information with current state.
        
        Args:
            state_update: Current state update from workflow
            iteration: Current iteration number
        """
        if not self.current_session:
            return
        
        try:
            # Extract relevant information from state update
            current_phase = self._determine_current_phase(state_update)
            progress_info = self._calculate_progress(state_update, iteration)
            
            # Update session progress
            self.current_session["progress"].update({
                "current_phase": current_phase,
                "completion_percentage": progress_info["completion_percentage"],
                "current_iteration": iteration,
                "last_update": time.time()
            })
            
            # Record iteration details
            iteration_info = {
                "iteration": iteration,
                "phase": current_phase,
                "timestamp": time.time(),
                "state_summary": self._summarize_state(state_update)
            }
            
            self.current_session["iterations"].append(iteration_info)
            
            # Call monitoring callbacks
            for callback in self.monitoring_callbacks:
                try:
                    callback({
                        "session_id": self.current_session["session_id"],
                        "iteration": iteration,
                        "phase": current_phase,
                        "progress": progress_info,
                        "state": state_update
                    })
                except Exception as e:
                    # Log callback errors but don't fail monitoring
                    log_experiment(
                        agent_name="Orchestrator",
                        model_used=self.model,
                        action=ActionType.ANALYSIS,
                        details={
                            "input_prompt": "Monitoring callback failed",
                            "output_response": f"Error: {str(e)}"
                        },
                        status="FAILURE"
                    )
            
        except Exception as e:
            # Log monitoring errors but don't fail execution
            log_experiment(
                agent_name="Orchestrator",
                model_used=self.model,
                action=ActionType.ANALYSIS,
                details={
                    "input_prompt": "Monitoring update failed",
                    "output_response": f"Error: {str(e)}"
                },
                status="FAILURE"
            )
    
    def _determine_current_phase(self, state_update: Dict[str, Any]) -> str:
        """
        Determine the current workflow phase from state update.
        
        Args:
            state_update: Current state update
            
        Returns:
            Current phase name
        """
        # Extract phase information from state update
        # This is a simplified implementation - in practice, you'd examine
        # the actual state structure from LangGraph
        
        if "agent_state" in state_update:
            agent_state = state_update["agent_state"]
            if hasattr(agent_state, 'status'):
                status = agent_state.status
                if status == AgentStatus.ANALYZING:
                    return "auditing"
                elif status == AgentStatus.FIXING:
                    return "fixing"
                elif status == AgentStatus.TESTING:
                    return "testing"
                elif status == AgentStatus.COMPLETE:
                    return "completed"
                elif status == AgentStatus.FAILED:
                    return "failed"
        
        # Check for specific node execution
        for key in state_update.keys():
            if "audit" in key.lower():
                return "auditing"
            elif "fix" in key.lower():
                return "fixing"
            elif "test" in key.lower():
                return "testing"
            elif "evaluate" in key.lower():
                return "evaluating"
            elif "finalize" in key.lower():
                return "finalizing"
        
        return "processing"
    
    def _calculate_progress(self, state_update: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """
        Calculate progress information from current state.
        
        Args:
            state_update: Current state update
            iteration: Current iteration number
            
        Returns:
            Progress information dictionary
        """
        # Base progress calculation
        max_iterations = self.max_iterations
        iteration_progress = min((iteration / max_iterations) * 100, 100)
        
        # Phase-based progress adjustment
        current_phase = self._determine_current_phase(state_update)
        phase_weights = {
            "initializing": 5,
            "auditing": 20,
            "fixing": 50,
            "testing": 15,
            "evaluating": 5,
            "finalizing": 5,
            "completed": 100,
            "failed": 100
        }
        
        base_progress = phase_weights.get(current_phase, 0)
        
        # Combine iteration and phase progress
        completion_percentage = min(base_progress + (iteration_progress * 0.5), 100)
        
        return {
            "completion_percentage": completion_percentage,
            "current_iteration": iteration,
            "max_iterations": max_iterations,
            "phase": current_phase,
            "iteration_progress": iteration_progress
        }
    
    def _summarize_state(self, state_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the current state for monitoring.
        
        Args:
            state_update: Current state update
            
        Returns:
            State summary dictionary
        """
        summary = {
            "timestamp": time.time(),
            "keys": list(state_update.keys()),
            "has_errors": False,
            "error_count": 0
        }
        
        # Check for agent state information
        if "agent_state" in state_update:
            agent_state = state_update["agent_state"]
            if hasattr(agent_state, 'status'):
                summary["agent_status"] = agent_state.status.value
            if hasattr(agent_state, 'iteration_count'):
                summary["iteration_count"] = agent_state.iteration_count
            if hasattr(agent_state, 'last_error') and agent_state.last_error:
                summary["has_errors"] = True
                summary["last_error"] = agent_state.last_error
        
        # Check for test results
        if "test_results" in state_update:
            test_results = state_update["test_results"]
            if test_results:
                summary["test_info"] = {
                    "passed": test_results.passed,
                    "total_tests": test_results.total_tests,
                    "failed_tests": test_results.failed_tests
                }
        
        # Check for error indicators
        error_indicators = ["error", "failed", "exception"]
        for key, value in state_update.items():
            if any(indicator in key.lower() for indicator in error_indicators):
                summary["has_errors"] = True
                summary["error_count"] += 1
        
        return summary
    
    def _should_terminate_execution(self, state_update: Dict[str, Any], 
                                   iteration: int) -> bool:
        """
        Check if execution should be terminated.
        
        Args:
            state_update: Current state update
            iteration: Current iteration number
            
        Returns:
            True if execution should terminate
        """
        # Check iteration limit
        if iteration > self.max_iterations * 2:  # Safety margin
            return True
        
        # Check for completion or failure states
        if "agent_state" in state_update:
            agent_state = state_update["agent_state"]
            if hasattr(agent_state, 'status'):
                if agent_state.status in [AgentStatus.COMPLETE, AgentStatus.FAILED]:
                    return True
        
        # Check for explicit termination signals
        if state_update.get("should_continue") is False:
            return True
        
        # Check for critical errors
        if state_update.get("error_count", 0) > 5:  # Too many errors
            return True
        
        return False
    
    def _process_final_results(self, final_state: Dict[str, Any], 
                              iteration_count: int) -> Dict[str, Any]:
        """
        Process final workflow results into a comprehensive report.
        
        Args:
            final_state: Final workflow state
            iteration_count: Total number of iterations
            
        Returns:
            Comprehensive results dictionary
        """
        execution_time = (self.end_time or time.time()) - (self.start_time or time.time())
        
        # Extract key information from final state
        success = False
        agent_status = "unknown"
        test_results = None
        refactoring_plans = []
        error_message = ""
        
        if final_state:
            # Extract agent state
            if "agent_state" in final_state:
                agent_state = final_state["agent_state"]
                if hasattr(agent_state, 'status'):
                    agent_status = agent_state.status.value
                    success = agent_state.status == AgentStatus.COMPLETE
                if hasattr(agent_state, 'last_error'):
                    error_message = agent_state.last_error
            
            # Extract test results
            if "test_results" in final_state:
                test_results = final_state["test_results"]
            
            # Extract refactoring plans
            if "refactoring_plans" in final_state:
                refactoring_plans = final_state["refactoring_plans"]
        
        # Create comprehensive results
        results = {
            "success": success,
            "status": agent_status,
            "execution_time": execution_time,
            "total_iterations": iteration_count,
            "max_iterations": self.max_iterations,
            "session_id": self.current_session["session_id"] if self.current_session else None,
            
            # File processing results
            "files_processed": len(refactoring_plans),
            "total_files": len(self.current_session.get("target_files", [])) if self.current_session else 0,
            
            # Test results
            "test_results": {
                "passed": test_results.passed if test_results else False,
                "total_tests": test_results.total_tests if test_results else 0,
                "failed_tests": test_results.failed_tests if test_results else 0,
                "success_rate": test_results.success_rate if test_results else 0.0
            } if test_results else None,
            
            # Refactoring results
            "refactoring_summary": {
                "plans_generated": len(refactoring_plans),
                "total_issues": sum(len(plan.issues) for plan in refactoring_plans),
                "critical_issues": sum(len(plan.get_critical_issues()) for plan in refactoring_plans)
            } if refactoring_plans else None,
            
            # Error information
            "error_message": error_message,
            "has_errors": bool(error_message),
            
            # Performance metrics
            "performance": {
                "execution_time_seconds": execution_time,
                "iterations_per_second": iteration_count / execution_time if execution_time > 0 else 0,
                "files_per_second": len(refactoring_plans) / execution_time if execution_time > 0 else 0
            },
            
            # Session information
            "session_info": self.current_session.copy() if self.current_session else None
        }
        
        return results
    
    def _finalize_session(self, execution_result: Dict[str, Any]) -> None:
        """
        Finalize the current execution session.
        
        Args:
            execution_result: Results from workflow execution
        """
        if not self.current_session:
            return
        
        # Update session with final results
        self.current_session.update({
            "end_time": time.time(),
            "status": "completed" if execution_result["success"] else "failed",
            "final_results": execution_result,
            "total_execution_time": execution_result["execution_time"]
        })
        
        # Add to execution history
        self.execution_history.append(self.current_session.copy())
        
        # Keep only last 10 sessions to prevent memory issues
        if len(self.execution_history) > 10:
            self.execution_history = self.execution_history[-10:]
        
        log_experiment(
            agent_name="Orchestrator",
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Finalized session {self.current_session['session_id']}",
                "output_response": f"Status: {self.current_session['status']}, Duration: {execution_result['execution_time']:.2f}s"
            },
            status="SUCCESS" if execution_result["success"] else "FAILURE"
        )
        
        # Clear current session
        self.current_session = None
    
    def _handle_execution_error(self, error_message: str) -> Dict[str, Any]:
        """
        Handle execution errors and create error result.
        
        Args:
            error_message: Error message
            
        Returns:
            Error result dictionary
        """
        execution_time = (self.end_time or time.time()) - (self.start_time or time.time())
        
        error_result = {
            "success": False,
            "status": "error",
            "error_message": error_message,
            "execution_time": execution_time,
            "total_iterations": 0,
            "session_id": self.current_session["session_id"] if self.current_session else None,
            "files_processed": 0,
            "total_files": 0,
            "test_results": None,
            "refactoring_summary": None,
            "has_errors": True
        }
        
        # Update current session if exists
        if self.current_session:
            self.current_session.update({
                "status": "error",
                "error_message": error_message,
                "end_time": time.time()
            })
            self.execution_history.append(self.current_session.copy())
            self.current_session = None
        
        log_experiment(
            agent_name="Orchestrator",
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Execution error occurred",
                "output_response": f"Error: {error_message}"
            },
            status="FAILURE"
        )
        
        return error_result
    
    def add_monitoring_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a monitoring callback function.
        
        Args:
            callback: Function to call with monitoring updates
        """
        self.monitoring_callbacks.append(callback)
    
    def remove_monitoring_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Remove a monitoring callback function.
        
        Args:
            callback: Function to remove from callbacks
        """
        if callback in self.monitoring_callbacks:
            self.monitoring_callbacks.remove(callback)
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of execution sessions.
        
        Returns:
            List of execution session dictionaries
        """
        return self.execution_history.copy()
    
    def get_current_session_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current execution session.
        
        Returns:
            Current session dictionary or None if no active session
        """
        return self.current_session.copy() if self.current_session else None
    
    def clear_execution_history(self) -> None:
        """Clear the execution history."""
        self.execution_history.clear()
        
        log_experiment(
            agent_name="Orchestrator",
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": "Cleared execution history",
                "output_response": "History cleared successfully"
            },
            status="SUCCESS"
        )