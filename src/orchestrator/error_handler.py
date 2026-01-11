"""
Error handling and recovery mechanisms for the Refactoring Swarm orchestrator.

This module provides comprehensive error handling, graceful failure modes,
cleanup procedures, and workflow state persistence and recovery.
"""

import os
import json
import traceback
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
from pathlib import Path

from ..models.core import AgentState, AgentStatus, RefactoringPlan, TestResult
from ..models.state_manager import StateManager, StateValidationError
from ..security.sandbox import SandboxManager
from ..utils.logger import log_experiment, ActionType
from ..exceptions import RefactoringError, AnalysisError, FixingError, TestingError


class ErrorSeverity:
    """Error severity levels for classification."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ErrorCategory:
    """Error categories for classification."""
    SYSTEM = "system"
    AGENT = "agent"
    WORKFLOW = "workflow"
    VALIDATION = "validation"
    RESOURCE = "resource"
    EXTERNAL = "external"


class WorkflowError:
    """Represents a workflow error with context and recovery information."""
    
    def __init__(self, error: Exception, context: Dict[str, Any], 
                 severity: str = ErrorSeverity.MEDIUM, 
                 category: str = ErrorCategory.WORKFLOW,
                 recoverable: bool = True):
        """
        Initialize workflow error.
        
        Args:
            error: The original exception
            context: Context information when error occurred
            severity: Error severity level
            category: Error category
            recoverable: Whether error is recoverable
        """
        self.error = error
        self.error_type = type(error).__name__
        self.message = str(error)
        self.context = context
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.timestamp = datetime.now().isoformat()
        self.traceback = traceback.format_exc()
        self.recovery_attempted = False
        self.recovery_successful = False
        self.recovery_details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "context": self.context,
            "severity": self.severity,
            "category": self.category,
            "recoverable": self.recoverable,
            "timestamp": self.timestamp,
            "traceback": self.traceback,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "recovery_details": self.recovery_details
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowError':
        """Create WorkflowError from dictionary."""
        # Create a generic exception from the stored data
        error = Exception(data["message"])
        
        workflow_error = cls(
            error=error,
            context=data["context"],
            severity=data["severity"],
            category=data["category"],
            recoverable=data["recoverable"]
        )
        
        # Restore additional fields
        workflow_error.error_type = data["error_type"]
        workflow_error.timestamp = data["timestamp"]
        workflow_error.traceback = data["traceback"]
        workflow_error.recovery_attempted = data["recovery_attempted"]
        workflow_error.recovery_successful = data["recovery_successful"]
        workflow_error.recovery_details = data["recovery_details"]
        
        return workflow_error


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies."""
    
    def can_recover(self, error: WorkflowError, state: Dict[str, Any]) -> bool:
        """
        Check if this strategy can recover from the error.
        
        Args:
            error: The workflow error
            state: Current workflow state
            
        Returns:
            True if recovery is possible
        """
        raise NotImplementedError
    
    def recover(self, error: WorkflowError, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to recover from the error.
        
        Args:
            error: The workflow error
            state: Current workflow state
            
        Returns:
            Updated state after recovery attempt
        """
        raise NotImplementedError


class RetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that retries the failed operation."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        """
        Initialize retry strategy.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff_factor: Exponential backoff factor
        """
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def can_recover(self, error: WorkflowError, state: Dict[str, Any]) -> bool:
        """Check if retry recovery is applicable."""
        # Don't retry critical errors or validation errors
        if error.severity == ErrorSeverity.CRITICAL:
            return False
        
        if error.category == ErrorCategory.VALIDATION:
            return False
        
        # Check retry count
        retry_count = state.get("retry_count", 0)
        return retry_count < self.max_retries
    
    def recover(self, error: WorkflowError, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery by retrying."""
        retry_count = state.get("retry_count", 0) + 1
        
        # Update state for retry
        state["retry_count"] = retry_count
        state["last_retry_error"] = error.to_dict()
        state["should_continue"] = True
        
        # Clear error flags to allow retry
        if "last_error" in state:
            state["last_error"] = ""
        if "error_count" in state:
            state["error_count"] = max(0, state["error_count"] - 1)
        
        error.recovery_attempted = True
        error.recovery_details = {
            "strategy": "retry",
            "attempt": retry_count,
            "max_retries": self.max_retries
        }
        
        return state


class RollbackStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that rolls back to a previous state."""
    
    def can_recover(self, error: WorkflowError, state: Dict[str, Any]) -> bool:
        """Check if rollback recovery is applicable."""
        # Can rollback if we have state history
        return "state_history" in state and len(state["state_history"]) > 0
    
    def recover(self, error: WorkflowError, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery by rolling back to previous state."""
        state_history = state.get("state_history", [])
        
        if not state_history:
            return state
        
        # Rollback to the last known good state
        previous_state = state_history[-1].copy()
        
        # Preserve error information
        previous_state["rollback_error"] = error.to_dict()
        previous_state["rollback_timestamp"] = datetime.now().isoformat()
        
        # Update recovery information
        error.recovery_attempted = True
        error.recovery_details = {
            "strategy": "rollback",
            "rollback_to": previous_state.get("timestamp", "unknown"),
            "states_available": len(state_history)
        }
        
        return previous_state


class SkipStrategy(ErrorRecoveryStrategy):
    """Recovery strategy that skips the failed operation and continues."""
    
    def can_recover(self, error: WorkflowError, state: Dict[str, Any]) -> bool:
        """Check if skip recovery is applicable."""
        # Can skip non-critical errors in certain categories
        if error.severity in [ErrorSeverity.LOW, ErrorSeverity.INFO]:
            return True
        
        # Can skip individual file processing errors
        if error.category == ErrorCategory.AGENT and "file_path" in error.context:
            return True
        
        return False
    
    def recover(self, error: WorkflowError, state: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt recovery by skipping the failed operation."""
        # Mark the operation as skipped
        skipped_operations = state.get("skipped_operations", [])
        skipped_operations.append({
            "error": error.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "context": error.context
        })
        state["skipped_operations"] = skipped_operations
        
        # Continue with workflow
        state["should_continue"] = True
        
        # Clear error flags
        if "last_error" in state:
            state["last_error"] = ""
        
        error.recovery_attempted = True
        error.recovery_details = {
            "strategy": "skip",
            "skipped_context": error.context
        }
        
        return state


class WorkflowErrorHandler:
    """
    Comprehensive error handler for workflow orchestration.
    
    Provides error classification, recovery strategies, cleanup procedures,
    and state persistence for workflow recovery.
    """
    
    def __init__(self, state_persistence_dir: str = ".refactoring_state"):
        """
        Initialize error handler.
        
        Args:
            state_persistence_dir: Directory for state persistence
        """
        self.state_persistence_dir = state_persistence_dir
        self.recovery_strategies: List[ErrorRecoveryStrategy] = [
            RetryStrategy(max_retries=3),
            RollbackStrategy(),
            SkipStrategy()
        ]
        self.error_history: List[WorkflowError] = []
        self.cleanup_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Ensure persistence directory exists
        os.makedirs(state_persistence_dir, exist_ok=True)
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an error with classification and recovery attempts.
        
        Args:
            error: The exception that occurred
            context: Context information when error occurred
            state: Current workflow state
            
        Returns:
            Updated state after error handling
        """
        try:
            # Classify the error
            workflow_error = self._classify_error(error, context)
            
            # Log the error
            self._log_error(workflow_error, context)
            
            # Add to error history
            self.error_history.append(workflow_error)
            
            # Persist state before recovery attempt
            self._persist_state(state, f"error_{workflow_error.timestamp}")
            
            # Attempt recovery
            recovered_state = self._attempt_recovery(workflow_error, state)
            
            # Update error with recovery results
            workflow_error.recovery_successful = recovered_state.get("should_continue", False)
            
            return recovered_state
            
        except Exception as handler_error:
            # If error handling itself fails, create a critical error
            critical_error = WorkflowError(
                error=handler_error,
                context={"original_error": str(error), "handler_context": context},
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.SYSTEM,
                recoverable=False
            )
            
            self._log_error(critical_error, context)
            
            # Return state with critical failure
            state["should_continue"] = False
            state["critical_error"] = critical_error.to_dict()
            
            return state
    
    def _classify_error(self, error: Exception, context: Dict[str, Any]) -> WorkflowError:
        """
        Classify an error based on type and context.
        
        Args:
            error: The exception
            context: Context information
            
        Returns:
            Classified WorkflowError
        """
        # Determine error category and severity based on exception type
        if isinstance(error, RefactoringError):
            if isinstance(error, AnalysisError):
                category = ErrorCategory.AGENT
                severity = ErrorSeverity.HIGH
            elif isinstance(error, FixingError):
                category = ErrorCategory.AGENT
                severity = ErrorSeverity.HIGH
            elif isinstance(error, TestingError):
                category = ErrorCategory.AGENT
                severity = ErrorSeverity.MEDIUM
            else:
                category = ErrorCategory.WORKFLOW
                severity = ErrorSeverity.MEDIUM
        elif isinstance(error, StateValidationError):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (OSError, IOError, FileNotFoundError)):
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.HIGH
        elif isinstance(error, (MemoryError, SystemError)):
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.CRITICAL
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.EXTERNAL
            severity = ErrorSeverity.MEDIUM
        else:
            category = ErrorCategory.WORKFLOW
            severity = ErrorSeverity.MEDIUM
        
        # Adjust severity based on context
        if context.get("iteration_count", 0) > 8:
            # Errors late in the process are more severe
            if severity == ErrorSeverity.MEDIUM:
                severity = ErrorSeverity.HIGH
        
        if context.get("critical_operation", False):
            # Critical operations have higher severity
            if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                severity = ErrorSeverity.HIGH
        
        # Determine if error is recoverable
        recoverable = severity != ErrorSeverity.CRITICAL and category != ErrorCategory.SYSTEM
        
        return WorkflowError(
            error=error,
            context=context,
            severity=severity,
            category=category,
            recoverable=recoverable
        )
    
    def _log_error(self, workflow_error: WorkflowError, context: Dict[str, Any]) -> None:
        """
        Log error information for debugging and monitoring.
        
        Args:
            workflow_error: The classified workflow error
            context: Additional context information
        """
        log_experiment(
            agent_name="ErrorHandler",
            model_used="N/A",
            action=ActionType.DEBUG,
            details={
                "input_prompt": f"Error occurred: {workflow_error.error_type}",
                "output_response": f"Severity: {workflow_error.severity}, Category: {workflow_error.category}, Recoverable: {workflow_error.recoverable}"
            },
            status="FAILURE"
        )
        
        # Log detailed error information
        error_details = {
            "error_type": workflow_error.error_type,
            "message": workflow_error.message,
            "severity": workflow_error.severity,
            "category": workflow_error.category,
            "recoverable": workflow_error.recoverable,
            "context": context,
            "timestamp": workflow_error.timestamp
        }
        
        # Write detailed error log
        error_log_path = os.path.join(self.state_persistence_dir, "error_log.json")
        try:
            if os.path.exists(error_log_path):
                with open(error_log_path, 'r') as f:
                    error_log = json.load(f)
            else:
                error_log = []
            
            error_log.append(error_details)
            
            # Keep only last 100 errors
            if len(error_log) > 100:
                error_log = error_log[-100:]
            
            with open(error_log_path, 'w') as f:
                json.dump(error_log, f, indent=2)
                
        except Exception as log_error:
            # If logging fails, just continue - don't fail the error handler
            print(f"Warning: Failed to write error log: {log_error}")
    
    def _attempt_recovery(self, workflow_error: WorkflowError, 
                         state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to recover from the error using available strategies.
        
        Args:
            workflow_error: The classified workflow error
            state: Current workflow state
            
        Returns:
            Updated state after recovery attempt
        """
        if not workflow_error.recoverable:
            # Mark as unrecoverable and stop workflow
            state["should_continue"] = False
            state["unrecoverable_error"] = workflow_error.to_dict()
            return state
        
        # Try each recovery strategy
        for strategy in self.recovery_strategies:
            try:
                if strategy.can_recover(workflow_error, state):
                    log_experiment(
                        agent_name="ErrorHandler",
                        model_used="N/A",
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": f"Attempting recovery with {type(strategy).__name__}",
                            "output_response": f"Error: {workflow_error.error_type}"
                        },
                        status="SUCCESS"
                    )
                    
                    recovered_state = strategy.recover(workflow_error, state)
                    
                    # If recovery was successful, return the recovered state
                    if recovered_state.get("should_continue", False):
                        workflow_error.recovery_successful = True
                        return recovered_state
                    
            except Exception as recovery_error:
                # If recovery strategy fails, log and try next strategy
                log_experiment(
                    agent_name="ErrorHandler",
                    model_used="N/A",
                    action=ActionType.DEBUG,
                    details={
                        "input_prompt": f"Recovery strategy {type(strategy).__name__} failed",
                        "output_response": f"Recovery error: {str(recovery_error)}"
                    },
                    status="FAILURE"
                )
                continue
        
        # If no recovery strategy worked, mark as failed
        state["should_continue"] = False
        state["recovery_failed"] = workflow_error.to_dict()
        
        return state
    
    def _persist_state(self, state: Dict[str, Any], checkpoint_name: str) -> None:
        """
        Persist workflow state for recovery purposes.
        
        Args:
            state: Current workflow state
            checkpoint_name: Name for the checkpoint
        """
        try:
            checkpoint_path = os.path.join(
                self.state_persistence_dir, 
                f"checkpoint_{checkpoint_name}.json"
            )
            
            # Create a serializable version of the state
            serializable_state = self._make_state_serializable(state)
            
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    "checkpoint_name": checkpoint_name,
                    "timestamp": datetime.now().isoformat(),
                    "state": serializable_state
                }, f, indent=2)
            
            # Keep only last 10 checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as persist_error:
            # If persistence fails, log but don't fail the workflow
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "State persistence failed",
                    "output_response": f"Error: {str(persist_error)}"
                },
                status="FAILURE"
            )
    
    def _make_state_serializable(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert state to a JSON-serializable format.
        
        Args:
            state: Original state dictionary
            
        Returns:
            Serializable state dictionary
        """
        serializable_state = {}
        
        for key, value in state.items():
            try:
                # Handle common non-serializable objects
                if hasattr(value, 'to_dict'):
                    serializable_state[key] = value.to_dict()
                elif hasattr(value, '__dict__'):
                    # For objects with __dict__, try to serialize their attributes
                    serializable_state[key] = {
                        "type": type(value).__name__,
                        "attributes": {k: v for k, v in value.__dict__.items() 
                                     if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                    }
                elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_state[key] = value
                else:
                    # For other types, store as string representation
                    serializable_state[key] = {
                        "type": type(value).__name__,
                        "string_repr": str(value)
                    }
            except Exception:
                # If serialization fails for this key, skip it
                serializable_state[key] = f"<non-serializable: {type(value).__name__}>"
        
        return serializable_state
    
    def _cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoint files to prevent disk space issues."""
        try:
            checkpoint_files = []
            for file in os.listdir(self.state_persistence_dir):
                if file.startswith("checkpoint_") and file.endswith(".json"):
                    file_path = os.path.join(self.state_persistence_dir, file)
                    checkpoint_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            checkpoint_files.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the 10 most recent checkpoints
            for file_path, _ in checkpoint_files[10:]:
                os.remove(file_path)
                
        except Exception as cleanup_error:
            # If cleanup fails, just log and continue
            print(f"Warning: Checkpoint cleanup failed: {cleanup_error}")
    
    def recover_from_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Recover workflow state from a checkpoint.
        
        Args:
            checkpoint_name: Specific checkpoint to recover from (optional)
            
        Returns:
            Recovered state dictionary or None if recovery fails
        """
        try:
            if checkpoint_name:
                checkpoint_path = os.path.join(
                    self.state_persistence_dir, 
                    f"checkpoint_{checkpoint_name}.json"
                )
            else:
                # Find the most recent checkpoint
                checkpoint_files = []
                for file in os.listdir(self.state_persistence_dir):
                    if file.startswith("checkpoint_") and file.endswith(".json"):
                        file_path = os.path.join(self.state_persistence_dir, file)
                        checkpoint_files.append((file_path, os.path.getmtime(file_path)))
                
                if not checkpoint_files:
                    return None
                
                # Get the most recent checkpoint
                checkpoint_path = max(checkpoint_files, key=lambda x: x[1])[0]
            
            if not os.path.exists(checkpoint_path):
                return None
            
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": f"Recovered from checkpoint: {checkpoint_data['checkpoint_name']}",
                    "output_response": f"Timestamp: {checkpoint_data['timestamp']}"
                },
                status="SUCCESS"
            )
            
            return checkpoint_data["state"]
            
        except Exception as recovery_error:
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Checkpoint recovery failed",
                    "output_response": f"Error: {str(recovery_error)}"
                },
                status="FAILURE"
            )
            return None
    
    def perform_graceful_cleanup(self, state: Dict[str, Any]) -> None:
        """
        Perform graceful cleanup when workflow terminates.
        
        Args:
            state: Final workflow state
        """
        try:
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Starting graceful cleanup",
                    "output_response": "Cleanup initiated"
                },
                status="SUCCESS"
            )
            
            # Call cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback(state)
                except Exception as callback_error:
                    log_experiment(
                        agent_name="ErrorHandler",
                        model_used="N/A",
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": "Cleanup callback failed",
                            "output_response": f"Error: {str(callback_error)}"
                        },
                        status="FAILURE"
                    )
            
            # Clean up sandbox if present
            if "sandbox_manager" in state and state["sandbox_manager"]:
                try:
                    state["sandbox_manager"].cleanup_sandbox()
                except Exception as sandbox_error:
                    log_experiment(
                        agent_name="ErrorHandler",
                        model_used="N/A",
                        action=ActionType.DEBUG,
                        details={
                            "input_prompt": "Sandbox cleanup failed",
                            "output_response": f"Error: {str(sandbox_error)}"
                        },
                        status="FAILURE"
                    )
            
            # Persist final state
            self._persist_state(state, "final_state")
            
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Graceful cleanup completed",
                    "output_response": "All cleanup operations finished"
                },
                status="SUCCESS"
            )
            
        except Exception as cleanup_error:
            log_experiment(
                agent_name="ErrorHandler",
                model_used="N/A",
                action=ActionType.DEBUG,
                details={
                    "input_prompt": "Graceful cleanup failed",
                    "output_response": f"Error: {str(cleanup_error)}"
                },
                status="FAILURE"
            )
    
    def add_cleanup_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a cleanup callback function.
        
        Args:
            callback: Function to call during cleanup
        """
        self.cleanup_callbacks.append(callback)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Error summary dictionary
        """
        if not self.error_history:
            return {"total_errors": 0, "error_categories": {}, "recovery_rate": 0.0}
        
        # Count errors by category and severity
        category_counts = {}
        severity_counts = {}
        recoverable_count = 0
        recovered_count = 0
        
        for error in self.error_history:
            # Count by category
            category_counts[error.category] = category_counts.get(error.category, 0) + 1
            
            # Count by severity
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
            
            # Count recovery statistics
            if error.recoverable:
                recoverable_count += 1
                if error.recovery_successful:
                    recovered_count += 1
        
        recovery_rate = (recovered_count / recoverable_count * 100) if recoverable_count > 0 else 0.0
        
        return {
            "total_errors": len(self.error_history),
            "error_categories": category_counts,
            "error_severities": severity_counts,
            "recoverable_errors": recoverable_count,
            "recovered_errors": recovered_count,
            "recovery_rate": recovery_rate,
            "recent_errors": [error.to_dict() for error in self.error_history[-5:]]
        }
    
    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()
        
        log_experiment(
            agent_name="ErrorHandler",
            model_used="N/A",
            action=ActionType.DEBUG,
            details={
                "input_prompt": "Cleared error history",
                "output_response": "Error history cleared successfully"
            },
            status="SUCCESS"
        )