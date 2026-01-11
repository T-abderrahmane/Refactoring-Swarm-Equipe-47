"""State validation and management utilities for the Refactoring Swarm system."""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from .core import AgentState, AgentStatus, RefactoringPlan, TestResult


class StateValidationError(Exception):
    """Raised when state validation fails."""
    pass


class StateManager:
    """Manages agent state persistence, validation, and transitions."""
    
    def __init__(self, state_file: Optional[str] = None):
        """Initialize state manager with optional state file path."""
        self.state_file = state_file or "agent_state.json"
        self._state: Optional[AgentState] = None
        self._state_history: List[AgentState] = []
    
    def create_initial_state(self, target_directory: str, sandbox_directory: str, 
                           current_files: List[str]) -> AgentState:
        """Create initial agent state."""
        state = AgentState(
            current_files=current_files,
            iteration_count=0,
            refactoring_plan=None,
            test_results=None,
            status=AgentStatus.IDLE,
            target_directory=target_directory,
            sandbox_directory=sandbox_directory,
            last_error=""
        )
        self._state = state
        self._add_to_history(state)
        return state
    
    def validate_state_transition(self, current_status: AgentStatus, 
                                 new_status: AgentStatus) -> bool:
        """Validate if a state transition is allowed."""
        valid_transitions = {
            AgentStatus.IDLE: [AgentStatus.ANALYZING],
            AgentStatus.ANALYZING: [AgentStatus.FIXING, AgentStatus.FAILED],
            AgentStatus.FIXING: [AgentStatus.TESTING, AgentStatus.FAILED],
            AgentStatus.TESTING: [AgentStatus.COMPLETE, AgentStatus.FIXING, AgentStatus.FAILED],
            AgentStatus.COMPLETE: [],  # Terminal state
            AgentStatus.FAILED: [AgentStatus.ANALYZING]  # Can retry from failed
        }
        
        allowed_next_states = valid_transitions.get(current_status, [])
        return new_status in allowed_next_states
    
    def transition_state(self, new_status: AgentStatus, 
                        error_message: str = "") -> AgentState:
        """Transition agent to new status with validation."""
        if not self._state:
            raise StateValidationError("No current state available for transition")
        
        if not self.validate_state_transition(self._state.status, new_status):
            raise StateValidationError(
                f"Invalid state transition from {self._state.status.value} to {new_status.value}"
            )
        
        # Create new state with updated status
        new_state = AgentState(
            current_files=self._state.current_files,
            iteration_count=self._state.iteration_count,
            refactoring_plan=self._state.refactoring_plan,
            test_results=self._state.test_results,
            status=new_status,
            target_directory=self._state.target_directory,
            sandbox_directory=self._state.sandbox_directory,
            last_error=error_message
        )
        
        self._state = new_state
        self._add_to_history(new_state)
        return new_state
    
    def update_refactoring_plan(self, plan: RefactoringPlan) -> AgentState:
        """Update the current refactoring plan."""
        if not self._state:
            raise StateValidationError("No current state available for update")
        
        new_state = AgentState(
            current_files=self._state.current_files,
            iteration_count=self._state.iteration_count,
            refactoring_plan=plan,
            test_results=self._state.test_results,
            status=self._state.status,
            target_directory=self._state.target_directory,
            sandbox_directory=self._state.sandbox_directory,
            last_error=self._state.last_error
        )
        
        self._state = new_state
        self._add_to_history(new_state)
        return new_state
    
    def update_test_results(self, results: TestResult) -> AgentState:
        """Update the current test results."""
        if not self._state:
            raise StateValidationError("No current state available for update")
        
        new_state = AgentState(
            current_files=self._state.current_files,
            iteration_count=self._state.iteration_count,
            refactoring_plan=self._state.refactoring_plan,
            test_results=results,
            status=self._state.status,
            target_directory=self._state.target_directory,
            sandbox_directory=self._state.sandbox_directory,
            last_error=self._state.last_error
        )
        
        self._state = new_state
        self._add_to_history(new_state)
        return new_state
    
    def increment_iteration(self) -> AgentState:
        """Increment iteration count and validate against limits."""
        if not self._state:
            raise StateValidationError("No current state available for iteration increment")
        
        new_iteration_count = self._state.iteration_count + 1
        
        # Check iteration limit
        if new_iteration_count > 10:
            raise StateValidationError("Maximum iteration limit (10) exceeded")
        
        new_state = AgentState(
            current_files=self._state.current_files,
            iteration_count=new_iteration_count,
            refactoring_plan=self._state.refactoring_plan,
            test_results=self._state.test_results,
            status=self._state.status,
            target_directory=self._state.target_directory,
            sandbox_directory=self._state.sandbox_directory,
            last_error=self._state.last_error
        )
        
        self._state = new_state
        self._add_to_history(new_state)
        return new_state
    
    def save_state(self, file_path: Optional[str] = None) -> None:
        """Persist current state to file."""
        if not self._state:
            raise StateValidationError("No state available to save")
        
        save_path = file_path or self.state_file
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "state": self._state.to_dict()
        }
        
        with open(save_path, 'w') as f:
            json.dump(state_data, f, indent=2)
    
    def load_state(self, file_path: Optional[str] = None) -> AgentState:
        """Load state from file."""
        load_path = file_path or self.state_file
        
        if not os.path.exists(load_path):
            raise StateValidationError(f"State file not found: {load_path}")
        
        with open(load_path, 'r') as f:
            state_data = json.load(f)
        
        self._state = AgentState.from_dict(state_data["state"])
        self._add_to_history(self._state)
        return self._state
    
    def get_current_state(self) -> Optional[AgentState]:
        """Get the current agent state."""
        return self._state
    
    def get_state_history(self) -> List[AgentState]:
        """Get the history of state changes."""
        return self._state_history.copy()
    
    def clear_state(self) -> None:
        """Clear current state and history."""
        self._state = None
        self._state_history.clear()
    
    def _add_to_history(self, state: AgentState) -> None:
        """Add state to history (internal method)."""
        # Keep only last 50 states to prevent memory issues
        if len(self._state_history) >= 50:
            self._state_history.pop(0)
        self._state_history.append(state)


class StateInspector:
    """Utilities for debugging and inspecting agent state."""
    
    @staticmethod
    def print_state_summary(state: AgentState) -> None:
        """Print a human-readable summary of the current state."""
        print(f"=== Agent State Summary ===")
        print(f"Status: {state.status.value}")
        print(f"Iteration: {state.iteration_count}/10")
        print(f"Files: {len(state.current_files)}")
        print(f"Target Directory: {state.target_directory}")
        print(f"Sandbox Directory: {state.sandbox_directory}")
        
        if state.refactoring_plan:
            print(f"Refactoring Plan: {len(state.refactoring_plan.issues)} issues")
        else:
            print("Refactoring Plan: None")
        
        if state.test_results:
            print(f"Test Results: {state.test_results.total_tests} tests, "
                  f"{state.test_results.failed_tests} failed")
        else:
            print("Test Results: None")
        
        if state.last_error:
            print(f"Last Error: {state.last_error}")
        
        print("=" * 27)
    
    @staticmethod
    def validate_state_integrity(state: AgentState) -> List[str]:
        """Validate state integrity and return list of issues."""
        issues = []
        
        # Check required fields
        if not state.current_files:
            issues.append("No current files specified")
        
        if not state.target_directory:
            issues.append("No target directory specified")
        
        if not state.sandbox_directory:
            issues.append("No sandbox directory specified")
        
        # Check iteration bounds
        if state.iteration_count < 0:
            issues.append("Negative iteration count")
        
        if state.iteration_count > 10:
            issues.append("Iteration count exceeds maximum (10)")
        
        # Check status consistency
        if state.status == AgentStatus.TESTING and not state.refactoring_plan:
            issues.append("Testing status without refactoring plan")
        
        if state.status == AgentStatus.COMPLETE and not state.test_results:
            issues.append("Complete status without test results")
        
        # Check file paths exist
        for file_path in state.current_files:
            if not os.path.exists(file_path):
                issues.append(f"File does not exist: {file_path}")
        
        return issues
    
    @staticmethod
    def export_state_report(state: AgentState, output_file: str) -> None:
        """Export detailed state report to file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "state": state.to_dict(),
            "integrity_issues": StateInspector.validate_state_integrity(state),
            "summary": {
                "status": state.status.value,
                "iteration": f"{state.iteration_count}/10",
                "files_count": len(state.current_files),
                "has_plan": state.refactoring_plan is not None,
                "has_results": state.test_results is not None,
                "is_complete": state.is_complete()
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)