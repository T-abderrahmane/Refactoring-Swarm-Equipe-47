"""
Incremental fixing strategy for the Fixer Agent.

This module implements step-by-step fix application logic with rollback
mechanisms and progress tracking for safe code modifications.
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..models.core import RefactoringPlan, CodeIssue, IssueSeverity
from ..exceptions import FixingError


class FixStrategy(Enum):
    """Strategies for applying fixes."""
    SEQUENTIAL = "sequential"  # Apply fixes one by one
    BATCH = "batch"  # Apply all fixes at once
    PRIORITY_BASED = "priority_based"  # Apply by severity/priority
    SAFE_MODE = "safe_mode"  # Apply with maximum validation


@dataclass
class FixStep:
    """Represents a single fix step in the incremental process."""
    step_number: int
    issue: CodeIssue
    fix_code: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back
    error_message: Optional[str] = None
    timestamp: Optional[datetime] = None
    backup_info: Optional[Any] = None
    validation_passed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'step_number': self.step_number,
            'issue': self.issue.to_dict(),
            'fix_code': self.fix_code,
            'status': self.status,
            'error_message': self.error_message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'validation_passed': self.validation_passed
        }


@dataclass
class FixingProgress:
    """Tracks progress of incremental fixing process."""
    total_steps: int
    completed_steps: int = 0
    failed_steps: int = 0
    rolled_back_steps: int = 0
    current_step: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    steps: List[FixStep] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_steps == 0:
            return 0.0
        return ((self.completed_steps - self.rolled_back_steps) / self.total_steps) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if all steps have been processed."""
        return self.current_step >= self.total_steps
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time in seconds."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'failed_steps': self.failed_steps,
            'rolled_back_steps': self.rolled_back_steps,
            'current_step': self.current_step,
            'success_rate': self.success_rate,
            'elapsed_time': self.elapsed_time,
            'steps': [step.to_dict() for step in self.steps]
        }


class IncrementalFixingStrategy:
    """Implements incremental fix application with rollback and progress tracking."""
    
    def __init__(self, strategy: FixStrategy = FixStrategy.SEQUENTIAL):
        """
        Initialize the incremental fixing strategy.
        
        Args:
            strategy: The fixing strategy to use
        """
        self.strategy = strategy
        self.progress: Optional[FixingProgress] = None
        self.rollback_stack: List[FixStep] = []
    
    def plan_fixes(self, refactoring_plan: RefactoringPlan) -> FixingProgress:
        """
        Plan the sequence of fixes to apply.
        
        Args:
            refactoring_plan: The refactoring plan containing issues
            
        Returns:
            FixingProgress object tracking the plan
        """
        # Sort issues based on strategy
        sorted_issues = self._sort_issues_by_strategy(refactoring_plan.issues)
        
        # Create fix steps
        steps = []
        for idx, issue in enumerate(sorted_issues, 1):
            step = FixStep(
                step_number=idx,
                issue=issue,
                status="pending"
            )
            steps.append(step)
        
        # Initialize progress tracking
        self.progress = FixingProgress(
            total_steps=len(steps),
            start_time=datetime.now(),
            steps=steps
        )
        
        return self.progress
    
    def _sort_issues_by_strategy(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """
        Sort issues based on the selected strategy.
        
        Args:
            issues: List of code issues to sort
            
        Returns:
            Sorted list of issues
        """
        if self.strategy == FixStrategy.SEQUENTIAL:
            # Sort by line number
            return sorted(issues, key=lambda x: x.line_number)
        
        elif self.strategy == FixStrategy.PRIORITY_BASED:
            # Sort by severity (critical first) then by line number
            severity_order = {
                IssueSeverity.CRITICAL: 5,
                IssueSeverity.HIGH: 4,
                IssueSeverity.MEDIUM: 3,
                IssueSeverity.LOW: 2,
                IssueSeverity.INFO: 1
            }
            return sorted(
                issues,
                key=lambda x: (-severity_order.get(x.severity, 0), x.line_number)
            )
        
        elif self.strategy == FixStrategy.SAFE_MODE:
            # Sort by severity (low risk first) then by line number
            severity_order = {
                IssueSeverity.INFO: 5,
                IssueSeverity.LOW: 4,
                IssueSeverity.MEDIUM: 3,
                IssueSeverity.HIGH: 2,
                IssueSeverity.CRITICAL: 1
            }
            return sorted(
                issues,
                key=lambda x: (-severity_order.get(x.severity, 0), x.line_number)
            )
        
        else:  # BATCH
            # No specific order for batch processing
            return issues
    
    def start_step(self, step_number: int) -> FixStep:
        """
        Mark a fix step as in progress.
        
        Args:
            step_number: The step number to start
            
        Returns:
            The FixStep object
            
        Raises:
            FixingError: If step cannot be started
        """
        if not self.progress:
            raise FixingError("No fixing plan initialized")
        
        if step_number < 1 or step_number > self.progress.total_steps:
            raise FixingError(f"Invalid step number: {step_number}")
        
        step = self.progress.steps[step_number - 1]
        step.status = "in_progress"
        step.timestamp = datetime.now()
        self.progress.current_step = step_number
        
        return step
    
    def complete_step(self, step_number: int, fix_code: str, 
                     backup_info: Optional[Any] = None, 
                     validation_passed: bool = True) -> FixStep:
        """
        Mark a fix step as completed.
        
        Args:
            step_number: The step number to complete
            fix_code: The fix code that was applied
            backup_info: Optional backup information for rollback
            validation_passed: Whether validation passed for this step
            
        Returns:
            The completed FixStep object
            
        Raises:
            FixingError: If step cannot be completed
        """
        if not self.progress:
            raise FixingError("No fixing plan initialized")
        
        if step_number < 1 or step_number > self.progress.total_steps:
            raise FixingError(f"Invalid step number: {step_number}")
        
        step = self.progress.steps[step_number - 1]
        step.status = "completed"
        step.fix_code = fix_code
        step.backup_info = backup_info
        step.validation_passed = validation_passed
        
        self.progress.completed_steps += 1
        self.rollback_stack.append(step)
        
        return step
    
    def fail_step(self, step_number: int, error_message: str) -> FixStep:
        """
        Mark a fix step as failed.
        
        Args:
            step_number: The step number that failed
            error_message: Description of the failure
            
        Returns:
            The failed FixStep object
            
        Raises:
            FixingError: If step cannot be marked as failed
        """
        if not self.progress:
            raise FixingError("No fixing plan initialized")
        
        if step_number < 1 or step_number > self.progress.total_steps:
            raise FixingError(f"Invalid step number: {step_number}")
        
        step = self.progress.steps[step_number - 1]
        step.status = "failed"
        step.error_message = error_message
        
        self.progress.failed_steps += 1
        
        return step
    
    def rollback_step(self, step_number: int) -> bool:
        """
        Rollback a specific fix step.
        
        Args:
            step_number: The step number to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.progress:
            return False
        
        if step_number < 1 or step_number > self.progress.total_steps:
            return False
        
        step = self.progress.steps[step_number - 1]
        
        if step.status != "completed":
            return False
        
        if not step.backup_info:
            return False
        
        try:
            # Perform rollback (actual implementation in FixerAgent)
            step.status = "rolled_back"
            self.progress.rolled_back_steps += 1
            self.progress.completed_steps -= 1
            
            # Remove from rollback stack
            if step in self.rollback_stack:
                self.rollback_stack.remove(step)
            
            return True
            
        except Exception:
            return False
    
    def rollback_all_steps(self) -> List[bool]:
        """
        Rollback all completed steps in reverse order.
        
        Returns:
            List of rollback success status for each step
        """
        results = []
        
        # Rollback in reverse order
        for step in reversed(self.rollback_stack):
            success = self.rollback_step(step.step_number)
            results.append(success)
        
        return results
    
    def get_next_step(self) -> Optional[FixStep]:
        """
        Get the next step to process.
        
        Returns:
            The next FixStep to process, or None if all steps are done
        """
        if not self.progress or self.progress.is_complete:
            return None
        
        next_step_number = self.progress.current_step + 1
        if next_step_number <= self.progress.total_steps:
            return self.progress.steps[next_step_number - 1]
        
        return None
    
    def get_current_step(self) -> Optional[FixStep]:
        """
        Get the current step being processed.
        
        Returns:
            The current FixStep, or None if no step is active
        """
        if not self.progress or self.progress.current_step == 0:
            return None
        
        if self.progress.current_step <= self.progress.total_steps:
            return self.progress.steps[self.progress.current_step - 1]
        
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current fixing progress.
        
        Returns:
            Dictionary containing progress information
        """
        if not self.progress:
            return {}
        
        return {
            'strategy': self.strategy.value,
            'total_steps': self.progress.total_steps,
            'completed_steps': self.progress.completed_steps,
            'failed_steps': self.progress.failed_steps,
            'rolled_back_steps': self.progress.rolled_back_steps,
            'current_step': self.progress.current_step,
            'success_rate': self.progress.success_rate,
            'elapsed_time': self.progress.elapsed_time,
            'is_complete': self.progress.is_complete,
            'pending_steps': self.progress.total_steps - self.progress.current_step
        }
    
    def finalize(self) -> None:
        """Finalize the fixing process and record end time."""
        if self.progress:
            self.progress.end_time = datetime.now()
    
    def reset(self) -> None:
        """Reset the strategy state."""
        self.progress = None
        self.rollback_stack.clear()
