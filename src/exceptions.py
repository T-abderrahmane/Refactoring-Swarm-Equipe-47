"""
Exception hierarchy for the Refactoring Swarm system.

This module defines the base exception classes and specialized exceptions
for different components of the refactoring system.
"""


class RefactoringError(Exception):
    """Base exception for all refactoring operations.
    
    This is the root exception class that all other refactoring-related
    exceptions inherit from. It provides a common interface for error
    handling across the entire system.
    """
    
    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.message = message
        self.details = details
    
    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        return self.message


class SecurityViolationError(RefactoringError):
    """Raised when attempting unauthorized file operations.
    
    This exception is raised by the security manager when code attempts
    to access files or directories outside the allowed sandbox area.
    """
    pass


class AnalysisError(RefactoringError):
    """Raised when code analysis fails.
    
    This exception is raised by the Auditor Agent when static analysis
    tools fail to process the code or when LLM analysis encounters errors.
    """
    pass


class FixingError(RefactoringError):
    """Raised when code fixing fails.
    
    This exception is raised by the Fixer Agent when code modifications
    fail due to syntax errors, logical issues, or other implementation problems.
    """
    pass


class TestingError(RefactoringError):
    """Raised when test execution fails.
    
    This exception is raised by the Judge Agent when test execution
    encounters system-level failures (not test failures, but execution issues).
    """
    pass


class OrchestrationError(RefactoringError):
    """Raised when workflow orchestration fails.
    
    This exception is raised by the Orchestrator when agent coordination
    fails or when workflow state transitions encounter errors.
    """
    pass