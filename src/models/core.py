"""Core data models for the Refactoring Swarm system."""

from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class IssueType(Enum):
    """Types of code issues that can be identified."""
    SYNTAX_ERROR = "syntax_error"
    STYLE_VIOLATION = "style_violation"
    LOGIC_ERROR = "logic_error"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_ISSUE = "security_issue"
    MAINTAINABILITY = "maintainability"


class IssueSeverity(Enum):
    """Severity levels for code issues."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AgentStatus(Enum):
    """Status values for agent state."""
    ANALYZING = "analyzing"
    FIXING = "fixing"
    TESTING = "testing"
    COMPLETE = "complete"
    FAILED = "failed"
    IDLE = "idle"


@dataclass
class CodeIssue:
    """Represents a code issue identified during analysis."""
    line_number: int
    issue_type: IssueType
    description: str
    suggested_fix: str
    severity: IssueSeverity
    file_path: str = ""
    column_number: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'line_number': self.line_number,
            'issue_type': self.issue_type.value,
            'description': self.description,
            'suggested_fix': self.suggested_fix,
            'severity': self.severity.value,
            'file_path': self.file_path,
            'column_number': self.column_number
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CodeIssue':
        """Create instance from dictionary."""
        return cls(
            line_number=data['line_number'],
            issue_type=IssueType(data['issue_type']),
            description=data['description'],
            suggested_fix=data['suggested_fix'],
            severity=IssueSeverity(data['severity']),
            file_path=data.get('file_path', ''),
            column_number=data.get('column_number', 0)
        )


@dataclass
class RefactoringPlan:
    """Represents a plan for refactoring a file or set of files."""
    file_path: str
    issues: List[CodeIssue]
    priority: int
    estimated_effort: str
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'file_path': self.file_path,
            'issues': [issue.to_dict() for issue in self.issues],
            'priority': self.priority,
            'estimated_effort': self.estimated_effort,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RefactoringPlan':
        """Create instance from dictionary."""
        return cls(
            file_path=data['file_path'],
            issues=[CodeIssue.from_dict(issue) for issue in data['issues']],
            priority=data['priority'],
            estimated_effort=data['estimated_effort'],
            created_at=data.get('created_at')
        )
    
    def get_critical_issues(self) -> List[CodeIssue]:
        """Get all critical severity issues."""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.CRITICAL]
    
    def get_issues_by_type(self, issue_type: IssueType) -> List[CodeIssue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]


@dataclass
class TestResult:
    """Represents the result of test execution."""
    passed: bool
    total_tests: int
    failed_tests: int
    error_details: List[str]
    execution_time: float
    test_file: str = ""
    stdout: str = ""
    stderr: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'passed': self.passed,
            'total_tests': self.total_tests,
            'failed_tests': self.failed_tests,
            'error_details': self.error_details,
            'execution_time': self.execution_time,
            'test_file': self.test_file,
            'stdout': self.stdout,
            'stderr': self.stderr
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestResult':
        """Create instance from dictionary."""
        return cls(
            passed=data['passed'],
            total_tests=data['total_tests'],
            failed_tests=data['failed_tests'],
            error_details=data['error_details'],
            execution_time=data['execution_time'],
            test_file=data.get('test_file', ''),
            stdout=data.get('stdout', ''),
            stderr=data.get('stderr', '')
        )
    
    @property
    def success_rate(self) -> float:
        """Calculate test success rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return ((self.total_tests - self.failed_tests) / self.total_tests) * 100


@dataclass
class AgentState:
    """Represents the current state of the agent workflow."""
    current_files: List[str]
    iteration_count: int
    refactoring_plan: Optional[RefactoringPlan]
    test_results: Optional[TestResult]
    status: AgentStatus
    target_directory: str = ""
    sandbox_directory: str = ""
    last_error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'current_files': self.current_files,
            'iteration_count': self.iteration_count,
            'refactoring_plan': self.refactoring_plan.to_dict() if self.refactoring_plan else None,
            'test_results': self.test_results.to_dict() if self.test_results else None,
            'status': self.status.value,
            'target_directory': self.target_directory,
            'sandbox_directory': self.sandbox_directory,
            'last_error': self.last_error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        """Create instance from dictionary."""
        return cls(
            current_files=data['current_files'],
            iteration_count=data['iteration_count'],
            refactoring_plan=RefactoringPlan.from_dict(data['refactoring_plan']) if data.get('refactoring_plan') else None,
            test_results=TestResult.from_dict(data['test_results']) if data.get('test_results') else None,
            status=AgentStatus(data['status']),
            target_directory=data.get('target_directory', ''),
            sandbox_directory=data.get('sandbox_directory', ''),
            last_error=data.get('last_error', '')
        )
    
    def to_json(self) -> str:
        """Convert to JSON string for persistence."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'AgentState':
        """Create instance from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def is_complete(self) -> bool:
        """Check if the workflow is complete."""
        return self.status in [AgentStatus.COMPLETE, AgentStatus.FAILED]
    
    def has_exceeded_max_iterations(self, max_iterations: int = 10) -> bool:
        """Check if maximum iterations have been exceeded."""
        return self.iteration_count >= max_iterations
    
    def increment_iteration(self) -> None:
        """Increment the iteration counter."""
        self.iteration_count += 1
    
    def reset_for_retry(self) -> None:
        """Reset state for retry attempt."""
        self.last_error = ""
        self.status = AgentStatus.ANALYZING