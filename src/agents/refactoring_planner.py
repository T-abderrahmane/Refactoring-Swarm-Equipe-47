"""
Refactoring Plan Generator for structured plan generation and issue categorization.

This module provides utilities for:
1. Generating structured refactoring plans
2. Categorizing and prioritizing issues
3. Assessing code quality comprehensively
4. Creating actionable fix sequences
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..models.core import (
    CodeIssue, RefactoringPlan, IssueType, IssueSeverity
)
from ..utils.logger import log_experiment, ActionType
from ..utils.llm_call import call_llm
from ..exceptions import AnalysisError


class IssueCategory(Enum):
    """Categories for organizing refactoring issues."""
    CRITICAL_BUGS = "critical_bugs"
    LOGIC_ERRORS = "logic_errors"
    STYLE_VIOLATIONS = "style_violations"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"


@dataclass
class CategorizedIssues:
    """Issues organized by category."""
    critical_bugs: List[CodeIssue]
    logic_errors: List[CodeIssue]
    style_violations: List[CodeIssue]
    performance: List[CodeIssue]
    security: List[CodeIssue]
    maintainability: List[CodeIssue]
    
    def get_all_issues(self) -> List[CodeIssue]:
        """Get all issues in priority order."""
        return (
            self.critical_bugs +
            self.logic_errors +
            self.security +
            self.performance +
            self.maintainability +
            self.style_violations
        )
    
    def get_category_summary(self) -> Dict[str, int]:
        """Get count of issues per category."""
        return {
            'critical_bugs': len(self.critical_bugs),
            'logic_errors': len(self.logic_errors),
            'style_violations': len(self.style_violations),
            'performance': len(self.performance),
            'security': len(self.security),
            'maintainability': len(self.maintainability),
        }


class RefactoringPlanner:
    """Generates comprehensive refactoring plans with issue categorization."""
    
    def __init__(self, model: str = "gemini-1.5-flash", llm_fn=None):
        """
        Initialize the refactoring planner.
        
        Args:
            model: LLM model to use for plan generation
            llm_fn: Function to call the LLM
        """
        self.model = model
        self.llm_fn = llm_fn
        self.agent_name = "RefactoringPlanner"
    
    def categorize_issues(self, issues: List[CodeIssue]) -> CategorizedIssues:
        """
        Categorize issues by type and severity.
        
        Args:
            issues: List of code issues to categorize
            
        Returns:
            CategorizedIssues object with organized issues
        """
        log_experiment(
            agent_name=self.agent_name,
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Categorizing {len(issues)} code issues",
                "output_response": "Issue categorization started"
            },
            status="SUCCESS"
        )
        
        categorized = CategorizedIssues(
            critical_bugs=[],
            logic_errors=[],
            style_violations=[],
            performance=[],
            security=[],
            maintainability=[]
        )
        
        for issue in issues:
            # Categorize by severity first
            if issue.severity == IssueSeverity.CRITICAL:
                categorized.critical_bugs.append(issue)
            # Then by type
            elif issue.issue_type == IssueType.LOGIC_ERROR:
                categorized.logic_errors.append(issue)
            elif issue.issue_type == IssueType.SECURITY_ISSUE:
                categorized.security.append(issue)
            elif issue.issue_type == IssueType.PERFORMANCE_ISSUE:
                categorized.performance.append(issue)
            elif issue.issue_type == IssueType.STYLE_VIOLATION:
                categorized.style_violations.append(issue)
            else:
                categorized.maintainability.append(issue)
        
        category_summary = categorized.get_category_summary()
        log_experiment(
            agent_name=self.agent_name,
            model_used=self.model,
            action=ActionType.ANALYSIS,
            details={
                "input_prompt": f"Issue categorization completed for {len(issues)} issues",
                "output_response": f"Categories: {category_summary}"
            },
            status="SUCCESS"
        )
        
        return categorized
    
    def generate_comprehensive_plan(self, file_path: str, 
                                   issues: List[CodeIssue],
                                   quality_score: float) -> RefactoringPlan:
        """
        Generate a comprehensive refactoring plan with detailed analysis.
        
        Args:
            file_path: Path to the file being refactored
            issues: List of identified code issues
            quality_score: Current code quality score (0-10)
            
        Returns:
            RefactoringPlan with comprehensive details
        """
        log_experiment(
            agent_name=self.agent_name,
            model_used=self.model,
            action=ActionType.GENERATION,
            details={
                "input_prompt": f"Generating comprehensive refactoring plan for {file_path} with {len(issues)} issues, quality score: {quality_score}",
                "output_response": "Plan generation started"
            },
            status="SUCCESS"
        )
        
        # Categorize issues
        categorized = self.categorize_issues(issues)
        
        # Calculate priority based on issue distribution
        priority = self._calculate_priority(categorized, quality_score)
        
        # Estimate effort
        estimated_effort = self._estimate_effort(categorized)
        
        # Create plan with categorized issues
        plan = RefactoringPlan(
            file_path=file_path,
            issues=categorized.get_all_issues(),
            priority=priority,
            estimated_effort=estimated_effort
        )
        
        log_experiment(
            agent_name=self.agent_name,
            model_used=self.model,
            action=ActionType.GENERATION,
            details={
                "input_prompt": f"Comprehensive refactoring plan completed for {file_path}",
                "output_response": f"Generated plan with priority {priority}, effort {estimated_effort}, {len(plan.issues)} issues"
            },
            status="SUCCESS"
        )
        
        return plan
    
    def _calculate_priority(self, categorized: CategorizedIssues, 
                           quality_score: float) -> int:
        """
        Calculate refactoring priority (1-10).
        
        Args:
            categorized: Categorized issues
            quality_score: Current quality score
            
        Returns:
            Priority value (1-10)
        """
        # Base priority on critical issues
        priority = len(categorized.critical_bugs) * 2
        
        # Add weight for logic errors
        priority += len(categorized.logic_errors)
        
        # Add weight for security issues
        priority += len(categorized.security) * 2
        
        # Adjust based on quality score
        if quality_score < 3:
            priority += 3
        elif quality_score < 5:
            priority += 2
        elif quality_score < 7:
            priority += 1
        
        # Cap at 10
        return min(10, max(1, priority))
    
    def _estimate_effort(self, categorized: CategorizedIssues) -> str:
        """
        Estimate refactoring effort.
        
        Args:
            categorized: Categorized issues
            
        Returns:
            Effort estimate (minimal, low, medium, high, very_high)
        """
        total_issues = sum(categorized.get_category_summary().values())
        critical_issues = len(categorized.critical_bugs) + len(categorized.logic_errors)
        
        if total_issues == 0:
            return "minimal"
        elif critical_issues > 10 or total_issues > 50:
            return "very_high"
        elif critical_issues > 5 or total_issues > 30:
            return "high"
        elif critical_issues > 2 or total_issues > 15:
            return "medium"
        elif total_issues > 5:
            return "low"
        else:
            return "minimal"
    
    def create_fix_sequence(self, plan: RefactoringPlan) -> List[List[CodeIssue]]:
        """
        Create a sequence of fix batches for incremental refactoring.
        
        Issues are grouped into batches that can be fixed together,
        with dependencies considered (e.g., fix critical bugs before style).
        
        Args:
            plan: RefactoringPlan to sequence
            
        Returns:
            List of issue batches in recommended fix order
        """
        categorized = self.categorize_issues(plan.issues)
        
        # Create fix sequence: critical first, then logic, then others
        sequence = []
        
        # Batch 1: Critical bugs
        if categorized.critical_bugs:
            sequence.append(categorized.critical_bugs)
        
        # Batch 2: Logic errors
        if categorized.logic_errors:
            sequence.append(categorized.logic_errors)
        
        # Batch 3: Security issues
        if categorized.security:
            sequence.append(categorized.security)
        
        # Batch 4: Performance issues
        if categorized.performance:
            sequence.append(categorized.performance)
        
        # Batch 5: Maintainability issues
        if categorized.maintainability:
            sequence.append(categorized.maintainability)
        
        # Batch 6: Style violations (lowest priority)
        if categorized.style_violations:
            sequence.append(categorized.style_violations)
        
        return sequence
    
    def generate_llm_refactoring_prompt(self, file_path: str, 
                                       file_content: str,
                                       plan: RefactoringPlan) -> str:
        """
        Generate a detailed prompt for LLM-based refactoring.
        
        Args:
            file_path: Path to the file
            file_content: Content of the file
            plan: RefactoringPlan with identified issues
            
        Returns:
            Formatted prompt for LLM
        """
        # Categorize issues for better organization
        categorized = self.categorize_issues(plan.issues)
        category_summary = categorized.get_category_summary()
        
        # Build issue summary
        issue_summary = "Issues to fix:\n"
        for category, count in category_summary.items():
            if count > 0:
                issue_summary += f"  - {category}: {count}\n"
        
        # Build detailed issue list
        issue_details = "Detailed issues:\n"
        for issue in plan.issues[:20]:  # Limit to first 20 for prompt size
            issue_details += f"  - Line {issue.line_number}: {issue.description}\n"
            issue_details += f"    Suggested fix: {issue.suggested_fix}\n"
        
        prompt = f"""You are a Python code refactoring expert. Refactor the following code to fix the identified issues.

File: {file_path}
Priority: {plan.priority}/10
Estimated Effort: {plan.estimated_effort}

{issue_summary}

{issue_details}

Original Code:
```python
{file_content}
```

Please provide the refactored code that:
1. Fixes all critical bugs and logic errors
2. Addresses security vulnerabilities
3. Improves performance where possible
4. Maintains the original functionality
5. Follows Python best practices

Return ONLY the refactored code in a code block, with no explanations."""
        
        return prompt
    
    def assess_code_quality(self, plan: RefactoringPlan) -> Dict[str, any]:
        """
        Provide comprehensive code quality assessment.
        
        Args:
            plan: RefactoringPlan with analysis results
            
        Returns:
            Dictionary with quality metrics
        """
        categorized = self.categorize_issues(plan.issues)
        category_summary = categorized.get_category_summary()
        
        total_issues = sum(category_summary.values())
        
        assessment = {
            'file': plan.file_path,
            'priority': plan.priority,
            'estimated_effort': plan.estimated_effort,
            'total_issues': total_issues,
            'issues_by_category': category_summary,
            'critical_issues': len(categorized.critical_bugs),
            'has_security_issues': len(categorized.security) > 0,
            'has_performance_issues': len(categorized.performance) > 0,
            'quality_level': self._determine_quality_level(total_issues, plan.priority),
            'recommendations': self._generate_recommendations(categorized)
        }
        
        return assessment
    
    def _determine_quality_level(self, total_issues: int, priority: int) -> str:
        """
        Determine overall code quality level.
        
        Args:
            total_issues: Total number of issues
            priority: Priority score
            
        Returns:
            Quality level (excellent, good, fair, poor, critical)
        """
        if total_issues == 0:
            return "excellent"
        elif total_issues <= 3 and priority <= 2:
            return "good"
        elif total_issues <= 10 and priority <= 5:
            return "fair"
        elif total_issues <= 25 and priority <= 8:
            return "poor"
        else:
            return "critical"
    
    def _generate_recommendations(self, categorized: CategorizedIssues) -> List[str]:
        """
        Generate actionable recommendations based on issues.
        
        Args:
            categorized: Categorized issues
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if categorized.critical_bugs:
            recommendations.append(
                f"Fix {len(categorized.critical_bugs)} critical bugs immediately"
            )
        
        if categorized.security:
            recommendations.append(
                f"Address {len(categorized.security)} security vulnerabilities"
            )
        
        if categorized.logic_errors:
            recommendations.append(
                f"Review and fix {len(categorized.logic_errors)} logic errors"
            )
        
        if categorized.performance:
            recommendations.append(
                f"Optimize {len(categorized.performance)} performance issues"
            )
        
        if categorized.maintainability:
            recommendations.append(
                f"Improve {len(categorized.maintainability)} maintainability issues"
            )
        
        if categorized.style_violations:
            recommendations.append(
                f"Clean up {len(categorized.style_violations)} style violations"
            )
        
        if not recommendations:
            recommendations.append("Code quality is excellent - minimal refactoring needed")
        
        return recommendations
    
    def generate_plan_summary(self, plan: RefactoringPlan) -> str:
        """
        Generate a human-readable summary of the refactoring plan.
        
        Args:
            plan: RefactoringPlan to summarize
            
        Returns:
            Formatted summary string
        """
        assessment = self.assess_code_quality(plan)
        
        summary = f"=== Refactoring Plan Summary ===\n\n"
        summary += f"File: {plan.file_path}\n"
        summary += f"Priority: {plan.priority}/10\n"
        summary += f"Estimated Effort: {plan.estimated_effort}\n"
        summary += f"Quality Level: {assessment['quality_level']}\n\n"
        
        summary += f"Issues Found: {assessment['total_issues']}\n"
        for category, count in assessment['issues_by_category'].items():
            if count > 0:
                summary += f"  - {category}: {count}\n"
        
        summary += f"\nRecommendations:\n"
        for i, rec in enumerate(assessment['recommendations'], 1):
            summary += f"  {i}. {rec}\n"
        
        return summary
