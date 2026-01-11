"""
Fixer Agent for code modification and refactoring implementation.

The Fixer Agent is responsible for:
1. Implementing code corrections based on refactoring plans
2. Maintaining code functionality while improving quality
3. Operating within sandbox constraints
4. Handling incremental fixes with rollback capabilities
"""

import os
import ast
import tempfile
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from ..models.core import (
    RefactoringPlan, CodeIssue, IssueType, IssueSeverity
)
from ..tools.file_manager import SecureFileManager
from ..security.sandbox import SandboxManager
from ..utils.llm_call import call_llm
from ..utils.logger import log_experiment, ActionType
from ..exceptions import FixingError
from .fixer_strategy import IncrementalFixingStrategy, FixStrategy, FixingProgress


class FixerAgent:
    """Agent responsible for code modification and refactoring implementation."""
    
    def __init__(self, sandbox_manager: SandboxManager, model: str = "gemini-1.5-flash", 
                 llm_fn=None, strategy: FixStrategy = FixStrategy.SEQUENTIAL):
        """
        Initialize the Fixer Agent.
        
        Args:
            sandbox_manager: SandboxManager instance for secure file operations
            model: LLM model to use for fix generation
            llm_fn: Function to call the LLM (for dependency injection)
            strategy: Incremental fixing strategy to use
        """
        self.model = model
        self.llm_fn = llm_fn
        self.sandbox_manager = sandbox_manager
        self.file_manager = SecureFileManager(sandbox_manager)
        self.agent_name = "Fixer"
        self.applied_fixes: List[Dict[str, Any]] = []
        
        # Initialize incremental fixing strategy
        self.incremental_strategy = IncrementalFixingStrategy(strategy)
        self.current_progress: Optional[FixingProgress] = None
    
    def apply_fixes_incrementally(self, refactoring_plan: RefactoringPlan) -> bool:
        """
        Apply fixes from a refactoring plan incrementally with progress tracking.
        
        Args:
            refactoring_plan: RefactoringPlan containing issues and suggested fixes
            
        Returns:
            True if all fixes applied successfully, False otherwise
            
        Raises:
            FixingError: If fixing fails
        """
        if not refactoring_plan.issues:
            return True  # No issues to fix
        
        try:
            # Plan the incremental fixes
            self.current_progress = self.incremental_strategy.plan_fixes(refactoring_plan)
            
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt=f"Starting incremental fixing for {refactoring_plan.file_path}",
                response=f"Planned {self.current_progress.total_steps} fix steps",
                model_used=self.model
            )
            
            # Read the original file
            file_path = refactoring_plan.file_path
            original_content = self.file_manager.safe_read_file(file_path)
            
            # Create initial backup
            main_backup = self.file_manager.create_backup(file_path)
            
            # Process each step incrementally
            success = True
            while not self.current_progress.is_complete:
                next_step = self.incremental_strategy.get_next_step()
                if not next_step:
                    break
                
                step_success = self._process_fix_step(file_path, next_step)
                if not step_success:
                    success = False
                    # Continue with remaining steps or break based on strategy
                    if self.incremental_strategy.strategy == FixStrategy.SAFE_MODE:
                        break
            
            # Finalize the process
            self.incremental_strategy.finalize()
            
            # Record the overall fix session
            self.applied_fixes.append({
                'file': file_path,
                'strategy': self.incremental_strategy.strategy.value,
                'progress': self.current_progress.to_dict(),
                'main_backup': main_backup,
                'success': success
            })
            
            return success
            
        except Exception as e:
            if self.current_progress:
                self.incremental_strategy.finalize()
            raise FixingError(f"Incremental fixing failed for {refactoring_plan.file_path}: {e}")
    
    def _process_fix_step(self, file_path: str, step) -> bool:
        """
        Process a single fix step.
        
        Args:
            file_path: Path to the file being fixed
            step: FixStep to process
            
        Returns:
            True if step processed successfully, False otherwise
        """
        try:
            # Start the step
            self.incremental_strategy.start_step(step.step_number)
            
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt=f"Processing step {step.step_number}: {step.issue.description}",
                response="Step started",
                model_used=self.model
            )
            
            # Read current file content
            current_content = self.file_manager.safe_read_file(file_path)
            
            # Create step-specific backup
            step_backup = self.file_manager.create_backup(file_path)
            
            # Generate fix using LLM
            fix_code = self._generate_fix_with_llm(file_path, current_content, step.issue)
            
            # Apply the fix
            modified_content = self._apply_single_fix(current_content, step.issue, fix_code)
            
            # Validate syntax
            validation_passed = self._validate_syntax(modified_content)
            
            if validation_passed:
                # Write the modified content
                self.file_manager.safe_write_file(file_path, modified_content, create_backup=False)
                
                # Mark step as completed
                self.incremental_strategy.complete_step(
                    step.step_number, 
                    fix_code, 
                    step_backup, 
                    validation_passed
                )
                
                log_experiment(
                    agent_name=self.agent_name,
                    action=ActionType.FIX,
                    prompt=f"Step {step.step_number} completed successfully",
                    response=f"Applied fix: {fix_code[:100]}...",
                    model_used=self.model
                )
                
                return True
            else:
                # Validation failed, mark as failed
                error_msg = f"Syntax validation failed for step {step.step_number}"
                self.incremental_strategy.fail_step(step.step_number, error_msg)
                
                log_experiment(
                    agent_name=self.agent_name,
                    action=ActionType.FIX,
                    prompt=f"Step {step.step_number} failed validation",
                    response=error_msg,
                    model_used=self.model
                )
                
                return False
                
        except Exception as e:
            # Mark step as failed
            error_msg = f"Step processing failed: {str(e)}"
            self.incremental_strategy.fail_step(step.step_number, error_msg)
            
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt=f"Step {step.step_number} encountered error",
                response=error_msg,
                model_used=self.model
            )
            
            return False
    
    def rollback_step(self, step_number: int) -> bool:
        """
        Rollback a specific fix step.
        
        Args:
            step_number: The step number to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.current_progress:
            return False
        
        try:
            # Get the step
            if step_number < 1 or step_number > len(self.current_progress.steps):
                return False
            
            step = self.current_progress.steps[step_number - 1]
            
            if step.status != "completed" or not step.backup_info:
                return False
            
            # Restore from backup
            self.file_manager.restore_backup(step.backup_info)
            
            # Update strategy state
            success = self.incremental_strategy.rollback_step(step_number)
            
            if success:
                log_experiment(
                    agent_name=self.agent_name,
                    action=ActionType.FIX,
                    prompt=f"Rolled back step {step_number}",
                    response="Rollback successful",
                    model_used=self.model
                )
            
            return success
            
        except Exception as e:
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt=f"Rollback failed for step {step_number}",
                response=f"Error: {str(e)}",
                model=self.model
            )
            return False
    
    def rollback_all_steps(self) -> bool:
        """
        Rollback all completed steps in reverse order.
        
        Returns:
            True if all rollbacks successful, False otherwise
        """
        if not self.current_progress:
            return False
        
        try:
            results = self.incremental_strategy.rollback_all_steps()
            all_successful = all(results)
            
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt="Rolling back all steps",
                response=f"Rollback results: {len([r for r in results if r])}/{len(results)} successful",
                model=self.model
            )
            
            return all_successful
            
        except Exception as e:
            log_experiment(
                agent_name=self.agent_name,
                action=ActionType.FIX,
                prompt="Rollback all steps failed",
                response=f"Error: {str(e)}",
                model=self.model
            )
            return False
    
    def get_fixing_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get current fixing progress information.
        
        Returns:
            Dictionary containing progress information, or None if no active progress
        """
        if not self.current_progress:
            return None
        
        return self.incremental_strategy.get_progress_summary()
    
    def get_current_step_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current step being processed.
        
        Returns:
            Dictionary containing current step information, or None if no active step
        """
        if not self.current_progress:
            return None
        
        current_step = self.incremental_strategy.get_current_step()
        if not current_step:
            return None
        
        return {
            'step_number': current_step.step_number,
            'issue_description': current_step.issue.description,
            'issue_type': current_step.issue.issue_type.value,
            'severity': current_step.issue.severity.value,
            'line_number': current_step.issue.line_number,
            'status': current_step.status,
            'error_message': current_step.error_message
        }
    def apply_fixes(self, refactoring_plan: RefactoringPlan, use_incremental: bool = True) -> bool:
        """
        Apply fixes from a refactoring plan to the target file.
        
        Args:
            refactoring_plan: RefactoringPlan containing issues and suggested fixes
            use_incremental: Whether to use incremental fixing strategy (default: True)
            
        Returns:
            True if all fixes applied successfully, False otherwise
            
        Raises:
            FixingError: If fixing fails
        """
        if use_incremental:
            return self.apply_fixes_incrementally(refactoring_plan)
        else:
            return self._apply_fixes_batch(refactoring_plan)
    
    def _apply_fixes_batch(self, refactoring_plan: RefactoringPlan) -> bool:
        """
        Apply fixes from a refactoring plan in batch mode (original implementation).
        
        Args:
            refactoring_plan: RefactoringPlan containing issues and suggested fixes
            
        Returns:
            True if all fixes applied successfully, False otherwise
            
        Raises:
            FixingError: If fixing fails
        """
        if not refactoring_plan.issues:
            return True  # No issues to fix
        
        file_path = refactoring_plan.file_path
        
        try:
            # Read the original file
            original_content = self.file_manager.safe_read_file(file_path)
            
            # Create backup before modifications
            backup = self.file_manager.create_backup(file_path)
            
            # Sort issues by line number (descending) to avoid line number shifts
            sorted_issues = sorted(
                refactoring_plan.issues,
                key=lambda x: x.line_number,
                reverse=True
            )
            
            # Apply fixes incrementally
            modified_content = original_content
            successful_fixes = []
            
            for issue in sorted_issues:
                try:
                    # Generate fix using LLM
                    fix_code = self._generate_fix_with_llm(file_path, modified_content, issue)
                    
                    # Apply the fix
                    modified_content = self._apply_single_fix(
                        modified_content, issue, fix_code
                    )
                    
                    successful_fixes.append({
                        'issue': issue,
                        'fix_code': fix_code,
                        'success': True
                    })
                    
                except Exception as e:
                    # Log failed fix but continue with others
                    successful_fixes.append({
                        'issue': issue,
                        'fix_code': None,
                        'success': False,
                        'error': str(e)
                    })
                    continue
            
            # Validate syntax of modified code
            if not self._validate_syntax(modified_content):
                raise FixingError("Modified code has syntax errors")
            
            # Write modified content back to file
            self.file_manager.safe_write_file(file_path, modified_content, create_backup=False)
            
            # Record applied fixes
            self.applied_fixes.append({
                'file': file_path,
                'fixes': successful_fixes,
                'backup': backup,
                'timestamp': backup.timestamp if backup else None
            })
            
            return True
            
        except FixingError:
            raise
        except Exception as e:
            raise FixingError(f"Failed to apply fixes to {file_path}: {e}")
    
    def _generate_fix_with_llm(self, file_path: str, file_content: str, 
                               issue: CodeIssue) -> str:
        """
        Generate a fix for a specific code issue using LLM.
        
        Args:
            file_path: Path to the file being fixed
            file_content: Current content of the file
            issue: CodeIssue to fix
            
        Returns:
            Generated fix code
            
        Raises:
            FixingError: If fix generation fails
        """
        if not self.llm_fn:
            # Fallback to suggested fix if no LLM available
            return issue.suggested_fix
        
        try:
            # Extract context around the issue
            lines = file_content.split('\n')
            start_line = max(0, issue.line_number - 3)
            end_line = min(len(lines), issue.line_number + 3)
            context_lines = lines[start_line:end_line]
            context = '\n'.join(context_lines)
            
            # Prepare prompt for LLM
            prompt = self._prepare_fix_prompt(file_path, context, issue)
            
            # Call LLM for fix generation
            response = call_llm(
                agent_name=self.agent_name,
                model=self.model,
                action=ActionType.FIX,
                prompt=prompt,
                llm_fn=self.llm_fn
            )
            
            # Extract fix code from response
            fix_code = self._extract_fix_code(response)
            
            return fix_code
            
        except Exception as e:
            raise FixingError(f"Failed to generate fix for issue at line {issue.line_number}: {e}")
    
    def _prepare_fix_prompt(self, file_path: str, context: str, issue: CodeIssue) -> str:
        """
        Prepare a prompt for LLM-based fix generation.
        
        Args:
            file_path: Path to the file
            context: Code context around the issue
            issue: The code issue to fix
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are a Python code refactoring expert. Fix the following code issue.

File: {file_path}
Issue Type: {issue.issue_type.value}
Severity: {issue.severity.value}
Line: {issue.line_number}
Description: {issue.description}

Code Context:
```python
{context}
```

Suggested Fix: {issue.suggested_fix}

Please provide the corrected code for the problematic section. Return ONLY the fixed code without explanations or markdown formatting. The code should:
1. Fix the identified issue
2. Maintain the original functionality
3. Follow Python best practices
4. Be syntactically correct

Fixed code:"""
        
        return prompt
    
    def _extract_fix_code(self, response: str) -> str:
        """
        Extract fix code from LLM response.
        
        Args:
            response: LLM response text
            
        Returns:
            Extracted fix code
        """
        # Remove markdown code blocks if present
        if "```python" in response:
            start = response.find("```python") + len("```python")
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                return response[start:end].strip()
        
        # Return the response as-is if no code blocks found
        return response.strip()
    
    def _apply_single_fix(self, file_content: str, issue: CodeIssue, 
                         fix_code: str) -> str:
        """
        Apply a single fix to the file content.
        
        Args:
            file_content: Current file content
            issue: The code issue being fixed
            fix_code: The fix code to apply
            
        Returns:
            Modified file content
            
        Raises:
            FixingError: If fix application fails
        """
        try:
            lines = file_content.split('\n')
            
            # Validate line number
            if issue.line_number < 1 or issue.line_number > len(lines):
                raise FixingError(f"Invalid line number: {issue.line_number}")
            
            # Get the problematic line (convert to 0-indexed)
            line_index = issue.line_number - 1
            original_line = lines[line_index]
            
            # Determine how many lines the fix spans
            fix_lines = fix_code.split('\n')
            
            # Replace the problematic line(s) with the fix
            lines[line_index:line_index + 1] = fix_lines
            
            # Reconstruct the file content
            modified_content = '\n'.join(lines)
            
            return modified_content
            
        except Exception as e:
            raise FixingError(f"Failed to apply fix at line {issue.line_number}: {e}")
    
    def _validate_syntax(self, code: str) -> bool:
        """
        Validate Python syntax of code.
        
        Args:
            code: Python code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            # If we can't parse for other reasons, assume it's valid
            return True
    
    def rollback_last_fixes(self) -> bool:
        """
        Rollback the last set of applied fixes.
        
        Returns:
            True if rollback successful, False otherwise
        """
        if not self.applied_fixes:
            return False
        
        last_fix_set = self.applied_fixes.pop()
        
        try:
            if last_fix_set['backup']:
                self.file_manager.restore_backup(last_fix_set['backup'])
                return True
            return False
        except Exception as e:
            # Re-add to list if rollback failed
            self.applied_fixes.append(last_fix_set)
            raise FixingError(f"Failed to rollback fixes: {e}")
    
    def get_applied_fixes_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all applied fixes.
        
        Returns:
            Dictionary containing fix statistics and details
        """
        # Calculate batch fixes statistics
        batch_fixes = [fix_set for fix_set in self.applied_fixes if 'fixes' in fix_set]
        total_batch_fixes = sum(len(fix_set['fixes']) for fix_set in batch_fixes)
        successful_batch_fixes = sum(
            len([f for f in fix_set['fixes'] if f['success']])
            for fix_set in batch_fixes
        )
        
        # Calculate incremental fixes statistics
        incremental_fixes = [fix_set for fix_set in self.applied_fixes if 'progress' in fix_set]
        total_incremental_steps = sum(
            fix_set['progress']['total_steps'] for fix_set in incremental_fixes
        )
        successful_incremental_steps = sum(
            fix_set['progress']['completed_steps'] for fix_set in incremental_fixes
        )
        
        total_fixes = total_batch_fixes + total_incremental_steps
        successful_fixes = successful_batch_fixes + successful_incremental_steps
        
        summary = {
            'total_fix_sessions': len(self.applied_fixes),
            'batch_sessions': len(batch_fixes),
            'incremental_sessions': len(incremental_fixes),
            'total_fixes': total_fixes,
            'successful_fixes': successful_fixes,
            'failed_fixes': total_fixes - successful_fixes,
            'success_rate': (successful_fixes / total_fixes * 100) if total_fixes > 0 else 0,
            'details': self.applied_fixes
        }
        
        # Add current progress if available
        if self.current_progress:
            summary['current_progress'] = self.get_fixing_progress()
        
        return summary
    
    def clear_applied_fixes_history(self) -> None:
        """Clear the history of applied fixes and reset incremental strategy."""
        self.applied_fixes.clear()
        self.incremental_strategy.reset()
        self.current_progress = None
