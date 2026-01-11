"""
Sandbox security manager for safe file operations.

This module provides security controls to ensure that all file operations
are restricted to designated areas and cannot access system files or
directories outside the intended scope.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from ..exceptions import SecurityViolationError


class SandboxManager:
    """Manages secure file operations within a restricted sandbox environment.
    
    This class enforces security policies by:
    - Restricting file operations to designated directories
    - Validating all file paths before operations
    - Preventing access to system files and directories
    - Managing temporary sandbox workspaces
    """
    
    def __init__(self, target_dir: str, sandbox_dir: Optional[str] = None):
        """Initialize the sandbox manager.
        
        Args:
            target_dir: The directory containing source code to be processed
            sandbox_dir: Optional custom sandbox directory. If None, creates temporary directory
        """
        self.target_dir = Path(target_dir).resolve()
        
        if not self.target_dir.exists():
            raise SecurityViolationError(f"Target directory does not exist: {target_dir}")
        
        if not self.target_dir.is_dir():
            raise SecurityViolationError(f"Target path is not a directory: {target_dir}")
        
        # Create or use provided sandbox directory
        if sandbox_dir:
            self.sandbox_dir = Path(sandbox_dir).resolve()
            self.sandbox_dir.mkdir(parents=True, exist_ok=True)
            self._temp_sandbox = False
        else:
            self.sandbox_dir = Path(tempfile.mkdtemp(prefix="refactoring_sandbox_"))
            self._temp_sandbox = True
        
        # Define allowed directories
        self.allowed_dirs = {
            self.target_dir,
            self.sandbox_dir,
        }
        
        # Define forbidden path patterns
        self.forbidden_patterns = [
            "/etc/",
            "/usr/",
            "/bin/",
            "/sbin/",
            "/sys/",
            "/proc/",
            "/dev/",
            "/var/",
            "/tmp/",
            "/root/",
            "~/.ssh/",
            "~/.config/",
        ]
    
    def validate_path(self, path: Union[str, Path]) -> Path:
        """Validate that a file path is allowed for operations.
        
        Args:
            path: The file path to validate
            
        Returns:
            Resolved Path object if valid
            
        Raises:
            SecurityViolationError: If path is not allowed
        """
        path = Path(path).resolve()
        
        # Check if path is within allowed directories
        path_allowed = False
        for allowed_dir in self.allowed_dirs:
            try:
                path.relative_to(allowed_dir)
                path_allowed = True
                break
            except ValueError:
                continue
        
        if not path_allowed:
            raise SecurityViolationError(
                f"Path access denied: {path}",
                f"Path must be within allowed directories: {[str(d) for d in self.allowed_dirs]}"
            )
        
        # Check for forbidden patterns
        path_str = str(path)
        for pattern in self.forbidden_patterns:
            if pattern in path_str:
                raise SecurityViolationError(
                    f"Path contains forbidden pattern: {path}",
                    f"Forbidden pattern: {pattern}"
                )
        
        return path
    
    def setup_sandbox(self) -> Path:
        """Set up the sandbox environment by copying target files.
        
        Returns:
            Path to the sandbox directory
        """
        try:
            # Copy all Python files from target to sandbox
            for py_file in self.target_dir.rglob("*.py"):
                relative_path = py_file.relative_to(self.target_dir)
                sandbox_file = self.sandbox_dir / relative_path
                
                # Create parent directories if needed
                sandbox_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy the file
                shutil.copy2(py_file, sandbox_file)
            
            return self.sandbox_dir
            
        except Exception as e:
            raise SecurityViolationError(
                "Failed to set up sandbox environment",
                str(e)
            )
    
    def safe_write(self, file_path: Union[str, Path], content: str) -> None:
        """Safely write content to a file within the sandbox.
        
        Args:
            file_path: Path to the file to write
            content: Content to write to the file
            
        Raises:
            SecurityViolationError: If path is not allowed
        """
        validated_path = self.validate_path(file_path)
        
        try:
            # Ensure parent directory exists
            validated_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the content
            with open(validated_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            raise SecurityViolationError(
                f"Failed to write file: {validated_path}",
                str(e)
            )
    
    def safe_read(self, file_path: Union[str, Path]) -> str:
        """Safely read content from a file within allowed directories.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            SecurityViolationError: If path is not allowed
        """
        validated_path = self.validate_path(file_path)
        
        try:
            with open(validated_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            raise SecurityViolationError(
                f"Failed to read file: {validated_path}",
                str(e)
            )
    
    def safe_copy(self, src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
        """Safely copy a file within allowed directories.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            
        Raises:
            SecurityViolationError: If either path is not allowed
        """
        validated_src = self.validate_path(src_path)
        validated_dst = self.validate_path(dst_path)
        
        try:
            # Ensure destination parent directory exists
            validated_dst.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy the file
            shutil.copy2(validated_src, validated_dst)
            
        except Exception as e:
            raise SecurityViolationError(
                f"Failed to copy file from {validated_src} to {validated_dst}",
                str(e)
            )
    
    def list_python_files(self, directory: Union[str, Path] = None) -> List[Path]:
        """List all Python files in a directory within the sandbox.
        
        Args:
            directory: Directory to search. If None, searches sandbox directory
            
        Returns:
            List of Python file paths
            
        Raises:
            SecurityViolationError: If directory is not allowed
        """
        if directory is None:
            directory = self.sandbox_dir
        
        validated_dir = self.validate_path(directory)
        
        try:
            return list(validated_dir.rglob("*.py"))
        except Exception as e:
            raise SecurityViolationError(
                f"Failed to list Python files in: {validated_dir}",
                str(e)
            )
    
    def cleanup_sandbox(self) -> None:
        """Clean up the sandbox directory if it was created temporarily."""
        if self._temp_sandbox and self.sandbox_dir.exists():
            try:
                shutil.rmtree(self.sandbox_dir)
            except Exception as e:
                # Log the error but don't raise - cleanup is best effort
                print(f"Warning: Failed to cleanup sandbox directory {self.sandbox_dir}: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.setup_sandbox()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_sandbox()