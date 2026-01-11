"""
Secure file manager with sandbox-restricted operations.
Implements file backup and restoration capabilities with path validation and security checks.
"""

import os
import shutil
import tempfile
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from ..exceptions import SecurityViolationError
from ..security.sandbox import SandboxManager


@dataclass
class FileBackup:
    """Represents a file backup with metadata."""
    original_path: str
    backup_path: str
    timestamp: datetime
    checksum: str
    size: int


@dataclass
class FileOperation:
    """Represents a file operation for tracking and rollback."""
    operation_type: str  # "create", "modify", "delete", "move"
    target_path: str
    backup_info: Optional[FileBackup]
    timestamp: datetime
    success: bool


class SecureFileManager:
    """
    Secure file manager that enforces sandbox restrictions and provides
    backup/restoration capabilities for safe file operations.
    """
    
    def __init__(self, sandbox_manager: SandboxManager):
        """
        Initialize secure file manager.
        
        Args:
            sandbox_manager: SandboxManager instance for path validation
        """
        self.sandbox_manager = sandbox_manager
        self.backup_dir = None
        self.operation_history: List[FileOperation] = []
        self._active_backups: Dict[str, FileBackup] = {}
        self._initialize_backup_directory()
    
    def _initialize_backup_directory(self) -> None:
        """Initialize temporary backup directory."""
        self.backup_dir = tempfile.mkdtemp(prefix="refactoring_backup_")
    
    def _validate_path(self, path: str, operation: str) -> str:
        """
        Validate file path against sandbox restrictions.
        
        Args:
            path: File path to validate
            operation: Type of operation being performed
            
        Returns:
            Normalized absolute path
            
        Raises:
            SecurityViolationError: If path is not allowed
        """
        try:
            normalized_path = os.path.abspath(path)
            
            # Use sandbox manager for validation
            self.sandbox_manager.validate_path(normalized_path)
            
            return normalized_path
            
        except Exception as e:
            if isinstance(e, SecurityViolationError):
                raise
            raise SecurityViolationError(f"Path validation failed for {path}: {e}")
    
    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculate MD5 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            MD5 checksum as hex string
        """
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def create_backup(self, file_path: str) -> Optional[FileBackup]:
        """
        Create a backup of the specified file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            FileBackup object if successful, None if file doesn't exist
            
        Raises:
            SecurityViolationError: If path is not allowed
        """
        validated_path = self._validate_path(file_path, "read")
        
        if not os.path.exists(validated_path):
            return None
        
        if not os.path.isfile(validated_path):
            raise SecurityViolationError(f"Path is not a file: {file_path}")
        
        try:
            # Generate backup filename
            timestamp = datetime.now()
            backup_filename = f"{os.path.basename(validated_path)}.{timestamp.strftime('%Y%m%d_%H%M%S')}.backup"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Copy file to backup location
            shutil.copy2(validated_path, backup_path)
            
            # Calculate checksum and size
            checksum = self._calculate_checksum(validated_path)
            size = os.path.getsize(validated_path)
            
            backup = FileBackup(
                original_path=validated_path,
                backup_path=backup_path,
                timestamp=timestamp,
                checksum=checksum,
                size=size
            )
            
            # Store backup reference
            self._active_backups[validated_path] = backup
            
            return backup
            
        except Exception as e:
            raise SecurityViolationError(f"Failed to create backup for {file_path}: {e}")
    
    def restore_backup(self, backup: FileBackup) -> bool:
        """
        Restore a file from backup.
        
        Args:
            backup: FileBackup object to restore
            
        Returns:
            True if restoration successful, False otherwise
            
        Raises:
            SecurityViolationError: If restoration is not allowed
        """
        if not os.path.exists(backup.backup_path):
            raise SecurityViolationError(f"Backup file not found: {backup.backup_path}")
        
        # Validate target path for writing
        self._validate_path(backup.original_path, "write")
        
        try:
            # Restore file
            shutil.copy2(backup.backup_path, backup.original_path)
            
            # Verify restoration
            restored_checksum = self._calculate_checksum(backup.original_path)
            if restored_checksum != backup.checksum:
                raise SecurityViolationError("Backup restoration failed: checksum mismatch")
            
            # Record operation
            operation = FileOperation(
                operation_type="restore",
                target_path=backup.original_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=True
            )
            self.operation_history.append(operation)
            
            return True
            
        except Exception as e:
            # Record failed operation
            operation = FileOperation(
                operation_type="restore",
                target_path=backup.original_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=False
            )
            self.operation_history.append(operation)
            
            raise SecurityViolationError(f"Failed to restore backup: {e}")
    
    def safe_write_file(self, file_path: str, content: str, create_backup: bool = True) -> bool:
        """
        Safely write content to a file with optional backup.
        
        Args:
            file_path: Path to the file to write
            content: Content to write
            create_backup: Whether to create backup before writing
            
        Returns:
            True if write successful, False otherwise
            
        Raises:
            SecurityViolationError: If write is not allowed
        """
        validated_path = self._validate_path(file_path, "write")
        
        backup = None
        if create_backup and os.path.exists(validated_path):
            backup = self.create_backup(validated_path)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(validated_path), exist_ok=True)
            
            # Write content
            with open(validated_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Record successful operation
            operation = FileOperation(
                operation_type="modify" if backup else "create",
                target_path=validated_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=True
            )
            self.operation_history.append(operation)
            
            return True
            
        except Exception as e:
            # Record failed operation
            operation = FileOperation(
                operation_type="modify" if backup else "create",
                target_path=validated_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=False
            )
            self.operation_history.append(operation)
            
            # Attempt to restore backup if write failed
            if backup:
                try:
                    self.restore_backup(backup)
                except:
                    pass  # Restoration failed, but we still need to report original error
            
            raise SecurityViolationError(f"Failed to write file {file_path}: {e}")
    
    def safe_read_file(self, file_path: str) -> str:
        """
        Safely read content from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content as string
            
        Raises:
            SecurityViolationError: If read is not allowed
        """
        validated_path = self._validate_path(file_path, "read")
        
        if not os.path.exists(validated_path):
            raise SecurityViolationError(f"File not found: {file_path}")
        
        if not os.path.isfile(validated_path):
            raise SecurityViolationError(f"Path is not a file: {file_path}")
        
        try:
            with open(validated_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise SecurityViolationError(f"Failed to read file {file_path}: {e}")
    
    def safe_delete_file(self, file_path: str, create_backup: bool = True) -> bool:
        """
        Safely delete a file with optional backup.
        
        Args:
            file_path: Path to the file to delete
            create_backup: Whether to create backup before deletion
            
        Returns:
            True if deletion successful, False otherwise
            
        Raises:
            SecurityViolationError: If deletion is not allowed
        """
        validated_path = self._validate_path(file_path, "write")
        
        if not os.path.exists(validated_path):
            return True  # File already doesn't exist
        
        backup = None
        if create_backup:
            backup = self.create_backup(validated_path)
        
        try:
            os.remove(validated_path)
            
            # Record successful operation
            operation = FileOperation(
                operation_type="delete",
                target_path=validated_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=True
            )
            self.operation_history.append(operation)
            
            return True
            
        except Exception as e:
            # Record failed operation
            operation = FileOperation(
                operation_type="delete",
                target_path=validated_path,
                backup_info=backup,
                timestamp=datetime.now(),
                success=False
            )
            self.operation_history.append(operation)
            
            raise SecurityViolationError(f"Failed to delete file {file_path}: {e}")
    
    def safe_move_file(self, src_path: str, dst_path: str, create_backup: bool = True) -> bool:
        """
        Safely move a file with optional backup.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            create_backup: Whether to create backup of destination if it exists
            
        Returns:
            True if move successful, False otherwise
            
        Raises:
            SecurityViolationError: If move is not allowed
        """
        validated_src = self._validate_path(src_path, "read")
        validated_dst = self._validate_path(dst_path, "write")
        
        if not os.path.exists(validated_src):
            raise SecurityViolationError(f"Source file not found: {src_path}")
        
        backup = None
        if create_backup and os.path.exists(validated_dst):
            backup = self.create_backup(validated_dst)
        
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(validated_dst), exist_ok=True)
            
            # Move file
            shutil.move(validated_src, validated_dst)
            
            # Record successful operation
            operation = FileOperation(
                operation_type="move",
                target_path=f"{validated_src} -> {validated_dst}",
                backup_info=backup,
                timestamp=datetime.now(),
                success=True
            )
            self.operation_history.append(operation)
            
            return True
            
        except Exception as e:
            # Record failed operation
            operation = FileOperation(
                operation_type="move",
                target_path=f"{validated_src} -> {validated_dst}",
                backup_info=backup,
                timestamp=datetime.now(),
                success=False
            )
            self.operation_history.append(operation)
            
            # Attempt to restore backup if move failed and destination was overwritten
            if backup and not os.path.exists(validated_dst):
                try:
                    self.restore_backup(backup)
                except:
                    pass
            
            raise SecurityViolationError(f"Failed to move file from {src_path} to {dst_path}: {e}")
    
    def rollback_operations(self, count: Optional[int] = None) -> List[bool]:
        """
        Rollback recent file operations using backups.
        
        Args:
            count: Number of operations to rollback (None for all)
            
        Returns:
            List of rollback success status for each operation
        """
        if not self.operation_history:
            return []
        
        operations_to_rollback = self.operation_history[-count:] if count else self.operation_history
        operations_to_rollback.reverse()  # Rollback in reverse order
        
        results = []
        
        for operation in operations_to_rollback:
            if not operation.success or not operation.backup_info:
                results.append(False)
                continue
            
            try:
                if operation.operation_type == "delete":
                    # Restore deleted file
                    self.restore_backup(operation.backup_info)
                    results.append(True)
                elif operation.operation_type in ["modify", "create"]:
                    # Restore previous version
                    self.restore_backup(operation.backup_info)
                    results.append(True)
                else:
                    results.append(False)
            except Exception:
                results.append(False)
        
        return results
    
    def get_operation_history(self) -> List[FileOperation]:
        """
        Get the history of file operations.
        
        Returns:
            List of FileOperation objects
        """
        return self.operation_history.copy()
    
    def get_active_backups(self) -> Dict[str, FileBackup]:
        """
        Get currently active backups.
        
        Returns:
            Dictionary mapping file paths to their backups
        """
        return self._active_backups.copy()
    
    def cleanup_backups(self) -> None:
        """Clean up all backup files and temporary directories."""
        try:
            if self.backup_dir and os.path.exists(self.backup_dir):
                shutil.rmtree(self.backup_dir)
            self._active_backups.clear()
        except Exception:
            pass  # Ignore cleanup errors
    
    def verify_file_integrity(self, file_path: str) -> bool:
        """
        Verify file integrity against its backup checksum.
        
        Args:
            file_path: Path to the file to verify
            
        Returns:
            True if file matches backup checksum, False otherwise
        """
        validated_path = self._validate_path(file_path, "read")
        
        if validated_path not in self._active_backups:
            return True  # No backup to compare against
        
        backup = self._active_backups[validated_path]
        current_checksum = self._calculate_checksum(validated_path)
        
        return current_checksum == backup.checksum
    
    def get_file_stats(self, file_path: str) -> Dict[str, any]:
        """
        Get file statistics and metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file statistics
        """
        validated_path = self._validate_path(file_path, "read")
        
        if not os.path.exists(validated_path):
            return {"exists": False}
        
        try:
            stat = os.stat(validated_path)
            return {
                "exists": True,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "created": datetime.fromtimestamp(stat.st_ctime),
                "is_file": os.path.isfile(validated_path),
                "is_directory": os.path.isdir(validated_path),
                "checksum": self._calculate_checksum(validated_path) if os.path.isfile(validated_path) else None
            }
        except Exception as e:
            raise SecurityViolationError(f"Failed to get file stats for {file_path}: {e}")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.cleanup_backups()