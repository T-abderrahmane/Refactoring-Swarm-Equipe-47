"""
Security and sandbox validation tests for the Refactoring Swarm system.

Tests security measures and sandbox isolation:
1. File operation restrictions and path validation
2. Sandbox isolation and security measures
3. Error handling for security violations
4. Prevention of unauthorized file access
"""

import os
import tempfile
import shutil
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.security.sandbox import SandboxManager
from src.tools.file_manager import SecureFileManager
from src.orchestrator.orchestrator import RefactoringOrchestrator
from src.exceptions import SecurityViolationError, RefactoringError


class TestSandboxSecurity(unittest.TestCase):
    """Test sandbox security and isolation measures."""
    
    def setUp(self):
        """Set up test environment with secure and insecure paths."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp(prefix="security_test_")
        self.target_dir = os.path.join(self.test_dir, "target")
        self.sandbox_dir = os.path.join(self.test_dir, "sandbox")
        
        # Create target directory with test files
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Create a test Python file
        test_file = os.path.join(self.target_dir, "test_file.py")
        with open(test_file, 'w') as f:
            f.write("def test_function():\n    return 'test'")
        
        # Initialize sandbox manager
        self.sandbox_manager = SandboxManager(self.target_dir, self.sandbox_dir)
        self.file_manager = SecureFileManager(self.sandbox_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_sandbox_initialization(self):
        """Test sandbox manager initialization and setup."""
        # Test successful initialization
        self.assertEqual(str(self.sandbox_manager.target_dir), str(Path(self.target_dir).resolve()))
        self.assertEqual(str(self.sandbox_manager.sandbox_dir), str(Path(self.sandbox_dir).resolve()))
        
        # Test sandbox setup
        sandbox_path = self.sandbox_manager.setup_sandbox()
        self.assertTrue(os.path.exists(sandbox_path))
        
        # Verify files were copied to sandbox
        sandbox_files = list(Path(sandbox_path).glob("*.py"))
        self.assertGreater(len(sandbox_files), 0)
    
    def test_sandbox_initialization_with_invalid_target(self):
        """Test sandbox initialization with invalid target directory."""
        invalid_target = "/nonexistent/directory"
        
        with self.assertRaises(SecurityViolationError):
            SandboxManager(invalid_target)
    
    def test_sandbox_initialization_with_file_as_target(self):
        """Test sandbox initialization with file instead of directory."""
        # Create a file instead of directory
        file_path = os.path.join(self.test_dir, "not_a_directory.txt")
        with open(file_path, 'w') as f:
            f.write("test")
        
        with self.assertRaises(SecurityViolationError):
            SandboxManager(file_path)
    
    def test_path_validation_allowed_paths(self):
        """Test path validation for allowed paths."""
        # Test target directory path
        target_file = os.path.join(self.target_dir, "allowed_file.py")
        validated_path = self.sandbox_manager.validate_path(target_file)
        self.assertEqual(str(validated_path), str(Path(target_file).resolve()))
        
        # Test sandbox directory path
        sandbox_file = os.path.join(self.sandbox_dir, "sandbox_file.py")
        validated_path = self.sandbox_manager.validate_path(sandbox_file)
        self.assertEqual(str(validated_path), str(Path(sandbox_file).resolve()))
    
    def test_path_validation_forbidden_paths(self):
        """Test path validation rejects forbidden paths."""
        forbidden_paths = [
            "/etc/passwd",
            "/usr/bin/python",
            "/bin/bash",
            "/sys/kernel",
            "/proc/version",
            "/dev/null",
            "/var/log/system.log",
            "/tmp/malicious_file",
            "/root/.bashrc",
            os.path.expanduser("~/.ssh/id_rsa"),
            os.path.expanduser("~/.config/secrets"),
        ]
        
        for forbidden_path in forbidden_paths:
            with self.assertRaises(SecurityViolationError):
                self.sandbox_manager.validate_path(forbidden_path)
    
    def test_path_validation_outside_allowed_directories(self):
        """Test path validation rejects paths outside allowed directories."""
        outside_paths = [
            "/home/user/documents/file.py",
            "/opt/application/config.py",
            "../../../etc/passwd",
            "../../sensitive_file.py",
            os.path.join(self.test_dir, "../outside_file.py"),
        ]
        
        for outside_path in outside_paths:
            with self.assertRaises(SecurityViolationError):
                self.sandbox_manager.validate_path(outside_path)
    
    def test_safe_file_operations_within_sandbox(self):
        """Test that safe file operations work within sandbox."""
        self.sandbox_manager.setup_sandbox()
        
        # Test safe write
        test_content = "def safe_function():\n    return 'safe'"
        safe_file = os.path.join(self.sandbox_dir, "safe_file.py")
        
        # Should not raise exception
        self.sandbox_manager.safe_write(safe_file, test_content)
        self.assertTrue(os.path.exists(safe_file))
        
        # Test safe read
        read_content = self.sandbox_manager.safe_read(safe_file)
        self.assertEqual(read_content, test_content)
        
        # Test safe copy
        copy_destination = os.path.join(self.sandbox_dir, "copied_file.py")
        self.sandbox_manager.safe_copy(safe_file, copy_destination)
        self.assertTrue(os.path.exists(copy_destination))
    
    def test_safe_file_operations_outside_sandbox(self):
        """Test that safe file operations reject paths outside sandbox."""
        # Test safe write to forbidden location
        forbidden_file = "/tmp/forbidden_file.py"
        test_content = "malicious content"
        
        with self.assertRaises(SecurityViolationError):
            self.sandbox_manager.safe_write(forbidden_file, test_content)
        
        # Test safe read from forbidden location
        with self.assertRaises(SecurityViolationError):
            self.sandbox_manager.safe_read("/etc/passwd")
        
        # Test safe copy to forbidden location
        source_file = os.path.join(self.target_dir, "test_file.py")
        forbidden_destination = "/tmp/malicious_copy.py"
        
        with self.assertRaises(SecurityViolationError):
            self.sandbox_manager.safe_copy(source_file, forbidden_destination)
    
    def test_list_python_files_security(self):
        """Test that listing Python files respects security boundaries."""
        self.sandbox_manager.setup_sandbox()
        
        # Test listing files in allowed directory
        sandbox_files = self.sandbox_manager.list_python_files(self.sandbox_dir)
        self.assertIsInstance(sandbox_files, list)
        self.assertGreater(len(sandbox_files), 0)
        
        # Test listing files in forbidden directory
        with self.assertRaises(SecurityViolationError):
            self.sandbox_manager.list_python_files("/etc")
    
    def test_sandbox_cleanup(self):
        """Test sandbox cleanup functionality."""
        # Setup sandbox
        sandbox_path = self.sandbox_manager.setup_sandbox()
        self.assertTrue(os.path.exists(sandbox_path))
        
        # Cleanup sandbox
        self.sandbox_manager.cleanup_sandbox()
        
        # For temporary sandboxes, directory should be removed
        if self.sandbox_manager._temp_sandbox:
            self.assertFalse(os.path.exists(sandbox_path))
    
    def test_context_manager_functionality(self):
        """Test sandbox manager as context manager."""
        with SandboxManager(self.target_dir) as sandbox:
            # Sandbox should be set up
            self.assertTrue(os.path.exists(sandbox.sandbox_dir))
            
            # Should be able to perform operations
            test_file = os.path.join(sandbox.sandbox_dir, "context_test.py")
            sandbox.safe_write(test_file, "# Context manager test")
            self.assertTrue(os.path.exists(test_file))
        
        # After context exit, temporary sandbox should be cleaned up
        # (Note: cleanup behavior depends on whether sandbox was temporary)


class TestSecureFileManager(unittest.TestCase):
    """Test secure file manager operations and security measures."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="file_manager_test_")
        self.target_dir = os.path.join(self.test_dir, "target")
        self.sandbox_dir = os.path.join(self.test_dir, "sandbox")
        
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Create test file
        self.test_file = os.path.join(self.target_dir, "test.py")
        with open(self.test_file, 'w') as f:
            f.write("def test():\n    return True")
        
        # Initialize managers
        self.sandbox_manager = SandboxManager(self.target_dir, self.sandbox_dir)
        self.file_manager = SecureFileManager(self.sandbox_manager)
    
    def tearDown(self):
        """Clean up test environment."""
        self.file_manager.cleanup_backups()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_secure_file_write_with_backup(self):
        """Test secure file writing with backup creation."""
        # Setup sandbox
        self.sandbox_manager.setup_sandbox()
        
        # Write to file in sandbox
        sandbox_file = os.path.join(self.sandbox_dir, "test.py")
        new_content = "def test():\n    return False"
        
        success = self.file_manager.safe_write_file(sandbox_file, new_content, create_backup=True)
        self.assertTrue(success)
        
        # Verify file was written
        self.assertTrue(os.path.exists(sandbox_file))
        with open(sandbox_file, 'r') as f:
            content = f.read()
        self.assertEqual(content, new_content)
        
        # Verify backup was created
        backups = self.file_manager.get_active_backups()
        self.assertGreater(len(backups), 0)
    
    def test_secure_file_write_forbidden_location(self):
        """Test that secure file writing rejects forbidden locations."""
        forbidden_file = "/etc/malicious_file.py"
        content = "malicious content"
        
        with self.assertRaises(SecurityViolationError):
            self.file_manager.safe_write_file(forbidden_file, content)
    
    def test_secure_file_read_allowed_location(self):
        """Test secure file reading from allowed locations."""
        # Read from target directory
        content = self.file_manager.safe_read_file(self.test_file)
        self.assertIn("def test()", content)
    
    def test_secure_file_read_forbidden_location(self):
        """Test that secure file reading rejects forbidden locations."""
        forbidden_files = [
            "/etc/passwd",
            "/usr/bin/python",
            "/root/.bashrc"
        ]
        
        for forbidden_file in forbidden_files:
            with self.assertRaises(SecurityViolationError):
                self.file_manager.safe_read_file(forbidden_file)
    
    def test_file_backup_and_restore(self):
        """Test file backup and restoration functionality."""
        # Create backup
        backup = self.file_manager.create_backup(self.test_file)
        self.assertIsNotNone(backup)
        self.assertTrue(os.path.exists(backup.backup_path))
        
        # Modify original file
        modified_content = "def test():\n    return 'modified'"
        with open(self.test_file, 'w') as f:
            f.write(modified_content)
        
        # Restore from backup
        success = self.file_manager.restore_backup(backup)
        self.assertTrue(success)
        
        # Verify restoration
        with open(self.test_file, 'r') as f:
            restored_content = f.read()
        self.assertIn("def test():\n    return True", restored_content)
    
    def test_file_operation_history_tracking(self):
        """Test that file operations are properly tracked."""
        # Setup sandbox
        self.sandbox_manager.setup_sandbox()
        
        # Perform several operations
        sandbox_file = os.path.join(self.sandbox_dir, "history_test.py")
        
        # Write file
        self.file_manager.safe_write_file(sandbox_file, "# Test 1")
        
        # Modify file
        self.file_manager.safe_write_file(sandbox_file, "# Test 2")
        
        # Get operation history
        history = self.file_manager.get_operation_history()
        self.assertGreater(len(history), 0)
        
        # Verify operation details
        for operation in history:
            self.assertIn(operation.operation_type, ["create", "modify", "delete", "move", "restore"])
            self.assertIsNotNone(operation.timestamp)
            self.assertIsInstance(operation.success, bool)
    
    def test_file_integrity_verification(self):
        """Test file integrity verification against backups."""
        # Create backup
        backup = self.file_manager.create_backup(self.test_file)
        
        # File should match backup initially
        self.assertTrue(self.file_manager.verify_file_integrity(self.test_file))
        
        # Modify file
        with open(self.test_file, 'w') as f:
            f.write("modified content")
        
        # File should no longer match backup
        self.assertFalse(self.file_manager.verify_file_integrity(self.test_file))
    
    def test_rollback_operations(self):
        """Test rollback of file operations."""
        # Setup sandbox
        self.sandbox_manager.setup_sandbox()
        
        # Perform operations
        sandbox_file = os.path.join(self.sandbox_dir, "rollback_test.py")
        
        # Create file
        self.file_manager.safe_write_file(sandbox_file, "# Original")
        
        # Modify file
        self.file_manager.safe_write_file(sandbox_file, "# Modified")
        
        # Rollback last operation
        results = self.file_manager.rollback_operations(count=1)
        self.assertGreater(len(results), 0)
        
        # Verify rollback worked (if backup was available)
        if any(results):
            with open(sandbox_file, 'r') as f:
                content = f.read()
            self.assertIn("Original", content)


class TestOrchestrationSecurity(unittest.TestCase):
    """Test security measures in the orchestration layer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp(prefix="orchestration_security_test_")
        self.target_dir = os.path.join(self.test_dir, "target")
        os.makedirs(self.target_dir, exist_ok=True)
        
        # Create test file
        test_file = os.path.join(self.target_dir, "secure_test.py")
        with open(test_file, 'w') as f:
            f.write("def secure_function():\n    return 'secure'")
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_orchestrator_rejects_invalid_target_directory(self):
        """Test that orchestrator rejects invalid target directories."""
        orchestrator = RefactoringOrchestrator(max_iterations=1)
        
        # Test nonexistent directory
        with self.assertRaises(RefactoringError):
            orchestrator.execute_refactoring("/nonexistent/directory")
        
        # Test file instead of directory
        test_file = os.path.join(self.test_dir, "not_directory.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        
        with self.assertRaises(RefactoringError):
            orchestrator.execute_refactoring(test_file)
    
    def test_orchestrator_sandbox_isolation(self):
        """Test that orchestrator properly isolates operations to sandbox."""
        mock_llm_fn = Mock(return_value="def secure_function():\n    return 'secure'")
        
        orchestrator = RefactoringOrchestrator(
            model="test-model",
            llm_fn=mock_llm_fn,
            max_iterations=1
        )
        
        # Execute refactoring
        results = orchestrator.execute_refactoring(self.target_dir)
        
        # Verify original files are unchanged
        original_file = os.path.join(self.target_dir, "secure_test.py")
        with open(original_file, 'r') as f:
            original_content = f.read()
        self.assertIn("def secure_function():\n    return 'secure'", original_content)
        
        # Verify sandbox was used (if results indicate success)
        if results.get("success"):
            # Sandbox should have been created and used
            self.assertIsNotNone(results.get("session_info"))
    
    def test_security_violation_error_handling(self):
        """Test proper handling of security violation errors."""
        # Create a mock that raises security violations
        mock_sandbox = Mock()
        mock_sandbox.validate_path.side_effect = SecurityViolationError("Access denied")
        
        # Test that security violations are properly caught and handled
        with patch('src.security.sandbox.SandboxManager') as mock_sandbox_class:
            mock_sandbox_class.return_value = mock_sandbox
            
            orchestrator = RefactoringOrchestrator(max_iterations=1)
            
            # Should handle security violation gracefully
            results = orchestrator.execute_refactoring(self.target_dir)
            
            # Should indicate failure due to security violation
            self.assertFalse(results["success"])
            self.assertTrue(results["has_errors"])
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        sandbox_manager = SandboxManager(self.target_dir)
        
        # Test various path traversal attempts
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/target/../../../etc/passwd",
            "target/../../etc/passwd",
            "./../../etc/passwd",
            "target/../sensitive_file.py",
        ]
        
        for attempt in traversal_attempts:
            with self.assertRaises(SecurityViolationError):
                sandbox_manager.validate_path(attempt)
    
    def test_symlink_attack_prevention(self):
        """Test prevention of symlink-based attacks."""
        # Create a symlink pointing outside allowed directories
        if os.name != 'nt':  # Skip on Windows where symlinks require special permissions
            symlink_path = os.path.join(self.target_dir, "malicious_symlink")
            try:
                os.symlink("/etc/passwd", symlink_path)
                
                sandbox_manager = SandboxManager(self.target_dir)
                
                # Should reject symlink pointing outside allowed directories
                with self.assertRaises(SecurityViolationError):
                    sandbox_manager.safe_read(symlink_path)
                    
            except OSError:
                # Skip test if symlink creation fails (e.g., insufficient permissions)
                self.skipTest("Cannot create symlinks in test environment")
    
    def test_resource_exhaustion_prevention(self):
        """Test prevention of resource exhaustion attacks."""
        sandbox_manager = SandboxManager(self.target_dir)
        
        # Test writing extremely large content
        large_content = "x" * (10 * 1024 * 1024)  # 10MB
        large_file = os.path.join(self.target_dir, "large_file.py")
        
        # Should handle large files gracefully (may succeed or fail based on system limits)
        try:
            sandbox_manager.safe_write(large_file, large_content)
            # If it succeeds, verify the file was created
            self.assertTrue(os.path.exists(large_file))
        except (SecurityViolationError, OSError):
            # If it fails, that's also acceptable for security
            pass
    
    def test_concurrent_access_safety(self):
        """Test safety of concurrent file operations."""
        import threading
        import time
        
        sandbox_manager = SandboxManager(self.target_dir)
        file_manager = SecureFileManager(sandbox_manager)
        
        test_file = os.path.join(self.target_dir, "concurrent_test.py")
        results = []
        
        def write_operation(content_id):
            try:
                content = f"# Content {content_id}\ndef func_{content_id}():\n    return {content_id}"
                success = file_manager.safe_write_file(test_file, content)
                results.append(("write", content_id, success))
            except Exception as e:
                results.append(("write", content_id, False, str(e)))
        
        def read_operation(read_id):
            try:
                content = file_manager.safe_read_file(test_file)
                results.append(("read", read_id, len(content) > 0))
            except Exception as e:
                results.append(("read", read_id, False, str(e)))
        
        # Create initial file
        file_manager.safe_write_file(test_file, "# Initial content")
        
        # Start concurrent operations
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=write_operation, args=(i,))
            t2 = threading.Thread(target=read_operation, args=(i,))
            threads.extend([t1, t2])
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # Verify no crashes occurred and file remains valid
        self.assertTrue(os.path.exists(test_file))
        final_content = file_manager.safe_read_file(test_file)
        self.assertGreater(len(final_content), 0)
        
        # Cleanup
        file_manager.cleanup_backups()


if __name__ == '__main__':
    unittest.main()