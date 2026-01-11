#!/usr/bin/env python3
"""
Integration test runner for the Refactoring Swarm system.

This script runs all integration tests and provides detailed reporting
on test results, including coverage of requirements validation.
"""

import os
import sys
import unittest
import json
import time
from pathlib import Path
from io import StringIO

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)


def run_integration_tests():
    """Run all integration tests and return results."""
    print("=" * 60)
    print("REFACTORING SWARM - INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Discover and run tests
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    
    # Load integration tests
    integration_suite = loader.discover(
        start_dir=str(test_dir / "integration"),
        pattern="test_*.py",
        top_level_dir=str(test_dir)
    )
    
    # Create test runner with detailed output
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        buffer=True
    )
    
    # Run tests
    print("Running integration tests...")
    start_time = time.time()
    result = runner.run(integration_suite)
    end_time = time.time()
    
    # Print results
    print(stream.getvalue())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"Execution Time: {end_time - start_time:.2f} seconds")
    
    # Success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Detailed failure information
    if result.failures:
        print("\n" + "=" * 60)
        print("FAILURES")
        print("=" * 60)
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print("-" * 40)
            print(traceback)
    
    if result.errors:
        print("\n" + "=" * 60)
        print("ERRORS")
        print("=" * 60)
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print("-" * 40)
            print(traceback)
    
    # Requirements coverage analysis
    print("\n" + "=" * 60)
    print("REQUIREMENTS COVERAGE ANALYSIS")
    print("=" * 60)
    
    requirements_coverage = analyze_requirements_coverage(result)
    for req_id, coverage_info in requirements_coverage.items():
        status = "✅ COVERED" if coverage_info["covered"] else "❌ NOT COVERED"
        print(f"{req_id}: {status} ({coverage_info['test_count']} tests)")
    
    # Overall assessment
    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ End-to-end workflow validation: COMPLETE")
        print("✅ Security and sandbox validation: COMPLETE")
        print("✅ Logging and data integrity: VALIDATED")
        return True
    else:
        print("⚠️  SOME INTEGRATION TESTS FAILED")
        print(f"❌ {len(result.failures)} test failures")
        print(f"❌ {len(result.errors)} test errors")
        print("\nPlease review the failures above and fix the issues.")
        return False


def analyze_requirements_coverage(test_result):
    """Analyze which requirements are covered by the tests."""
    # Map requirements to test coverage
    # This is based on the requirements specified in the task details
    requirements_map = {
        "1.1": {
            "description": "Process all Python files in directory",
            "covered": False,
            "test_count": 0
        },
        "1.2": {
            "description": "Produce clean, functional Python code",
            "covered": False,
            "test_count": 0
        },
        "1.3": {
            "description": "Improve Pylint quality score",
            "covered": False,
            "test_count": 0
        },
        "1.5": {
            "description": "Operate without human intervention",
            "covered": False,
            "test_count": 0
        },
        "4.1": {
            "description": "Only write files within sandbox directory",
            "covered": False,
            "test_count": 0
        },
        "4.2": {
            "description": "Prevent access outside target/sandbox",
            "covered": False,
            "test_count": 0
        },
        "4.3": {
            "description": "Validate file paths before operations",
            "covered": False,
            "test_count": 0
        },
        "4.4": {
            "description": "Reject system file modifications",
            "covered": False,
            "test_count": 0
        }
    }
    
    # Analyze test names and methods to determine coverage
    # This is a simplified analysis based on test method names
    for test, _ in test_result.failures + test_result.errors:
        test_name = str(test).lower()
        
        # Map test names to requirements
        if "workflow" in test_name or "end_to_end" in test_name:
            requirements_map["1.1"]["test_count"] += 1
            requirements_map["1.2"]["test_count"] += 1
            requirements_map["1.5"]["test_count"] += 1
        
        if "security" in test_name or "sandbox" in test_name:
            requirements_map["4.1"]["test_count"] += 1
            requirements_map["4.2"]["test_count"] += 1
            requirements_map["4.3"]["test_count"] += 1
            requirements_map["4.4"]["test_count"] += 1
    
    # Mark as covered if tests exist (even if they failed)
    for req_id, req_info in requirements_map.items():
        if req_info["test_count"] > 0:
            req_info["covered"] = True
    
    return requirements_map


def validate_test_environment():
    """Validate that the test environment is properly set up."""
    print("Validating test environment...")
    
    # Check that source code is available
    src_dir = Path(__file__).parent.parent / "src"
    if not src_dir.exists():
        print("❌ Source directory not found")
        return False
    
    # Check for required modules
    required_modules = [
        "src.orchestrator.orchestrator",
        "src.security.sandbox",
        "src.tools.file_manager",
        "src.exceptions"
    ]
    
    for module_name in required_modules:
        try:
            __import__(module_name)
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            return False
    
    # Check for test fixtures
    fixtures_dir = Path(__file__).parent / "fixtures"
    if not fixtures_dir.exists():
        print("❌ Test fixtures directory not found")
        return False
    
    print("✅ Test environment validation passed")
    return True


def main():
    """Main entry point for test runner."""
    print("Refactoring Swarm Integration Test Runner")
    print("=" * 60)
    
    # Validate environment
    if not validate_test_environment():
        print("\n❌ Test environment validation failed")
        sys.exit(1)
    
    # Run tests
    success = run_integration_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()