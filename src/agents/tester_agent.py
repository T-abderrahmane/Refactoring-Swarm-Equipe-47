"""
Tester Agent - Generates test scripts for source files that lack them.

This agent is responsible for:
1. Checking which source files have corresponding test scripts
2. Reading the source code of files missing tests
3. Using LLM to generate comprehensive test scripts
4. Writing the generated test files to the sandbox
"""

import os
import time
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from src.models.graph_state import RefactoringState
from src.tools.refactoring_tools import read_python_file, write_python_file, list_python_files
from src.utils.logger import log_llm_interaction, ActionType


def get_llm():
    """Get configured Gemini LLM instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )


def tester_node(state: RefactoringState) -> Dict:
    """
    Tester agent node for LangGraph.

    Generates test scripts for source files that don't already have one.

    Args:
        state: Current workflow state

    Returns:
        Updated state with generated test file information
    """
    print(f"\n{'='*60}")
    print("🧪 TESTER AGENT: Generating missing test scripts...")
    print(f"{'='*60}")

    python_files = state.get("python_files", [])
    iteration = state.get("iteration", 0)

    if not python_files:
        print("❌ No source files to generate tests for")
        return {
            "generated_test_files": []
        }

    # Step 1: Discover existing test files in the sandbox
    print("\n📂 Discovering existing test files...")
    files_result = list_python_files.invoke({"directory": "."})
    existing_files = [f.strip() for f in files_result.split("\n") if f.strip()] if "Error" not in files_result else []
    existing_test_files = {os.path.basename(f) for f in existing_files if os.path.basename(f).startswith("test_")}
    print(f"  Found {len(existing_test_files)} existing test files: {existing_test_files}")

    # Step 2: Identify source files missing a corresponding test script
    source_files = [f for f in python_files if not os.path.basename(f).startswith("test_")]
    missing_tests = []
    for src_file in source_files:
        src_basename = os.path.basename(src_file)
        expected_test = f"test_{src_basename}"
        if expected_test not in existing_test_files:
            missing_tests.append(src_file)

    if not missing_tests:
        print("✅ All source files already have corresponding test scripts")
        return {
            "generated_test_files": []
        }

    print(f"⚠️  {len(missing_tests)} source files missing tests: {missing_tests}")

    # Step 3: Generate test scripts for each file missing one
    generated_test_files = []

    for file_path in missing_tests:
        src_basename = os.path.basename(file_path)
        src_dir = os.path.dirname(file_path)
        test_file_name = f"test_{src_basename}"
        test_file_path = os.path.join(src_dir, test_file_name) if src_dir else test_file_name

        print(f"\n  • Generating tests for: {file_path}")

        # Read the source file
        source_content = read_python_file.invoke({"file_path": file_path})
        if "Error" in source_content:
            print(f"    ❌ Could not read file: {source_content}")
            continue

        # Create prompt for LLM
        system_prompt = """You are an expert Python test engineer specializing in writing comprehensive pytest test suites.

Your task is to generate a complete test script for the provided Python source file.

Guidelines:
1. Import the module's public functions/classes using relative imports (e.g., from module_name import func)
2. Write tests using plain assert statements (pytest style)
3. Cover normal cases, edge cases, and error cases
4. Each test function should test one specific behavior
5. Use descriptive test function names starting with test_
6. Keep tests simple, readable, and independent
7. Test error handling by verifying exceptions are raised when expected

IMPORTANT: Return ONLY the complete Python test code, nothing else.
Do not include explanations, comments about changes, or markdown formatting.
Just return the raw Python code for the test file."""

        module_name = os.path.splitext(src_basename)[0]
        user_prompt = f"""Generate a comprehensive pytest test script for the following Python module.

Module name: {module_name}
File path: {file_path}

Source code:
```python
{source_content}
```

Generate a test file that thoroughly tests all public functions and classes in this module."""

        try:
            llm = get_llm()

            # Call LLM to generate tests with retry logic
            @retry(
                stop=stop_after_attempt(5),
                wait=wait_exponential(multiplier=1, min=4, max=60),
                retry=retry_if_exception_type((Exception,)),
                reraise=True
            )
            def call_llm_with_retry():
                time.sleep(2)  # Rate limiting: 2 second delay before each call
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt)
                ]
                return llm.invoke(messages)

            response = call_llm_with_retry()
            test_code = response.content

            # Clean up the response (remove markdown code blocks if present)
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0].strip()

            # Log the LLM interaction
            log_llm_interaction(
                agent_name="Tester",
                model_used="gemini-2.5-flash",
                action=ActionType.GENERATION,
                input_prompt=f"{system_prompt}\n\n{user_prompt}",
                output_response=test_code,
                status="SUCCESS"
            )

            # Write the test file
            write_result = write_python_file.invoke({
                "file_path": test_file_path,
                "content": test_code
            })

            if "Success" in write_result:
                print(f"    ✅ Generated and saved: {test_file_path}")
                generated_test_files.append(test_file_path)
            else:
                print(f"    ❌ Failed to write test file: {write_result}")

        except Exception as e:
            error_msg = f"Failed to generate tests for {file_path}: {str(e)}"
            print(f"    ❌ {error_msg}")

            # Log failed interaction
            log_llm_interaction(
                agent_name="Tester",
                model_used="gemini-2.5-flash",
                action=ActionType.GENERATION,
                input_prompt=f"{system_prompt}\n\n{user_prompt}",
                output_response=str(e),
                status="FAILURE"
            )

    # Create summary
    print(f"\n📋 Tester Summary:")
    print(f"  • Source files missing tests: {len(missing_tests)}")
    print(f"  • Test files generated: {len(generated_test_files)}")
    print(f"{'='*60}\n")

    return {
        "generated_test_files": generated_test_files
    }
