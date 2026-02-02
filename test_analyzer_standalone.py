"""
Standalone Analyzer Test Script
Tests the Analyzer agent in isolation with comprehensive logging.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment
load_dotenv()

def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80 + "\n")


def get_llm():
    """Get configured Gemini LLM instance with logging."""
    print_section("INITIALIZING LLM")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
    
    print(f"✓ API Key found: {api_key[:20]}...{api_key[-4:]}")
    print(f"✓ API Key length: {len(api_key)} characters")
    
    print("\n🔧 Creating ChatGoogleGenerativeAI instance...")
    print(f"   Model: gemini-2.5-flash")
    print(f"   Temperature: 0.3")
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=api_key
    )
    
    print("✓ LLM instance created successfully")
    return llm


def read_test_file():
    """Read a test Python file."""
    print_section("READING TEST FILE")
    
    test_file = Path("./sandbox/test_code.py")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        sys.exit(1)
    
    print(f"📄 Reading: {test_file}")
    
    with open(test_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"✓ File read successfully")
    print(f"✓ File size: {len(content)} characters")
    print(f"✓ Lines: {len(content.splitlines())}")
    
    return str(test_file), content


def run_simple_analysis():
    """Run a simple code analysis with one LLM call."""
    print_section("STANDALONE ANALYZER TEST")
    print("This script tests the Analyzer agent in isolation")
    print("Making ONE LLM request with comprehensive logging\n")
    
    # Read test file
    file_path, file_content = read_test_file()
    
    # Prepare prompts
    print_section("PREPARING PROMPTS")
    
    system_prompt = """You are an expert Python code analyzer.

Analyze the provided Python code and identify issues such as:
1. Syntax errors and bugs
2. Missing or incorrect documentation
3. Code style issues (PEP 8)
4. Unused imports or variables
5. Code complexity issues

Provide a brief analysis with the top 3-5 issues found."""

    user_prompt = f"""Analyze this Python file:

File: {file_path}

Code:
```python
{file_content}
```

Please provide a concise analysis listing the main issues."""

    print("📝 System Prompt:")
    print("-" * 80)
    print(system_prompt)
    print("-" * 80)
    
    print("\n📝 User Prompt:")
    print("-" * 80)
    print(user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
    print("-" * 80)
    
    print(f"\n📊 Prompt Stats:")
    print(f"   System prompt: {len(system_prompt)} characters")
    print(f"   User prompt: {len(user_prompt)} characters")
    print(f"   Total: {len(system_prompt) + len(user_prompt)} characters")
    print(f"   Estimated tokens: ~{(len(system_prompt) + len(user_prompt)) // 4}")
    
    # Initialize LLM
    llm = get_llm()
    
    # Make LLM call with retry and comprehensive logging
    print_section("MAKING LLM REQUEST")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def call_llm_with_retry(attempt=1):
        print(f"\n🔄 Attempt {attempt}/5")
        print(f"⏰ Adding 3-second delay before request...")
        time.sleep(3)  # Rate limiting
        
        print(f"📤 Sending request to Gemini API...")
        start_time = time.time()
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            print(f"   Message count: {len(messages)}")
            print(f"   Message types: {[type(m).__name__ for m in messages]}")
            
            response = llm.invoke(messages)
            
            elapsed = time.time() - start_time
            print(f"✓ Response received in {elapsed:.2f} seconds")
            
            return response
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"❌ Request failed after {elapsed:.2f} seconds")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Error message: {str(e)}")
            
            # Check for specific error codes
            error_str = str(e).lower()
            if '429' in error_str or 'rate limit' in error_str or 'quota' in error_str:
                print("\n⚠️  RATE LIMIT ERROR DETECTED!")
                print("   This is a 429 error - too many requests")
                print("   Waiting before retry...")
            elif '401' in error_str or 'unauthorized' in error_str:
                print("\n⚠️  AUTHENTICATION ERROR!")
                print("   Check your API key")
            elif '400' in error_str or 'bad request' in error_str:
                print("\n⚠️  BAD REQUEST ERROR!")
                print("   There may be an issue with the request format")
            
            raise
    
    # Attempt the call
    try:
        response = call_llm_with_retry()
        
        # Process response
        print_section("RESPONSE RECEIVED")
        
        content = response.content
        
        print(f"✓ Response content length: {len(content)} characters")
        print(f"✓ Response type: {type(response).__name__}")
        
        print("\n📥 Response Content:")
        print("-" * 80)
        print(content)
        print("-" * 80)
        
        # Additional response metadata if available
        if hasattr(response, 'response_metadata'):
            print("\n📊 Response Metadata:")
            for key, value in response.response_metadata.items():
                print(f"   {key}: {value}")
        
        print_section("TEST COMPLETED SUCCESSFULLY")
        print("✓ Analyzer test completed without errors")
        print("✓ LLM responded successfully")
        return content
        
    except Exception as e:
        print_section("TEST FAILED")
        print(f"❌ Error: {type(e).__name__}")
        print(f"❌ Message: {str(e)}")
        
        # Print full traceback
        import traceback
        print("\n📋 Full Traceback:")
        print("-" * 80)
        traceback.print_exc()
        print("-" * 80)
        
        sys.exit(1)


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" "*25 + "ANALYZER STANDALONE TEST")
    print(" "*20 + "Single Request Debugging Tool")
    print("="*80)
    
    run_simple_analysis()
    
    print("\n" + "="*80)
    print("✓ Script completed")
    print("="*80 + "\n")
