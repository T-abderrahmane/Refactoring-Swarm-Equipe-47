#!/usr/bin/env python3
"""
Test script to verify LLM integration is working correctly.
"""
import os
import sys
from dotenv import load_dotenv
from google import genai

load_dotenv()

def test_llm_connection():
    """Test if LLM connection works."""
    print("🧪 Testing LLM Integration...")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in .env file")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    
    try:
        # Initialize client
        print("\n📡 Connecting to Google Gemini API...")
        client = genai.Client(api_key=api_key)
        
        # Test with a simple prompt
        print("🤖 Sending test prompt to gemini-flash-latest...")
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents="Say 'Hello from Gemini!' in one sentence."
        )
        
        print(f"✅ Response received: {response.text}")
        print("\n" + "=" * 60)
        print("🎉 LLM Integration Test PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        error_str = str(e)
        print(f"\n❌ LLM Integration Test FAILED: {error_str}")
        
       
       
if __name__ == "__main__":
    success = test_llm_connection()
    sys.exit(0 if success else 1)
