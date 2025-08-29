#!/usr/bin/env python3
"""
Test script to verify Claude setup and API connectivity.
"""

import os
import json
from anthropic import Anthropic

def test_claude_setup():
    """Test Claude API setup and basic functionality."""
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment variables")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return False
    
    print("✅ API key found")
    
    # Test client creation
    try:
        client = Anthropic(api_key=api_key)
        print("✅ Anthropic client created successfully")
    except Exception as e:
        print(f"❌ Failed to create client: {e}")
        return False
    
    # Test basic API call
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[
                {"role": "user", "content": "Say 'Hello, Claude is working!'"}
            ]
        )
        print(f"✅ API call successful: {response.content[0].text.strip()}")
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False
    
    # Test prompt data file
    try:
        with open('prompt_test_2_grouped.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Prompt data loaded: {len(data['outputs'])} prompts found")
    except Exception as e:
        print(f"❌ Failed to load prompt data: {e}")
        return False
    
    print("\n🎉 All tests passed! Claude setup is ready.")
    return True

if __name__ == "__main__":
    test_claude_setup() 