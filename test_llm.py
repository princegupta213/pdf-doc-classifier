#!/usr/bin/env python3
"""
Test script for LLM functionality in PDF Document Classifier.
"""

import os
import sys
from openai import OpenAI

def test_llm_connection():
    """Test LLM connection and functionality."""
    print("🧪 Testing LLM Connection for PDF Document Classifier")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found in environment variables")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print(f"✅ API key found: {api_key[:20]}...")
    
    # Test connection
    try:
        client = OpenAI(api_key=api_key)
        print("✅ OpenAI client created successfully")
        
        # Test simple call
        print("🧪 Testing simple API call...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'Hello'"}],
            max_tokens=5
        )
        
        print("✅ API call successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        # Test document classification prompt
        print("\n🧪 Testing document classification prompt...")
        classification_prompt = """
        You are an expert document classifier. Analyze this text and classify it:
        
        "Invoice #12345
        Bill To: John Doe
        Total Amount: $150.00
        Due Date: 2024-01-15"
        
        Respond with JSON: {"classification": "invoice", "confidence": 0.95, "evidence": ["invoice number", "bill to", "total amount"]}
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert document classifier. Always respond with valid JSON."},
                {"role": "user", "content": classification_prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        print("✅ Document classification test successful!")
        print(f"Response: {response.choices[0].message.content}")
        
        print("\n🎉 All LLM tests passed! Your app should work with LLM features.")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        
        # Provide specific error guidance
        if "quota" in str(e).lower():
            print("\n💡 Quota Issue Solutions:")
            print("1. Check usage: https://platform.openai.com/usage")
            print("2. Add credits: https://platform.openai.com/account/billing")
            print("3. Wait for monthly reset if on free tier")
        elif "api_key" in str(e).lower():
            print("\n💡 API Key Issue Solutions:")
            print("1. Verify key at: https://platform.openai.com/api-keys")
            print("2. Check key permissions")
            print("3. Generate new key if needed")
        elif "model" in str(e).lower():
            print("\n💡 Model Issue Solutions:")
            print("1. Check model availability")
            print("2. Try different model (gpt-3.5-turbo)")
        
        return False

if __name__ == "__main__":
    success = test_llm_connection()
    if success:
        print("\n🚀 Ready to test your Streamlit app with LLM features!")
        print("Run: python3 -m streamlit run streamlit_app.py")
    else:
        print("\n🔧 Please resolve the issues above before using LLM features.")
