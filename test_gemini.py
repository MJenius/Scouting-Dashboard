"""
Test script to verify Gemini API integration
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is loaded
api_key = os.getenv('GEMINI_API_KEY')
print(f"API Key loaded: {api_key is not None}")
if api_key:
    print(f"API Key starts with: {api_key[:10]}...")
else:
    print("❌ No API key found!")

# Try to import and initialize Gemini
try:
    import google.generativeai as genai
    print("✓ google-generativeai package imported successfully")
    
    if api_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✓ Gemini model initialized successfully")
        
        # Test a simple generation
        print("\nTesting API with a simple prompt...")
        response = model.generate_content("Say 'Hello, the API is working!'")
        print(f"✓ API Response: {response.text}")
    else:
        print("⚠ Cannot test API without key")
        
except ImportError as e:
    print(f"❌ Failed to import google-generativeai: {e}")
except Exception as e:
    print(f"❌ Error: {e}")
