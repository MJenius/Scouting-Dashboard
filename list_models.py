"""
List available Gemini models
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key
api_key = os.getenv('GEMINI_API_KEY')
if api_key:
    print(f"✓ API key found (starts with: {api_key[:10]}...)")
    genai.configure(api_key=api_key)
    
    print("\nAvailable models:")
    for model in genai.list_models():
        if 'generateContent' in model.supported_generation_methods:
            print(f"  - {model.name}")
else:
    print("❌ No API key found!")
