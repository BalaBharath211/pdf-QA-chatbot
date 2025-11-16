import google.generativeai as genai

print("Testing Gemini Models")
print("=" * 50)

api_key = input("Enter your Gemini API key: ").strip()
genai.configure(api_key=api_key)

# Test different model name formats
test_models = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-1.5-flash", 
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-pro",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash"
]

print("\nTesting models...")
working_model = None

for model_name in test_models:
    try:
        print(f"Trying: {model_name}... ", end="")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello, I work!'")
        print(f"✅ WORKS! Response: {response.text[:50]}")
        if not working_model:
            working_model = model_name
    except Exception as e:
        print(f"❌ Failed: {str(e)[:50]}")

print("\n" + "=" * 50)
if working_model:
    print(f"✅ USE THIS MODEL: '{working_model}'")
    print(f"\nUpdate your rag_core.py file:")
    print(f"  model = genai.GenerativeModel('{working_model}')")
else:
    print("❌ No working models found!")
    print("\nTrying to list available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"  Available: {m.name}")
    except Exception as e:
        print(f"  Error listing models: {e}")
        