"""
Simple test script to verify the FastAPI application works correctly.
Run this after starting the server with: uvicorn main:app --reload
"""
import requests
import json
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint."""
    try:
        print(f"\nTesting {name}...")
        url = f"{BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests."""
    print("="*50)
    print("FastAPI Application Test Suite")
    print("="*50)
    
    results = []
    
    # Test 1: Root endpoint
    results.append(test_endpoint("Root", "GET", "/"))
    
    # Test 2: Health check
    results.append(test_endpoint("Health Check", "GET", "/health"))
    
    # Note: The following tests require OpenAI API key to be configured
    print("\n" + "="*50)
    print("Note: Chat, Query, and Embed endpoints require OPENAI_API_KEY")
    print("="*50)
    
    # Test 3: Chat endpoint (will fail without API key)
    test_endpoint(
        "Chat",
        "POST",
        "/api/chat",
        {"message": "Hello!"}
    )
    
    # Test 4: Query endpoint (will fail without API key and documents)
    test_endpoint(
        "Query Documents",
        "POST",
        "/api/query",
        {"question": "What is FastAPI?"}
    )
    
    # Test 5: Embed endpoint (will fail without API key)
    test_endpoint(
        "Create Embedding",
        "POST",
        "/api/embed",
        {"text": "Hello world"}
    )
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    successful = sum(results)
    total = len(results)
    print(f"Basic tests passed: {successful}/{total}")
    
    if successful == total:
        print("✓ All basic tests passed!")
        return 0
    else:
        print("✗ Some tests failed. Check server logs.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
