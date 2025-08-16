#!/usr/bin/env python3
"""
Simple API Test Script
Tests the FastAPI service endpoints to ensure everything is working.
"""

import requests
import json
from datetime import datetime

def test_api_endpoints():
    """Test all available API endpoints."""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing FastAPI Service Endpoints")
    print("=" * 50)
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Root endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    print()
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health endpoint: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health endpoint failed: {e}")
    
    print()
    
    # Test API docs
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"✅ API docs: {response.status_code}")
        print(f"   Docs available at: {base_url}/docs")
    except Exception as e:
        print(f"❌ API docs failed: {e}")
    
    print()
    
    # Test OpenAPI schema
    try:
        response = requests.get(f"{base_url}/openapi.json")
        print(f"✅ OpenAPI schema: {response.status_code}")
        print(f"   Schema available at: {base_url}/openapi.json")
    except Exception as e:
        print(f"❌ OpenAPI schema failed: {e}")
    
    print()
    print("🎉 API testing completed!")
    print(f"📚 Full API documentation: {base_url}/docs")
    print(f"🔍 Health check: {base_url}/health")

if __name__ == "__main__":
    test_api_endpoints()
