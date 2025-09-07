#!/usr/bin/env python3
"""
Apollo API Debug Test
Tests the exact same pattern that works in your notebook
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_apollo_exact_notebook_pattern():
    """Test Apollo API using the exact same pattern from your working notebook"""
    
    # Get API key exactly as in notebook
    APOLLO_API_KEY = os.getenv("apollo_key") or os.getenv("APOLLO_API_KEY")
    
    print("🔍 Apollo API Debug Test")
    print("=" * 40)
    print(f"API Key: {APOLLO_API_KEY[:10]}..." if APOLLO_API_KEY else "❌ No API key found")
    
    if not APOLLO_API_KEY:
        print("❌ No Apollo API key found in environment")
        return False
    
    # Test 1: Simple health/auth check
    print("\n1️⃣ Testing basic API authentication...")
    try:
        url = "https://api.apollo.io/api/v1/auth/health"
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "x-api-key": APOLLO_API_KEY
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
        if response.status_code == 200:
            print("   ✅ Basic authentication works")
        else:
            print(f"   ❌ Basic authentication failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Try the exact LinkedIn enrichment from your notebook
    print("\n2️⃣ Testing LinkedIn enrichment (exact notebook pattern)...")
    try:
        url = "https://api.apollo.io/api/v1/people/match"
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "x-api-key": APOLLO_API_KEY
        }
        
        # Use exact payload from your notebook
        payload = {
            "linkedin_url": "https://www.linkedin.com/in/sameh-suleiman-asaad-22a8b359",
            "reveal_personal_emails": False,
            "reveal_phone_number": False
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Headers: {dict(response.headers)}")
        print(f"   Response: {response.text[:500]}")
        
        if response.status_code == 200:
            data = response.json()
            person = data.get('person', {})
            if person:
                print(f"   ✅ LinkedIn enrichment works!")
                print(f"   Name: {person.get('name')}")
                print(f"   Title: {person.get('title')}")
                return True
            else:
                print("   ⚠️ Empty person data")
        else:
            print(f"   ❌ LinkedIn enrichment failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 3: Try with uppercase headers
    print("\n3️⃣ Testing with uppercase X-Api-Key header...")
    try:
        headers_alt = {
            "accept": "application/json",
            "Cache-Control": "no-cache",
            "Content-Type": "application/json",
            "X-Api-Key": APOLLO_API_KEY  # Uppercase version
        }
        
        response = requests.post(url, headers=headers_alt, json=payload, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        
        if response.status_code == 200:
            print(f"   ✅ Uppercase header works!")
            return True
        else:
            print(f"   ❌ Uppercase header failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 4: Check rate limits and account info
    print("\n4️⃣ Checking account limits and permissions...")
    try:
        # Try a minimal request to see account info
        url = "https://api.apollo.io/api/v1/mixed_people/search"
        headers = {
            "accept": "application/json",
            "Cache-Control": "no-cache", 
            "Content-Type": "application/json",
            "x-api-key": APOLLO_API_KEY
        }
        
        # Minimal search to test permissions
        minimal_payload = {
            "page": 1,
            "per_page": 1
        }
        
        response = requests.post(url, headers=headers, json=minimal_payload, timeout=10)
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.text[:300]}")
        
        # Check response headers for rate limit info
        rate_headers = {k: v for k, v in response.headers.items() if 'rate' in k.lower() or 'limit' in k.lower()}
        if rate_headers:
            print(f"   Rate Limit Headers: {rate_headers}")
        
        if response.status_code == 200:
            print(f"   ✅ People search works!")
            return True
        elif response.status_code == 403:
            print(f"   ❌ 403 Forbidden - API key may lack permissions for this endpoint")
        elif response.status_code == 429:
            print(f"   ❌ 429 Rate Limited - API key hit rate limits")
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    return False

def check_environment_variables():
    """Check all Apollo-related environment variables"""
    print("\n🔧 Environment Variables Check")
    print("=" * 40)
    
    apollo_vars = ['APOLLO_API_KEY', 'apollo_key']
    for var in apollo_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {value[:10]}...")
        else:
            print(f"❌ {var}: Not set")

if __name__ == "__main__":
    check_environment_variables()
    success = test_apollo_exact_notebook_pattern()
    
    if success:
        print("\n🎉 Apollo API is working!")
    else:
        print("\n❌ Apollo API tests failed")
        print("\n💡 Possible issues:")
        print("   1. API key may have limited permissions")
        print("   2. Account may not have access to these endpoints")
        print("   3. Rate limits may be exceeded")
        print("   4. API key might be expired or invalid")
        print("   5. Account might be on a free plan with restricted access")
        
        print("\n🔍 Next steps:")
        print("   1. Check your Apollo dashboard for API permissions")
        print("   2. Verify your plan includes API access")
        print("   3. Try with a fresh API key")
        print("   4. Contact Apollo support if the issue persists")

