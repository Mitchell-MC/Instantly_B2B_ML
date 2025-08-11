"""
Test script for Instantly V2 API.
"""

import requests
import logging
import os
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_api_key_from_env_file(env_file_path: str = r"C:\Users\mccal\Downloads\Instantly B2B Main\Instantly.env") -> str:
    """Load API key from the specified .env file."""
    try:
        if not os.path.exists(env_file_path):
            logger.error(f"Environment file not found at: {env_file_path}")
            return None
        
        with open(env_file_path, 'r') as file:
            content = file.read()
        
        # Look for API key patterns in the file
        api_key_patterns = [
            r'INSTANTLY_API_KEY\s*=\s*["\']?([^"\'\s]+)["\']?',
            r'API_KEY\s*=\s*["\']?([^"\'\s]+)["\']?',
            r'INSTANTLY_AI_API_KEY\s*=\s*["\']?([^"\'\s]+)["\']?',
            r'BEARER_TOKEN\s*=\s*["\']?([^"\'\s]+)["\']?'
        ]
        
        for pattern in api_key_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                api_key = match.group(1)
                logger.info(f"Successfully loaded API key from {env_file_path}")
                return api_key
        
        logger.error(f"No API key found in {env_file_path}")
        return None
        
    except Exception as e:
        logger.error(f"Error reading environment file {env_file_path}: {e}")
        return None

def test_instantly_v2_api():
    """Test the API key with Instantly V2 API endpoints."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return None, None, None
    
    logger.info(f"🔑 Testing API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and endpoints
    base_url = "https://api.instantly.ai"
    
    # V2 API endpoints (based on documentation)
    v2_endpoints = [
        "/api/v2/campaigns",
        "/api/v2/contacts",
        "/api/v2/leads",
        "/api/v2/accounts",
        "/api/v2/workspaces",
        "/api/v2/emails"
    ]
    
    # V2 API uses Bearer token authentication
    auth_headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    logger.info("🔍 Testing Instantly V2 API endpoints...")
    
    for endpoint in v2_endpoints:
        url = f"{base_url}{endpoint}"
        logger.info(f"\n📋 Testing endpoint: {endpoint}")
        
        try:
            response = requests.get(url, headers=auth_headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ SUCCESS! Endpoint: {endpoint}")
                logger.info(f"   Response: {str(data)[:200]}...")
                return "Instantly V2", url, auth_headers
            elif response.status_code == 401:
                logger.info(f"❌ Authentication failed for {endpoint}")
            elif response.status_code == 403:
                logger.info(f"⚠️  Forbidden for {endpoint} (might be valid but no permission)")
            elif response.status_code == 404:
                logger.info(f"❌ Endpoint not found: {endpoint}")
            else:
                logger.info(f"⚠️  Status {response.status_code} for {endpoint}")
                
        except Exception as e:
            logger.info(f"❌ Error with {endpoint}: {e}")
            continue
    
    logger.error("❌ No working V2 endpoints found.")
    return None, None, None

if __name__ == "__main__":
    service, url, auth_header = test_instantly_v2_api()
    
    if service:
        logger.info(f"\n🎉 SUCCESS! Your API key works with {service}!")
        logger.info(f"   Working endpoint: {url}")
        logger.info(f"   Auth header: {auth_header}")
    else:
        logger.error("\n❌ API key doesn't work with Instantly V2 API.")
        logger.info("💡 This might be for a different service or the key might be invalid.") 