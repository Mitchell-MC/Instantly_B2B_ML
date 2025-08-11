"""
Test script for the new API key provided by the user.
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

def test_new_api_key():
    """Test the new API key against various services."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return None, None, None
    
    logger.info(f"🔑 Testing new API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Common API services and their test endpoints
    services = {
        "Instantly": {
            "base_urls": [
                "https://api.instantly.ai",
                "https://api.instantly.com",
                "https://instantly.ai/api",
                "https://app.instantly.ai/api"
            ],
            "endpoints": ["/api/v1/campaigns", "/campaigns", "/api/campaigns"]
        },
        "Apollo": {
            "base_urls": ["https://api.apollo.io"],
            "endpoints": ["/v1/people/search", "/v1/organizations/search", "/v1/people", "/v1/organizations"]
        },
        "Hunter.io": {
            "base_urls": ["https://api.hunter.io"],
            "endpoints": ["/v2/domain-search"]
        },
        "Clearbit": {
            "base_urls": ["https://api.clearbit.com"],
            "endpoints": ["/v1/people/find", "/v1/companies/find"]
        },
        "ZoomInfo": {
            "base_urls": ["https://api.zoominfo.com"],
            "endpoints": ["/v1/people/search", "/v1/companies/search"]
        },
        "LinkedIn": {
            "base_urls": ["https://api.linkedin.com"],
            "endpoints": ["/v2/people", "/v2/organizations"]
        },
        "Salesforce": {
            "base_urls": ["https://api.salesforce.com"],
            "endpoints": ["/services/data/v58.0/sobjects/Contact", "/services/data/v58.0/sobjects/Account"]
        },
        "HubSpot": {
            "base_urls": ["https://api.hubapi.com"],
            "endpoints": ["/crm/v3/objects/contacts", "/crm/v3/objects/companies"]
        }
    }
    
    # Different auth patterns
    auth_patterns = [
        {'Authorization': f'Bearer {api_key}'},
        {'X-API-Key': api_key},
        {'api-key': api_key},
        {'x-api-key': api_key},
        {'Authorization': f'Token {api_key}'}
    ]
    
    logger.info("🔍 Testing API key against various services...")
    
    for service_name, service_config in services.items():
        logger.info(f"\n📋 Testing {service_name}...")
        
        for base_url in service_config["base_urls"]:
            for endpoint in service_config["endpoints"]:
                url = f"{base_url}{endpoint}"
                
                for auth_header in auth_patterns:
                    try:
                        response = requests.get(url, headers=auth_header, timeout=5)
                        
                        if response.status_code == 200:
                            logger.info(f"✅ SUCCESS! {service_name} - {url}")
                            logger.info(f"   Auth header: {list(auth_header.keys())[0]}")
                            logger.info(f"   Response: {response.text[:200]}...")
                            return service_name, url, auth_header
                        elif response.status_code == 401:
                            logger.info(f"❌ Authentication failed for {service_name} - {url}")
                            break  # Don't try other auth patterns for this endpoint
                        elif response.status_code == 403:
                            logger.info(f"⚠️  Forbidden for {service_name} - {url} (might be valid but no permission)")
                        elif response.status_code == 404:
                            logger.info(f"❌ Not found: {service_name} - {url}")
                        else:
                            logger.info(f"⚠️  Status {response.status_code} for {service_name} - {url}")
                            
                    except requests.exceptions.RequestException as e:
                        logger.info(f"❌ Error with {service_name} - {url}: {e}")
                        continue
    
    logger.error("❌ Could not identify the service for this API key.")
    return None, None, None

if __name__ == "__main__":
    service, url, auth_header = test_new_api_key()
    
    if service:
        logger.info(f"\n🎉 SUCCESS! Your API key is for: {service}")
        logger.info(f"   Working endpoint: {url}")
        logger.info(f"   Auth header: {auth_header}")
    else:
        logger.error("\n❌ Could not identify the service for your API key.")
        logger.info("💡 Suggestions:")
        logger.info("   1. Check if the API key is for a different service")
        logger.info("   2. Verify the API key is still valid")
        logger.info("   3. Check the service's API documentation for correct endpoints") 