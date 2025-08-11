"""
Test script specifically for Apollo API.
"""

import requests
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_apollo_api():
    """Test the API key specifically with Apollo API."""
    
    # Your API key (base64 encoded)
    encoded_key = "ZjQ0Mjc3ZGUtNjliMi00YmMzLWE2OWMtMjhhZmQ0MDk0MTIzOkx5VWZ6UnB6RmR3Zw=="
    
    # Decode it
    decoded_key = base64.b64decode(encoded_key).decode('utf-8')
    logger.info(f"Decoded key: {decoded_key}")
    
    # Extract parts
    key_part = decoded_key.split(':')[0]
    secret_part = decoded_key.split(':')[1]
    
    logger.info(f"Key part: {key_part}")
    logger.info(f"Secret part: {secret_part}")
    
    # Apollo API endpoints to test
    endpoints = [
        "/v1/people/search",
        "/v1/organizations/search", 
        "/v1/people",
        "/v1/organizations",
        "/v1/people/enrich",
        "/v1/organizations/enrich"
    ]
    
    # Different auth patterns for Apollo
    auth_patterns = [
        {'X-API-Key': key_part},
        {'Authorization': f'Bearer {key_part}'},
        {'api-key': key_part},
        {'x-api-key': key_part},
        {'X-API-Key': decoded_key},
        {'Authorization': f'Bearer {decoded_key}'}
    ]
    
    base_url = "https://api.apollo.io"
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        logger.info(f"\n🔍 Testing endpoint: {endpoint}")
        
        for i, auth_header in enumerate(auth_patterns):
            try:
                logger.info(f"  Trying auth pattern {i+1}: {list(auth_header.keys())[0]}")
                
                response = requests.get(url, headers=auth_header, timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ SUCCESS! Endpoint: {endpoint}")
                    logger.info(f"   Auth pattern: {list(auth_header.keys())[0]}")
                    logger.info(f"   Response: {response.text[:200]}...")
                    return True, endpoint, auth_header
                elif response.status_code == 401:
                    logger.info(f"❌ Authentication failed")
                elif response.status_code == 403:
                    logger.info(f"⚠️  Forbidden (might be valid but no permission)")
                elif response.status_code == 404:
                    logger.info(f"❌ Endpoint not found")
                else:
                    logger.info(f"⚠️  Status {response.status_code}")
                    
            except Exception as e:
                logger.info(f"❌ Error: {e}")
                continue
    
    logger.error("❌ No working Apollo endpoints found.")
    return False, None, None

if __name__ == "__main__":
    success, endpoint, auth_header = test_apollo_api()
    
    if success:
        logger.info(f"\n🎉 SUCCESS! Your API key works with Apollo!")
        logger.info(f"   Working endpoint: {endpoint}")
        logger.info(f"   Auth header: {auth_header}")
    else:
        logger.error("\n❌ API key doesn't work with Apollo.")
        logger.info("💡 This might be for a different service or the key might be invalid.") 