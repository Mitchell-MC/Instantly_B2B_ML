"""
Test script to explore campaign data and find contacts.
"""

import requests
import logging
import os
import re
import json

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

def explore_campaign_data():
    """Explore campaign data to understand the structure and find contacts."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # Get campaigns first
    logger.info("🔍 Getting campaigns...")
    try:
        campaigns_url = f"{base_url}/api/v2/campaigns"
        response = requests.get(campaigns_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        campaigns_data = response.json()
        campaigns = campaigns_data.get('items', [])
        
        logger.info(f"✅ Found {len(campaigns)} campaigns")
        
        if campaigns:
            # Examine the first campaign structure
            first_campaign = campaigns[0]
            logger.info(f"📋 First campaign structure:")
            logger.info(f"   ID: {first_campaign.get('id')}")
            logger.info(f"   Name: {first_campaign.get('name')}")
            logger.info(f"   Status: {first_campaign.get('status')}")
            
            # Check if campaign has any contact-related fields
            campaign_keys = list(first_campaign.keys())
            logger.info(f"   Available fields: {campaign_keys}")
            
            # Look for contact-related fields
            contact_fields = [k for k in campaign_keys if any(term in k.lower() for term in ['contact', 'lead', 'person', 'email', 'recipient'])]
            if contact_fields:
                logger.info(f"   Contact-related fields: {contact_fields}")
            
            # Try to get contacts for the first campaign
            campaign_id = first_campaign.get('id')
            if campaign_id:
                logger.info(f"🔍 Trying to get contacts for campaign {campaign_id}...")
                
                # Try different possible endpoints for campaign contacts
                contact_endpoints = [
                    f"{base_url}/api/v2/campaigns/{campaign_id}/contacts",
                    f"{base_url}/api/v2/campaigns/{campaign_id}/leads",
                    f"{base_url}/api/v2/campaigns/{campaign_id}/recipients",
                    f"{base_url}/api/v2/campaigns/{campaign_id}/people",
                    f"{base_url}/api/v2/campaigns/{campaign_id}/emails"
                ]
                
                for endpoint in contact_endpoints:
                    try:
                        logger.info(f"   Trying: {endpoint}")
                        response = requests.get(endpoint, headers=headers, timeout=10)
                        
                        if response.status_code == 200:
                            data = response.json()
                            items = data.get('items', [])
                            logger.info(f"   ✅ SUCCESS! Found {len(items)} items")
                            if items:
                                logger.info(f"   First item keys: {list(items[0].keys())}")
                            return endpoint, data
                        elif response.status_code == 404:
                            logger.info(f"   ❌ Not found")
                        else:
                            logger.info(f"   ⚠️  Status {response.status_code}")
                            
                    except Exception as e:
                        logger.info(f"   ❌ Error: {e}")
                        continue
                
                logger.info("❌ No working contact endpoints found for this campaign")
        
        # Also try accounts endpoint
        logger.info("🔍 Checking accounts endpoint...")
        try:
            accounts_url = f"{base_url}/api/v2/accounts"
            response = requests.get(accounts_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            accounts_data = response.json()
            accounts = accounts_data.get('items', [])
            
            logger.info(f"✅ Found {len(accounts)} accounts")
            
            if accounts:
                first_account = accounts[0]
                logger.info(f"📋 First account structure:")
                logger.info(f"   Available fields: {list(first_account.keys())}")
        
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
        
        # Try emails endpoint
        logger.info("🔍 Checking emails endpoint...")
        try:
            emails_url = f"{base_url}/api/v2/emails"
            response = requests.get(emails_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            emails_data = response.json()
            emails = emails_data.get('items', [])
            
            logger.info(f"✅ Found {len(emails)} emails")
            
            if emails:
                first_email = emails[0]
                logger.info(f"📋 First email structure:")
                logger.info(f"   Available fields: {list(first_email.keys())}")
                
                # Look for contact information in email data
                contact_info = [k for k in first_email.keys() if any(term in k.lower() for term in ['contact', 'lead', 'person', 'recipient', 'to', 'from'])]
                if contact_info:
                    logger.info(f"   Contact-related fields: {contact_info}")
        
        except Exception as e:
            logger.error(f"Error getting emails: {e}")
            
    except Exception as e:
        logger.error(f"Error exploring campaign data: {e}")

if __name__ == "__main__":
    explore_campaign_data() 