"""
Test script to examine email data and extract lead/contact information.
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

def examine_email_leads():
    """Examine email data to extract lead/contact information."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # Get emails with lead information
    logger.info("🔍 Getting emails with lead data...")
    try:
        emails_url = f"{base_url}/api/v2/emails"
        response = requests.get(emails_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        emails_data = response.json()
        emails = emails_data.get('items', [])
        
        logger.info(f"✅ Found {len(emails)} emails")
        
        if emails:
            # Examine the first few emails to understand the lead structure
            for i, email in enumerate(emails[:3]):  # Look at first 3 emails
                logger.info(f"\n📧 Email {i+1}:")
                logger.info(f"   Subject: {email.get('subject', 'N/A')}")
                logger.info(f"   To: {email.get('to_address_email_list', 'N/A')}")
                logger.info(f"   From: {email.get('from_address_email', 'N/A')}")
                logger.info(f"   Campaign ID: {email.get('campaign_id', 'N/A')}")
                
                # Examine the lead field
                lead = email.get('lead')
                if lead:
                    logger.info(f"   ✅ Lead data found!")
                    logger.info(f"   Lead type: {type(lead)}")
                    if isinstance(lead, dict):
                        logger.info(f"   Lead fields: {list(lead.keys())}")
                        # Show some key lead information
                        for key, value in lead.items():
                            if key in ['first_name', 'last_name', 'email', 'company', 'title', 'phone']:
                                logger.info(f"   {key}: {value}")
                    else:
                        logger.info(f"   Lead value: {lead}")
                else:
                    logger.info(f"   ❌ No lead data")
            
            # Collect all unique leads
            unique_leads = {}
            lead_count = 0
            
            for email in emails:
                lead = email.get('lead')
                if lead and isinstance(lead, dict):
                    lead_id = lead.get('id')
                    if lead_id and lead_id not in unique_leads:
                        unique_leads[lead_id] = lead
                        lead_count += 1
            
            logger.info(f"\n📊 Lead Summary:")
            logger.info(f"   Total emails: {len(emails)}")
            logger.info(f"   Unique leads found: {lead_count}")
            
            if unique_leads:
                logger.info(f"   ✅ Successfully extracted {len(unique_leads)} unique leads!")
                
                # Show sample lead structure
                first_lead = list(unique_leads.values())[0]
                logger.info(f"\n📋 Sample lead structure:")
                logger.info(f"   Available fields: {list(first_lead.keys())}")
                
                # Show contact information from first lead
                contact_fields = ['first_name', 'last_name', 'email', 'company', 'title', 'phone', 'linkedin_url']
                logger.info(f"   Contact information:")
                for field in contact_fields:
                    if field in first_lead:
                        logger.info(f"   {field}: {first_lead[field]}")
                
                return list(unique_leads.values())
            else:
                logger.info("❌ No unique leads found in email data")
                return []
        
    except Exception as e:
        logger.error(f"Error examining email leads: {e}")
        return []

if __name__ == "__main__":
    leads = examine_email_leads()
    
    if leads:
        logger.info(f"\n🎉 Successfully extracted {len(leads)} leads from email data!")
        logger.info("This can be used as an alternative to the contacts endpoint.")
    else:
        logger.info("\n❌ No leads could be extracted from email data.") 