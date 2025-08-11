"""
Script to extract ESP codes from Instantly accounts endpoint.
"""

import requests
import logging
import os
import re
import pandas as pd

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

def get_esp_codes():
    """Get ESP codes from Instantly accounts endpoint."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # Get accounts with ESP information
    logger.info("🔍 Getting accounts with ESP codes...")
    try:
        accounts_url = f"{base_url}/api/v2/accounts"
        response = requests.get(accounts_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        accounts_data = response.json()
        accounts = accounts_data.get('items', [])
        
        logger.info(f"✅ Found {len(accounts)} accounts")
        
        if accounts:
            # Extract ESP information
            esp_data = []
            
            for account in accounts:
                esp_info = {
                    'email': account.get('email', ''),
                    'first_name': account.get('first_name', ''),
                    'last_name': account.get('last_name', ''),
                    'organization': account.get('organization', ''),
                    'provider_code': account.get('provider_code', ''),  # This is the ESP code
                    'warmup_status': account.get('warmup_status', ''),
                    'setup_pending': account.get('setup_pending', False),
                    'is_managed_account': account.get('is_managed_account', False),
                    'status': account.get('status', ''),
                    'stat_warmup_score': account.get('stat_warmup_score', ''),
                    'timestamp_created': account.get('timestamp_created', ''),
                    'timestamp_updated': account.get('timestamp_updated', '')
                }
                esp_data.append(esp_info)
            
            # Create DataFrame
            df = pd.DataFrame(esp_data)
            
            # Save to CSV
            output_file = "instantly_esp_codes.csv"
            df.to_csv(output_file, index=False)
            
            logger.info(f"💾 Saved ESP codes to {output_file}")
            
            # Show ESP code summary
            logger.info("\n📊 ESP Code Summary:")
            esp_counts = df['provider_code'].value_counts()
            for esp_code, count in esp_counts.items():
                logger.info(f"   {esp_code}: {count} accounts")
            
            # Show sample data
            logger.info("\n📋 Sample ESP Data:")
            logger.info(f"   Total accounts: {len(df)}")
            logger.info(f"   Unique ESP codes: {len(esp_counts)}")
            logger.info(f"   Available fields: {list(df.columns)}")
            
            return df
        
    except Exception as e:
        logger.error(f"Error getting ESP codes: {e}")
        return None

if __name__ == "__main__":
    df = get_esp_codes()
    
    if df is not None:
        logger.info(f"\n🎉 Successfully extracted ESP codes!")
        logger.info(f"📁 File saved as: instantly_esp_codes.csv")
    else:
        logger.info("\n❌ Could not extract ESP codes.") 