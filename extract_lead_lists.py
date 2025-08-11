"""
Extract Lead Lists data from Instantly API V2.
Based on the official documentation at https://developer.instantly.ai/api/v2/leadlist
"""

import requests
import logging
import os
import re
import pandas as pd
import json
from datetime import datetime

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

def extract_lead_lists():
    """Extract Lead Lists data from Instantly API V2."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # Lead Lists endpoint from documentation
    endpoint = "/api/v2/lead-lists"
    url = f"{base_url}{endpoint}"
    
    logger.info(f"🔍 Extracting data from: {endpoint}")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get('items', [])
            
            # Handle different response structures
            if not items and isinstance(data, list):
                items = data
            elif not items and isinstance(data, dict):
                items = [data]
            
            logger.info(f"✅ SUCCESS! Found {len(items)} lead lists")
            
            if items:
                # Analyze the data structure
                all_keys = set()
                for item in items:
                    all_keys.update(item.keys())
                
                logger.info(f"📋 Available columns: {sorted(list(all_keys))}")
                
                # Show sample data
                if items:
                    logger.info(f"\n📋 Sample Lead List structure:")
                    sample = items[0]
                    for key, value in sample.items():
                        logger.info(f"   {key}: {value}")
                
                # Create DataFrame
                df = pd.DataFrame(items)
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = f"lead_lists_data_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                
                logger.info(f"\n💾 Saved Lead Lists data to: {csv_file}")
                logger.info(f"📊 Total lead lists: {len(df)}")
                logger.info(f"📋 Total columns: {len(df.columns)}")
                
                # Show column summary
                logger.info("\n📋 Column Summary:")
                for i, col in enumerate(df.columns, 1):
                    non_null_count = df[col].notna().sum()
                    null_count = df[col].isna().sum()
                    logger.info(f"  {i:3d}. {col:<30} | Non-null: {non_null_count:4d} | Null: {null_count:4d}")
                
                return df
            
            else:
                logger.info("❌ No lead lists found")
                return None
                
        elif response.status_code == 404:
            logger.error("❌ Lead Lists endpoint not found")
            return None
        elif response.status_code == 401:
            logger.error("❌ Authentication failed - check your API key")
            return None
        elif response.status_code == 403:
            logger.error("⚠️  Forbidden - you might not have permission to access lead lists")
            return None
        else:
            logger.error(f"❌ API request failed. Status: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error extracting lead lists: {e}")
        return None

def extract_leads_from_lists():
    """Extract leads from lead lists if available."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # First get lead lists
    lead_lists_url = f"{base_url}/api/v2/lead-lists"
    
    try:
        response = requests.get(lead_lists_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            lead_lists = data.get('items', [])
            
            if not lead_lists and isinstance(data, list):
                lead_lists = data
            elif not lead_lists and isinstance(data, dict):
                lead_lists = [data]
            
            logger.info(f"✅ Found {len(lead_lists)} lead lists")
            
            all_leads = []
            
            for lead_list in lead_lists:
                list_id = lead_list.get('id')
                list_name = lead_list.get('name', 'Unknown')
                
                logger.info(f"\n🔍 Extracting leads from list: {list_name} (ID: {list_id})")
                
                # Try to get leads from this list
                leads_url = f"{base_url}/api/v2/lead-lists/{list_id}/leads"
                
                try:
                    leads_response = requests.get(leads_url, headers=headers, timeout=15)
                    
                    if leads_response.status_code == 200:
                        leads_data = leads_response.json()
                        leads = leads_data.get('items', [])
                        
                        if not leads and isinstance(leads_data, list):
                            leads = leads_data
                        elif not leads and isinstance(leads_data, dict):
                            leads = [leads_data]
                        
                        logger.info(f"   ✅ Found {len(leads)} leads in this list")
                        
                        # Add list information to each lead
                        for lead in leads:
                            lead['lead_list_id'] = list_id
                            lead['lead_list_name'] = list_name
                            all_leads.append(lead)
                        
                    elif leads_response.status_code == 404:
                        logger.info(f"   ❌ No leads endpoint found for this list")
                    else:
                        logger.info(f"   ⚠️  Status {leads_response.status_code} for leads endpoint")
                        
                except Exception as e:
                    logger.info(f"   ❌ Error getting leads from list: {e}")
            
            if all_leads:
                # Create DataFrame
                df = pd.DataFrame(all_leads)
                
                # Save to CSV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_file = f"leads_from_lists_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                
                logger.info(f"\n💾 Saved leads data to: {csv_file}")
                logger.info(f"📊 Total leads: {len(df)}")
                logger.info(f"📋 Total columns: {len(df.columns)}")
                
                # Show column summary
                logger.info("\n📋 Column Summary:")
                for i, col in enumerate(df.columns, 1):
                    non_null_count = df[col].notna().sum()
                    null_count = df[col].isna().sum()
                    logger.info(f"  {i:3d}. {col:<30} | Non-null: {non_null_count:4d} | Null: {null_count:4d}")
                
                return df
            else:
                logger.info("❌ No leads found in any lists")
                return None
                
        else:
            logger.error(f"❌ Error getting lead lists. Status: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Error extracting leads from lists: {e}")
        return None

if __name__ == "__main__":
    logger.info("🚀 Starting Lead Lists data extraction...")
    
    # Extract lead lists
    lead_lists_df = extract_lead_lists()
    
    if lead_lists_df is not None:
        logger.info(f"\n🎉 Successfully extracted {len(lead_lists_df)} lead lists!")
        
        # Try to extract leads from the lists
        logger.info("\n🔍 Attempting to extract leads from lead lists...")
        leads_df = extract_leads_from_lists()
        
        if leads_df is not None:
            logger.info(f"\n🎉 Successfully extracted {len(leads_df)} leads from lists!")
        else:
            logger.info("\n❌ Could not extract leads from lists.")
    else:
        logger.info("\n❌ Could not extract lead lists.") 