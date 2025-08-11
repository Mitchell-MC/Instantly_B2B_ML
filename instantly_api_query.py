"""
Instantly API Query Script
Queries the Instantly API and returns a CSV with all available columns from Instantly's response.
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path
import time
from typing import Dict, List, Optional
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_api_key_from_env_file(env_file_path: str = r"C:\Users\mccal\Downloads\Instantly B2B Main\Instantly.env") -> Optional[str]:
    """
    Load API key from the specified .env file.
    
    Args:
        env_file_path: Path to the .env file containing the API key
        
    Returns:
        API key string if found, None otherwise
    """
    try:
        if not os.path.exists(env_file_path):
            logger.error(f"Environment file not found at: {env_file_path}")
            return None
        
        with open(env_file_path, 'r') as file:
            content = file.read()
        
        # Look for API key patterns in the file
        # Common patterns: INSTANTLY_API_KEY=, API_KEY=, etc.
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
        logger.info("Available patterns in file:")
        for line in content.split('\n'):
            if '=' in line and any(keyword in line.upper() for keyword in ['API', 'KEY', 'TOKEN', 'BEARER']):
                logger.info(f"  Found: {line.strip()}")
        
        return None
        
    except Exception as e:
        logger.error(f"Error reading environment file {env_file_path}: {e}")
        return None

class InstantlyAPI:
    """Instantly API client for querying contact data."""
    
    def __init__(self, api_key: str, base_url: str = "https://api.instantly.ai"):
        """
        Initialize Instantly API client.
        
        Args:
            api_key: Your Instantly API key
            base_url: Instantly API base URL
        """
        self.api_key = api_key
        self.base_url = base_url
        
        # V2 API uses Bearer token authentication
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # Log the configuration (without exposing the full API key)
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        logger.info(f"Initialized Instantly V2 API client with base URL: {base_url}")
        logger.info(f"API Key: {masked_key}")
    
    def get_campaigns(self) -> List[Dict]:
        """Get all campaigns from Instantly V2 API."""
        try:
            url = f"{self.base_url}/api/v2/campaigns"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            campaigns = data.get('items', [])
            logger.info(f"Retrieved {len(campaigns)} campaigns from V2 API")
            return campaigns
            
        except Exception as e:
            logger.error(f"Error fetching campaigns: {e}")
            return []
    
    def get_contacts(self, campaign_id: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """
        Get contacts from Instantly V2 API by extracting from emails endpoint.
        
        Args:
            campaign_id: Specific campaign ID to query (optional)
            limit: Number of contacts to retrieve
            
        Returns:
            List of contact dictionaries
        """
        try:
            # Add query parameters for V2 API
            params = {
                'limit': limit,
                'page': 1
            }
            
            # Get emails and extract contact information
            url = f"{self.base_url}/api/v2/emails"
            
            if campaign_id:
                # Filter by campaign if provided
                params['campaign_id'] = campaign_id
            
            try:
                logger.info(f"Getting emails from {url}")
                response = requests.get(url, headers=self.headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                emails = data.get('items', [])
                logger.info(f"Retrieved {len(emails)} emails")
                
                # Extract unique contacts from emails
                contacts = []
                seen_emails = set()
                
                for email in emails:
                    # Extract contact information from email data
                    contact = {
                        'email': email.get('to_address_email_list', ''),
                        'from_email': email.get('from_address_email', ''),
                        'subject': email.get('subject', ''),
                        'campaign_id': email.get('campaign_id', ''),
                        'message_id': email.get('message_id', ''),
                        'timestamp_created': email.get('timestamp_created', ''),
                        'timestamp_email': email.get('timestamp_email', ''),
                        'lead_email': email.get('lead', ''),  # This is the lead email
                        'thread_id': email.get('thread_id', ''),
                        'is_unread': email.get('is_unread', False),
                        'is_focused': email.get('is_focused', False),
                        'ue_type': email.get('ue_type', ''),
                        'step': email.get('step', ''),
                        'organization_id': email.get('organization_id', ''),
                        'eaccount': email.get('eaccount', '')
                    }
                    
                    # Use lead email as primary identifier to avoid duplicates
                    lead_email = email.get('lead', '')
                    if lead_email and lead_email not in seen_emails:
                        seen_emails.add(lead_email)
                        contacts.append(contact)
                
                logger.info(f"Extracted {len(contacts)} unique contacts from emails")
                return contacts
                
            except Exception as e:
                logger.error(f"Error fetching emails: {e}")
                return []
            
        except Exception as e:
            logger.error(f"Error fetching contacts: {e}")
            return []
    
    def get_contact_details(self, contact_id: str) -> Optional[Dict]:
        """
        Get detailed information for a specific contact.
        
        Args:
            contact_id: The contact ID to retrieve
            
        Returns:
            Contact details dictionary or None
        """
        try:
            url = f"{self.base_url}/api/v2/contacts/{contact_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            contact = response.json()
            logger.info(f"Retrieved details for contact {contact_id}")
            return contact
            
        except Exception as e:
            logger.error(f"Error fetching contact {contact_id}: {e}")
            return None
    
    def test_api_connection(self) -> bool:
        """
        Test the API connection and return available endpoints.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test V2 API connection
            url = f"{self.base_url}/api/v2/campaigns"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                campaigns = data.get('items', [])
                logger.info(f"✅ V2 API connection successful! Found {len(campaigns)} campaigns")
                return True
            elif response.status_code == 401:
                logger.error("❌ Authentication failed - check your API key")
                return False
            elif response.status_code == 404:
                logger.error("❌ V2 API endpoint not found")
                return False
            else:
                logger.error(f"❌ API connection failed. Status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ API connection error: {e}")
            return False
    
    def get_available_endpoints(self) -> List[str]:
        """
        Test common Instantly V2 API endpoints to find which ones are available.
        
        Returns:
            List of working endpoints
        """
        available_endpoints = []
        test_endpoints = [
            "/api/v2/campaigns",
            "/api/v2/contacts", 
            "/api/v2/leads",
            "/api/v2/people",
            "/api/v2/accounts",
            "/api/v2/workspaces",
            "/api/v2/emails"
        ]
        
        for endpoint in test_endpoints:
            url = f"{self.base_url}{endpoint}"
            try:
                response = requests.get(url, headers=self.headers)
                
                if response.status_code == 200:
                    available_endpoints.append(endpoint)
                    logger.info(f"✅ Available endpoint: {endpoint}")
                elif response.status_code == 401:
                    logger.info(f"❌ Authentication failed for {endpoint}")
                else:
                    logger.info(f"❌ Not available: {endpoint} (Status: {response.status_code})")
                    
            except Exception as e:
                logger.info(f"❌ Error testing {endpoint}: {e}")
        
        return available_endpoints
    
    def search_contacts(self, query: str, limit: int = 1000) -> List[Dict]:
        """
        Search contacts using Instantly V2 API search functionality.
        
        Args:
            query: Search query string
            limit: Number of results to return
            
        Returns:
            List of matching contacts
        """
        try:
            url = f"{self.base_url}/api/v2/contacts/search"
            params = {
                'q': query,
                'limit': limit,
                'page': 1
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            contacts = data.get('items', [])
            logger.info(f"Search returned {len(contacts)} contacts for query: {query}")
            return contacts
            
        except Exception as e:
            logger.error(f"Error searching contacts: {e}")
            return []

def analyze_contact_structure(contacts: List[Dict]) -> Dict:
    """
    Analyze the structure of contacts to identify all available columns.
    
    Args:
        contacts: List of contact dictionaries
        
    Returns:
        Dictionary with column analysis
    """
    if not contacts:
        return {}
    
    # Collect all unique keys from all contacts
    all_keys = set()
    jsonb_keys = set()
    nested_structures = {}
    
    for contact in contacts:
        # Add all top-level keys
        all_keys.update(contact.keys())
        
        # Identify JSONB/nested structures
        for key, value in contact.items():
            if isinstance(value, dict):
                jsonb_keys.add(key)
                if key not in nested_structures:
                    nested_structures[key] = set()
                nested_structures[key].update(value.keys())
            elif isinstance(value, list):
                jsonb_keys.add(key)
                # Analyze list items if they're dictionaries
                if value and isinstance(value[0], dict):
                    if key not in nested_structures:
                        nested_structures[key] = set()
                    for item in value[:5]:  # Sample first 5 items
                        nested_structures[key].update(item.keys())
    
    return {
        'all_columns': sorted(list(all_keys)),
        'jsonb_columns': sorted(list(jsonb_keys)),
        'nested_structures': {k: sorted(list(v)) for k, v in nested_structures.items()},
        'total_contacts': len(contacts)
    }

def flatten_jsonb_columns(df: pd.DataFrame, jsonb_columns: List[str]) -> pd.DataFrame:
    """
    Flatten JSONB columns to create additional features.
    
    Args:
        df: DataFrame with JSONB columns
        jsonb_columns: List of column names that contain JSONB data
        
    Returns:
        DataFrame with flattened JSONB columns
    """
    df_flattened = df.copy()
    
    for col in jsonb_columns:
        if col in df_flattened.columns:
            try:
                # Create presence indicators
                df_flattened[f'has_{col}'] = df_flattened[col].notna().astype(int)
                
                # Create length indicators
                df_flattened[f'{col}_length'] = df_flattened[col].astype(str).str.len()
                
                # Try to extract specific fields from JSON
                if df_flattened[col].dtype == 'object':
                    # Sample a non-null value to understand structure
                    sample_value = df_flattened[col].dropna().iloc[0] if not df_flattened[col].dropna().empty else None
                    
                    if sample_value and isinstance(sample_value, dict):
                        # Extract common fields
                        for field in ['id', 'name', 'email', 'title', 'company', 'industry']:
                            if field in sample_value:
                                df_flattened[f'{col}_{field}'] = df_flattened[col].apply(
                                    lambda x: x.get(field, '') if isinstance(x, dict) else ''
                                )
                    
                    elif sample_value and isinstance(sample_value, list) and sample_value:
                        # Handle list of dictionaries
                        if isinstance(sample_value[0], dict):
                            # Extract count and sample fields
                            df_flattened[f'{col}_count'] = df_flattened[col].apply(
                                lambda x: len(x) if isinstance(x, list) else 0
                            )
                            
                            # Extract common fields from first item
                            first_item = sample_value[0]
                            for field in ['id', 'name', 'title', 'company']:
                                if field in first_item:
                                    df_flattened[f'{col}_first_{field}'] = df_flattened[col].apply(
                                        lambda x: x[0].get(field, '') if isinstance(x, list) and x else ''
                                    )
                
                logger.info(f"Flattened JSONB column: {col}")
                
            except Exception as e:
                logger.warning(f"Could not flatten JSONB column {col}: {e}")
    
    return df_flattened

def query_instantly_api(api_key: str, output_file: str = "instantly_contacts_export.csv", 
                       limit: int = 1000, search_query: Optional[str] = None) -> pd.DataFrame:
    """
    Query Instantly API and return a CSV with all available columns.
    
    Args:
        api_key: Your Instantly API key
        output_file: Output CSV filename
        limit: Number of contacts to retrieve
        search_query: Optional search query to filter contacts
        
    Returns:
        DataFrame with all Instantly contact data
    """
    logger.info("🚀 Starting Instantly API query...")
    
    # Initialize API client
    api = InstantlyAPI(api_key)
    
    # Test API connection first
    logger.info("🔍 Testing API connection...")
    if not api.test_api_connection():
        logger.error("❌ API connection failed. Please check your API key and try again.")
        return pd.DataFrame()
    
    # Get available endpoints
    logger.info("🔍 Discovering available endpoints...")
    available_endpoints = api.get_available_endpoints()
    
    if not available_endpoints:
        logger.error("❌ No available endpoints found. Please check your API key permissions.")
        return pd.DataFrame()
    
    # Get contacts based on search query or all contacts
    if search_query:
        logger.info(f"Searching contacts with query: {search_query}")
        contacts = api.search_contacts(search_query, limit=limit)
    else:
        logger.info("Retrieving all contacts...")
        contacts = api.get_contacts(limit=limit)
    
    if not contacts:
        logger.error("No contacts retrieved from API")
        return pd.DataFrame()
    
    # Analyze contact structure
    logger.info("📊 Analyzing contact structure...")
    structure_analysis = analyze_contact_structure(contacts)
    
    logger.info(f"Found {len(structure_analysis['all_columns'])} total columns")
    logger.info(f"Found {len(structure_analysis['jsonb_columns'])} JSONB columns")
    
    # Convert to DataFrame
    logger.info("🔄 Converting to DataFrame...")
    df = pd.DataFrame(contacts)
    
    # Flatten JSONB columns
    if structure_analysis['jsonb_columns']:
        logger.info("🔧 Flattening JSONB columns...")
        df = flatten_jsonb_columns(df, structure_analysis['jsonb_columns'])
    
    # Save to CSV
    logger.info(f"💾 Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Print summary
    logger.info("="*60)
    logger.info("INSTANTLY API QUERY SUMMARY")
    logger.info("="*60)
    logger.info(f"📊 Total contacts: {len(df)}")
    logger.info(f"🔧 Total columns: {len(df.columns)}")
    logger.info(f"📁 Output file: {output_file}")
    logger.info(f"📏 File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    # Print column summary
    logger.info("\n📋 Column Summary:")
    for i, col in enumerate(df.columns, 1):
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        logger.info(f"  {i:3d}. {col:<30} | Non-null: {non_null_count:4d} | Null: {null_count:4d}")
    
    # Print JSONB structure details
    if structure_analysis['nested_structures']:
        logger.info("\n🔍 JSONB Structure Details:")
        for col, nested_keys in structure_analysis['nested_structures'].items():
            logger.info(f"  {col}: {', '.join(nested_keys[:10])}{'...' if len(nested_keys) > 10 else ''}")
    
    return df

def main():
    """Main function to run the Instantly API query."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Query Instantly API and export contacts to CSV')
    parser.add_argument('--api-key', help='Your Instantly API key (optional if using env file)')
    parser.add_argument('--env-file', default=r"C:\Users\mccal\Downloads\Instantly B2B Main\Instantly.env", 
                       help='Path to .env file containing API key')
    parser.add_argument('--output', default='instantly_contacts_export.csv', help='Output CSV filename')
    parser.add_argument('--limit', type=int, default=100, help='Number of contacts to retrieve')
    parser.add_argument('--search', help='Optional search query to filter contacts')
    
    args = parser.parse_args()
    
    # Get API key from command line or environment file
    api_key = args.api_key
    if not api_key:
        logger.info("No API key provided via command line, attempting to load from environment file...")
        api_key = load_api_key_from_env_file(args.env_file)
        
        if not api_key:
            logger.error("❌ No API key found. Please provide --api-key or ensure the environment file contains a valid API key.")
            return
    
    # Run the query
    df = query_instantly_api(
        api_key=api_key,
        output_file=args.output,
        limit=args.limit,
        search_query=args.search
    )
    
    if not df.empty:
        logger.info("✅ Successfully exported Instantly contacts to CSV!")
        logger.info(f"📁 File saved as: {args.output}")
    else:
        logger.error("❌ Failed to retrieve contacts from Instantly API")

if __name__ == "__main__":
    # Example usage (uncomment and modify as needed)
    # df = query_instantly_api(
    #     api_key="your_api_key_here",
    #     output_file="instantly_contacts_export.csv",
    #     limit=1000
    # )
    
    # You can also run without arguments to use the default env file
    # python instantly_api_query.py
    
    main() 