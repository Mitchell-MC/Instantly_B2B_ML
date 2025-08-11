"""
Comprehensive data extraction script for Instantly API.
Extracts all potential columns from all available endpoints.
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

def analyze_data_structure(data_list, endpoint_name):
    """Analyze the structure of data to identify all available columns."""
    if not data_list:
        return {}
    
    # Collect all unique keys from all entries
    all_keys = set()
    nested_structures = {}
    
    for item in data_list:
        # Add all top-level keys
        all_keys.update(item.keys())
        
        # Identify nested structures
        for key, value in item.items():
            if isinstance(value, dict):
                if key not in nested_structures:
                    nested_structures[key] = set()
                nested_structures[key].update(value.keys())
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                if key not in nested_structures:
                    nested_structures[key] = set()
                for list_item in value[:5]:  # Sample first 5 items
                    nested_structures[key].update(list_item.keys())
    
    return {
        'endpoint': endpoint_name,
        'total_entries': len(data_list),
        'all_columns': sorted(list(all_keys)),
        'nested_structures': {k: sorted(list(v)) for k, v in nested_structures.items()},
        'sample_data': data_list[0] if data_list else {}
    }

def extract_all_data():
    """Extract all available data from all Instantly API endpoints."""
    
    # Load API key from env file
    api_key = load_api_key_from_env_file()
    
    if not api_key:
        logger.error("❌ Could not load API key from environment file.")
        return
    
    logger.info(f"🔑 Using API key: {api_key[:8]}...{api_key[-4:]}")
    
    # V2 API base URL and headers
    base_url = "https://api.instantly.ai"
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    # Define all endpoints to test
    endpoints = [
        "/api/v2/campaigns",
        "/api/v2/accounts", 
        "/api/v2/emails",
        "/api/v2/contacts",
        "/api/v2/leads",
        "/api/v2/people",
        "/api/v2/workspaces"
    ]
    
    all_data = {}
    comprehensive_summary = []
    
    logger.info("🔍 Extracting data from all available endpoints...")
    
    for endpoint in endpoints:
        try:
            url = f"{base_url}{endpoint}"
            logger.info(f"\n📋 Testing endpoint: {endpoint}")
            
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                logger.info(f"✅ SUCCESS! Found {len(items)} entries")
                
                if items:
                    # Analyze the data structure
                    structure = analyze_data_structure(items, endpoint)
                    all_data[endpoint] = structure
                    comprehensive_summary.append(structure)
                    
                    # Show sample of available columns
                    logger.info(f"   Available columns: {len(structure['all_columns'])}")
                    logger.info(f"   Sample columns: {structure['all_columns'][:10]}{'...' if len(structure['all_columns']) > 10 else ''}")
                    
                    # Show nested structures if any
                    if structure['nested_structures']:
                        logger.info(f"   Nested structures: {list(structure['nested_structures'].keys())}")
                
            elif response.status_code == 404:
                logger.info(f"❌ Endpoint not found: {endpoint}")
            elif response.status_code == 401:
                logger.info(f"❌ Authentication failed for {endpoint}")
            else:
                logger.info(f"⚠️  Status {response.status_code} for {endpoint}")
                
        except Exception as e:
            logger.info(f"❌ Error with {endpoint}: {e}")
    
    # Create comprehensive report
    logger.info("\n" + "="*80)
    logger.info("COMPREHENSIVE DATA STRUCTURE ANALYSIS")
    logger.info("="*80)
    
    # Save detailed analysis to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_file = f"comprehensive_data_analysis_{timestamp}.json"
    
    with open(analysis_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    logger.info(f"💾 Detailed analysis saved to: {analysis_file}")
    
    # Create summary DataFrame
    summary_data = []
    for structure in comprehensive_summary:
        summary_data.append({
            'endpoint': structure['endpoint'],
            'total_entries': structure['total_entries'],
            'total_columns': len(structure['all_columns']),
            'columns': ', '.join(structure['all_columns']),
            'nested_structures': ', '.join(structure['nested_structures'].keys()) if structure['nested_structures'] else 'None'
        })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_file = f"data_structure_summary_{timestamp}.csv"
        df_summary.to_csv(summary_file, index=False)
        
        logger.info(f"💾 Summary saved to: {summary_file}")
        
        # Print comprehensive summary
        logger.info("\n📊 COMPREHENSIVE DATA SUMMARY:")
        logger.info("="*60)
        
        for _, row in df_summary.iterrows():
            logger.info(f"\n🔗 {row['endpoint']}")
            logger.info(f"   📊 Entries: {row['total_entries']}")
            logger.info(f"   📋 Columns: {row['total_columns']}")
            logger.info(f"   📝 Available fields: {row['columns']}")
            if row['nested_structures'] != 'None':
                logger.info(f"   🗂️  Nested structures: {row['nested_structures']}")
        
        # Create individual CSV files for each endpoint
        logger.info("\n💾 Creating individual CSV files...")
        
        for endpoint, structure in all_data.items():
            if structure['total_entries'] > 0:
                # Create DataFrame from the data
                df = pd.DataFrame(structure['sample_data'] if structure['total_entries'] == 1 else items)
                
                # Clean endpoint name for filename
                clean_endpoint = endpoint.replace('/api/v2/', '').replace('/', '_')
                csv_file = f"{clean_endpoint}_data_{timestamp}.csv"
                df.to_csv(csv_file, index=False)
                
                logger.info(f"   ✅ {csv_file}: {len(df)} rows, {len(df.columns)} columns")
        
        return comprehensive_summary
    
    else:
        logger.error("❌ No data could be extracted from any endpoints")
        return None

if __name__ == "__main__":
    logger.info("🚀 Starting comprehensive data extraction...")
    
    result = extract_all_data()
    
    if result:
        logger.info(f"\n🎉 Successfully extracted data from {len(result)} endpoints!")
        logger.info("📁 Check the generated files for complete data structure analysis.")
    else:
        logger.info("\n❌ No data could be extracted.") 