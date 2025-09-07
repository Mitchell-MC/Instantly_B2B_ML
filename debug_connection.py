#!/usr/bin/env python3
"""
Debug database connection configuration
"""

import yaml
from bronze_to_silver_pipeline import BronzeToSilverPipeline

def debug_connection():
    print("🔍 Debugging database connection...")
    
    # Load config directly
    with open('config/silver_layer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 Config from file:")
    db_config = config['database']
    for key, value in db_config.items():
        if key == 'password':
            print(f"   {key}: {'*' * len(str(value))}")
        else:
            print(f"   {key}: {value}")
    
    # Test connection string
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    print(f"\n🔗 Connection string: postgresql://{db_config['user']}:***@{db_config['host']}:{db_config['port']}/{db_config['database']}")
    
    # Initialize pipeline and check its config
    print("\n🔧 Pipeline configuration:")
    try:
        pipeline = BronzeToSilverPipeline()
        pipeline_db_config = pipeline.config['database']
        for key, value in pipeline_db_config.items():
            if key == 'password':
                print(f"   {key}: {'*' * len(str(value))}")
            else:
                print(f"   {key}: {value}")
    except Exception as e:
        print(f"   ❌ Pipeline init failed: {e}")

if __name__ == "__main__":
    debug_connection()
