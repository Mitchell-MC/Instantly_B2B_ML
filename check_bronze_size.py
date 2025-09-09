#!/usr/bin/env python3
"""
Check the full size of the bronze layer to process all available data
"""

import yaml
from sqlalchemy import create_engine, text

def check_bronze_size():
    """Check the bronze layer size"""
    
    # Load config
    with open('config/silver_layer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db_config = config['database']
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_string)

    print("🔍 CHECKING BRONZE LAYER SIZE")
    print("=" * 40)

    with engine.connect() as conn:
        # Check total bronze records
        result = conn.execute(text("""
            SELECT COUNT(*) as total_records
            FROM leads.enriched_contacts
        """))
        
        total = result.fetchone()[0]
        print(f"📋 Total records in bronze layer: {total:,}")
        
        # Check records with timestamps (processable)
        result = conn.execute(text("""
            SELECT COUNT(*) as processable_records
            FROM leads.enriched_contacts
            WHERE timestamp_created IS NOT NULL
        """))
        
        processable = result.fetchone()[0]
        print(f"⚡ Processable records (with timestamps): {processable:,}")
        
        # Check what's already in silver
        result = conn.execute(text("""
            SELECT COUNT(*) as silver_records
            FROM leads.silver_ml_features
        """))
        
        silver = result.fetchone()[0]
        print(f"💎 Current silver layer records: {silver:,}")
        
        remaining = processable - silver
        print(f"🚀 Remaining records to process: {remaining:,}")
        
        # Sample date range
        result = conn.execute(text("""
            SELECT 
                MIN(timestamp_created) as earliest,
                MAX(timestamp_created) as latest
            FROM leads.enriched_contacts
            WHERE timestamp_created IS NOT NULL
        """))
        
        dates = result.fetchone()
        print(f"📅 Date range: {dates[0]} to {dates[1]}")

if __name__ == "__main__":
    check_bronze_size()
