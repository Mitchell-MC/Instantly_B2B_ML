#!/usr/bin/env python3
"""
Get actual schema from enriched_contacts table
"""

from sqlalchemy import create_engine, text
import yaml

def get_actual_schema():
    with open('config/silver_layer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db_config = config['database']
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_string)

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'leads' 
            AND table_name = 'enriched_contacts' 
            ORDER BY column_name
        """))
        columns = result.fetchall()

    print('=== ACTUAL SCHEMA FROM enriched_contacts ===')
    print(f'Total columns: {len(columns)}')
    
    # Group columns by type
    engagement_cols = []
    apollo_cols = []
    timestamp_cols = []
    text_cols = []
    other_cols = []
    
    for col_name, data_type in columns:
        if 'email_' in col_name and ('open' in col_name or 'click' in col_name or 'reply' in col_name):
            engagement_cols.append((col_name, data_type))
        elif col_name.startswith('apollo_') or col_name in ['title', 'city', 'state', 'country', 'organization_name', 'organization_industry', 'seniority', 'departments', 'functions']:
            apollo_cols.append((col_name, data_type))
        elif 'timestamp' in col_name or col_name.endswith('_at'):
            timestamp_cols.append((col_name, data_type))
        elif data_type in ['text', 'character varying']:
            text_cols.append((col_name, data_type))
        else:
            other_cols.append((col_name, data_type))
    
    print(f"\n📊 ENGAGEMENT COLUMNS ({len(engagement_cols)}):")
    for col_name, data_type in engagement_cols:
        print(f"  {col_name}: {data_type}")
    
    print(f"\n🌟 APOLLO COLUMNS ({len(apollo_cols)}):")
    for col_name, data_type in apollo_cols:
        print(f"  {col_name}: {data_type}")
    
    print(f"\n⏰ TIMESTAMP COLUMNS ({len(timestamp_cols)}):")
    for col_name, data_type in timestamp_cols:
        print(f"  {col_name}: {data_type}")
    
    print(f"\n📝 TEXT COLUMNS ({len(text_cols)}):")
    for col_name, data_type in text_cols[:10]:  # Show first 10
        print(f"  {col_name}: {data_type}")
    if len(text_cols) > 10:
        print(f"  ... and {len(text_cols) - 10} more")
    
    print(f"\n📋 OTHER COLUMNS ({len(other_cols)}):")
    for col_name, data_type in other_cols[:10]:  # Show first 10
        print(f"  {col_name}: {data_type}")
    if len(other_cols) > 10:
        print(f"  ... and {len(other_cols) - 10} more")

if __name__ == "__main__":
    get_actual_schema()
