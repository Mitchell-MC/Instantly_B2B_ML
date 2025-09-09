#!/usr/bin/env python3
"""
Drop the basic gold table so it can be recreated with full schema
"""

import yaml
from sqlalchemy import create_engine, text

# Load config
with open('config/silver_layer_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(connection_string)

try:
    with engine.connect() as conn:
        print("🗑️  Dropping existing basic gold table...")
        conn.execute(text('DROP TABLE IF EXISTS leads.gold_ml_features CASCADE'))
        conn.commit()
        print("✅ Basic gold table dropped successfully")
        print("💡 Next run will create comprehensive gold table with all features")

except Exception as e:
    print(f"❌ Error: {e}")
