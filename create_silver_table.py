#!/usr/bin/env python3
"""
Create silver layer table in leads schema
"""

from sqlalchemy import create_engine, text
import yaml

def create_silver_table():
    with open('config/silver_layer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db_config = config['database']
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_string)

    create_table_sql = """
    CREATE TABLE IF NOT EXISTS leads.silver_ml_features (
        id TEXT PRIMARY KEY,
        email VARCHAR(255),
        campaign VARCHAR(255),
        engagement_level INTEGER,
        processed_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """

    with engine.connect() as conn:
        conn.execute(text(create_table_sql))
        conn.commit()
        print('✅ Silver layer table created successfully!')

if __name__ == "__main__":
    create_silver_table()
