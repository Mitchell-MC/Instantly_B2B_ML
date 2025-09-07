#!/usr/bin/env python3
"""
Get all actual column names from the database
"""

from sqlalchemy import create_engine, text
import yaml

def get_column_names():
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
            AND table_name = 'instantly_enriched_contacts' 
            ORDER BY column_name
        """))
        columns = result.fetchall()

    print('All column names and types:')
    
    # Group by prefix
    i_columns = []
    l_columns = []
    a_columns = []
    other_columns = []
    
    for col_name, data_type in columns:
        if col_name.startswith('I_'):
            i_columns.append((col_name, data_type))
        elif col_name.startswith('L_'):
            l_columns.append((col_name, data_type))
        elif col_name.startswith('A_'):
            a_columns.append((col_name, data_type))
        else:
            other_columns.append((col_name, data_type))
    
    print(f"\n🔵 I_ prefix columns ({len(i_columns)}):")
    for col_name, data_type in i_columns:
        print(f"  {col_name} ({data_type})")
    
    print(f"\n🟡 L_ prefix columns ({len(l_columns)}):")
    for col_name, data_type in l_columns:
        print(f"  {col_name} ({data_type})")
    
    print(f"\n🟢 A_ prefix columns ({len(a_columns)}):")
    for col_name, data_type in a_columns:
        print(f"  {col_name} ({data_type})")
    
    print(f"\n⚪ Other columns ({len(other_columns)}):")
    for col_name, data_type in other_columns:
        print(f"  {col_name} ({data_type})")

if __name__ == "__main__":
    get_column_names()
