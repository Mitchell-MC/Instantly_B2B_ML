#!/usr/bin/env python3
"""
Database Connection Test Script
Tests connectivity and permissions for the bronze-to-silver pipeline
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
from pathlib import Path

def load_config(config_path: str = "config/silver_layer_config.yaml"):
    """Load configuration"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None

def test_database_connection():
    """Test database connectivity and permissions"""
    print("🔍 Testing Database Connection and Permissions...")
    print("=" * 60)
    
    # Load config
    config = load_config()
    if not config:
        return False
    
    db_config = config['database']
    
    # Test 1: Basic connection
    print("1️⃣  Testing basic database connection...")
    try:
        connection_string = (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        engine = create_engine(connection_string)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            if result[0] == 1:
                print("   ✅ Basic connection successful")
            else:
                print("   ❌ Basic connection failed")
                return False
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # Test 2: Schema access
    print("\n2️⃣  Testing schema access...")
    try:
        with engine.connect() as conn:
            # List available schemas
            schemas_query = text("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('leads', 'public', 'ml_lead_scoring')
            ORDER BY schema_name
            """)
            schemas = conn.execute(schemas_query).fetchall()
            available_schemas = [row[0] for row in schemas]
            print(f"   📋 Available schemas: {available_schemas}")
            
            if 'leads' in available_schemas:
                print("   ✅ 'leads' schema found")
            else:
                print("   ❌ 'leads' schema not found")
                return False
    except Exception as e:
        print(f"   ❌ Schema access failed: {e}")
        return False
    
    # Test 3: Table access
    print("\n3️⃣  Testing table access...")
    try:
        with engine.connect() as conn:
            # Check if bronze table exists and is accessible
            table_query = text("""
            SELECT COUNT(*) as row_count
            FROM leads.instantly_enriched_contacts
            LIMIT 1
            """)
            result = conn.execute(table_query).fetchone()
            print(f"   ✅ Bronze table accessible (sample count: {result[0]})")
    except Exception as e:
        print(f"   ❌ Bronze table access failed: {e}")
        print(f"   💡 This might be a permissions issue or table doesn't exist")
        return False
    
    # Test 4: Write permissions
    print("\n4️⃣  Testing write permissions...")
    try:
        with engine.connect() as conn:
            # Try to create a test table
            test_table_query = text("""
            CREATE TABLE IF NOT EXISTS leads.pipeline_test_table (
                id SERIAL PRIMARY KEY,
                test_column TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.execute(test_table_query)
            conn.commit()
            
            # Try to insert a test record
            insert_query = text("""
            INSERT INTO leads.pipeline_test_table (test_column) 
            VALUES ('connection_test')
            """)
            conn.execute(insert_query)
            conn.commit()
            
            # Clean up test table
            cleanup_query = text("DROP TABLE IF EXISTS leads.pipeline_test_table")
            conn.execute(cleanup_query)
            conn.commit()
            
            print("   ✅ Write permissions confirmed")
    except Exception as e:
        print(f"   ❌ Write permission test failed: {e}")
        print(f"   💡 You may need GRANT permissions on the leads schema")
        return False
    
    # Test 5: Sample data extraction
    print("\n5️⃣  Testing sample data extraction...")
    try:
        query = text("SELECT * FROM leads.instantly_enriched_contacts LIMIT 3")
        df = pd.read_sql(query, engine)
        print(f"   ✅ Sample extraction successful: {len(df)} rows, {len(df.columns)} columns")
        print(f"   📊 Sample columns: {list(df.columns)[:5]}...")
        
        # Check for key columns
        required_columns = ['id', 'email', 'campaign', 'email_open_count', 'email_click_count', 'email_reply_count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"   ⚠️  Missing required columns: {missing_columns}")
        else:
            print("   ✅ All required columns present")
            
    except Exception as e:
        print(f"   ❌ Sample data extraction failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All database tests passed! Pipeline should work correctly.")
    return True

def main():
    """Main test function"""
    success = test_database_connection()
    
    if success:
        print("\n💡 Next steps:")
        print("   1. Run: python validate_schema_alignment.py")
        print("   2. Create silver table: psql -f sql/create_silver_table_in_leads_schema.sql")
        print("   3. Test pipeline: python pipeline_runner.py --mode incremental --lookback-days 1")
        sys.exit(0)
    else:
        print("\n🔧 Troubleshooting suggestions:")
        print("   1. Check database credentials in config/silver_layer_config.yaml")
        print("   2. Ensure user has permissions on 'leads' schema")
        print("   3. Verify 'instantly_enriched_contacts' table exists")
        print("   4. Contact your database administrator for permissions")
        sys.exit(1)

if __name__ == "__main__":
    main()
