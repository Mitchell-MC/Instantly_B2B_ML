"""
Test RDS Connection Script
Simple script to test database connectivity before running main scripts.
"""

import psycopg2
import pandas as pd

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',  # Use localhost since we're connecting through SSH tunnel
    'database': 'postgres',
    'user': 'mitchell',
    'password': 'CTej3Ba8uBrx6o',
    'port': 5431  # Local port forwarded through SSH tunnel
}

def test_connection():
    """Test the database connection."""
    print("🔌 Testing RDS connection...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("✅ Successfully connected to RDS database")
        
        # Test a simple query
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"📊 Database version: {version[0]}")
        
        # Test table access
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'leads'
            LIMIT 5;
        """)
        tables = cursor.fetchall()
        print(f"📋 Available tables in 'leads' schema:")
        for table in tables:
            print(f"  - {table[0]}")
        
        # Test data access
        cursor.execute("""
            SELECT COUNT(*) 
            FROM leads.enriched_contacts 
            WHERE email_open_count IS NOT NULL;
        """)
        count = cursor.fetchone()
        print(f"📊 Total records in enriched_contacts: {count[0]:,}")
        
        # Test sample data
        cursor.execute("""
            SELECT * 
            FROM leads.enriched_contacts 
            WHERE email_open_count IS NOT NULL 
            LIMIT 1;
        """)
        sample = cursor.fetchone()
        if sample:
            print("✅ Sample data retrieved successfully")
        
        cursor.close()
        conn.close()
        
        print("🎉 All connection tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def test_data_loading():
    """Test loading a small sample of data."""
    print("\n📊 Testing data loading...")
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        
        # Load small sample
        query = """
        SELECT * FROM leads.enriched_contacts
        WHERE email_open_count IS NOT NULL
        LIMIT 100
        """
        
        df = pd.read_sql(query, conn)
        print(f"✅ Data loaded successfully. Shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        # Show data types
        print(f"📊 Data types:")
        print(df.dtypes.value_counts())
        
        # Show sample values
        print(f"📋 Sample data (first 3 rows):")
        print(df.head(3))
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 RDS Connection Test")
    print("=" * 50)
    
    # Test connection
    connection_ok = test_connection()
    
    if connection_ok:
        # Test data loading
        data_ok = test_data_loading()
        
        if data_ok:
            print("\n🎉 All tests passed! Ready to run RDS scripts.")
            return True
        else:
            print("\n❌ Data loading test failed.")
            return False
    else:
        print("\n❌ Connection test failed.")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ You can now run the RDS scripts:")
        print("  - src/train_rds_advanced.py")
        print("  - src/feature_engineering_rds.py")
        print("  - src/predict_rds.py")
    else:
        print("\n❌ Please fix the connection issues before running RDS scripts.")
