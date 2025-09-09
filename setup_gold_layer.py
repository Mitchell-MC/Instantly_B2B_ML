#!/usr/bin/env python3
"""
Setup Gold Layer Table and Test Pipeline
Run this when database is available to create the gold layer
"""

from sqlalchemy import create_engine, text
import yaml
import logging

def setup_gold_layer():
    """Create gold layer table and test pipeline"""
    
    print("🏗️  SETTING UP GOLD LAYER")
    print("=" * 40)
    
    try:
        # Load config
        with open('config/silver_layer_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        db_config = config['database']
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        engine = create_engine(connection_string)
        
        print("✅ Database connection established")
        
        # First, add timestamp columns to silver layer
        print("📋 Adding timestamp columns to silver layer...")
        with open('sql/add_timestamp_columns_to_silver.sql', 'r') as f:
            silver_update_script = f.read()
        
        with engine.connect() as conn:
            conn.execute(text(silver_update_script))
            conn.commit()
            print("✅ Silver layer timestamp columns added")
        
        # Create gold layer table
        print("📋 Creating gold layer table...")
        with open('sql/create_gold_layer_table.sql', 'r') as f:
            sql_script = f.read()
        
        with engine.connect() as conn:
            # Execute table creation
            conn.execute(text(sql_script))
            conn.commit()
            
            # Verify table creation
            result = conn.execute(text("""
                SELECT COUNT(*) 
                FROM information_schema.tables 
                WHERE table_schema = 'leads' 
                AND table_name = 'gold_ml_features'
            """))
            
            if result.fetchone()[0] > 0:
                print("✅ Gold layer table created successfully")
                
                # Check table structure
                columns_result = conn.execute(text("""
                    SELECT COUNT(*) 
                    FROM information_schema.columns 
                    WHERE table_schema = 'leads' 
                    AND table_name = 'gold_ml_features'
                """))
                
                column_count = columns_result.fetchone()[0]
                print(f"📊 Gold table has {column_count} columns")
                
                # Test pipeline
                print("\n🧪 Testing gold layer pipeline...")
                from silver_to_gold_pipeline import SilverToGoldPipeline
                
                pipeline = SilverToGoldPipeline()
                result = pipeline.run_pipeline(incremental=True, lookback_days=60)
                
                if result is not None:
                    print(f"✅ Pipeline test successful - processed {len(result)} mature leads")
                else:
                    print("ℹ️  No mature leads found for processing")
                
            else:
                print("❌ Gold layer table creation failed")
                
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    setup_gold_layer()
