#!/usr/bin/env python3
"""
Complete pipeline test - Bronze to Silver ETL
"""

from bronze_to_silver_pipeline import BronzeToSilverPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete bronze to silver pipeline"""
    try:
        print("🚀 COMPLETE BRONZE-TO-SILVER PIPELINE TEST")
        print("=" * 50)
        
        # Initialize pipeline
        print("1️⃣  Initializing pipeline...")
        pipeline = BronzeToSilverPipeline(config_path="config/silver_layer_config.yaml")
        
        # Test small batch first
        print("2️⃣  Running pipeline on small batch (limit 100)...")
        pipeline.run_pipeline(incremental=False, write_to_silver=True, limit=100)
        
        print("3️⃣  Verifying silver layer data...")
        from sqlalchemy import text
        with pipeline.engine.connect() as conn:
            # Check record count
            result = conn.execute(text("SELECT COUNT(*) FROM leads.silver_ml_features"))
            count = result.scalar()
            print(f"   📊 Silver layer records: {count}")
            
            # Check feature columns
            result = conn.execute(text("""
                SELECT engagement_level, COUNT(*) as count 
                FROM leads.silver_ml_features 
                GROUP BY engagement_level 
                ORDER BY engagement_level
            """))
            print("   🎯 Target distribution:")
            for row in result:
                print(f"      Level {row[0]}: {row[1]} records")
            
            # Check Apollo features
            result = conn.execute(text("""
                SELECT 
                    AVG(has_apollo_enrichment) as apollo_rate,
                    AVG(data_quality_score) as avg_quality,
                    AVG(text_length) as avg_text_len
                FROM leads.silver_ml_features
            """))
            row = result.fetchone()
            print(f"   🌟 Apollo enrichment rate: {row[0]:.2%}")
            print(f"   📈 Average data quality: {row[1]:.3f}")
            print(f"   📝 Average text length: {row[2]:.0f}")
            
        print("\n✅ PIPELINE TEST SUCCESSFUL!")
        print("🎉 Ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🚀 Next steps:")
        print("   1. Run full pipeline: python pipeline_runner.py --mode incremental")
        print("   2. Set up scheduling: python pipeline_runner.py --mode scheduler")
        print("   3. Use silver layer for ML: SELECT * FROM leads.v_ml_training_features")
    else:
        print("\n🔧 Fix the issues above before proceeding")

