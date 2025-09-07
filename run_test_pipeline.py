#!/usr/bin/env python3
"""
Run the bronze-to-silver pipeline with a small test batch
"""

from bronze_to_silver_pipeline import BronzeToSilverPipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test_pipeline():
    """Run the pipeline with a limited batch"""
    try:
        print("🚀 RUNNING BRONZE-TO-SILVER ETL PIPELINE")
        print("=" * 50)
        
        # Initialize pipeline
        print("1️⃣  Initializing pipeline...")
        pipeline = BronzeToSilverPipeline(config_path="config/silver_layer_config.yaml")
        
        # Run with small test batch
        print("2️⃣  Running pipeline (test batch of 1000 records)...")
        
        # Extract bronze data with limit
        df_bronze = pipeline.extract_bronze_data(incremental=False)
        print(f"   📊 Extracted {len(df_bronze)} records from bronze layer")
        
        # Take first 1000 for testing
        if len(df_bronze) > 1000:
            df_bronze = df_bronze.head(1000)
            print(f"   🔬 Using first 1000 records for testing")
        
        # Create target variable
        df_bronze = pipeline.create_target_variable(df_bronze)
        
        # Feature engineering
        print("3️⃣  Running feature engineering...")
        df_features = pipeline.engineer_timestamp_features(df_bronze)
        df_features = pipeline.engineer_text_features(df_features)
        df_features = pipeline.engineer_categorical_features(df_features)
        df_features = pipeline.engineer_apollo_features(df_features)
        df_features = pipeline.engineer_jsonb_features(df_features)
        
        # Data quality checks
        df_features = pipeline.apply_data_quality_checks(df_features)
        
        # Prepare silver layer
        silver_df = pipeline.prepare_silver_features(df_features)
        print(f"   ✨ Prepared {len(silver_df)} records with {len(silver_df.columns)} features")
        
        # Write to silver layer
        print("4️⃣  Writing to silver layer...")
        pipeline.write_to_silver_layer(silver_df)
        
        print(f"\n🎉 SUCCESS! Pipeline completed successfully!")
        print(f"📊 {len(silver_df)} records written to leads.silver_ml_features")
        print(f"🔧 Run verification: python quick_verify.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_test_pipeline()
    if success:
        print("\n🔍 Next steps:")
        print("   1. Verify data: python quick_verify.py")
        print("   2. Check SQL: psql ... -f simple_verify.sql")
        print("   3. Run full pipeline: python pipeline_runner.py --mode incremental")
    else:
        print("\n🔧 Fix the issues above before proceeding")
