#!/usr/bin/env python3
"""
Simple test of the bronze to silver pipeline
"""

from bronze_to_silver_pipeline import BronzeToSilverPipeline
import pandas as pd

def test_pipeline():
    print("🔧 Testing Bronze to Silver Pipeline...")
    
    try:
        # Initialize pipeline
        print("1️⃣  Initializing pipeline...")
        pipeline = BronzeToSilverPipeline()
        
        # Try to extract data
        print("2️⃣  Extracting bronze data...")
        df = pipeline.extract_bronze_data(incremental=False)
        print(f"   ✅ Extracted {len(df)} rows with {len(df.columns)} columns")
        
        if len(df) == 0:
            print("   ⚠️  No data found - table might be empty")
            print("   📊 Available columns:")
            for col in df.columns[:10]:
                print(f"      - {col}")
            return
        
        # Test target variable creation
        print("3️⃣  Testing target variable creation...")
        df_target = pipeline.create_target_variable(df.copy())
        
        if 'engagement_level' in df_target.columns:
            target_dist = df_target['engagement_level'].value_counts().sort_index()
            print(f"   ✅ Target variable created successfully")
            print(f"   📊 Target distribution: {dict(target_dist)}")
        else:
            print("   ❌ Target variable creation failed")
        
        # Test feature engineering steps
        print("4️⃣  Testing feature engineering...")
        
        # Timestamp features
        df_time = pipeline.engineer_timestamp_features(df_target.copy())
        timestamp_features = [col for col in df_time.columns if col.startswith('created_') or col.startswith('days_')]
        print(f"   ✅ Created {len(timestamp_features)} timestamp features")
        
        # Text features  
        df_text = pipeline.engineer_text_features(df_time.copy())
        text_features = [col for col in df_text.columns if 'text_' in col]
        print(f"   ✅ Created {len(text_features)} text features")
        
        # Categorical features
        df_cat = pipeline.engineer_categorical_features(df_text.copy())
        cat_features = [col for col in df_cat.columns if '_interaction' in col or '_grouped' in col]
        print(f"   ✅ Created {len(cat_features)} categorical features")
        
        # Apollo features
        df_apollo = pipeline.engineer_apollo_features(df_cat.copy())
        apollo_features = [col for col in df_apollo.columns if col.startswith('has_apollo') or col.startswith('apollo_') or col.startswith('company_') or col.startswith('is_')]
        print(f"   ✅ Created {len(apollo_features)} Apollo features")
        
        print("\n🎉 Pipeline test completed successfully!")
        print(f"📈 Final feature count: {len(df_apollo.columns)}")
        
        # Test small silver layer write
        print("5️⃣  Testing silver layer preparation...")
        silver_df = pipeline.prepare_silver_features(df_apollo.copy())
        print(f"   ✅ Prepared {len(silver_df)} records for silver layer")
        print(f"   📊 Silver features: {len(silver_df.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        print("\n💡 Next step: Run incremental pipeline with real data")
        print("   Command: python pipeline_runner.py --mode incremental --lookback-days 1")
    else:
        print("\n🔧 Fix the issues above before proceeding")

