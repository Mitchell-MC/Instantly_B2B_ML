"""
Test Advanced Feature Engineering
Validate the feature engineering module before full training
"""

import pandas as pd
import numpy as np
from advanced_feature_engineering import apply_advanced_feature_engineering

def test_advanced_feature_engineering():
    """Test the advanced feature engineering module."""
    print("🧪 Testing Advanced Feature Engineering")
    print("="*50)
    
    # Load a small sample of data
    print("📊 Loading sample data...")
    try:
        df = pd.read_csv("merged_contacts.csv", low_memory=False, nrows=1000)
        print(f"✅ Sample data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ merged_contacts.csv not found!")
        return
    
    # Display original columns
    print(f"📋 Original columns: {len(df.columns)}")
    print("Sample columns:", list(df.columns[:10]))
    
    # Test each feature engineering function individually
    print("\n🔧 Testing individual feature engineering functions...")
    
    # 1. Test JSONB features
    print("\n1. Testing JSONB features...")
    try:
        from advanced_feature_engineering import parse_jsonb_features
        df_test = df.copy()
        df_test = parse_jsonb_features(df_test)
        print(f"✅ JSONB features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ JSONB features failed: {e}")
    
    # 2. Test temporal features
    print("\n2. Testing temporal features...")
    try:
        from advanced_feature_engineering import create_advanced_temporal_features
        df_test = df.copy()
        df_test = create_advanced_temporal_features(df_test)
        print(f"✅ Temporal features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Temporal features failed: {e}")
    
    # 3. Test text features
    print("\n3. Testing text features...")
    try:
        from advanced_feature_engineering import create_text_embedding_features
        df_test = df.copy()
        df_test = create_text_embedding_features(df_test)
        print(f"✅ Text features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Text features failed: {e}")
    
    # 4. Test interaction features
    print("\n4. Testing interaction features...")
    try:
        from advanced_feature_engineering import create_interaction_features_advanced
        df_test = df.copy()
        df_test = create_interaction_features_advanced(df_test)
        print(f"✅ Interaction features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Interaction features failed: {e}")
    
    # 5. Test engagement features
    print("\n5. Testing engagement features...")
    try:
        from advanced_feature_engineering import create_engagement_pattern_features
        df_test = df.copy()
        df_test = create_engagement_pattern_features(df_test)
        print(f"✅ Engagement features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Engagement features failed: {e}")
    
    # 6. Test company maturity features
    print("\n6. Testing company maturity features...")
    try:
        from advanced_feature_engineering import create_company_maturity_features
        df_test = df.copy()
        df_test = create_company_maturity_features(df_test)
        print(f"✅ Company maturity features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Company maturity features failed: {e}")
    
    # 7. Test geographic features
    print("\n7. Testing geographic features...")
    try:
        from advanced_feature_engineering import create_advanced_geographic_features
        df_test = df.copy()
        df_test = create_advanced_geographic_features(df_test)
        print(f"✅ Geographic features added. New shape: {df_test.shape}")
    except Exception as e:
        print(f"❌ Geographic features failed: {e}")
    
    # Test full feature engineering pipeline
    print("\n🚀 Testing full feature engineering pipeline...")
    try:
        df_full = df.copy()
        df_full = apply_advanced_feature_engineering(df_full)
        print(f"✅ Full feature engineering completed!")
        print(f"   Original shape: {df.shape}")
        print(f"   Final shape: {df_full.shape}")
        print(f"   Features added: {df_full.shape[1] - df.shape[1]}")
        
        # Show new features
        original_cols = set(df.columns)
        new_cols = set(df_full.columns) - original_cols
        print(f"   New features: {len(new_cols)}")
        if new_cols:
            print("   Sample new features:", list(new_cols)[:10])
        
    except Exception as e:
        print(f"❌ Full feature engineering failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_advanced_feature_engineering() 