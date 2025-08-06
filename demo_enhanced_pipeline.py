"""
Enhanced Pipeline Demonstration
Shows the integration of comprehensive preprocessing with production pipeline.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data for demonstration."""
    print("📊 Creating sample data...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic sample data
    data = {
        'email': [f'user{i}@example.com' for i in range(n_samples)],
        'first_name': np.random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David'], n_samples),
        'last_name': np.random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], n_samples),
        'organization_employees': np.random.choice([10, 50, 200, 500, 1000, 5000], n_samples),
        'daily_limit': np.random.choice([50, 100, 150, 200, 250, 300, 400, 500], n_samples),
        'country': np.random.choice(['United States', 'Canada', 'United Kingdom', 'Germany', 'France'], n_samples),
        'title': np.random.choice(['Manager', 'Director', 'VP', 'CEO', 'Engineer', 'Analyst'], n_samples),
        'organization_industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Manufacturing', 'Retail'], n_samples),
        'esp_code': np.random.choice([2.0, 8.0, 11.0, 3.0, 1.0, 999.0], n_samples),
        'timestamp_created': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'campaign_id': [f'campaign_{i%10}' for i in range(n_samples)],
        'email_subjects': [f'Subject line {i}' for i in range(n_samples)],
        'email_bodies': [f'Email body content for lead {i}' for i in range(n_samples)],
        'seniority': np.random.choice(['Junior', 'Mid-level', 'Senior', 'Executive'], n_samples),
        'city': np.random.choice(['New York', 'London', 'Toronto', 'Berlin', 'Paris'], n_samples),
        'state': np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL'], n_samples),
        'page_retrieved': np.random.randint(1, 100, n_samples),
        'organization_founded_year': np.random.randint(1980, 2020, n_samples),
        'employment_history': ['{"company": "ABC Corp"}' if i % 3 == 0 else None for i in range(n_samples)],
        'organization_data': ['{"industry": "Tech"}' if i % 4 == 0 else None for i in range(n_samples)],
        'account_data': ['{"status": "active"}' if i % 5 == 0 else None for i in range(n_samples)],
        'api_response_raw': ['{"enriched": true}' if i % 6 == 0 else None for i in range(n_samples)]
    }
    
    df = pd.DataFrame(data)
    
    # Create target variables
    # Binary target: opened
    df['opened'] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Multi-class target: engagement_level
    df['email_open_count'] = np.where(df['opened'] == 1, np.random.randint(1, 5), 0)
    df['email_click_count'] = np.where(df['opened'] == 1, np.random.randint(0, 3), 0)
    df['email_reply_count'] = np.where(df['opened'] == 1, np.random.randint(0, 2), 0)
    
    print(f"✅ Sample data created. Shape: {df.shape}")
    return df

def demonstrate_binary_target():
    """Demonstrate the pipeline with binary target."""
    print("\n" + "="*60)
    print("BINARY TARGET DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    df = create_sample_data()
    
    # Save sample data
    df.to_csv('data/sample_data.csv', index=False)
    print("✅ Sample data saved to data/sample_data.csv")
    
    # Update config for binary target
    config_path = Path("config/main_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['target_type'] = 'binary'
    config['data']['target_variable'] = 'opened'
    config['data']['input_file'] = 'data/sample_data.csv'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration updated for binary target")
    print("🎯 Target: opened (0/1)")
    print("📊 Target distribution:")
    print(df['opened'].value_counts().sort_index())
    
    return df

def demonstrate_multiclass_target():
    """Demonstrate the pipeline with multi-class target."""
    print("\n" + "="*60)
    print("MULTI-CLASS TARGET DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    df = create_sample_data()
    
    # Create multi-class target
    conditions = [
        ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
        (df['email_open_count'] > 0)                                       # Tier 1: Open
    ]
    choices = [2, 1]
    df['engagement_level'] = np.select(conditions, choices, default=0)
    
    # Save sample data
    df.to_csv('data/sample_data_multiclass.csv', index=False)
    print("✅ Sample data saved to data/sample_data_multiclass.csv")
    
    # Update config for multi-class target
    config_path = Path("config/main_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['target_type'] = 'multiclass'
    config['data']['target_variable'] = 'engagement_level'
    config['data']['input_file'] = 'data/sample_data_multiclass.csv'
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("✅ Configuration updated for multi-class target")
    print("🎯 Target: engagement_level (0, 1, 2)")
    print("📊 Target distribution:")
    print(df['engagement_level'].value_counts().sort_index())
    
    return df

def show_enhanced_features():
    """Show the enhanced feature engineering capabilities."""
    print("\n" + "="*60)
    print("ENHANCED FEATURE ENGINEERING CAPABILITIES")
    print("="*60)
    
    from src.feature_engineering import (
        enhanced_text_preprocessing, advanced_timestamp_features,
        create_interaction_features, create_jsonb_features,
        handle_outliers, create_xgboost_optimized_features
    )
    
    # Create sample data
    df = create_sample_data()
    
    print("🔧 Original features:", len(df.columns))
    print("Sample features:", list(df.columns[:10]))
    
    # Apply enhanced feature engineering
    print("\n🔧 Applying enhanced feature engineering...")
    
    # 1. Enhanced text preprocessing
    df = enhanced_text_preprocessing(df)
    print("✅ Text preprocessing complete")
    
    # 2. Advanced timestamp features
    df = advanced_timestamp_features(df)
    print("✅ Timestamp features complete")
    
    # 3. Interaction features
    df = create_interaction_features(df)
    print("✅ Interaction features complete")
    
    # 4. JSONB features
    df = create_jsonb_features(df)
    print("✅ JSONB features complete")
    
    # 5. Domain-specific features
    df = create_xgboost_optimized_features(df)
    print("✅ Domain-specific features complete")
    
    # 6. Outlier handling
    df = handle_outliers(df)
    print("✅ Outlier handling complete")
    
    print(f"\n✅ Enhanced feature engineering complete!")
    print(f"📊 Final features: {len(df.columns)}")
    print(f"📈 Feature increase: {len(df.columns) - 21} new features")
    
    # Show some new features
    new_features = [col for col in df.columns if col not in [
        'email', 'first_name', 'last_name', 'organization_employees', 'daily_limit',
        'country', 'title', 'organization_industry', 'esp_code', 'timestamp_created',
        'campaign_id', 'email_subjects', 'email_bodies', 'seniority', 'city', 'state',
        'page_retrieved', 'organization_founded_year', 'employment_history',
        'organization_data', 'account_data', 'api_response_raw'
    ]]
    
    print(f"\n🔧 New feature categories:")
    print(f"  Text features: {len([f for f in new_features if 'text' in f.lower()])}")
    print(f"  Temporal features: {len([f for f in new_features if 'created' in f.lower() or 'days' in f.lower()])}")
    print(f"  Interaction features: {len([f for f in new_features if 'interaction' in f.lower()])}")
    print(f"  JSONB features: {len([f for f in new_features if 'has_' in f.lower() or 'enrichment' in f.lower()])}")
    print(f"  Domain features: {len([f for f in new_features if 'daily_limit' in f or 'employees' in f or 'esp' in f])}")
    
    return df

def show_preprocessing_pipeline():
    """Show the advanced preprocessing pipeline."""
    print("\n" + "="*60)
    print("ADVANCED PREPROCESSING PIPELINE")
    print("="*60)
    
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, OneHotEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
    
    # Create sample data
    df = create_sample_data()
    df = show_enhanced_features()  # Apply feature engineering
    
    # Prepare features
    from src.feature_engineering import prepare_features_for_model
    X, y, selected_features = prepare_features_for_model(df, target_variable='opened')
    
    print(f"\n🔧 Creating advanced preprocessing pipeline...")
    print(f"📊 Input features: {X.shape[1]}")
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=30, sparse_output=False))
    ])
    
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(max_features=300, stop_words='english', ngram_range=(1, 2)))
    ])
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col.endswith('_interaction') or 
                          col in ['title', 'seniority', 'organization_industry', 'country', 'city']]
    text_feature = 'combined_text'
    
    # Remove categorical features from numeric features
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"📊 Feature types:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Text feature: {text_feature if text_feature in X.columns else 'None'}")
    
    # Create final pipeline
    transformers = [('num', numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    if text_feature in X.columns:
        transformers.append(('text', text_transformer, text_feature))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', SelectKBest(f_classif, k=60))
    ])
    
    print(f"\n✅ Advanced preprocessing pipeline created!")
    print(f"🔧 Pipeline steps:")
    print(f"  1. ColumnTransformer (numeric, categorical, text)")
    print(f"  2. VarianceThreshold (remove low-variance features)")
    print(f"  3. SelectKBest (select top 60 features)")
    
    return final_pipeline

def main():
    """Main demonstration function."""
    print("🚀 ENHANCED PIPELINE DEMONSTRATION")
    print("="*60)
    print("This demonstration shows the integration of comprehensive")
    print("preprocessing with the production pipeline.")
    print("="*60)
    
    try:
        # Ensure directories exist
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # 1. Show enhanced feature engineering
        show_enhanced_features()
        
        # 2. Show preprocessing pipeline
        show_preprocessing_pipeline()
        
        # 3. Demonstrate binary target
        demonstrate_binary_target()
        
        # 4. Demonstrate multi-class target
        demonstrate_multiclass_target()
        
        print("\n" + "="*60)
        print("🎉 DEMONSTRATION COMPLETE!")
        print("="*60)
        print("✅ Enhanced feature engineering integrated")
        print("✅ Advanced preprocessing pipeline created")
        print("✅ Binary and multi-class target support")
        print("✅ Configuration flexibility demonstrated")
        print("\n📁 Files created:")
        print("  - data/sample_data.csv (binary target)")
        print("  - data/sample_data_multiclass.csv (multi-class target)")
        print("  - config/main_config.yaml (updated)")
        
        print("\n🚀 Next steps:")
        print("  1. Run: python src/train.py (to train the model)")
        print("  2. Run: python src/predict.py (to make predictions)")
        print("  3. Run: python src/monitor.py (to monitor performance)")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 