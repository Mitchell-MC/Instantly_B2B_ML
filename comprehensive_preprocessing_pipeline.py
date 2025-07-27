"""
Comprehensive Preprocessing Pipeline for B2B Email Marketing Dataset

This script provides enhanced preprocessing steps specifically designed for the
merged_contacts.csv dataset, incorporating best practices for feature engineering,
data quality checks, and advanced preprocessing techniques.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
TEXT_COLS_FOR_FEATURE = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

# List of columns to drop due to leakage, irrelevance, or being empty
COLS_TO_DROP = [
    # General/ Personal Identifiers
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    
    # Leakage Columns
    'email_reply_count', 'email_opened_variant', 'email_opened_step', 
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated', 
    'status_summary', 'email_clicked_variant', 'email_clicked_step',
    
    # Metadata and other unused columns
    'personalization', 'payload', 'list_id', 'assigned_to', 'campaign', 'uploaded_by_user',

    # Empty columns identified from logs
    'auto_variant_select', 'verification_status'
]

def load_data(file_path: Path) -> pd.DataFrame:
    """Enhanced data loading with comprehensive datetime handling."""
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
    print(f"Data successfully loaded. Shape: {df.shape}")

    # Standardize all potential timestamp columns to UTC
    all_potential_timestamps = list(set(TIMESTAMP_COLS + [col for col in COLS_TO_DROP if 'timestamp' in col]))
    for col in all_potential_timestamps:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Timezone handling
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.tz_localize('UTC', nonexistent='NaT')
    for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        if df[col].dt.tz != 'UTC':
            df[col] = df[col].dt.tz_convert('UTC')
    
    return df

def data_quality_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Comprehensive data quality assessment."""
    print("\n=== DATA QUALITY REPORT ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Missing values analysis
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (missing_stats / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Missing_Count': missing_stats,
        'Missing_Percent': missing_percent
    })
    print(f"\nTop 10 columns with missing values:\n{missing_df.head(10)}")
    
    # Duplicate analysis
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    
    # High cardinality check
    print("\nHigh cardinality categorical columns:")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            unique_count = df[col].nunique()
            if unique_count > 50:
                print(f"  {col}: {unique_count} unique values")
    
    return df

def enhanced_text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced text preprocessing with additional features."""
    print("\nEnhanced text preprocessing...")
    
    # Clean and normalize text columns
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            # Basic cleaning
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
    
    # Create combined text column
    df['combined_text'] = ""
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '
    
    # Create text-based features
    df['combined_text_length'] = df['combined_text'].str.len()
    df['combined_text_word_count'] = df['combined_text'].str.split().str.len()
    df['has_numbers_in_text'] = df['combined_text'].str.contains(r'\d', regex=True).astype(int)
    df['has_email_in_text'] = df['combined_text'].str.contains(r'@', regex=True).astype(int)
    df['has_url_in_text'] = df['combined_text'].str.contains(r'http', regex=True).astype(int)
    
    # Text quality indicators
    df['text_quality_score'] = (
        (df['combined_text_length'] > 10).astype(int) +
        (df['combined_text_word_count'] > 3).astype(int) +
        (~df['combined_text'].str.contains('nan', regex=True)).astype(int)
    )
    
    print(f"Text features created. Combined text length range: {df['combined_text_length'].min()}-{df['combined_text_length'].max()}")
    
    return df

def advanced_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced timestamp feature engineering with business insights."""
    print("\nAdvanced timestamp feature engineering...")
    
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        # Basic datetime features
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_hour'] = df['timestamp_created'].dt.hour
        df['created_quarter'] = df['timestamp_created'].dt.quarter
        
        # Business-relevant features
        df['created_is_weekend'] = (df['timestamp_created'].dt.dayofweek >= 5).astype(int)
        df['created_is_business_hours'] = (
            (df['timestamp_created'].dt.hour >= 9) & 
            (df['timestamp_created'].dt.hour <= 17)
        ).astype(int)
        df['created_is_morning'] = (df['timestamp_created'].dt.hour <= 12).astype(int)
        
        # Seasonal features
        df['created_season'] = df['timestamp_created'].dt.month % 12 // 3
        
        # Time differences relative to creation
        ref_date = df['timestamp_created']
        for col in TIMESTAMP_COLS:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days
                
                # Additional temporal features
                df[f"{feature_name}_abs"] = df[feature_name].abs()
                df[f"has_{col.replace('timestamp_', '')}"] = df[col].notna().astype(int)
        
        # Recency from current time
        current_time = pd.Timestamp.now(tz='UTC')
        df['days_since_creation'] = (current_time - df['timestamp_created']).dt.days
        df['weeks_since_creation'] = df['days_since_creation'] // 7
        
        print(f"Timestamp features created. Creation date range: {df['timestamp_created'].min()} to {df['timestamp_created'].max()}")
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-relevant interaction features."""
    print("\nCreating interaction features...")
    
    # High-value B2B interactions
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        df['industry_seniority_interaction'] = (
            df['organization_industry'].fillna('Unknown') + '_' + 
            df['seniority'].fillna('Unknown')
        )
    
    # Geographic-industry interaction
    if 'country' in df.columns and 'organization_industry' in df.columns:
        df['geo_industry_interaction'] = (
            df['country'].fillna('Unknown') + '_' + 
            df['organization_industry'].fillna('Unknown')
        )
    
    # Title-industry interaction
    if 'title' in df.columns and 'organization_industry' in df.columns:
        df['title_industry_interaction'] = (
            df['title'].fillna('Unknown') + '_' + 
            df['organization_industry'].fillna('Unknown')
        )
    
    return df

def create_jsonb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced JSONB feature engineering."""
    print("\nCreating JSONB features...")
    
    # JSONB presence indicators
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    # Enrichment completeness score
    enrichment_cols = [f'has_{col}' for col in JSONB_COLS if f'has_{col}' in df.columns]
    if enrichment_cols:
        df['enrichment_completeness'] = df[enrichment_cols].sum(axis=1)
        df['enrichment_completeness_pct'] = df['enrichment_completeness'] / len(enrichment_cols)
    
    return df

def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Outlier detection and treatment for numerical features."""
    print("\nHandling outliers...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col.startswith('days_between_') or col.startswith('days_since_'):
            # Cap extreme day differences
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            if q99 - q01 > 0:  # Only cap if there's variation
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        elif col.endswith('_length') or col.endswith('_count'):
            # Cap extreme text lengths/counts
            q95 = df[col].quantile(0.95)
            df[col] = df[col].clip(upper=q95)
    
    return df

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Comprehensive feature engineering pipeline."""
    print("\n=== COMPREHENSIVE FEATURE ENGINEERING ===")
    
    # 1. Data quality checks
    df = data_quality_checks(df)
    
    # 2. Create target variable
    if 'email_reply_count' not in df.columns or 'email_click_count' not in df.columns or 'email_open_count' not in df.columns:
        print("Error: Source columns for engagement level not found.")
        sys.exit(1)
    
    conditions = [
        ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
        (df['email_open_count'] > 0)                                       # Tier 1: Open
    ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)
    
    # Check target distribution
    target_dist = df[TARGET_VARIABLE].value_counts().sort_index()
    print(f"\nTarget variable distribution:\n{target_dist}")
    print(f"Class proportions: {target_dist / len(df)}")
    
    # 3. Enhanced text preprocessing
    df = enhanced_text_preprocessing(df)
    
    # 4. Advanced timestamp features
    df = advanced_timestamp_features(df)
    
    # 5. Create interaction features
    df = create_interaction_features(df)
    
    # 6. JSONB features
    df = create_jsonb_features(df)
    
    # 7. Handle outliers
    df = handle_outliers(df)
    
    # 8. Separate features and target
    y = pd.Series(df[TARGET_VARIABLE].copy())
    
    # Define all columns to drop
    all_source_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS_FOR_FEATURE + all_source_cols + [TARGET_VARIABLE]))
    
    # Create feature set
    X = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')
    
    print(f"\nFeature engineering complete. Final shape: {X.shape}")
    print(f"Features include: {X.columns.tolist()[:10]}...")
    
    return X, y

def create_advanced_preprocessor():
    """Create advanced preprocessing pipeline."""
    print("\n=== CREATING ADVANCED PREPROCESSOR ===")
    
    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # More robust to outliers than StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            max_categories=30,  # Reduce to prevent overfitting
            sparse_output=False,
            drop='if_binary'
        ))
    ])
    
    # Enhanced text transformer
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(
            max_features=300,  # Balanced number of features
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        ))
    ])
    
    return numeric_transformer, categorical_transformer, text_transformer

def main():
    """Main function to demonstrate comprehensive preprocessing."""
    # Load data
    df = load_data(CSV_FILE_PATH)
    
    # Apply comprehensive feature engineering
    X, y = engineer_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Create preprocessor
    numeric_transformer, categorical_transformer, text_transformer = create_advanced_preprocessor()
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col.endswith('_interaction') or col in CATEGORICAL_COLS]
    text_feature = 'combined_text'
    
    # Remove categorical features from numeric features
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"\nFeature types:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Text feature: {text_feature}")
    
    # Create final preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text', text_transformer, text_feature)
        ],
        remainder='drop'
    )
    
    # Add feature selection
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', SelectKBest(f_classif, k=500))
    ])
    
    print("\n=== PREPROCESSING PIPELINE SUMMARY ===")
    print("✅ Data quality checks")
    print("✅ Enhanced text preprocessing")
    print("✅ Advanced timestamp features")
    print("✅ Business interaction features")
    print("✅ JSONB enrichment features")
    print("✅ Outlier handling")
    print("✅ Robust scaling")
    print("✅ Feature selection")
    print("✅ Class imbalance awareness")
    
    return final_pipeline, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    pipeline, X_train, X_test, y_train, y_test = main()
    
    print(f"\nReady for model training with:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Test set: {X_test.shape}")
    if hasattr(y_train, 'unique'):
        print(f"  Target classes: {sorted(y_train.unique())}")
    else:
        print(f"  Target classes: {sorted(np.unique(y_train))}") 