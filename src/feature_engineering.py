"""
Enhanced Feature Engineering Module for Email Opening Prediction
Production-ready feature engineering that incorporates advanced preprocessing techniques.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def enhanced_text_preprocessing(df):
    """Advanced text preprocessing with quality scoring and combined features."""
    print("🔧 Enhanced text preprocessing...")
    
    # Text columns for feature engineering
    text_cols = ['campaign_id', 'email_subjects', 'email_bodies']
    
    # Clean and normalize text columns
    for col in text_cols:
        if col in df.columns:
            # Basic cleaning
            df[col] = df[col].astype(str).str.lower()
            df[col] = df[col].str.replace(r'[^\w\s]', '', regex=True)
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.strip()
    
    # Create combined text column
    df['combined_text'] = ""
    for col in text_cols:
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

def advanced_timestamp_features(df):
    """Enhanced timestamp feature engineering with business insights."""
    print("🔧 Advanced timestamp feature engineering...")
    
    timestamp_cols = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 
                     'enriched_at', 'inserted_at', 'last_contacted_from']
    
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
        for col in timestamp_cols:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days
                
                # Additional temporal features
                df[f"{feature_name}_abs"] = df[feature_name].abs()
                df[f"has_{col.replace('timestamp_', '')}"] = df[col].notna().astype(int)
        
        # Recency from current time (handle timezone consistently)
        try:
            # Try to get current time with timezone
            current_time = pd.Timestamp.now(tz='UTC')
            # Ensure timestamp_created has timezone info
            if df['timestamp_created'].dt.tz is None:
                # If naive, assume UTC
                df['days_since_creation'] = (current_time - df['timestamp_created'].dt.tz_localize('UTC')).dt.days
            else:
                # If already has timezone, convert to UTC
                df['days_since_creation'] = (current_time - df['timestamp_created'].dt.tz_convert('UTC')).dt.days
        except Exception as e:
            # Fallback: use naive datetime
            current_time = pd.Timestamp.now()
            df['days_since_creation'] = (current_time - df['timestamp_created']).dt.days
        
        df['weeks_since_creation'] = df['days_since_creation'] // 7
        
        print(f"Timestamp features created. Creation date range: {df['timestamp_created'].min()} to {df['timestamp_created'].max()}")
    
    return df

def create_interaction_features(df):
    """Create business-relevant interaction features."""
    print("🔧 Creating interaction features...")
    
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

def create_jsonb_features(df):
    """Enhanced JSONB feature engineering."""
    print("🔧 Creating JSONB features...")
    
    jsonb_cols = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
    
    # JSONB presence indicators
    for col in jsonb_cols:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    
    # Enrichment completeness score
    enrichment_cols = [f'has_{col}' for col in jsonb_cols if f'has_{col}' in df.columns]
    if enrichment_cols:
        df['enrichment_completeness'] = df[enrichment_cols].sum(axis=1)
        df['enrichment_completeness_pct'] = df['enrichment_completeness'] / len(enrichment_cols)
    
    return df

def handle_outliers(df):
    """Outlier detection and treatment for numerical features."""
    print("🔧 Handling outliers...")
    
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

def create_xgboost_optimized_features(df):
    """
    Create XGBoost-optimized features for email opening prediction.
    """
    print("🔧 Creating XGBoost-optimized features...")
    
    # Create a copy to avoid modifying original
    df_optimized = df.copy()
    
    # 1. Daily limit features (handle missing column)
    if 'daily_limit' in df_optimized.columns:
        # Log transformation
        df_optimized['daily_limit_log'] = np.log1p(df_optimized['daily_limit'].fillna(0))
        
        # Square transformation
        df_optimized['daily_limit_squared'] = df_optimized['daily_limit'].fillna(0) ** 2
        
        # Performance tier (based on quantiles) - handle edge cases
        daily_limit_filled = df_optimized['daily_limit'].fillna(0)
        if daily_limit_filled.nunique() > 1:
            try:
                daily_limit_quantiles = daily_limit_filled.quantile([0.25, 0.5, 0.75])
                # Ensure quantiles are monotonically increasing
                quantiles_list = [0] + [q for q in daily_limit_quantiles.tolist() if q > 0] + [float('inf')]
                if len(quantiles_list) >= 3:  # Need at least 3 bins
                    df_optimized['daily_limit_performance'] = pd.cut(
                        daily_limit_filled,
                        bins=quantiles_list,
                        labels=['Low', 'Medium', 'High', 'Very High'][:len(quantiles_list)-1]
                    ).cat.codes
                else:
                    df_optimized['daily_limit_performance'] = 0
            except:
                df_optimized['daily_limit_performance'] = 0
        else:
            df_optimized['daily_limit_performance'] = 0
        
        # Optimal threshold features
        df_optimized['daily_limit_is_optimal'] = (df_optimized['daily_limit'] >= 200).astype(int)
        df_optimized['daily_limit_is_high'] = (df_optimized['daily_limit'] >= 500).astype(int)
        
        # Quantile-based features - handle edge cases
        if daily_limit_filled.nunique() > 1:
            try:
                df_optimized['daily_limit_quantile'] = pd.qcut(
                    daily_limit_filled, 
                    q=min(5, daily_limit_filled.nunique()), 
                    labels=False, 
                    duplicates='drop'
                )
            except:
                df_optimized['daily_limit_quantile'] = 0
        else:
            df_optimized['daily_limit_quantile'] = 0
    else:
        # Add default values if daily_limit doesn't exist
        df_optimized['daily_limit_log'] = 0
        df_optimized['daily_limit_squared'] = 0
        df_optimized['daily_limit_performance'] = 0
        df_optimized['daily_limit_is_optimal'] = 0
        df_optimized['daily_limit_is_high'] = 0
        df_optimized['daily_limit_quantile'] = 0
    
    # 2. Employee count features (handle missing column)
    if 'organization_employees' in df_optimized.columns:
        # Log transformation
        df_optimized['employees_log'] = np.log1p(df_optimized['organization_employees'].fillna(0))
        
        # Square root transformation
        df_optimized['employees_sqrt'] = np.sqrt(df_optimized['organization_employees'].fillna(0))
        
        # Quantile-based features - handle edge cases
        employees_filled = df_optimized['organization_employees'].fillna(0)
        if employees_filled.nunique() > 1:
            try:
                df_optimized['employees_quantile'] = pd.qcut(
                    employees_filled, 
                    q=min(5, employees_filled.nunique()), 
                    labels=False, 
                    duplicates='drop'
                )
            except:
                df_optimized['employees_quantile'] = 0
        else:
            df_optimized['employees_quantile'] = 0
        
        # Company size categories
        df_optimized['is_enterprise'] = (df_optimized['organization_employees'] >= 1000).astype(int)
        df_optimized['is_mid_market'] = ((df_optimized['organization_employees'] >= 100) & 
                                        (df_optimized['organization_employees'] < 1000)).astype(int)
        df_optimized['is_smb'] = ((df_optimized['organization_employees'] >= 10) & 
                                  (df_optimized['organization_employees'] < 100)).astype(int)
        df_optimized['is_very_small'] = (df_optimized['organization_employees'] < 10).astype(int)
    else:
        # Add default values if organization_employees doesn't exist
        df_optimized['employees_log'] = 0
        df_optimized['employees_sqrt'] = 0
        df_optimized['employees_quantile'] = 0
        df_optimized['is_enterprise'] = 0
        df_optimized['is_mid_market'] = 0
        df_optimized['is_smb'] = 0
        df_optimized['is_very_small'] = 0
    
    # 3. Geographic frequency features (handle missing columns)
    for col in ['country', 'state', 'city']:
        if col in df_optimized.columns:
            # Frequency encoding
            freq_map = df_optimized[col].value_counts(normalize=True)
            df_optimized[f'{col}_frequency'] = df_optimized[col].map(freq_map).fillna(0)
            
            # Normalized frequency
            df_optimized[f'{col}_frequency_norm'] = (df_optimized[f'{col}_frequency'] - 
                                                   freq_map.min()) / (freq_map.max() - freq_map.min())
        else:
            # Add default values if column doesn't exist
            df_optimized[f'{col}_frequency'] = 0
            df_optimized[f'{col}_frequency_norm'] = 0
    
    # 4. Industry-title interaction (handle missing columns)
    if 'organization_industry' in df_optimized.columns and 'title' in df_optimized.columns:
        # Create interaction feature
        df_optimized['industry_title_frequency'] = (
            df_optimized['organization_industry'].astype(str) + '_' + 
            df_optimized['title'].astype(str)
        ).map(
            (df_optimized['organization_industry'].astype(str) + '_' + 
             df_optimized['title'].astype(str)).value_counts(normalize=True)
        ).fillna(0)
    else:
        df_optimized['industry_title_frequency'] = 0
    
    # 5. Tech leadership indicator (handle missing column)
    if 'title' in df_optimized.columns:
        tech_leadership_keywords = ['cto', 'cio', 'vp engineering', 'head of engineering', 
                                  'director of engineering', 'lead engineer', 'principal engineer']
        df_optimized['is_tech_leadership'] = df_optimized['title'].str.lower().str.contains(
            '|'.join(tech_leadership_keywords), na=False
        ).astype(int)
    else:
        df_optimized['is_tech_leadership'] = 0
    
    # 6. ESP performance features (handle missing column)
    if 'esp_code' in df_optimized.columns:
        # High performance ESP indicator
        high_performance_esps = [1, 2, 3, 4, 5]  # Example high-performance ESP codes
        df_optimized['esp_is_high_performance'] = df_optimized['esp_code'].isin(high_performance_esps).astype(int)
        
        # ESP frequency encoding
        esp_freq = df_optimized['esp_code'].value_counts(normalize=True)
        df_optimized['esp_frequency'] = df_optimized['esp_code'].map(esp_freq).fillna(0)
        
        # ESP performance tier (handle None values)
        if df_optimized['esp_code'].notna().any():
            esp_quantiles = df_optimized['esp_code'].dropna().quantile([0.33, 0.66])
            df_optimized['esp_performance_tier'] = pd.cut(
                df_optimized['esp_code'].fillna(df_optimized['esp_code'].median()),
                bins=[0] + esp_quantiles.tolist() + [float('inf')],
                labels=['Low', 'Medium', 'High']
            ).cat.codes
        else:
            df_optimized['esp_performance_tier'] = 0
    else:
        # Add default values if esp_code doesn't exist
        df_optimized['esp_is_high_performance'] = 0
        df_optimized['esp_frequency'] = 0
        df_optimized['esp_performance_tier'] = 0
    
    # 7. Campaign features (handle missing column)
    if 'campaign_id' in df_optimized.columns:
        # Campaign frequency
        campaign_freq = df_optimized['campaign_id'].value_counts(normalize=True)
        df_optimized['campaign_frequency'] = df_optimized['campaign_id'].map(campaign_freq).fillna(0)
        
        # High volume campaign indicator
        high_volume_threshold = campaign_freq.quantile(0.8)
        df_optimized['is_high_volume_campaign'] = (
            df_optimized['campaign_id'].map(campaign_freq) >= high_volume_threshold
        ).astype(int)
    else:
        # Add default values if campaign_id doesn't exist
        df_optimized['campaign_frequency'] = 0
        df_optimized['is_high_volume_campaign'] = 0
    
    # 8. Page retrieved features (handle missing column)
    if 'page_retrieved' in df_optimized.columns:
        # Log transformation
        df_optimized['page_retrieved_log'] = np.log1p(df_optimized['page_retrieved'].fillna(0))
        
        # Square root transformation
        df_optimized['page_retrieved_sqrt'] = np.sqrt(df_optimized['page_retrieved'].fillna(0))
    else:
        # Add default values if page_retrieved doesn't exist
        df_optimized['page_retrieved_log'] = 0
        df_optimized['page_retrieved_sqrt'] = 0
    
    # 9. Company age features (handle missing column)
    if 'organization_founded_year' in df_optimized.columns:
        current_year = pd.Timestamp.now().year
        df_optimized['company_age'] = current_year - df_optimized['organization_founded_year'].fillna(current_year)
        df_optimized['company_age_log'] = np.log1p(df_optimized['company_age'])
        
        # Company maturity indicators
        df_optimized['is_startup'] = (df_optimized['company_age'] <= 5).astype(int)
        df_optimized['is_established'] = (df_optimized['company_age'] >= 20).astype(int)
    else:
        # Add default values if organization_founded_year doesn't exist
        df_optimized['company_age'] = 0
        df_optimized['company_age_log'] = 0
        df_optimized['is_startup'] = 0
        df_optimized['is_established'] = 0
    
    return df_optimized

def encode_categorical_features(df, max_categories=100):
    """
    Encode categorical features using frequency encoding for high cardinality
    and Label Encoding for low cardinality features.
    """
    print("🔧 Encoding categorical features...")
    
    # Create a copy to avoid modifying original
    df_encoded = df.copy()
    label_encoders = {}
    
    # Get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Handle NaN values properly for categorical data
            if df_encoded[col].dtype == 'category':
                # For categorical data, add 'Unknown' to categories first
                df_encoded[col] = df_encoded[col].cat.add_categories(['Unknown'])
                df_encoded[col] = df_encoded[col].fillna('Unknown')
            else:
                # For object data, fill NaN with 'Unknown'
                df_encoded[col] = df_encoded[col].fillna('Unknown')
            
            unique_count = df_encoded[col].nunique()
            
            if unique_count > max_categories:
                # Frequency encoding for high cardinality
                freq_encoding = df_encoded[col].value_counts(normalize=True)
                df_encoded[f'{col}_frequency'] = df_encoded[col].map(freq_encoding)
                df_encoded[f'{col}_frequency_norm'] = (df_encoded[col].map(freq_encoding) - freq_encoding.min()) / (freq_encoding.max() - freq_encoding.min())
                print(f"  Frequency encoded {col} ({unique_count} unique values)")
            else:
                # Label encoding for low cardinality
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                label_encoders[col] = le
                print(f"  Label encoded {col} ({unique_count} unique values)")
    
    return df_encoded, label_encoders

def prepare_features_for_model(df, target_variable='opened', cols_to_drop=None):
    """
    Prepare final feature set for model training.
    
    Args:
        df (pd.DataFrame): Input dataframe with all features
        target_variable (str): Name of target variable
        cols_to_drop (list): Additional columns to drop
        
    Returns:
        tuple: (X, y, selected_features)
    """
    print("🔧 Preparing features for model...")
    
    # Default columns to drop
    default_cols_to_drop = [
        'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
        'website', 'headline', 'company_domain', 'phone', 'apollo_id',
        'apollo_name', 'organization', 'photo_url', 'organization_name',
        'organization_website', 'organization_phone', 'combined_text'  # Drop raw text
    ]
    
    if cols_to_drop:
        default_cols_to_drop.extend(cols_to_drop)
    
    # Create feature set
    X = df.drop(columns=[col for col in default_cols_to_drop if col in df.columns], errors='ignore')
    
    # Remove target variable if present
    if target_variable in X.columns:
        X = X.drop(columns=[target_variable])
    
    # Select only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Get target variable
    y = df[target_variable] if target_variable in df.columns else None
    
    print(f"Final feature set shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y, list(X.columns) 