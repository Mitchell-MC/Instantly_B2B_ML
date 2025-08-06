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
    Advanced XGBoost-specific feature engineering based on data analysis insights.
    
    Args:
        df (pd.DataFrame): Input dataframe with raw features
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    print("🔧 Creating XGBoost-optimized features...")
    
    # 1. DAILY LIMIT ENGINEERING (Top predictor from daily limit analysis)
    if 'daily_limit' in df.columns:
        # Performance-based encoding from daily limit analysis
        def daily_limit_performance_score(limit):
            if pd.isna(limit):
                return 0
            elif limit <= 100:
                return 5  # Best performance (3.958 open rate from analysis)
            elif limit <= 250:
                return 2  # Lower performance
            elif limit <= 350:
                return 3  # Medium performance  
            elif limit <= 450:
                return 2  # Medium-low performance
            elif limit <= 650:
                return 4  # Good performance
            else:
                return 4  # High performance (2.723 open rate)
        
        df['daily_limit_performance'] = df['daily_limit'].apply(daily_limit_performance_score)
        df['daily_limit_log'] = np.log1p(df['daily_limit'].fillna(0))
        df['daily_limit_squared'] = df['daily_limit'].fillna(0) ** 2
        df['daily_limit_is_optimal'] = ((df['daily_limit'] >= 100) & (df['daily_limit'] <= 150)).astype(int)
        df['daily_limit_is_high'] = (df['daily_limit'] >= 500).astype(int)
        df['daily_limit_quantile'] = pd.qcut(df['daily_limit'].fillna(-1), 
                                           q=10, labels=False, duplicates='drop')
    
    # 2. COMPANY SIZE OPTIMIZATION (Your #1 feature: organization_employees)
    if 'organization_employees' in df.columns:
        df['employees_log'] = np.log1p(df['organization_employees'].fillna(0))
        df['employees_sqrt'] = np.sqrt(df['organization_employees'].fillna(0))
        df['employees_quantile'] = pd.qcut(df['organization_employees'].fillna(-1), 
                                         q=20, labels=False, duplicates='drop')
        
        # Size category features
        df['is_enterprise'] = (df['organization_employees'] >= 1000).astype(int)
        df['is_mid_market'] = ((df['organization_employees'] >= 200) & 
                              (df['organization_employees'] < 1000)).astype(int)
        df['is_smb'] = (df['organization_employees'] < 200).astype(int)
        df['is_very_small'] = (df['organization_employees'] <= 10).astype(int)
    
    # 3. GEOGRAPHIC FEATURES (country, state, city are top features)
    location_cols = ['country', 'state', 'city']
    for col in location_cols:
        if col in df.columns:
            # Frequency encoding (very effective for XGBoost)
            freq_map = df[col].value_counts().to_dict()
            df[f'{col}_frequency'] = df[col].map(freq_map).fillna(0)
            
            # Normalized frequency (0-1 scale)
            max_freq = df[f'{col}_frequency'].max()
            df[f'{col}_frequency_norm'] = df[f'{col}_frequency'] / max_freq if max_freq > 0 else 0
    
    # 4. INDUSTRY-TITLE INTERACTIONS (Top features)
    if 'organization_industry' in df.columns and 'title' in df.columns:
        # Create high-value combinations
        df['industry_title_combo'] = (df['organization_industry'].astype(str) + '_' + 
                                     df['title'].astype(str))
        
        # Frequency encode the combinations
        combo_freq = df['industry_title_combo'].value_counts().to_dict()
        df['industry_title_frequency'] = df['industry_title_combo'].map(combo_freq).fillna(0)
        
        # High-performing industry-title combinations
        tech_titles = ['CEO', 'CTO', 'VP', 'Director', 'Manager', 'Founder']
        df['is_tech_leadership'] = ((df['organization_industry'].str.contains('Technology|Software|Tech', na=False)) &
                                   (df['title'].str.contains('|'.join(tech_titles), na=False))).astype(int)
        
        # Drop the text combo column
        df = df.drop(columns=['industry_title_combo'])
    
    # 5. ESP CODE OPTIMIZATION (Based on ESP analysis)
    if 'esp_code' in df.columns:
        # From ESP analysis: ESP 8.0, 11.0, 2.0 perform best
        high_performance_esp = [8.0, 11.0, 2.0]
        df['esp_is_high_performance'] = df['esp_code'].isin(high_performance_esp).astype(int)
        
        # ESP frequency encoding
        esp_freq = df['esp_code'].value_counts().to_dict()
        df['esp_frequency'] = df['esp_code'].map(esp_freq).fillna(0)
        
        # ESP performance tiers
        df['esp_performance_tier'] = pd.cut(df['esp_code'], 
                                          bins=[0, 5, 10, 15, 20], 
                                          labels=['low', 'medium', 'high', 'very_high'],
                                          include_lowest=True)
    
    # 6. CAMPAIGN FEATURES
    if 'campaign_id' in df.columns:
        # Campaign frequency encoding
        campaign_freq = df['campaign_id'].value_counts().to_dict()
        df['campaign_frequency'] = df['campaign_id'].map(campaign_freq).fillna(0)
        
        # Campaign performance indicators
        df['is_high_volume_campaign'] = (df['campaign_frequency'] >= df['campaign_frequency'].quantile(0.8)).astype(int)
    
    # 7. TECHNICAL FEATURES
    if 'page_retrieved' in df.columns:
        df['page_retrieved_log'] = np.log1p(df['page_retrieved'].fillna(0))
        df['page_retrieved_sqrt'] = np.sqrt(df['page_retrieved'].fillna(0))
    
    # 8. ORGANIZATION FEATURES
    if 'organization_founded_year' in df.columns:
        current_year = pd.Timestamp.now().year
        df['company_age'] = current_year - df['organization_founded_year'].fillna(current_year)
        df['company_age_log'] = np.log1p(df['company_age'])
        
        # Company maturity categories
        df['is_startup'] = (df['company_age'] <= 5).astype(int)
        df['is_established'] = (df['company_age'] > 10).astype(int)
    
    return df

def encode_categorical_features(df, target_variable='opened', max_categories=100):
    """
    Encode categorical features using Label Encoding and frequency encoding.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_variable (str): Name of target variable
        max_categories (int): Maximum categories to keep for high-cardinality features
        
    Returns:
        tuple: (encoded_df, label_encoders_dict)
    """
    print("🔧 Encoding categorical features...")
    
    # Create copy to avoid modifying original
    df_encoded = df.copy()
    label_encoders = {}
    
    # Get categorical columns (excluding target and already encoded features)
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_cols = [col for col in categorical_cols if col != target_variable]
    
    # Remove interaction columns that are already encoded
    interaction_cols = [col for col in categorical_cols if col.endswith('_interaction')]
    categorical_cols = [col for col in categorical_cols if col not in interaction_cols]
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Handle high cardinality
            unique_count = df_encoded[col].nunique()
            
            if unique_count > max_categories:
                # For high cardinality, use frequency encoding
                freq_map = df_encoded[col].value_counts().to_dict()
                df_encoded[f'{col}_freq_encoded'] = df_encoded[col].map(freq_map).fillna(0)
                # Drop original column
                df_encoded = df_encoded.drop(columns=[col])
                print(f"  Frequency encoded {col} ({unique_count} unique values)")
            else:
                # For low cardinality, use Label Encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].fillna('Unknown'))
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