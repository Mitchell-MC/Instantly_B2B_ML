"""
Enhanced Feature Engineering Module for Email Opening Prediction
Production-ready feature engineering that incorporates advanced preprocessing techniques.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
import json # Added for JSONB parsing

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
            # Create bins and handle duplicates
            bins = [0] + esp_quantiles.tolist() + [float('inf')]
            # Remove duplicates from bins
            unique_bins = list(dict.fromkeys(bins))  # Preserve order while removing duplicates
            
            # Adjust labels based on number of bins
            if len(unique_bins) == 2:  # Only 0 and inf
                df_optimized['esp_performance_tier'] = 0
            elif len(unique_bins) == 3:  # 0, quantile, inf
                df_optimized['esp_performance_tier'] = pd.cut(
                    df_optimized['esp_code'].fillna(df_optimized['esp_code'].median()),
                    bins=unique_bins,
                    labels=['Low', 'High'],
                    duplicates='drop'
                ).cat.codes
            else:  # 0, quantile1, quantile2, inf
                df_optimized['esp_performance_tier'] = pd.cut(
                    df_optimized['esp_code'].fillna(df_optimized['esp_code'].median()),
                    bins=unique_bins,
                    labels=['Low', 'Medium', 'High'][:len(unique_bins)-1],
                    duplicates='drop'
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

def create_comprehensive_organization_data(df):
    """
    Create comprehensive organization data by combining all organization-related fields.
    """
    print("🏢 Creating comprehensive organization data...")
    
    df_org = df.copy()
    
    # Initialize comprehensive organization data
    df_org['organization_comprehensive'] = ""
    
    # Combine all organization-related fields
    org_fields = [
        'organization_x', 'organization_y', 'organization_name', 'company_name',
        'organization_industry', 'organization_website', 'organization_phone',
        'company_domain', 'organization_data'
    ]
    
    for field in org_fields:
        if field in df_org.columns:
            # Add field value if not null
            mask = df_org[field].notna() & (df_org[field] != '')
            df_org.loc[mask, 'organization_comprehensive'] += df_org.loc[mask, field].astype(str) + ' | '
    
    # Add employee count information
    if 'organization_employees' in df_org.columns:
        mask = df_org['organization_employees'].notna()
        df_org.loc[mask, 'organization_comprehensive'] += 'Employees: ' + df_org.loc[mask, 'organization_employees'].astype(str) + ' | '
    
    # Add founded year information
    if 'organization_founded_year' in df_org.columns:
        mask = df_org['organization_founded_year'].notna()
        df_org.loc[mask, 'organization_comprehensive'] += 'Founded: ' + df_org.loc[mask, 'organization_founded_year'].astype(str) + ' | '
    
    # Clean up the combined data
    df_org['organization_comprehensive'] = df_org['organization_comprehensive'].str.rstrip(' | ')
    
    # Create organization data quality indicators
    df_org['org_data_completeness'] = 0
    org_quality_fields = [
        'organization_name', 'organization_industry', 'organization_employees',
        'organization_founded_year', 'organization_website', 'organization_phone'
    ]
    
    for field in org_quality_fields:
        if field in df_org.columns:
            df_org['org_data_completeness'] += df_org[field].notna().astype(int)
    
    # Normalize completeness score (0-6 scale)
    df_org['org_data_completeness_pct'] = df_org['org_data_completeness'] / len(org_quality_fields)
    
    # Create organization maturity indicators
    if 'organization_employees' in df_org.columns and 'organization_founded_year' in df_org.columns:
        current_year = pd.Timestamp.now().year
        df_org['company_age'] = current_year - df_org['organization_founded_year'].fillna(current_year)
        
        # Organization maturity score (combination of size and age)
        df_org['org_maturity_score'] = (
            (df_org['organization_employees'].fillna(0) / 1000) * 0.5 +  # Size component
            (df_org['company_age'] / 50) * 0.5  # Age component
        )
        
        # Maturity categories
        df_org['org_maturity_category'] = pd.cut(
            df_org['org_maturity_score'],
            bins=[0, 0.2, 0.5, 1.0, float('inf')],
            labels=['Startup', 'Growth', 'Established', 'Enterprise'],
            include_lowest=True
        ).cat.codes
    
    # Create organization data length features
    df_org['org_data_length'] = df_org['organization_comprehensive'].str.len()
    df_org['org_data_word_count'] = df_org['organization_comprehensive'].str.split().str.len()
    
    # Organization data presence indicators
    df_org['has_org_name'] = df_org['organization_name'].notna().astype(int) if 'organization_name' in df_org.columns else 0
    df_org['has_org_industry'] = df_org['organization_industry'].notna().astype(int) if 'organization_industry' in df_org.columns else 0
    df_org['has_org_employees'] = df_org['organization_employees'].notna().astype(int) if 'organization_employees' in df_org.columns else 0
    df_org['has_org_website'] = df_org['organization_website'].notna().astype(int) if 'organization_website' in df_org.columns else 0
    df_org['has_org_phone'] = df_org['organization_phone'].notna().astype(int) if 'organization_phone' in df_org.columns else 0
    df_org['has_org_domain'] = df_org['company_domain'].notna().astype(int) if 'company_domain' in df_org.columns else 0
    
    # Parse JSONB organization_data if available
    if 'organization_data' in df_org.columns:
        try:
            # Basic JSONB parsing
            df_org['has_org_json_data'] = df_org['organization_data'].notna().astype(int)
            df_org['org_json_length'] = df_org['organization_data'].astype(str).str.len()
            
            # Parse once per row, then derive fields to avoid repeated json.loads
            def _safe_parse(obj):
                try:
                    if pd.notna(obj) and obj != 'null':
                        return json.loads(obj) if isinstance(obj, str) else obj
                except Exception:
                    return {}
                return {}

            _parsed = df_org['organization_data'].apply(_safe_parse)
            df_org['org_json_has_website'] = _parsed.apply(
                lambda d: 1 if isinstance(d, dict) and d.get('website') else 0
            )
            df_org['org_json_has_phone'] = _parsed.apply(
                lambda d: 1 if isinstance(d, dict) and d.get('phone') else 0
            )
            df_org['org_json_has_address'] = _parsed.apply(
                lambda d: 1 if isinstance(d, dict) and d.get('address') else 0
            )
            
        except Exception as e:
            print(f"⚠️ Warning: Could not parse organization_data JSON: {e}")
            df_org['has_org_json_data'] = 0
            df_org['org_json_length'] = 0
            df_org['org_json_has_website'] = 0
            df_org['org_json_has_phone'] = 0
            df_org['org_json_has_address'] = 0
    
    print(f"✅ Organization data created. Shape: {df_org.shape}")
    print(f"📊 Organization data completeness: {df_org['org_data_completeness'].mean():.2f}/6 fields")
    
    return df_org

def create_advanced_engagement_features(df):
    """
    Create advanced engagement pattern and behavioral features (NO LEAKAGE).
    """
    print("🎯 Creating advanced engagement features (no leakage)...")
    
    df_eng = df.copy()
    
    # 1. Temporal engagement patterns (safe - no target variables)
    if 'timestamp_created' in df_eng.columns:
        # Time-based engagement features
        df_eng['created_hour'] = df_eng['timestamp_created'].dt.hour
        df_eng['created_day_of_week'] = df_eng['timestamp_created'].dt.dayofweek
        df_eng['created_month'] = df_eng['timestamp_created'].dt.month
        
        # Business hours engagement
        df_eng['business_hours_created'] = (
            (df_eng['created_hour'] >= 9) & (df_eng['created_hour'] <= 17) &
            (df_eng['created_day_of_week'] < 5)
        ).astype(int)
        
        # Weekend engagement
        df_eng['weekend_created'] = (df_eng['created_day_of_week'] >= 5).astype(int)
    
    # 2. Company engagement context (safe - no target variables)
    if 'organization_employees' in df_eng.columns:
        # Company size engagement potential
        df_eng['company_engagement_potential'] = df_eng['organization_employees'] / 1000
        
        # Company maturity engagement potential
        if 'organization_founded_year' in df_eng.columns:
            current_year = pd.Timestamp.now().year
            df_eng['company_age'] = current_year - df_eng['organization_founded_year'].fillna(current_year)
            df_eng['company_maturity_engagement'] = df_eng['company_age'] / 50
    
    # 3. Advanced text engagement signals (safe - no target variables)
    if 'combined_text_length' in df_eng.columns:
        # Text complexity score
        df_eng['text_complexity_score'] = df_eng['combined_text_length'] / 1000
        
        # Text quality engagement potential
        df_eng['text_quality_engagement'] = df_eng['text_quality_score'] / 10
    
    # 4. Campaign engagement context (safe - no target variables)
    if 'campaign_frequency' in df_eng.columns:
        # Campaign engagement potential
        df_eng['campaign_engagement_potential'] = df_eng['campaign_frequency'] / 100
        
        # High-volume campaign indicator
        df_eng['high_volume_campaign_potential'] = (
            df_eng['campaign_frequency'] > df_eng['campaign_frequency'].quantile(0.75)
        ).astype(int)
    
    # 5. Geographic engagement patterns (safe - no target variables)
    if 'country_frequency' in df_eng.columns:
        # Geographic engagement potential
        df_eng['geo_engagement_potential'] = df_eng['country_frequency'] / 100
    
    # 6. Industry engagement patterns (safe - no target variables)
    if 'industry_title_frequency' in df_eng.columns:
        # Industry engagement potential
        df_eng['industry_engagement_potential'] = df_eng['industry_title_frequency'] / 100
    
    # 7. ESP engagement patterns (safe - no target variables)
    if 'esp_frequency' in df_eng.columns:
        # ESP engagement potential
        df_eng['esp_engagement_potential'] = df_eng['esp_frequency'] / 100
    
    # 8. Daily limit engagement patterns (safe - no target variables)
    if 'daily_limit' in df_eng.columns:
        # Daily limit engagement potential
        df_eng['daily_limit_engagement_potential'] = df_eng['daily_limit'] / 1000
    
    # 9. Organization data engagement patterns (safe - no target variables)
    if 'org_data_completeness' in df_eng.columns:
        # Organization data engagement potential
        df_eng['org_data_engagement_potential'] = df_eng['org_data_completeness'] / 6
    
    # 10. Text features engagement patterns (safe - no target variables)
    if 'combined_text_word_count' in df_eng.columns:
        # Text richness engagement potential
        df_eng['text_richness_engagement'] = df_eng['combined_text_word_count'] / 100
    
    print(f"✅ Advanced engagement features created (no leakage). Shape: {df_eng.shape}")
    
    return df_eng

def encode_categorical_features(df, max_categories=100):
    """
    Encode categorical features with frequency encoding for high cardinality
    and Label Encoding for low cardinality, handling unhashable types.
    """
    print("🔧 Encoding categorical features...")
    
    df_encoded = df.copy()
    
    # Get categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            # Skip columns that might contain unhashable types (like lists from JSONB)
            try:
                # Check if column contains unhashable types
                sample_values = df_encoded[col].dropna().head(100)
                if any(isinstance(val, (list, dict, set)) for val in sample_values):
                    print(f"⚠️ Skipping {col} - contains unhashable types (likely JSONB data)")
                    continue
                
                # Get unique values
                unique_values = df_encoded[col].dropna().unique()
                
                if len(unique_values) <= max_categories:
                    # Use Label Encoding for low cardinality
                    le = LabelEncoder()
                    df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].fillna('Unknown'))
                    print(f"  ✅ Label encoded {col} ({len(unique_values)} categories)")
                else:
                    # Use Frequency Encoding for high cardinality
                    freq_encoding = df_encoded[col].value_counts(normalize=True)
                    df_encoded[f'{col}_freq_encoded'] = df_encoded[col].map(freq_encoding).fillna(0)
                    print(f"  ✅ Frequency encoded {col} ({len(unique_values)} categories)")
                    
            except (TypeError, ValueError) as e:
                print(f"⚠️ Skipping {col} - encoding failed: {e}")
                continue
    
    print(f"✅ Categorical encoding complete. Shape: {df_encoded.shape}")
    return df_encoded

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

def apply_enhanced_feature_engineering(df):
    """
    Apply all enhanced feature engineering techniques.
    """
    print("🚀 Applying enhanced feature engineering...")
    
    # Apply all feature engineering functions
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_jsonb_features(df)
    df = create_comprehensive_organization_data(df)  # Add comprehensive organization data
    df = create_xgboost_optimized_features(df)
    df = handle_outliers(df)
    
    print(f"✅ Enhanced feature engineering complete. Shape: {df.shape}")
    return df 

def create_comprehensive_jsonb_features(df):
    """
    Create comprehensive JSONB features based on actual database structure analysis.
    """
    print("🔧 Creating comprehensive JSONB features...")
    
    df_jsonb = df.copy()
    
    # 1. EMPLOYMENT_HISTORY ANALYSIS (Array of job objects)
    if 'employment_history' in df_jsonb.columns:
        print("  📊 Processing employment_history...")
        
        # Basic presence and complexity
        df_jsonb['has_employment_history'] = df_jsonb['employment_history'].notna().astype(int)
        df_jsonb['employment_history_length'] = df_jsonb['employment_history'].astype(str).str.len()
        
        # Employment history features
        def extract_employment_features(json_str):
            try:
                if pd.notna(json_str) and json_str != 'null':
                    jobs = json.loads(json_str)
                    if isinstance(jobs, list) and jobs:
                        # Job count
                        job_count = len(jobs)
                        
                        # Current job analysis
                        current_jobs = [job for job in jobs if job.get('current', False)]
                        current_job_count = len(current_jobs)
                        
                        # Title analysis
                        titles = [job.get('title', '') for job in jobs if job.get('title')]
                        senior_titles = ['ceo', 'cto', 'cfo', 'coo', 'vp', 'director', 'head', 'chief', 'president', 'founder']
                        senior_count = sum(1 for title in titles if any(senior in title.lower() for senior in senior_titles))
                        
                        # Duration analysis
                        durations = []
                        for job in jobs:
                            start_date = job.get('start_date')
                            end_date = job.get('end_date')
                            if start_date and end_date:
                                try:
                                    start = pd.to_datetime(start_date)
                                    end = pd.to_datetime(end_date)
                                    duration = (end - start).days
                                    durations.append(duration)
                                except:
                                    continue
                        
                        avg_duration = np.mean(durations) if durations else 0
                        max_duration = max(durations) if durations else 0
                        
                        return {
                            'job_count': job_count,
                            'current_job_count': current_job_count,
                            'senior_title_count': senior_count,
                            'avg_job_duration': avg_duration,
                            'max_job_duration': max_duration,
                            'total_experience_years': sum(durations) / 365 if durations else 0
                        }
                return {
                    'job_count': 0, 'current_job_count': 0, 'senior_title_count': 0,
                    'avg_job_duration': 0, 'max_job_duration': 0, 'total_experience_years': 0
                }
            except:
                return {
                    'job_count': 0, 'current_job_count': 0, 'senior_title_count': 0,
                    'avg_job_duration': 0, 'max_job_duration': 0, 'total_experience_years': 0
                }
        
        # Apply employment feature extraction once, then expand efficiently
        employment_features = df_jsonb['employment_history'].apply(extract_employment_features)
        employment_df = pd.DataFrame(list(employment_features))
        for key in ['job_count', 'current_job_count', 'senior_title_count', 'avg_job_duration', 'max_job_duration', 'total_experience_years']:
            if key in employment_df.columns:
                df_jsonb[f'employment_{key}'] = employment_df[key].fillna(0)
    
    # 2. ORGANIZATION_DATA ANALYSIS (Company information object)
    if 'organization_data' in df_jsonb.columns:
        print("  📊 Processing organization_data...")
        
        # Basic presence and complexity
        df_jsonb['has_organization_data'] = df_jsonb['organization_data'].notna().astype(int)
        df_jsonb['organization_data_length'] = df_jsonb['organization_data'].astype(str).str.len()
        
        # Organization features
        def extract_organization_features(json_str):
            try:
                if pd.notna(json_str) and json_str != 'null':
                    org = json.loads(json_str)
                    if isinstance(org, dict):
                        return {
                            'org_has_industry': 1 if org.get('industry') else 0,
                            'org_has_keywords': 1 if org.get('keywords') else 0,
                            'org_has_linkedin': 1 if org.get('linkedin_url') else 0,
                            'org_has_phone': 1 if org.get('phone') else 0,
                            'org_has_alexa_ranking': 1 if org.get('alexa_ranking') else 0,
                            'org_is_publicly_traded': 1 if org.get('publicly_traded_exchange') else 0,
                            'org_has_retail_locations': 1 if org.get('retail_location_count') else 0,
                            'org_has_secondary_industries': 1 if org.get('secondary_industries') else 0,
                            'org_keyword_count': len(org.get('keywords', [])),
                            'org_secondary_industry_count': len(org.get('secondary_industries', []))
                        }
                return {
                    'org_has_industry': 0, 'org_has_keywords': 0, 'org_has_linkedin': 0,
                    'org_has_phone': 0, 'org_has_alexa_ranking': 0, 'org_is_publicly_traded': 0,
                    'org_has_retail_locations': 0, 'org_has_secondary_industries': 0,
                    'org_keyword_count': 0, 'org_secondary_industry_count': 0
                }
            except:
                return {
                    'org_has_industry': 0, 'org_has_keywords': 0, 'org_has_linkedin': 0,
                    'org_has_phone': 0, 'org_has_alexa_ranking': 0, 'org_is_publicly_traded': 0,
                    'org_has_retail_locations': 0, 'org_has_secondary_industries': 0,
                    'org_keyword_count': 0, 'org_secondary_industry_count': 0
                }
        
        # Apply organization feature extraction once, then expand efficiently
        org_features = df_jsonb['organization_data'].apply(extract_organization_features)
        org_df = pd.DataFrame(list(org_features))
        for key in ['org_has_industry', 'org_has_keywords', 'org_has_linkedin', 'org_has_phone', 
                   'org_has_alexa_ranking', 'org_is_publicly_traded', 'org_has_retail_locations',
                   'org_has_secondary_industries', 'org_keyword_count', 'org_secondary_industry_count']:
            if key in org_df.columns:
                df_jsonb[key] = org_df[key].fillna(0)
    
    # 3. ACCOUNT_DATA ANALYSIS (Account information object)
    if 'account_data' in df_jsonb.columns:
        print("  📊 Processing account_data...")
        
        # Basic presence and complexity
        df_jsonb['has_account_data'] = df_jsonb['account_data'].notna().astype(int)
        df_jsonb['account_data_length'] = df_jsonb['account_data'].astype(str).str.len()
        
        # Account features
        def extract_account_features(json_str):
            try:
                if pd.notna(json_str) and json_str != 'null':
                    account = json.loads(json_str)
                    if isinstance(account, dict):
                        return {
                            'account_has_domain': 1 if account.get('domain') else 0,
                            'account_has_source': 1 if account.get('source') else 0,
                            'account_has_team_id': 1 if account.get('team_id') else 0,
                            'account_has_owner_id': 1 if account.get('owner_id') else 0,
                            'account_has_linkedin': 1 if account.get('linkedin_url') else 0,
                            'account_has_alexa_ranking': 1 if account.get('alexa_ranking') else 0,
                            'account_is_publicly_traded': 1 if account.get('publicly_traded_exchange') else 0,
                            'account_has_custom_fields': 1 if account.get('typed_custom_fields') else 0,
                            'account_existence_level': account.get('existence_level', 0),
                            'account_custom_field_count': len(account.get('typed_custom_fields', {}))
                        }
                return {
                    'account_has_domain': 0, 'account_has_source': 0, 'account_has_team_id': 0,
                    'account_has_owner_id': 0, 'account_has_linkedin': 0, 'account_has_alexa_ranking': 0,
                    'account_is_publicly_traded': 0, 'account_has_custom_fields': 0,
                    'account_existence_level': 0, 'account_custom_field_count': 0
                }
            except:
                return {
                    'account_has_domain': 0, 'account_has_source': 0, 'account_has_team_id': 0,
                    'account_has_owner_id': 0, 'account_has_linkedin': 0, 'account_has_alexa_ranking': 0,
                    'account_is_publicly_traded': 0, 'account_has_custom_fields': 0,
                    'account_existence_level': 0, 'account_custom_field_count': 0
                }
        
        # Apply account feature extraction once, then expand efficiently
        account_features = df_jsonb['account_data'].apply(extract_account_features)
        account_df = pd.DataFrame(list(account_features))
        for key in ['account_has_domain', 'account_has_source', 'account_has_team_id', 'account_has_owner_id',
                   'account_has_linkedin', 'account_has_alexa_ranking', 'account_is_publicly_traded',
                   'account_has_custom_fields', 'account_existence_level', 'account_custom_field_count']:
            if key in account_df.columns:
                df_jsonb[key] = account_df[key].fillna(0)
    
    # 4. API_RESPONSE_RAW ANALYSIS (Contact enrichment data object)
    if 'api_response_raw' in df_jsonb.columns:
        print("  📊 Processing api_response_raw...")
        
        # Basic presence and complexity
        df_jsonb['has_api_response'] = df_jsonb['api_response_raw'].notna().astype(int)
        df_jsonb['api_response_length'] = df_jsonb['api_response_raw'].astype(str).str.len()
        
        # API response features
        def extract_api_features(json_str):
            try:
                if pd.notna(json_str) and json_str != 'null':
                    api_data = json.loads(json_str)
                    if isinstance(api_data, dict):
                        # Contact features
                        contact_features = {
                            'api_has_photo': 1 if api_data.get('photo_url') else 0,
                            'api_has_linkedin': 1 if api_data.get('linkedin_url') else 0,
                            'api_has_email': 1 if api_data.get('email') else 0,
                            'api_has_functions': 1 if api_data.get('functions') else 0,
                            'api_has_departments': 1 if api_data.get('departments') else 0,
                            'api_has_account': 1 if api_data.get('account') else 0,
                            'api_email_status': 1 if api_data.get('email_status') == 'valid' else 0,
                            'api_function_count': len(api_data.get('functions', [])),
                            'api_department_count': len(api_data.get('departments', []))
                        }
                        
                        # Account features from API response
                        account = api_data.get('account', {})
                        if isinstance(account, dict):
                            contact_features.update({
                                'api_account_has_id': 1 if account.get('id') else 0,
                                'api_account_has_name': 1 if account.get('name') else 0,
                                'api_account_has_domain': 1 if account.get('domain') else 0,
                                'api_account_has_industry': 1 if account.get('industry') else 0
                            })
                        
                        return contact_features
                return {
                    'api_has_photo': 0, 'api_has_linkedin': 0, 'api_has_email': 0,
                    'api_has_functions': 0, 'api_has_departments': 0, 'api_has_account': 0,
                    'api_email_status': 0, 'api_function_count': 0, 'api_department_count': 0,
                    'api_account_has_id': 0, 'api_account_has_name': 0, 'api_account_has_domain': 0,
                    'api_account_has_industry': 0
                }
            except:
                return {
                    'api_has_photo': 0, 'api_has_linkedin': 0, 'api_has_email': 0,
                    'api_has_functions': 0, 'api_has_departments': 0, 'api_has_account': 0,
                    'api_email_status': 0, 'api_function_count': 0, 'api_department_count': 0,
                    'api_account_has_id': 0, 'api_account_has_name': 0, 'api_account_has_domain': 0,
                    'api_account_has_industry': 0
                }
        
        # Apply API feature extraction once, then expand efficiently
        api_features = df_jsonb['api_response_raw'].apply(extract_api_features)
        api_df = pd.DataFrame(list(api_features))
        for key in ['api_has_photo', 'api_has_linkedin', 'api_has_email', 'api_has_functions',
                   'api_has_departments', 'api_has_account', 'api_email_status', 'api_function_count',
                   'api_department_count', 'api_account_has_id', 'api_account_has_name',
                   'api_account_has_domain', 'api_account_has_industry']:
            if key in api_df.columns:
                df_jsonb[key] = api_df[key].fillna(0)
    
    # 5. COMPREHENSIVE ENRICHMENT SCORES
    print("  📊 Creating comprehensive enrichment scores...")
    
    # Data completeness scores
    jsonb_cols = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
    presence_cols = [f'has_{col.replace("_", "")}' for col in jsonb_cols]
    available_presence_cols = [col for col in presence_cols if col in df_jsonb.columns]
    
    if available_presence_cols:
        df_jsonb['jsonb_completeness_score'] = df_jsonb[available_presence_cols].sum(axis=1)
        df_jsonb['jsonb_completeness_pct'] = df_jsonb['jsonb_completeness_score'] / len(available_presence_cols)
    
    # Data richness scores
    length_cols = [f'{col.replace("_", "")}_length' for col in jsonb_cols]
    available_length_cols = [col for col in length_cols if col in df_jsonb.columns]
    
    if available_length_cols:
        df_jsonb['jsonb_richness_score'] = df_jsonb[available_length_cols].sum(axis=1)
        df_jsonb['jsonb_richness_normalized'] = (df_jsonb['jsonb_richness_score'] - df_jsonb['jsonb_richness_score'].min()) / (df_jsonb['jsonb_richness_score'].max() - df_jsonb['jsonb_richness_score'].min())
    
    print(f"✅ Comprehensive JSONB features created. Shape: {df_jsonb.shape}")
    return df_jsonb 