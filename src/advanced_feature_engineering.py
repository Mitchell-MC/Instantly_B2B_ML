"""
Advanced Feature Engineering for Email Engagement Prediction
Enhanced features to improve model accuracy beyond 67.8%
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def parse_jsonb_features(df):
    """
    Advanced JSONB feature engineering with detailed parsing.
    """
    print("🔧 Advanced JSONB feature engineering...")
    
    jsonb_columns = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
    
    for col in jsonb_columns:
        if col in df.columns:
            # Basic presence indicator
            df[f'has_{col}'] = df[col].notna().astype(int)
            
            # JSON complexity features
            df[f'{col}_length'] = df[col].astype(str).str.len()
            df[f'{col}_word_count'] = df[col].astype(str).str.split().str.len()
            
            # Try to parse JSON and extract features
            try:
                # Employment history features
                if col == 'employment_history':
                    df[f'{col}_company_count'] = df[col].apply(
                        lambda x: len(json.loads(x)) if pd.notna(x) and x != 'null' else 0
                    )
                    df[f'{col}_total_duration'] = df[col].apply(
                        lambda x: sum(job.get('duration', 0) for job in json.loads(x)) 
                        if pd.notna(x) and x != 'null' else 0
                    )
                    df[f'{col}_avg_duration'] = df[f'{col}_total_duration'] / df[f'{col}_company_count'].replace(0, 1)
                
                # Organization data features
                elif col == 'organization_data':
                    df[f'{col}_has_website'] = df[col].apply(
                        lambda x: 1 if pd.notna(x) and 'website' in json.loads(x) else 0
                    )
                    df[f'{col}_has_phone'] = df[col].apply(
                        lambda x: 1 if pd.notna(x) and 'phone' in json.loads(x) else 0
                    )
                    df[f'{col}_has_address'] = df[col].apply(
                        lambda x: 1 if pd.notna(x) and 'address' in json.loads(x) else 0
                    )
                
                # Account data features
                elif col == 'account_data':
                    df[f'{col}_account_age'] = df[col].apply(
                        lambda x: json.loads(x).get('account_age', 0) if pd.notna(x) and x != 'null' else 0
                    )
                    df[f'{col}_is_verified'] = df[col].apply(
                        lambda x: 1 if pd.notna(x) and json.loads(x).get('verified', False) else 0
                    )
                
                # API response features
                elif col == 'api_response_raw':
                    df[f'{col}_response_time'] = df[col].apply(
                        lambda x: json.loads(x).get('response_time', 0) if pd.notna(x) and x != 'null' else 0
                    )
                    df[f'{col}_status_code'] = df[col].apply(
                        lambda x: json.loads(x).get('status_code', 0) if pd.notna(x) and x != 'null' else 0
                    )
                    df[f'{col}_is_success'] = (df[f'{col}_status_code'] == 200).astype(int)
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not parse {col} JSON: {e}")
    
    return df

def create_advanced_temporal_features(df):
    """
    Advanced temporal feature engineering with business logic.
    """
    print("🔧 Advanced temporal feature engineering...")
    
    if 'timestamp_created' in df.columns:
        # Convert to datetime
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'])
        
        # Business-relevant temporal features
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_quarter'] = df['timestamp_created'].dt.quarter
        df['created_hour'] = df['timestamp_created'].dt.hour
        
        # Advanced business features
        df['is_monday'] = (df['created_day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['created_day_of_week'] == 4).astype(int)
        df['is_weekend'] = (df['created_day_of_week'] >= 5).astype(int)
        df['is_business_hours'] = ((df['created_hour'] >= 9) & (df['created_hour'] <= 17)).astype(int)
        df['is_morning'] = (df['created_hour'] <= 12).astype(int)
        df['is_afternoon'] = ((df['created_hour'] > 12) & (df['created_hour'] <= 17)).astype(int)
        df['is_evening'] = (df['created_hour'] > 17).astype(int)
        
        # Seasonal features
        df['is_q1'] = (df['created_quarter'] == 1).astype(int)
        df['is_q4'] = (df['created_quarter'] == 4).astype(int)
        
        # Recency features
        current_time = pd.Timestamp.now()
        df['days_since_created'] = (current_time - df['timestamp_created']).dt.days
        df['weeks_since_created'] = df['days_since_created'] // 7
        df['months_since_created'] = df['days_since_created'] // 30
        
        # Recency buckets
        df['is_recent'] = (df['days_since_created'] <= 7).astype(int)
        df['is_old'] = (df['days_since_created'] > 90).astype(int)
    
    return df

def create_text_embedding_features(df):
    """
    Advanced text feature engineering with embeddings.
    """
    print("🔧 Advanced text feature engineering...")
    
    # Combine all text fields efficiently
    text_columns = ['email_subjects', 'email_bodies', 'title', 'organization_industry']
    df['combined_text_advanced'] = ""
    
    for col in text_columns:
        if col in df.columns:
            df['combined_text_advanced'] += df[col].fillna('').astype(str) + ' '
    
    # Advanced text features (more efficient)
    df['text_length'] = df['combined_text_advanced'].str.len()
    df['text_word_count'] = df['combined_text_advanced'].str.split().str.len()
    
    # More efficient sentence counting
    df['text_sentence_count'] = df['combined_text_advanced'].str.count(r'[.!?]') + 1
    
    # Text quality indicators
    df['has_numbers'] = df['combined_text_advanced'].str.contains(r'\d', na=False).astype(int)
    df['has_emails'] = df['combined_text_advanced'].str.contains(r'@', na=False).astype(int)
    df['has_urls'] = df['combined_text_advanced'].str.contains(r'http', na=False).astype(int)
    df['has_phone'] = df['combined_text_advanced'].str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}', na=False).astype(int)
    
    # Business-specific keywords (more efficient)
    business_keywords = ['ceo', 'cto', 'vp', 'director', 'manager', 'founder', 'co-founder']
    for keyword in business_keywords:
        df[f'has_{keyword}'] = df['combined_text_advanced'].str.lower().str.contains(keyword, na=False).astype(int)
    
    # Text complexity score
    df['text_complexity'] = (
        df['text_word_count'] * 0.3 +
        df['text_sentence_count'] * 0.2 +
        df['has_numbers'] * 0.1 +
        df['has_emails'] * 0.1 +
        df['has_urls'] * 0.1 +
        df['has_phone'] * 0.1
    )
    
    return df

def create_interaction_features_advanced(df):
    """
    Advanced interaction features with business logic.
    """
    print("🔧 Advanced interaction features...")
    
    # Industry-seniority interactions
    if 'organization_industry' in df.columns and 'title' in df.columns:
        # High-value industry-title combinations
        high_value_combinations = [
            ('technology', 'ceo'), ('technology', 'cto'), ('technology', 'vp'),
            ('finance', 'ceo'), ('finance', 'director'), ('finance', 'manager'),
            ('healthcare', 'director'), ('healthcare', 'manager'),
            ('education', 'director'), ('education', 'manager')
        ]
        
        for industry, title in high_value_combinations:
            mask = (
                df['organization_industry'].str.lower().str.contains(industry, na=False) &
                df['title'].str.lower().str.contains(title, na=False)
            )
            df[f'is_{industry}_{title}'] = mask.astype(int)
    
    # Geographic-industry interactions
    if 'country' in df.columns and 'organization_industry' in df.columns:
        # US tech companies
        df['is_us_tech'] = (
            (df['country'] == 'United States') &
            (df['organization_industry'].str.lower().str.contains('technology|software|tech', na=False))
        ).astype(int)
        
        # European finance companies
        df['is_eu_finance'] = (
            (df['country'].isin(['United Kingdom', 'Germany', 'France', 'Netherlands'])) &
            (df['organization_industry'].str.lower().str.contains('finance|banking|financial', na=False))
        ).astype(int)
    
    return df

def create_engagement_pattern_features(df):
    """
    Create features based on engagement patterns and sequences.
    """
    print("🔧 Engagement pattern features...")
    
    # Engagement intensity features
    if 'email_open_count' in df.columns:
        df['open_intensity'] = df['email_open_count'] / df['email_open_count'].max()
        df['is_high_engager'] = (df['email_open_count'] >= 3).astype(int)
        df['is_medium_engager'] = ((df['email_open_count'] >= 1) & (df['email_open_count'] <= 2)).astype(int)
        df['is_low_engager'] = (df['email_open_count'] == 0).astype(int)
    
    # Click-through rate (if email_sent_count exists)
    if 'email_click_count' in df.columns and 'email_open_count' in df.columns:
        df['ctr'] = df['email_click_count'] / (df['email_open_count'] + 1)  # Add 1 to avoid division by zero
        df['has_clicks'] = (df['email_click_count'] > 0).astype(int)
    
    # Reply rate
    if 'email_reply_count' in df.columns and 'email_open_count' in df.columns:
        df['reply_rate'] = df['email_reply_count'] / (df['email_open_count'] + 1)
        df['has_replies'] = (df['email_reply_count'] > 0).astype(int)
    
    # Engagement sequence features
    if all(col in df.columns for col in ['email_open_count', 'email_click_count', 'email_reply_count']):
        # Engagement funnel: Open -> Click -> Reply
        df['engagement_funnel_stage'] = 0
        df.loc[df['email_open_count'] > 0, 'engagement_funnel_stage'] = 1
        df.loc[df['email_click_count'] > 0, 'engagement_funnel_stage'] = 2
        df.loc[df['email_reply_count'] > 0, 'engagement_funnel_stage'] = 3
        
        # Multi-channel engagement
        df['multi_channel_engagement'] = (
            (df['email_open_count'] > 0).astype(int) +
            (df['email_click_count'] > 0).astype(int) +
            (df['email_reply_count'] > 0).astype(int)
        )
    
    return df

def create_company_maturity_features(df):
    """
    Advanced company maturity and lifecycle features.
    """
    print("🔧 Company maturity features...")
    
    if 'organization_founded_year' in df.columns:
        current_year = datetime.now().year
        df['company_age'] = current_year - df['organization_founded_year'].fillna(current_year)
        
        # Company lifecycle stages
        df['is_startup'] = (df['company_age'] <= 3).astype(int)
        df['is_growth'] = ((df['company_age'] > 3) & (df['company_age'] <= 10)).astype(int)
        df['is_mature'] = (df['company_age'] > 10).astype(int)
        
        # Company age buckets
        df['company_age_bucket'] = pd.cut(
            df['company_age'],
            bins=[0, 2, 5, 10, 20, 100],
            labels=['Very Young', 'Young', 'Growth', 'Mature', 'Established']
        ).cat.codes
    
    if 'organization_employees' in df.columns:
        # Company size maturity
        df['employee_density'] = df['organization_employees'] / df['organization_employees'].max()
        
        # Size-age interaction
        if 'company_age' in df.columns:
            df['size_age_ratio'] = df['organization_employees'] / (df['company_age'] + 1)
    
    return df

def create_advanced_geographic_features(df):
    """
    Advanced geographic and market features.
    """
    print("🔧 Advanced geographic features...")
    
    # Market tier classification
    tier1_markets = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia']
    tier2_markets = ['Netherlands', 'Sweden', 'Switzerland', 'Norway', 'Denmark', 'Singapore']
    
    df['market_tier'] = 3  # Default to tier 3
    df.loc[df['country'].isin(tier1_markets), 'market_tier'] = 1
    df.loc[df['country'].isin(tier2_markets), 'market_tier'] = 2
    
    # Geographic clustering
    if 'state' in df.columns and 'country' in df.columns:
        # US state clustering
        us_tech_states = ['California', 'New York', 'Texas', 'Washington', 'Massachusetts']
        df['is_us_tech_state'] = (
            (df['country'] == 'United States') & 
            (df['state'].isin(us_tech_states))
        ).astype(int)
    
    return df

def apply_advanced_feature_engineering(df):
    """
    Apply all advanced feature engineering techniques.
    """
    print("🚀 Applying advanced feature engineering...")
    
    # Apply all feature engineering functions
    df = parse_jsonb_features(df)
    df = create_advanced_temporal_features(df)
    df = create_text_embedding_features(df)
    df = create_interaction_features_advanced(df)
    df = create_engagement_pattern_features(df)
    df = create_company_maturity_features(df)
    df = create_advanced_geographic_features(df)
    
    print(f"✅ Advanced feature engineering complete. Shape: {df.shape}")
    return df 