"""
Feature Engineering Without Data Leakage
Focuses on predictive features only, avoiding target-related leakage.
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_predictive_features_only(df):
    """
    Create features that predict engagement WITHOUT using engagement data.
    """
    print("🔧 Creating predictive features (no leakage)...")
    
    df_predictive = df.copy()
    
    # 1. Company/Organization Features (predictive)
    if 'organization_employees' in df_predictive.columns:
        df_predictive['employees_log'] = np.log1p(df_predictive['organization_employees'].fillna(0))
        df_predictive['employees_sqrt'] = np.sqrt(df_predictive['organization_employees'].fillna(0))
        
        # Company size categories
        df_predictive['is_enterprise'] = (df_predictive['organization_employees'] >= 1000).astype(int)
        df_predictive['is_mid_market'] = ((df_predictive['organization_employees'] >= 100) & 
                                         (df_predictive['organization_employees'] < 1000)).astype(int)
        df_predictive['is_smb'] = ((df_predictive['organization_employees'] >= 10) & 
                                  (df_predictive['organization_employees'] < 100)).astype(int)
        df_predictive['is_very_small'] = (df_predictive['organization_employees'] < 10).astype(int)
    
    # 2. Company Age Features
    if 'organization_founded_year' in df_predictive.columns:
        current_year = datetime.now().year
        df_predictive['company_age'] = current_year - df_predictive['organization_founded_year'].fillna(current_year)
        df_predictive['is_startup'] = (df_predictive['company_age'] <= 3).astype(int)
        df_predictive['is_growth'] = ((df_predictive['company_age'] > 3) & (df_predictive['company_age'] <= 10)).astype(int)
        df_predictive['is_mature'] = (df_predictive['company_age'] > 10).astype(int)
    
    # 3. Geographic Features
    if 'country' in df_predictive.columns:
        # Market tier classification
        tier1_markets = ['United States', 'United Kingdom', 'Germany', 'France', 'Canada', 'Australia']
        tier2_markets = ['Netherlands', 'Sweden', 'Switzerland', 'Norway', 'Denmark', 'Singapore']
        
        df_predictive['market_tier'] = 3  # Default to tier 3
        df_predictive.loc[df_predictive['country'].isin(tier1_markets), 'market_tier'] = 1
        df_predictive.loc[df_predictive['country'].isin(tier2_markets), 'market_tier'] = 2
    
    # 4. Title/Position Features
    if 'title' in df_predictive.columns:
        # Seniority indicators
        senior_titles = ['ceo', 'cto', 'vp', 'director', 'head', 'chief', 'president']
        mid_titles = ['manager', 'lead', 'senior', 'principal']
        junior_titles = ['associate', 'junior', 'entry', 'intern']
        
        title_lower = df_predictive['title'].str.lower().fillna('')
        
        df_predictive['is_senior'] = title_lower.str.contains('|'.join(senior_titles), na=False).astype(int)
        df_predictive['is_mid_level'] = title_lower.str.contains('|'.join(mid_titles), na=False).astype(int)
        df_predictive['is_junior'] = title_lower.str.contains('|'.join(junior_titles), na=False).astype(int)
        
        # Department indicators
        tech_titles = ['engineer', 'developer', 'programmer', 'architect', 'devops', 'data']
        sales_titles = ['sales', 'account', 'business development', 'revenue']
        marketing_titles = ['marketing', 'growth', 'brand', 'content']
        
        df_predictive['is_tech_role'] = title_lower.str.contains('|'.join(tech_titles), na=False).astype(int)
        df_predictive['is_sales_role'] = title_lower.str.contains('|'.join(sales_titles), na=False).astype(int)
        df_predictive['is_marketing_role'] = title_lower.str.contains('|'.join(marketing_titles), na=False).astype(int)
    
    # 5. Industry Features
    if 'organization_industry' in df_predictive.columns:
        industry_lower = df_predictive['organization_industry'].str.lower().fillna('')
        
        # Industry categories
        tech_industries = ['technology', 'software', 'internet', 'information technology', 'saas']
        finance_industries = ['finance', 'banking', 'financial services', 'insurance']
        healthcare_industries = ['healthcare', 'medical', 'pharmaceutical', 'biotechnology']
        manufacturing_industries = ['manufacturing', 'industrial', 'automotive', 'aerospace']
        
        df_predictive['is_tech_industry'] = industry_lower.str.contains('|'.join(tech_industries), na=False).astype(int)
        df_predictive['is_finance_industry'] = industry_lower.str.contains('|'.join(finance_industries), na=False).astype(int)
        df_predictive['is_healthcare_industry'] = industry_lower.str.contains('|'.join(healthcare_industries), na=False).astype(int)
        df_predictive['is_manufacturing_industry'] = industry_lower.str.contains('|'.join(manufacturing_industries), na=False).astype(int)
    
    # 6. Temporal Features (predictive)
    if 'timestamp_created' in df_predictive.columns:
        df_predictive['timestamp_created'] = pd.to_datetime(df_predictive['timestamp_created'])
        
        # Time-based features
        df_predictive['created_day_of_week'] = df_predictive['timestamp_created'].dt.dayofweek
        df_predictive['created_month'] = df_predictive['timestamp_created'].dt.month
        df_predictive['created_quarter'] = df_predictive['timestamp_created'].dt.quarter
        df_predictive['created_hour'] = df_predictive['timestamp_created'].dt.hour
        
        # Business-relevant temporal features
        df_predictive['is_monday'] = (df_predictive['created_day_of_week'] == 0).astype(int)
        df_predictive['is_friday'] = (df_predictive['created_day_of_week'] == 4).astype(int)
        df_predictive['is_weekend'] = (df_predictive['created_day_of_week'] >= 5).astype(int)
        df_predictive['is_business_hours'] = ((df_predictive['created_hour'] >= 9) & (df_predictive['created_hour'] <= 17)).astype(int)
        
        # Recency features
        current_time = pd.Timestamp.now()
        df_predictive['days_since_created'] = (current_time - df_predictive['timestamp_created']).dt.days
        df_predictive['is_recent'] = (df_predictive['days_since_created'] <= 7).astype(int)
        df_predictive['is_old'] = (df_predictive['days_since_created'] > 90).astype(int)
    
    # 7. Text Features (predictive)
    text_columns = ['email_subjects', 'email_bodies', 'title', 'organization_industry']
    df_predictive['combined_text'] = ""
    
    for col in text_columns:
        if col in df_predictive.columns:
            df_predictive['combined_text'] += df_predictive[col].fillna('').astype(str) + ' '
    
    # Text complexity features
    df_predictive['text_length'] = df_predictive['combined_text'].str.len()
    df_predictive['text_word_count'] = df_predictive['combined_text'].str.split().str.len()
    df_predictive['text_sentence_count'] = df_predictive['combined_text'].str.count(r'[.!?]') + 1
    
    # Text quality indicators
    df_predictive['has_numbers'] = df_predictive['combined_text'].str.contains(r'\d', na=False).astype(int)
    df_predictive['has_emails'] = df_predictive['combined_text'].str.contains(r'@', na=False).astype(int)
    df_predictive['has_urls'] = df_predictive['combined_text'].str.contains(r'http', na=False).astype(int)
    
    # Business keywords
    business_keywords = ['ceo', 'cto', 'vp', 'director', 'manager', 'founder']
    for keyword in business_keywords:
        df_predictive[f'has_{keyword}'] = df_predictive['combined_text'].str.lower().str.contains(keyword, na=False).astype(int)
    
    # 8. Interaction Features (predictive)
    if 'organization_industry' in df_predictive.columns and 'title' in df_predictive.columns:
        # Industry-title combinations
        high_value_combinations = [
            ('technology', 'ceo'), ('technology', 'cto'), ('technology', 'vp'),
            ('finance', 'ceo'), ('finance', 'director'), ('finance', 'manager'),
            ('healthcare', 'director'), ('healthcare', 'manager')
        ]
        
        for industry, title in high_value_combinations:
            mask = (
                df_predictive['organization_industry'].str.lower().str.contains(industry, na=False) &
                df_predictive['title'].str.lower().str.contains(title, na=False)
            )
            df_predictive[f'is_{industry}_{title}'] = mask.astype(int)
    
    # 9. Geographic-Industry Interactions
    if 'country' in df_predictive.columns and 'organization_industry' in df_predictive.columns:
        # US tech companies
        df_predictive['is_us_tech'] = (
            (df_predictive['country'] == 'United States') &
            (df_predictive['organization_industry'].str.lower().str.contains('technology|software|tech', na=False))
        ).astype(int)
        
        # European finance companies
        df_predictive['is_eu_finance'] = (
            (df_predictive['country'].isin(['United Kingdom', 'Germany', 'France', 'Netherlands'])) &
            (df_predictive['organization_industry'].str.lower().str.contains('finance|banking|financial', na=False))
        ).astype(int)
    
    # 10. Company Maturity Features
    if 'organization_employees' in df_predictive.columns and 'company_age' in df_predictive.columns:
        # Size-age ratio (predictive of engagement potential)
        df_predictive['size_age_ratio'] = df_predictive['organization_employees'] / (df_predictive['company_age'] + 1)
        
        # Employee density
        df_predictive['employee_density'] = df_predictive['organization_employees'] / df_predictive['organization_employees'].max()
    
    return df_predictive

def remove_leakage_features(df):
    """
    Remove any features that could cause data leakage.
    """
    print("🚫 Removing leakage features...")
    
    # List of features that directly relate to engagement outcomes
    leakage_features = [
        'email_open_count', 'email_click_count', 'email_reply_count',
        'email_opened_step', 'email_opened_variant', 'email_clicked_step', 
        'email_clicked_variant', 'email_replied_step', 'email_replied_variant',
        'engagement_funnel_stage', 'is_high_engager', 'is_medium_engager', 
        'is_low_engager', 'open_intensity', 'ctr', 'reply_rate', 'has_clicks', 
        'has_replies', 'multi_channel_engagement'
    ]
    
    # Remove leakage features if they exist
    existing_leakage = [col for col in leakage_features if col in df.columns]
    if existing_leakage:
        print(f"Removing {len(existing_leakage)} leakage features: {existing_leakage}")
        df = df.drop(columns=existing_leakage)
    
    return df

def apply_predictive_feature_engineering(df):
    """
    Apply feature engineering focused on predictive features only.
    """
    print("🚀 Applying predictive feature engineering...")
    
    # Create predictive features
    df = create_predictive_features_only(df)
    
    # Remove any leakage features
    df = remove_leakage_features(df)
    
    print(f"✅ Predictive feature engineering complete. Shape: {df.shape}")
    return df 