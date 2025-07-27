"""
Analyze the top 3 features from SHAP plot to understand their actual values and meanings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer

# Configuration (same as SHAP script)
CSV_FILE_PATH = Path("merged_contacts.csv")
TEXT_COLS = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'verification_status', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

COLS_TO_DROP = [
    'id', 'email_clicked_variant', 'email_clicked_step', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    'email_reply_count', 'email_opened_variant', 'email_opened_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click',
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated',
    'personalization', 'status_summary', 'payload', 'list_id',
    'assigned_to', 'campaign', 'uploaded_by_user',
]

def load_and_analyze_data():
    """Load data and analyze the top 3 SHAP features"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Analyze the top 3 features from SHAP plot
    print("\n" + "="*60)
    print("ANALYSIS OF TOP 3 SHAP FEATURES")
    print("="*60)
    
    # 1. Analyze tfidf_com (TF-IDF feature for 'com')
    print("\n1. TF-IDF 'com' Feature Analysis:")
    print("-" * 40)
    
    # Get the original text data
    combined_text = ""
    for col in TEXT_COLS:
        if col in df.columns:
            combined_text += df[col].fillna('') + ' '
    
    # Check what 'com' refers to in the text
    com_mentions = df[df.apply(lambda row: any('com' in str(val).lower() for val in row[TEXT_COLS] if pd.notna(val)), axis=1)]
    print(f"Records containing 'com': {len(com_mentions)}")
    
    if len(com_mentions) > 0:
        print("\nSample text containing 'com':")
        for col in TEXT_COLS:
            if col in com_mentions.columns:
                sample_texts = com_mentions[col].dropna().head(3)
                for i, text in enumerate(sample_texts):
                    print(f"  {col} {i+1}: {str(text)[:100]}...")
    
    # 2. Analyze esp_code
    print("\n2. ESP Code Analysis:")
    print("-" * 40)
    if 'esp_code' in df.columns:
        esp_stats = df['esp_code'].describe()
        print(f"ESP Code Statistics:")
        print(f"  Mean: {esp_stats['mean']:.2f}")
        print(f"  Median: {esp_stats['50%']:.2f}")
        print(f"  Min: {esp_stats['min']:.2f}")
        print(f"  Max: {esp_stats['max']:.2f}")
        print(f"  Std: {esp_stats['std']:.2f}")
        
        # Show value distribution
        esp_counts = df['esp_code'].value_counts().head(10)
        print(f"\nTop 10 ESP Code values:")
        for code, count in esp_counts.items():
            print(f"  {code}: {count} records")
        
        # Check if esp_code correlates with engagement
        if 'email_click_count' in df.columns and 'email_open_count' in df.columns:
            high_engagement = df[(df['email_click_count'] > 0) | (df['email_open_count'] > 0)]
            low_engagement = df[(df['email_click_count'] == 0) & (df['email_open_count'] == 0)]
            
            print(f"\nESP Code by Engagement:")
            print(f"  High Engagement ESP Code Mean: {high_engagement['esp_code'].mean():.2f}")
            print(f"  Low Engagement ESP Code Mean: {low_engagement['esp_code'].mean():.2f}")
    
    # 3. Analyze status_x
    print("\n3. Status X Analysis:")
    print("-" * 40)
    if 'status_x' in df.columns:
        status_stats = df['status_x'].describe()
        print(f"Status X Statistics:")
        print(f"  Mean: {status_stats['mean']:.2f}")
        print(f"  Median: {status_stats['50%']:.2f}")
        print(f"  Min: {status_stats['min']:.2f}")
        print(f"  Max: {status_stats['max']:.2f}")
        print(f"  Std: {status_stats['std']:.2f}")
        
        # Show value distribution
        status_counts = df['status_x'].value_counts().sort_index()
        print(f"\nStatus X value distribution:")
        for status, count in status_counts.items():
            print(f"  {status}: {count} records")
        
        # Check if status_x correlates with engagement
        if 'email_click_count' in df.columns and 'email_open_count' in df.columns:
            high_engagement = df[(df['email_click_count'] > 0) | (df['email_open_count'] > 0)]
            low_engagement = df[(df['email_click_count'] == 0) & (df['email_open_count'] == 0)]
            
            print(f"\nStatus X by Engagement:")
            print(f"  High Engagement Status Mean: {high_engagement['status_x'].mean():.2f}")
            print(f"  Low Engagement Status Mean: {low_engagement['status_x'].mean():.2f}")
    
    # Additional analysis of text content
    print("\n4. Text Content Analysis:")
    print("-" * 40)
    
    # Check what words are most common in high vs low engagement
    if 'email_click_count' in df.columns and 'email_open_count' in df.columns:
        high_engagement_text = df[(df['email_click_count'] > 0) | (df['email_open_count'] > 0)]
        low_engagement_text = df[(df['email_click_count'] == 0) & (df['email_open_count'] == 0)]
        
        # Sample text from each group
        print("Sample text from high engagement contacts:")
        for col in TEXT_COLS:
            if col in high_engagement_text.columns:
                sample_texts = high_engagement_text[col].dropna().head(2)
                for i, text in enumerate(sample_texts):
                    print(f"  {col} {i+1}: {str(text)[:80]}...")
        
        print("\nSample text from low engagement contacts:")
        for col in TEXT_COLS:
            if col in low_engagement_text.columns:
                sample_texts = low_engagement_text[col].dropna().head(2)
                for i, text in enumerate(sample_texts):
                    print(f"  {col} {i+1}: {str(text)[:80]}...")

if __name__ == "__main__":
    load_and_analyze_data() 