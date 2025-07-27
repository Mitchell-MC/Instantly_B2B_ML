"""
Inspect the top 10 features from binary SHAP analysis
Understand what each variable represents and its relationship to email opens
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def inspect_features():
    """Inspect the top 10 features from SHAP analysis"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Top 10 features from SHAP analysis
    top_features = [
        'email_list',
        'link_tracking', 
        'esp_code',
        'organization_employees',
        'daily_limit',
        'page_retrieved',
        'status_x',
        'organization_founded_year',
        'upload_method',
        'campaign_schedule_schedules'
    ]
    
    print("\n" + "="*80)
    print("FEATURE INSPECTION: TOP 10 SHAP FEATURES")
    print("="*80)
    
    for i, feature in enumerate(top_features, 1):
        print(f"\n{i}. {feature.upper()}")
        print("-" * 50)
        
        if feature in df.columns:
            # Basic statistics
            print(f"Data type: {df[feature].dtype}")
            print(f"Missing values: {df[feature].isnull().sum()} ({df[feature].isnull().sum()/len(df)*100:.1f}%)")
            
            if df[feature].dtype == 'object':
                # Categorical feature
                unique_vals = df[feature].value_counts()
                print(f"Unique values: {len(unique_vals)}")
                print("Top 10 values:")
                for val, count in unique_vals.head(10).items():
                    print(f"  '{val}': {count:,} ({count/len(df)*100:.1f}%)")
                    
                # Check relationship with email_open_count
                if 'email_open_count' in df.columns:
                    open_rate_by_val = df.groupby(feature)['email_open_count'].agg(['count', 'sum', 'mean'])
                    open_rate_by_val['open_rate'] = open_rate_by_val['sum'] / open_rate_by_val['count']
                    open_rate_by_val = open_rate_by_val.sort_values('open_rate', ascending=False)
                    print("\nOpen rates by value (top 5):")
                    for val, row in open_rate_by_val.head(5).iterrows():
                        print(f"  '{val}': {row['open_rate']:.3f} ({row['count']:,} contacts)")
                        
            else:
                # Numerical feature
                print(f"Min: {df[feature].min()}")
                print(f"Max: {df[feature].max()}")
                print(f"Mean: {df[feature].mean():.2f}")
                print(f"Median: {df[feature].median():.2f}")
                
                # Check relationship with email_open_count
                if 'email_open_count' in df.columns:
                    # Create bins for numerical features
                    if feature == 'organization_employees':
                        bins = [0, 10, 50, 200, 1000, float('inf')]
                        labels = ['1-10', '11-50', '51-200', '201-1000', '1000+']
                        df[f'{feature}_binned'] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True)
                        open_rate_by_bin = df.groupby(f'{feature}_binned')['email_open_count'].agg(['count', 'sum', 'mean'])
                        open_rate_by_bin['open_rate'] = open_rate_by_bin['sum'] / open_rate_by_bin['count']
                        print("\nOpen rates by company size:")
                        for bin_name, row in open_rate_by_bin.iterrows():
                            print(f"  {bin_name}: {row['open_rate']:.3f} ({row['count']:,} contacts)")
                    elif feature == 'organization_founded_year':
                        # Handle year data
                        current_year = 2024
                        df['company_age'] = current_year - df[feature]
                        bins = [0, 5, 10, 20, 50, float('inf')]
                        labels = ['0-5 years', '6-10 years', '11-20 years', '21-50 years', '50+ years']
                        df['age_binned'] = pd.cut(df['company_age'], bins=bins, labels=labels, include_lowest=True)
                        open_rate_by_age = df.groupby('age_binned')['email_open_count'].agg(['count', 'sum', 'mean'])
                        open_rate_by_age['open_rate'] = open_rate_by_age['sum'] / open_rate_by_age['count']
                        print("\nOpen rates by company age:")
                        for age_bin, row in open_rate_by_age.iterrows():
                            print(f"  {age_bin}: {row['open_rate']:.3f} ({row['count']:,} contacts)")
                    else:
                        # For other numerical features, show correlation
                        correlation = df[feature].corr(df['email_open_count'])
                        print(f"Correlation with email_open_count: {correlation:.3f}")
        else:
            print(f"Feature '{feature}' not found in dataset")
    
    print("\n" + "="*80)
    print("FEATURE INTERPRETATION SUMMARY")
    print("="*80)
    
    interpretations = {
        'email_list': 'Email list identifier - different lists may have different quality/engagement levels',
        'link_tracking': 'Whether link tracking is enabled - affects email deliverability and engagement',
        'esp_code': 'Email Service Provider code - different ESPs have different deliverability rates',
        'organization_employees': 'Company size - larger companies may have different email behavior',
        'daily_limit': 'Daily sending limits - may affect email timing and frequency',
        'page_retrieved': 'Whether contact data was retrieved from a page - indicates data quality',
        'status_x': 'Apollo contact quality score - higher quality contacts may engage more',
        'organization_founded_year': 'Company age - newer vs older companies may have different email behavior',
        'upload_method': 'How contacts were uploaded - may indicate data quality and source',
        'campaign_schedule_schedules': 'Campaign scheduling settings - timing affects open rates'
    }
    
    for feature, interpretation in interpretations.items():
        print(f"\n• {feature}: {interpretation}")

if __name__ == "__main__":
    inspect_features() 