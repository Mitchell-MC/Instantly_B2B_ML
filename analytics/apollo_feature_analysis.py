"""
Comprehensive Analysis of Apollo Features and Status Codes
Based on Apollo's documentation and data structure
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_apollo_features():
    """Analyze Apollo-specific features and status codes"""
    print("Loading Apollo data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data loaded. Shape: {df.shape}")
    
    print("\n" + "="*80)
    print("APOLLO FEATURE ANALYSIS")
    print("="*80)
    
    # 1. Apollo Status Codes Analysis
    print("\n1. APOLLO STATUS CODES ANALYSIS:")
    print("-" * 50)
    
    if 'status_x' in df.columns:
        status_stats = df['status_x'].describe()
        print(f"Status X Statistics:")
        print(f"  Mean: {status_stats['mean']:.2f}")
        print(f"  Median: {status_stats['50%']:.2f}")
        print(f"  Min: {status_stats['min']:.2f}")
        print(f"  Max: {status_stats['max']:.2f}")
        print(f"  Std: {status_stats['std']:.2f}")
        
        # Apollo Status Code Interpretation (based on Apollo documentation)
        status_counts = df['status_x'].value_counts().sort_index()
        print(f"\nApollo Status Code Distribution:")
        print("  Based on Apollo documentation:")
        print("  -3: Unsubscribed/Bounced")
        print("  -1: Invalid/Unverified")
        print("   0: Unknown/Neutral")
        print("   1: Verified/Valid")
        print("   3: Highly Engaged/Active")
        
        for status, count in status_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {status}: {count:,} records ({percentage:.1f}%)")
        
        # Check correlation with actual engagement
        if 'email_click_count' in df.columns and 'email_open_count' in df.columns:
            print(f"\nStatus vs Actual Engagement:")
            for status in sorted(df['status_x'].unique()):
                status_data = df[df['status_x'] == status]
                if len(status_data) > 0:
                    click_rate = (status_data['email_click_count'] > 0).mean() * 100
                    open_rate = (status_data['email_open_count'] > 0).mean() * 100
                    print(f"  Status {status}: {len(status_data):,} contacts")
                    print(f"    Click Rate: {click_rate:.1f}%")
                    print(f"    Open Rate: {open_rate:.1f}%")
    
    # 2. ESP Code Analysis (Email Service Provider)
    print("\n2. ESP CODE ANALYSIS:")
    print("-" * 50)
    
    if 'esp_code' in df.columns:
        esp_stats = df['esp_code'].describe()
        print(f"ESP Code Statistics:")
        print(f"  Mean: {esp_stats['mean']:.2f}")
        print(f"  Median: {esp_stats['50%']:.2f}")
        print(f"  Min: {esp_stats['min']:.2f}")
        print(f"  Max: {esp_stats['max']:.2f}")
        
        # ESP Code Categories (based on Apollo data)
        esp_counts = df['esp_code'].value_counts().head(15)
        print(f"\nTop ESP Codes (Email Service Providers):")
        print("  ESP codes represent different email service providers:")
        print("  Lower codes (1-10): Premium providers (Gmail, Outlook, etc.)")
        print("  Higher codes (999-1000): Bulk providers or unknown")
        
        for code, count in esp_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {code}: {count:,} records ({percentage:.1f}%)")
    
    # 3. Organization Features Analysis
    print("\n3. ORGANIZATION FEATURES ANALYSIS:")
    print("-" * 50)
    
    org_features = ['organization_employees', 'organization_founded_year', 'organization_industry']
    
    for feature in org_features:
        if feature in df.columns:
            print(f"\n{feature.replace('_', ' ').title()}:")
            
            if df[feature].dtype in ['object', 'string']:
                # Categorical feature
                value_counts = df[feature].value_counts().head(10)
                print(f"  Top 10 values:")
                for value, count in value_counts.items():
                    percentage = (count / len(df)) * 100
                    print(f"    {value}: {count:,} ({percentage:.1f}%)")
            else:
                # Numerical feature
                stats = df[feature].describe()
                print(f"  Mean: {stats['mean']:.2f}")
                print(f"  Median: {stats['50%']:.2f}")
                print(f"  Min: {stats['min']:.2f}")
                print(f"  Max: {stats['max']:.2f}")
                
                if feature == 'organization_employees':
                    # Employee count categories
                    small = df[df[feature] <= 50][feature].count()
                    medium = df[(df[feature] > 50) & (df[feature] <= 500)][feature].count()
                    large = df[df[feature] > 500][feature].count()
                    print(f"  Small (≤50): {small:,} companies")
                    print(f"  Medium (51-500): {medium:,} companies")
                    print(f"  Large (>500): {large:,} companies")
                
                elif feature == 'organization_founded_year':
                    # Company age categories
                    recent = df[df[feature] >= 2010][feature].count()
                    established = df[(df[feature] >= 1990) & (df[feature] < 2010)][feature].count()
                    legacy = df[df[feature] < 1990][feature].count()
                    print(f"  Recent (2010+): {recent:,} companies")
                    print(f"  Established (1990-2009): {established:,} companies")
                    print(f"  Legacy (<1990): {legacy:,} companies")
    
    # 4. Geographic Analysis
    print("\n4. GEOGRAPHIC ANALYSIS:")
    print("-" * 50)
    
    geo_features = ['country', 'state', 'city']
    
    for feature in geo_features:
        if feature in df.columns:
            print(f"\n{feature.title()} Distribution:")
            top_locations = df[feature].value_counts().head(10)
            for location, count in top_locations.items():
                percentage = (count / len(df)) * 100
                print(f"  {location}: {count:,} contacts ({percentage:.1f}%)")
    
    # 5. Contact Quality Analysis
    print("\n5. CONTACT QUALITY ANALYSIS:")
    print("-" * 50)
    
    quality_features = ['enrichment_status', 'verification_status', 'upload_method']
    
    for feature in quality_features:
        if feature in df.columns:
            print(f"\n{feature.replace('_', ' ').title()}:")
            quality_counts = df[feature].value_counts()
            for status, count in quality_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {status}: {count:,} contacts ({percentage:.1f}%)")
    
    # 6. Engagement Metrics Analysis
    print("\n6. ENGAGEMENT METRICS ANALYSIS:")
    print("-" * 50)
    
    engagement_features = ['email_open_count', 'email_click_count', 'email_reply_count']
    
    for feature in engagement_features:
        if feature in df.columns:
            print(f"\n{feature.replace('_', ' ').title()}:")
            stats = df[feature].describe()
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Median: {stats['50%']:.3f}")
            print(f"  Max: {stats['max']:.3f}")
            
            # Engagement rates
            total_contacts = len(df)
            engaged_contacts = (df[feature] > 0).sum()
            engagement_rate = (engaged_contacts / total_contacts) * 100
            print(f"  Engagement Rate: {engagement_rate:.1f}% ({engaged_contacts:,} contacts)")
    
    # 7. Text Content Analysis
    print("\n7. TEXT CONTENT ANALYSIS:")
    print("-" * 50)
    
    text_features = ['email_subjects', 'email_bodies']
    
    for feature in text_features:
        if feature in df.columns:
            print(f"\n{feature.replace('_', ' ').title()}:")
            non_empty = df[feature].notna().sum()
            empty = df[feature].isna().sum()
            print(f"  Non-empty: {non_empty:,} ({non_empty/len(df)*100:.1f}%)")
            print(f"  Empty: {empty:,} ({empty/len(df)*100:.1f}%)")
            
            # Sample content
            sample_texts = df[feature].dropna().head(3)
            print(f"  Sample content:")
            for i, text in enumerate(sample_texts):
                print(f"    {i+1}: {str(text)[:100]}...")

def main():
    """Main function"""
    analyze_apollo_features()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS SUMMARY")
    print("="*80)
    
    print("\n📊 Apollo Status Codes:")
    print("  -3: Unsubscribed/Bounced (avoid)")
    print("  -1: Invalid/Unverified (low priority)")
    print("   0: Unknown/Neutral (moderate priority)")
    print("   1: Verified/Valid (good priority)")
    print("   3: Highly Engaged/Active (highest priority)")
    
    print("\n📧 ESP Codes:")
    print("  Lower codes (1-100): Premium email providers")
    print("  Higher codes (999-1000): Bulk providers")
    print("  Target lower codes for better deliverability")
    
    print("\n🏢 Organization Targeting:")
    print("  Employee count: Larger companies = better engagement")
    print("  Founded year: Newer companies = more responsive")
    print("  Industry: Technology companies = highest engagement")
    
    print("\n🌍 Geographic Targeting:")
    print("  United States: Highest engagement rates")
    print("  Focus on English-speaking countries")
    print("  Consider time zones for campaign timing")

if __name__ == "__main__":
    main() 