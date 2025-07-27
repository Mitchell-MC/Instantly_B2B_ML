"""
Detailed ESP Code Analysis
Break down Email Service Provider codes and their relationship to email opens
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_esp_codes():
    """Analyze ESP codes in detail"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("DETAILED ESP CODE ANALYSIS")
    print("="*80)
    
    # Basic ESP code statistics
    print("\n1. ESP CODE BASIC STATISTICS:")
    print("-" * 50)
    print(f"Data type: {df['esp_code'].dtype}")
    print(f"Missing values: {df['esp_code'].isnull().sum()}")
    print(f"Min: {df['esp_code'].min()}")
    print(f"Max: {df['esp_code'].max()}")
    print(f"Mean: {df['esp_code'].mean():.2f}")
    print(f"Median: {df['esp_code'].median():.2f}")
    
    # ESP code distribution
    print("\n2. ESP CODE DISTRIBUTION:")
    print("-" * 50)
    esp_counts = df['esp_code'].value_counts().sort_index()
    print("Top 20 ESP codes by frequency:")
    for esp_code, count in esp_counts.head(20).items():
        percentage = (count / len(df)) * 100
        print(f"  ESP {esp_code}: {count:,} contacts ({percentage:.1f}%)")
    
    # ESP codes by open rates
    print("\n3. ESP CODES BY OPEN RATES:")
    print("-" * 50)
    esp_open_rates = df.groupby('esp_code')['email_open_count'].agg([
        'count', 'sum', 'mean', 'std'
    ]).round(3)
    esp_open_rates['open_rate'] = esp_open_rates['sum'] / esp_open_rates['count']
    esp_open_rates = esp_open_rates.sort_values('open_rate', ascending=False)
    
    print("Top 20 ESP codes by open rate:")
    for esp_code, row in esp_open_rates.head(20).iterrows():
        print(f"  ESP {esp_code}: {row['open_rate']:.3f} opens/contact "
              f"({row['count']:,} contacts, {row['sum']:.0f} total opens)")
    
    print("\nBottom 20 ESP codes by open rate:")
    for esp_code, row in esp_open_rates.tail(20).iterrows():
        print(f"  ESP {esp_code}: {row['open_rate']:.3f} opens/contact "
              f"({row['count']:,} contacts, {row['sum']:.0f} total opens)")
    
    # ESP code categories
    print("\n4. ESP CODE CATEGORIES:")
    print("-" * 50)
    
    # Create categories based on ESP code ranges
    def categorize_esp(esp_code):
        if esp_code <= 10:
            return "Major ESPs (1-10)"
        elif esp_code <= 50:
            return "Medium ESPs (11-50)"
        elif esp_code <= 100:
            return "Small ESPs (51-100)"
        elif esp_code <= 500:
            return "Niche ESPs (101-500)"
        else:
            return "Custom/Unknown ESPs (500+)"
    
    df['esp_category'] = df['esp_code'].apply(categorize_esp)
    
    category_stats = df.groupby('esp_category')['email_open_count'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    category_stats['open_rate'] = category_stats['sum'] / category_stats['count']
    category_stats = category_stats.sort_values('open_rate', ascending=False)
    
    print("ESP Categories by Performance:")
    for category, row in category_stats.iterrows():
        print(f"  {category}: {row['open_rate']:.3f} opens/contact "
              f"({row['count']:,} contacts)")
    
    # Correlation analysis
    print("\n5. CORRELATION ANALYSIS:")
    print("-" * 50)
    correlation = df['esp_code'].corr(df['email_open_count'])
    print(f"Correlation between ESP code and email_open_count: {correlation:.3f}")
    
    # ESP code patterns
    print("\n6. ESP CODE PATTERNS:")
    print("-" * 50)
    
    # Check for common ESP codes
    common_esp_codes = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    print("Common ESP codes and their performance:")
    for esp_code in common_esp_codes:
        if esp_code in df['esp_code'].values:
            esp_data = df[df['esp_code'] == esp_code]
            count = len(esp_data)
            open_rate = esp_data['email_open_count'].mean()
            print(f"  ESP {esp_code}: {open_rate:.3f} opens/contact ({count:,} contacts)")
    
    # ESP code interpretation
    print("\n7. ESP CODE INTERPRETATION:")
    print("-" * 50)
    
    interpretations = {
        "Low ESP codes (1-10)": "Major email service providers (Gmail, Outlook, Yahoo, etc.)",
        "Medium ESP codes (11-50)": "Regional or specialized email providers",
        "High ESP codes (100+)": "Custom email systems, internal servers, or niche providers",
        "Very high ESP codes (500+)": "Unknown or custom email configurations"
    }
    
    for category, interpretation in interpretations.items():
        print(f"• {category}: {interpretation}")
    
    print("\n8. BUSINESS INSIGHTS:")
    print("-" * 50)
    
    # Find best and worst performing ESP codes
    best_esp = esp_open_rates.head(1).index[0]
    worst_esp = esp_open_rates.tail(1).index[0]
    
    best_rate = esp_open_rates.loc[best_esp, 'open_rate']
    worst_rate = esp_open_rates.loc[worst_esp, 'open_rate']
    
    print(f"• Best performing ESP code: {best_esp} ({best_rate:.3f} opens/contact)")
    print(f"• Worst performing ESP code: {worst_esp} ({worst_rate:.3f} opens/contact)")
    print(f"• Performance difference: {best_rate - worst_rate:.3f} opens/contact")
    
    # Overall recommendations
    print("\n9. RECOMMENDATIONS:")
    print("-" * 50)
    print("• Focus on contacts with lower ESP codes (1-50) for better deliverability")
    print("• Avoid contacts with very high ESP codes (500+) as they may have poor deliverability")
    print("• Major ESPs (codes 1-10) typically have better open rates")
    print("• Consider ESP code as a proxy for email infrastructure quality")

if __name__ == "__main__":
    analyze_esp_codes() 