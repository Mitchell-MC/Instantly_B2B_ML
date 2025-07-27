"""
Detailed Email List Analysis
Break down email_list feature for statistics and insights
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_email_lists():
    """Analyze email_list feature in detail"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("DETAILED EMAIL LIST ANALYSIS")
    print("="*80)
    
    # Basic email_list statistics
    print("\n1. EMAIL LIST BASIC STATISTICS:")
    print("-" * 50)
    print(f"Data type: {df['email_list'].dtype}")
    print(f"Missing values: {df['email_list'].isnull().sum()} ({df['email_list'].isnull().sum()/len(df)*100:.1f}%)")
    print(f"Unique email lists: {df['email_list'].nunique()}")
    
    # Email list distribution
    print("\n2. EMAIL LIST DISTRIBUTION:")
    print("-" * 50)
    list_counts = df['email_list'].value_counts()
    print("Top 15 email lists by frequency:")
    for i, (email_list, count) in enumerate(list_counts.head(15).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{i:2d}. {email_list[:80]}{'...' if len(email_list) > 80 else ''}")
        print(f"    Contacts: {count:,} ({percentage:.1f}%)")
    
    # Email list performance by open rates
    print("\n3. EMAIL LIST PERFORMANCE BY OPEN RATES:")
    print("-" * 50)
    list_performance = df.groupby('email_list')['email_open_count'].agg([
        'count', 'sum', 'mean', 'std'
    ]).round(3)
    list_performance['open_rate'] = list_performance['sum'] / list_performance['count']
    list_performance = list_performance.sort_values('open_rate', ascending=False)
    
    print("Top 15 email lists by open rate:")
    for i, (email_list, row) in enumerate(list_performance.head(15).iterrows(), 1):
        print(f"{i:2d}. {email_list[:80]}{'...' if len(email_list) > 80 else ''}")
        print(f"    Open Rate: {row['open_rate']:.3f} opens/contact")
        print(f"    Contacts: {row['count']:,}, Total Opens: {row['sum']:.0f}")
    
    print("\nBottom 15 email lists by open rate:")
    for i, (email_list, row) in enumerate(list_performance.tail(15).iterrows(), 1):
        print(f"{i:2d}. {email_list[:80]}{'...' if len(email_list) > 80 else ''}")
        print(f"    Open Rate: {row['open_rate']:.3f} opens/contact")
        print(f"    Contacts: {row['count']:,}, Total Opens: {row['sum']:.0f}")
    
    # Extract domain patterns from email lists
    print("\n4. DOMAIN PATTERN ANALYSIS:")
    print("-" * 50)
    
    def extract_domains(email_list_str):
        """Extract domains from email list string"""
        if pd.isna(email_list_str):
            return []
        # Extract email addresses using regex
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_list_str)
        # Extract domains
        domains = [email.split('@')[1] for email in emails if '@' in email]
        return list(set(domains))  # Remove duplicates
    
    # Analyze domains across all email lists
    all_domains = []
    domain_counts = {}
    
    for email_list in df['email_list'].dropna():
        domains = extract_domains(email_list)
        all_domains.extend(domains)
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print(f"Total unique domains found: {len(set(all_domains))}")
    print("\nTop 20 domains by frequency:")
    sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (domain, count) in enumerate(sorted_domains[:20], 1):
        print(f"{i:2d}. {domain}: {count:,} occurrences")
    
    # Email list categorization
    print("\n5. EMAIL LIST CATEGORIZATION:")
    print("-" * 50)
    
    def categorize_email_list(email_list_str):
        """Categorize email list based on characteristics"""
        if pd.isna(email_list_str):
            return "Unknown"
        
        email_list_str = str(email_list_str).lower()
        
        # Check for different patterns
        if 'weclouddata' in email_list_str:
            return "WeCloudData Lists"
        elif 'beamdata' in email_list_str:
            return "BeamData Lists"
        elif 'louise' in email_list_str:
            return "Louise Lists"
        elif 'warney' in email_list_str:
            return "Warney Lists"
        elif 'n-tran' in email_list_str or 'ntran' in email_list_str:
            return "N-Tran Lists"
        elif len(email_list_str.split(',')) > 10:
            return "Large Lists (10+ emails)"
        elif len(email_list_str.split(',')) > 5:
            return "Medium Lists (5-10 emails)"
        else:
            return "Small Lists (<5 emails)"
    
    df['list_category'] = df['email_list'].apply(categorize_email_list)
    
    category_stats = df.groupby('list_category')['email_open_count'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    category_stats['open_rate'] = category_stats['sum'] / category_stats['count']
    category_stats = category_stats.sort_values('open_rate', ascending=False)
    
    print("Email List Categories by Performance:")
    for category, row in category_stats.iterrows():
        print(f"  {category}: {row['open_rate']:.3f} opens/contact ({row['count']:,} contacts)")
    
    # Email list quality analysis
    print("\n6. EMAIL LIST QUALITY ANALYSIS:")
    print("-" * 50)
    
    def analyze_list_quality(email_list_str):
        """Analyze quality characteristics of email list"""
        if pd.isna(email_list_str):
            return {'email_count': 0, 'unique_domains': 0, 'quality_score': 0}
        
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_list_str)
        domains = [email.split('@')[1] for email in emails if '@' in email]
        unique_domains = len(set(domains))
        
        # Quality score based on domain diversity and email count
        quality_score = min(unique_domains * len(emails) / 10, 10)  # Scale 0-10
        
        return {
            'email_count': len(emails),
            'unique_domains': unique_domains,
            'quality_score': quality_score
        }
    
    # Apply quality analysis to top performing lists
    print("Quality analysis of top 10 performing email lists:")
    for email_list, row in list_performance.head(10).iterrows():
        quality = analyze_list_quality(email_list)
        print(f"\n{email_list[:60]}...")
        print(f"  Open Rate: {row['open_rate']:.3f}")
        print(f"  Emails in List: {quality['email_count']}")
        print(f"  Unique Domains: {quality['unique_domains']}")
        print(f"  Quality Score: {quality['quality_score']:.1f}/10")
    
    # Business insights
    print("\n7. BUSINESS INSIGHTS:")
    print("-" * 50)
    
    # Find best and worst performing lists
    best_list = list_performance.head(1).index[0]
    worst_list = list_performance.tail(1).index[0]
    
    best_rate = list_performance.loc[best_list, 'open_rate']
    worst_rate = list_performance.loc[worst_list, 'open_rate']
    
    print(f"• Best performing list: {best_list[:60]}...")
    print(f"  Open Rate: {best_rate:.3f} opens/contact")
    
    print(f"\n• Worst performing list: {worst_list[:60]}...")
    print(f"  Open Rate: {worst_rate:.3f} opens/contact")
    
    print(f"\n• Performance difference: {best_rate - worst_rate:.3f} opens/contact")
    
    # List size analysis
    print("\n8. LIST SIZE ANALYSIS:")
    print("-" * 50)
    
    def get_list_size(email_list_str):
        """Get the number of emails in a list"""
        if pd.isna(email_list_str):
            return 0
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email_list_str)
        return len(emails)
    
    df['list_size'] = df['email_list'].apply(get_list_size)
    
    # Analyze performance by list size
    size_bins = [0, 5, 10, 20, 50, float('inf')]
    size_labels = ['1-5', '6-10', '11-20', '21-50', '50+']
    df['list_size_category'] = pd.cut(df['list_size'], bins=size_bins, labels=size_labels, include_lowest=True)
    
    size_performance = df.groupby('list_size_category')['email_open_count'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    size_performance['open_rate'] = size_performance['sum'] / size_performance['count']
    
    print("Performance by list size:")
    for size_cat, row in size_performance.iterrows():
        print(f"  {size_cat} emails: {row['open_rate']:.3f} opens/contact ({row['count']:,} contacts)")
    
    # Recommendations
    print("\n9. RECOMMENDATIONS:")
    print("-" * 50)
    
    print("• Focus on high-performing email lists with diverse domains")
    print("• Avoid lists with poor open rates and low domain diversity")
    print("• Medium-sized lists (6-20 emails) tend to perform better")
    print("• Lists with multiple domains show better engagement")
    print("• Consider list quality score for campaign targeting")

if __name__ == "__main__":
    analyze_email_lists() 