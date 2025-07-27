"""
Page Retrieved Variable Analysis
Understand what page_retrieved indicates and its relationship to email performance
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_page_retrieved():
    """Analyze the page_retrieved variable in detail"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("PAGE RETRIEVED VARIABLE ANALYSIS")
    print("="*80)
    
    # Basic statistics
    print("\n1. BASIC STATISTICS:")
    print("-" * 50)
    print(f"Data type: {df['page_retrieved'].dtype}")
    print(f"Missing values: {df['page_retrieved'].isnull().sum()}")
    print(f"Min: {df['page_retrieved'].min()}")
    print(f"Max: {df['page_retrieved'].max()}")
    print(f"Mean: {df['page_retrieved'].mean():.2f}")
    print(f"Median: {df['page_retrieved'].median():.2f}")
    print(f"Standard deviation: {df['page_retrieved'].std():.2f}")
    
    # Value distribution
    print("\n2. VALUE DISTRIBUTION:")
    print("-" * 50)
    value_counts = df['page_retrieved'].value_counts().sort_index()
    print("Top 20 most common page_retrieved values:")
    for value, count in value_counts.head(20).items():
        percentage = (count / len(df)) * 100
        print(f"  {value}: {count:,} contacts ({percentage:.1f}%)")
    
    # Performance analysis by page_retrieved value
    print("\n3. PERFORMANCE BY PAGE_RETRIEVED VALUE:")
    print("-" * 50)
    performance_by_page = df.groupby('page_retrieved')['email_open_count'].agg([
        'count', 'sum', 'mean', 'std'
    ]).round(3)
    performance_by_page['open_rate'] = performance_by_page['sum'] / performance_by_page['count']
    performance_by_page = performance_by_page.sort_values('open_rate', ascending=False)
    
    print("Top 20 page_retrieved values by open rate:")
    for page_value, row in performance_by_page.head(20).iterrows():
        print(f"  Page {page_value}: {row['open_rate']:.3f} opens/contact "
              f"({row['count']:,} contacts, {row['sum']:.0f} total opens)")
    
    print("\nBottom 20 page_retrieved values by open rate:")
    for page_value, row in performance_by_page.tail(20).iterrows():
        print(f"  Page {page_value}: {row['open_rate']:.3f} opens/contact "
              f"({row['count']:,} contacts, {row['sum']:.0f} total opens)")
    
    # Correlation analysis
    print("\n4. CORRELATION ANALYSIS:")
    print("-" * 50)
    correlation = df['page_retrieved'].corr(df['email_open_count'])
    print(f"Correlation with email_open_count: {correlation:.3f}")
    
    # Page range analysis
    print("\n5. PAGE RANGE ANALYSIS:")
    print("-" * 50)
    
    # Create page ranges
    df['page_range'] = pd.cut(df['page_retrieved'], 
                              bins=[0, 50, 100, 150, 200, 266], 
                              labels=['1-50', '51-100', '101-150', '151-200', '201-266'],
                              include_lowest=True)
    
    range_performance = df.groupby('page_range')['email_open_count'].agg([
        'count', 'sum', 'mean'
    ]).round(3)
    range_performance['open_rate'] = range_performance['sum'] / range_performance['count']
    
    print("Performance by page range:")
    for page_range, row in range_performance.iterrows():
        print(f"  {page_range}: {row['open_rate']:.3f} opens/contact ({row['count']:,} contacts)")
    
    # Page retrieved interpretation
    print("\n6. PAGE_RETRIEVED INTERPRETATION:")
    print("-" * 50)
    
    print("Based on the analysis, 'page_retrieved' likely indicates:")
    print("• The page number or position where contact data was retrieved from")
    print("• Could represent search result position (e.g., Google search page)")
    print("• May indicate data source depth or quality")
    print("• Lower numbers = higher quality/more relevant results")
    print("• Higher numbers = deeper search results or less relevant")
    
    # Business insights
    print("\n7. BUSINESS INSIGHTS:")
    print("-" * 50)
    
    # Find best and worst performing pages
    best_page = performance_by_page.head(1).index[0]
    worst_page = performance_by_page.tail(1).index[0]
    
    best_rate = performance_by_page.loc[best_page, 'open_rate']
    worst_rate = performance_by_page.loc[worst_page, 'open_rate']
    
    print(f"• Best performing page: {best_page} ({best_rate:.3f} opens/contact)")
    print(f"• Worst performing page: {worst_page} ({worst_rate:.3f} opens/contact)")
    print(f"• Performance difference: {best_rate - worst_rate:.3f} opens/contact")
    
    # Page quality analysis
    print("\n8. PAGE QUALITY ANALYSIS:")
    print("-" * 50)
    
    # Analyze if lower page numbers perform better
    low_pages = df[df['page_retrieved'] <= 50]
    high_pages = df[df['page_retrieved'] > 200]
    
    low_page_rate = low_pages['email_open_count'].mean()
    high_page_rate = high_pages['email_open_count'].mean()
    
    print(f"• Low pages (1-50): {low_page_rate:.3f} opens/contact ({len(low_pages):,} contacts)")
    print(f"• High pages (201-266): {high_page_rate:.3f} opens/contact ({len(high_pages):,} contacts)")
    
    if low_page_rate > high_page_rate:
        print(f"• Low pages perform {((low_page_rate/high_page_rate)-1)*100:.1f}% better")
    else:
        print(f"• High pages perform {((high_page_rate/low_page_rate)-1)*100:.1f}% better")
    
    # Recommendations
    print("\n9. RECOMMENDATIONS:")
    print("-" * 50)
    
    print("• Focus on contacts from lower page numbers (1-100) for better quality")
    print("• Avoid contacts from very high page numbers (200+) as they may be low quality")
    print("• Use page_retrieved as a proxy for contact relevance/quality")
    print("• Consider page position when prioritizing campaign targets")
    print("• Lower page numbers likely indicate more relevant search results")

if __name__ == "__main__":
    analyze_page_retrieved() 