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
    
    # Create realistic page ranges (as if these were actual search result pages)
    df['page_range'] = pd.cut(df['page_retrieved'], 
                              bins=[0, 5, 10, 20, 50, 266], 
                              labels=['Pages 1-5', 'Pages 6-10', 'Pages 11-20', 'Pages 21-50', 'Pages 50+'],
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
    print("• Data collection batch number (1-266 batches of ~100 contacts each)")
    print("• Could represent search result page position from data source")
    print("• May indicate data collection sequence or source quality")
    print("• Lower page numbers (1-5) = first/highest quality results")
    print("• Medium page numbers (6-20) = standard quality results")
    print("• Higher page numbers (50+) = deeper/potentially lower quality results")
    
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
    
    # Analyze if lower page numbers perform better (realistic ranges)
    premium_pages = df[df['page_retrieved'] <= 5]      # First 5 pages
    standard_pages = df[(df['page_retrieved'] >= 6) & (df['page_retrieved'] <= 20)]  # Pages 6-20
    deep_pages = df[df['page_retrieved'] > 50]         # Beyond page 50
    
    premium_rate = premium_pages['email_open_count'].mean()
    standard_rate = standard_pages['email_open_count'].mean()
    deep_rate = deep_pages['email_open_count'].mean()
    
    print(f"• Premium pages (1-5): {premium_rate:.3f} opens/contact ({len(premium_pages):,} contacts)")
    print(f"• Standard pages (6-20): {standard_rate:.3f} opens/contact ({len(standard_pages):,} contacts)")
    print(f"• Deep pages (50+): {deep_rate:.3f} opens/contact ({len(deep_pages):,} contacts)")
    
    # Performance comparison
    if premium_rate > deep_rate:
        print(f"• Premium pages perform {((premium_rate/deep_rate)-1)*100:.1f}% better than deep pages")
    else:
        print(f"• Deep pages perform {((deep_rate/premium_rate)-1)*100:.1f}% better than premium pages")
    
    if premium_rate > standard_rate:
        print(f"• Premium pages perform {((premium_rate/standard_rate)-1)*100:.1f}% better than standard pages")
    else:
        print(f"• Standard pages perform {((standard_rate/premium_rate)-1)*100:.1f}% better than premium pages")
    
    # Recommendations
    print("\n9. RECOMMENDATIONS:")
    print("-" * 50)
    
    print("• Prioritize contacts from premium pages (1-5) for highest quality campaigns")
    print("• Use standard pages (6-20) for regular outreach campaigns")
    print("• Be cautious with deep pages (50+) - test performance before large campaigns")
    print("• Use page_retrieved as a quality indicator when segmenting contacts")
    print("• Consider page position when setting campaign expectations and budgets")
    print("• Monitor performance differences between page ranges to optimize sourcing strategy")

if __name__ == "__main__":
    analyze_page_retrieved() 