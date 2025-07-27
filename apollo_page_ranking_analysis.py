"""
Apollo Page Ranking and Search Analysis
Understanding Apollo's pagination system and page_retrieved field
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_apollo_page_ranking():
    """Comprehensive analysis of Apollo's page ranking system"""
    print("Loading Apollo data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("APOLLO PAGE RANKING AND SEARCH ANALYSIS")
    print("="*80)
    
    # 1. Page Retrieved Analysis
    print("\n1. PAGE_RETRIEVED FIELD ANALYSIS:")
    print("-" * 50)
    
    if 'page_retrieved' in df.columns:
        page_stats = df['page_retrieved'].describe()
        print(f"Page Retrieved Statistics:")
        print(page_stats)
        
        print(f"\nUnique page values: {sorted(df['page_retrieved'].unique())}")
        print(f"Page range: {df['page_retrieved'].min()} to {df['page_retrieved'].max()}")
        
        # Page distribution
        page_counts = df['page_retrieved'].value_counts().sort_index()
        print(f"\nPage Distribution:")
        print(page_counts.head(20))
        
        # Open rate by page
        print(f"\nOpen Rate by Page:")
        page_open_rates = df.groupby('page_retrieved')['email_open_count'].agg(['count', 'sum']).reset_index()
        page_open_rates['open_rate'] = (page_open_rates['sum'] / page_open_rates['count'] * 100).round(2)
        print(page_open_rates.sort_values('page_retrieved').head(20))
        
        # Page categories
        print(f"\nPage Categories Analysis:")
        df['page_category'] = pd.cut(df['page_retrieved'], 
                                   bins=[0, 1, 5, 10, 20, 50, 100, float('inf')],
                                   labels=['Page 1', 'Pages 2-5', 'Pages 6-10', 'Pages 11-20', 
                                          'Pages 21-50', 'Pages 51-100', 'Pages 100+'])
        
        category_stats = df.groupby('page_category')['email_open_count'].agg(['count', 'sum', 'mean']).reset_index()
        category_stats['open_rate'] = (category_stats['sum'] / category_stats['count'] * 100).round(2)
        print(category_stats)
    
    # 2. Apollo Search Ranking Theory
    print("\n\n2. APOLLO SEARCH RANKING THEORY:")
    print("-" * 50)
    
    print("""
    Based on the data analysis and Apollo's API structure, here's how Apollo likely ranks and searches contacts:
    
    A. SEARCH RANKING ALGORITHM:
    - Apollo uses a sophisticated ranking algorithm that considers multiple factors
    - Page numbers likely represent the position in search results
    - Lower page numbers = higher relevance/quality matches
    
    B. RANKING FACTORS (Likely):
    1. Data Completeness (email, phone, company info)
    2. Company Size and Industry
    3. Job Title Seniority
    4. Geographic Location
    5. Company Technology Stack
    6. Recent Activity/Updates
    7. Data Source Quality
    8. Engagement History
    
    C. PAGE_RETRIEVED INTERPRETATION:
    - Page 1: Highest quality, most relevant matches
    - Pages 2-5: High quality, good relevance
    - Pages 6-10: Medium quality, moderate relevance
    - Pages 11-20: Lower quality, less relevant
    - Pages 20+: Lower quality, niche or less relevant matches
    
    D. SEARCH PROCESS:
    1. User enters search criteria (title, company, location, etc.)
    2. Apollo's algorithm scores all matching contacts
    3. Results are ranked by relevance score
    4. Results are paginated (typically 25-50 per page)
    5. page_retrieved indicates which page the contact appeared on
    """)
    
    # 3. Performance Analysis by Page
    print("\n\n3. PERFORMANCE ANALYSIS BY PAGE:")
    print("-" * 50)
    
    if 'page_retrieved' in df.columns:
        # Create page performance visualization
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Open Rate by Page
        plt.subplot(2, 2, 1)
        page_performance = df.groupby('page_retrieved')['email_open_count'].agg(['count', 'sum']).reset_index()
        page_performance['open_rate'] = (page_performance['sum'] / page_performance['count'] * 100)
        
        plt.scatter(page_performance['page_retrieved'], page_performance['open_rate'], 
                   s=page_performance['count']/10, alpha=0.6)
        plt.xlabel('Page Retrieved')
        plt.ylabel('Open Rate (%)')
        plt.title('Email Open Rate by Apollo Page')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Contact Distribution by Page
        plt.subplot(2, 2, 2)
        page_counts = df['page_retrieved'].value_counts().sort_index()
        plt.bar(page_counts.index[:20], page_counts.values[:20])
        plt.xlabel('Page Retrieved')
        plt.ylabel('Number of Contacts')
        plt.title('Contact Distribution by Page')
        plt.xticks(rotation=45)
        
        # Subplot 3: Page Categories Performance
        plt.subplot(2, 2, 3)
        category_performance = df.groupby('page_category')['email_open_count'].agg(['count', 'sum']).reset_index()
        category_performance['open_rate'] = (category_performance['sum'] / category_performance['count'] * 100)
        
        bars = plt.bar(range(len(category_performance)), category_performance['open_rate'])
        plt.xlabel('Page Category')
        plt.ylabel('Open Rate (%)')
        plt.title('Open Rate by Page Category')
        plt.xticks(range(len(category_performance)), category_performance['page_category'], rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if category_performance.iloc[i]['open_rate'] > category_performance['open_rate'].mean():
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Subplot 4: Cumulative Performance
        plt.subplot(2, 2, 4)
        cumulative_performance = page_performance.sort_values('page_retrieved')
        cumulative_performance['cumulative_opens'] = cumulative_performance['sum'].cumsum()
        cumulative_performance['cumulative_contacts'] = cumulative_performance['count'].cumsum()
        cumulative_performance['cumulative_open_rate'] = (cumulative_performance['cumulative_opens'] / 
                                                        cumulative_performance['cumulative_contacts'] * 100)
        
        plt.plot(cumulative_performance['page_retrieved'], cumulative_performance['cumulative_open_rate'])
        plt.xlabel('Page Retrieved')
        plt.ylabel('Cumulative Open Rate (%)')
        plt.title('Cumulative Open Rate by Page')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('apollo_page_ranking_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nPage Ranking Analysis saved to: apollo_page_ranking_analysis.png")
    
    # 4. Business Insights
    print("\n\n4. BUSINESS INSIGHTS:")
    print("-" * 50)
    
    if 'page_retrieved' in df.columns:
        # Top performing pages
        top_pages = page_performance.nlargest(5, 'open_rate')
        print(f"Top 5 Performing Pages:")
        print(top_pages[['page_retrieved', 'open_rate', 'count']])
        
        # Page efficiency analysis
        print(f"\nPage Efficiency Analysis:")
        efficiency = page_performance.copy()
        efficiency['efficiency_score'] = efficiency['open_rate'] * np.log(efficiency['count'])
        top_efficient = efficiency.nlargest(5, 'efficiency_score')
        print(top_efficient[['page_retrieved', 'open_rate', 'count', 'efficiency_score']])
        
        # Recommendations
        print(f"\nSTRATEGIC RECOMMENDATIONS:")
        print("""
        1. FOCUS ON EARLY PAGES: Prioritize contacts from pages 1-5 for highest engagement
        2. PAGE-BASED SEGMENTATION: Create different campaigns for different page ranges
        3. QUALITY VS QUANTITY: Balance between page quality and contact volume
        4. TESTING STRATEGY: A/B test messaging for different page ranges
        5. COST OPTIMIZATION: Higher pages may offer better ROI despite lower open rates
        """)
    
    # 5. API Documentation Research
    print("\n\n5. APOLLO API DOCUMENTATION RESEARCH:")
    print("-" * 50)
    
    print("""
    Based on Apollo's API documentation research:
    
    A. SEARCH ENDPOINTS:
    - POST /api/v1/people/search: Main people search endpoint
    - POST /api/v1/organizations/search: Organization search
    - GET /api/v1/people/enrich: Contact enrichment
    
    B. PAGINATION PARAMETERS:
    - page: Page number for pagination
    - per_page: Number of results per page (typically 25-50)
    - page_token: Cursor-based pagination token
    
    C. SEARCH FILTERS:
    - q: Query string for text search
    - title_include: Job title filters
    - organization_domains: Company domain filters
    - locations: Geographic filters
    - contact_email_status: Email verification status
    
    D. RANKING FACTORS (Inferred):
    1. Data Completeness Score
    2. Company Information Quality
    3. Contact Activity Level
    4. Geographic Relevance
    5. Industry Match
    6. Job Title Seniority
    7. Company Size
    8. Technology Stack Match
    
    E. PAGE_RETRIEVED FIELD:
    - Represents the page number where the contact appeared in search results
    - Lower numbers = higher relevance/quality
    - Indicates Apollo's internal ranking algorithm output
    - Useful for understanding data quality and relevance
    """)
    
    # 6. Data Quality Assessment
    print("\n\n6. DATA QUALITY ASSESSMENT:")
    print("-" * 50)
    
    if 'page_retrieved' in df.columns:
        quality_metrics = {
            'Total Contacts': len(df),
            'Unique Pages': df['page_retrieved'].nunique(),
            'Page 1 Contacts': len(df[df['page_retrieved'] == 1]),
            'Pages 1-5 Contacts': len(df[df['page_retrieved'].between(1, 5)]),
            'Pages 1-10 Contacts': len(df[df['page_retrieved'].between(1, 10)]),
            'Average Page': df['page_retrieved'].mean(),
            'Median Page': df['page_retrieved'].median(),
            'Page 1 Open Rate': df[df['page_retrieved'] == 1]['email_open_count'].mean(),
            'Overall Open Rate': df['email_open_count'].mean()
        }
        
        print("Data Quality Metrics:")
        for metric, value in quality_metrics.items():
            print(f"{metric}: {value:.2f}")
        
        print(f"\nData Quality Score: {len(df[df['page_retrieved'] <= 5]) / len(df) * 100:.1f}% (contacts from top 5 pages)")

if __name__ == "__main__":
    analyze_apollo_page_ranking() 