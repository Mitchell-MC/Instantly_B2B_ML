"""
Granular Segmentation Analysis for B2B Email Targeting
Breaking down engagement patterns by Industry + Company Size + Seniority

This analysis will help Alex identify the most responsive, granular segments
to focus their targeting efforts on.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def create_segmentation_features(df):
    """Create segmentation features for analysis"""
    print("🔧 Creating segmentation features...")
    
    # 1. Company Size Categories
    if 'organization_employees' in df.columns:
        df['company_size'] = pd.cut(
            df['organization_employees'].fillna(-1),
            bins=[-1, 0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['Unknown', '1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
    else:
        df['company_size'] = 'Unknown'
    
    # 2. Industry Categories (group similar industries)
    if 'organization_industry' in df.columns:
        # Create industry categories
        industry_mapping = {
            'Software': ['Software', 'Technology', 'SaaS', 'IT Services', 'Computer Software'],
            'Finance': ['Financial Services', 'Banking', 'Insurance', 'Investment'],
            'Healthcare': ['Healthcare', 'Medical', 'Pharmaceuticals', 'Biotechnology'],
            'Manufacturing': ['Manufacturing', 'Industrial', 'Automotive', 'Aerospace'],
            'Retail': ['Retail', 'E-commerce', 'Consumer Goods'],
            'Consulting': ['Consulting', 'Professional Services', 'Management Consulting'],
            'Education': ['Education', 'Training', 'E-learning'],
            'Media': ['Media', 'Entertainment', 'Publishing', 'Marketing'],
            'Real Estate': ['Real Estate', 'Construction', 'Property'],
            'Energy': ['Energy', 'Oil & Gas', 'Utilities'],
            'Government': ['Government', 'Public Sector', 'Non-profit'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        df['industry_category'] = 'Other'
        for category, keywords in industry_mapping.items():
            mask = df['organization_industry'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'industry_category'] = category
    
    # 3. Seniority Categories
    if 'seniority' in df.columns:
        seniority_mapping = {
            'C-Level': ['C-Level', 'CEO', 'CTO', 'CFO', 'COO', 'CMO', 'CIO', 'Founder', 'Owner'],
            'VP/Director': ['VP', 'Vice President', 'Director', 'Head of'],
            'Manager': ['Manager', 'Lead', 'Senior Manager'],
            'Senior': ['Senior', 'Senior Engineer', 'Senior Developer', 'Senior Analyst'],
            'Mid-Level': ['Mid', 'Engineer', 'Developer', 'Analyst', 'Specialist'],
            'Junior': ['Junior', 'Entry', 'Associate'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        df['seniority_category'] = 'Other'
        for category, keywords in seniority_mapping.items():
            mask = df['seniority'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'seniority_category'] = category
    
    # 4. Engagement Metrics
    df['opened'] = (df['email_open_count'] > 0).astype(int)
    df['clicked'] = (df['email_click_count'] > 0).astype(int)
    df['replied'] = (df['email_reply_count'] > 0).astype(int)
    
    return df

def analyze_granular_segments(df):
    """Analyze engagement patterns by granular segments"""
    print("\n" + "="*100)
    print("GRANULAR SEGMENTATION ANALYSIS")
    print("="*100)
    
    # 1. Overall Segment Performance
    print("\n1. OVERALL SEGMENT PERFORMANCE:")
    print("-" * 80)
    
    # Create combined segments (convert categorical to string first)
    df['segment'] = df['industry_category'].astype(str) + ' | ' + df['company_size'].astype(str) + ' | ' + df['seniority_category'].astype(str)
    
    segment_performance = df.groupby('segment').agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    segment_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                 'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for segments with sufficient data (at least 50 contacts)
    significant_segments = segment_performance[segment_performance['contact_count'] >= 50]
    significant_segments = significant_segments.sort_values('open_rate', ascending=False)
    
    print(f"Top 20 Most Responsive Segments (min 50 contacts):")
    print(significant_segments.head(20))
    
    # 2. Industry + Company Size Analysis
    print("\n\n2. INDUSTRY + COMPANY SIZE ANALYSIS:")
    print("-" * 80)
    
    industry_size_performance = df.groupby(['industry_category', 'company_size']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    industry_size_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                       'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_industry_size = industry_size_performance[industry_size_performance['contact_count'] >= 30]
    significant_industry_size = significant_industry_size.sort_values('open_rate', ascending=False)
    
    print(f"Top Industry + Company Size Combinations (min 30 contacts):")
    print(significant_industry_size.head(15))
    
    # 3. Seniority within Company Size Analysis
    print("\n\n3. SENIORITY WITHIN COMPANY SIZE ANALYSIS:")
    print("-" * 80)
    
    seniority_size_performance = df.groupby(['company_size', 'seniority_category']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    seniority_size_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                        'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_seniority_size = seniority_size_performance[seniority_size_performance['contact_count'] >= 20]
    significant_seniority_size = significant_seniority_size.sort_values('open_rate', ascending=False)
    
    print(f"Top Seniority + Company Size Combinations (min 20 contacts):")
    print(significant_seniority_size.head(15))
    
    # 4. Industry + Seniority Analysis
    print("\n\n4. INDUSTRY + SENIORITY ANALYSIS:")
    print("-" * 80)
    
    industry_seniority_performance = df.groupby(['industry_category', 'seniority_category']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    industry_seniority_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                            'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_industry_seniority = industry_seniority_performance[industry_seniority_performance['contact_count'] >= 25]
    significant_industry_seniority = significant_industry_seniority.sort_values('open_rate', ascending=False)
    
    print(f"Top Industry + Seniority Combinations (min 25 contacts):")
    print(significant_industry_seniority.head(15))
    
    return significant_segments, significant_industry_size, significant_seniority_size, significant_industry_seniority

def analyze_campaign_patterns(df):
    """Analyze campaign-specific patterns within segments"""
    print("\n\n5. CAMPAIGN-SPECIFIC SEGMENT ANALYSIS:")
    print("-" * 80)
    
    if 'campaign_id' in df.columns:
        # Top performing campaigns by segment
        campaign_segment_performance = df.groupby(['campaign_id', 'segment']).agg({
            'opened': ['count', 'sum', 'mean']
        }).round(4)
        
        campaign_segment_performance.columns = ['contact_count', 'total_opens', 'open_rate']
        
        # Filter for significant campaign-segment combinations
        significant_campaign_segments = campaign_segment_performance[
            (campaign_segment_performance['contact_count'] >= 20) & 
            (campaign_segment_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"Top Campaign-Segment Combinations (min 20 contacts, 5%+ open rate):")
        print(significant_campaign_segments.head(10))
    
    if 'email_list' in df.columns:
        # Email list performance by segment
        list_segment_performance = df.groupby(['email_list', 'segment']).agg({
            'opened': ['count', 'sum', 'mean']
        }).round(4)
        
        list_segment_performance.columns = ['contact_count', 'total_opens', 'open_rate']
        
        # Filter for significant list-segment combinations
        significant_list_segments = list_segment_performance[
            (list_segment_performance['contact_count'] >= 15) & 
            (list_segment_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"\nTop Email List-Segment Combinations (min 15 contacts, 5%+ open rate):")
        print(significant_list_segments.head(10))

def create_visualizations(df, top_segments):
    """Create visualizations for the segmentation analysis"""
    print("\n\n6. CREATING VISUALIZATIONS:")
    print("-" * 80)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Granular Segmentation Analysis - Engagement Patterns', fontsize=16, fontweight='bold')
    
    # 1. Top Segments by Open Rate
    top_20_segments = top_segments.head(20)
    axes[0, 0].barh(range(len(top_20_segments)), top_20_segments['open_rate'])
    axes[0, 0].set_yticks(range(len(top_20_segments)))
    axes[0, 0].set_yticklabels([seg[:40] + '...' if len(seg) > 40 else seg for seg in top_20_segments.index])
    axes[0, 0].set_xlabel('Open Rate')
    axes[0, 0].set_title('Top 20 Segments by Open Rate')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Company Size Performance
    size_performance = df.groupby('company_size')['opened'].agg(['count', 'mean']).round(4)
    size_performance = size_performance[size_performance['count'] >= 50]
    axes[0, 1].bar(range(len(size_performance)), size_performance['mean'])
    axes[0, 1].set_xticks(range(len(size_performance)))
    axes[0, 1].set_xticklabels(size_performance.index, rotation=45)
    axes[0, 1].set_ylabel('Open Rate')
    axes[0, 1].set_title('Open Rate by Company Size')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Industry Performance
    industry_performance = df.groupby('industry_category')['opened'].agg(['count', 'mean']).round(4)
    industry_performance = industry_performance[industry_performance['count'] >= 100]
    axes[1, 0].bar(range(len(industry_performance)), industry_performance['mean'])
    axes[1, 0].set_xticks(range(len(industry_performance)))
    axes[1, 0].set_xticklabels(industry_performance.index, rotation=45)
    axes[1, 0].set_ylabel('Open Rate')
    axes[1, 0].set_title('Open Rate by Industry')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Seniority Performance
    seniority_performance = df.groupby('seniority_category')['opened'].agg(['count', 'mean']).round(4)
    seniority_performance = seniority_performance[seniority_performance['count'] >= 50]
    axes[1, 1].bar(range(len(seniority_performance)), seniority_performance['mean'])
    axes[1, 1].set_xticks(range(len(seniority_performance)))
    axes[1, 1].set_xticklabels(seniority_performance.index, rotation=45)
    axes[1, 1].set_ylabel('Open Rate')
    axes[1, 1].set_title('Open Rate by Seniority')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('granular_segmentation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'granular_segmentation_analysis.png'")

def generate_targeting_recommendations(top_segments, industry_size, seniority_size, industry_seniority):
    """Generate specific targeting recommendations for Alex"""
    print("\n\n7. TARGETING RECOMMENDATIONS FOR ALEX:")
    print("-" * 80)
    
    print("🎯 HIGH-PRIORITY SEGMENTS TO FOCUS ON:")
    print("=" * 50)
    
    # Top 10 overall segments
    print("\n1. TOP 10 OVERALL SEGMENTS (Industry + Company Size + Seniority):")
    for i, (segment, data) in enumerate(top_segments.head(10).iterrows(), 1):
        print(f"   {i}. {segment}")
        print(f"      📊 Open Rate: {data['open_rate']:.1%} | Contacts: {data['contact_count']:,}")
        print(f"      📈 Click Rate: {data['click_rate']:.1%} | Reply Rate: {data['reply_rate']:.1%}")
        print()
    
    print("\n2. TOP INDUSTRY + COMPANY SIZE COMBINATIONS:")
    for i, ((industry, size), data) in enumerate(industry_size.head(8).iterrows(), 1):
        print(f"   {i}. {industry} | {size}")
        print(f"      📊 Open Rate: {data['open_rate']:.1%} | Contacts: {data['contact_count']:,}")
        print()
    
    print("\n3. TOP SENIORITY + COMPANY SIZE COMBINATIONS:")
    for i, ((size, seniority), data) in enumerate(seniority_size.head(8).iterrows(), 1):
        print(f"   {i}. {size} | {seniority}")
        print(f"      📊 Open Rate: {data['open_rate']:.1%} | Contacts: {data['contact_count']:,}")
        print()
    
    print("\n4. TOP INDUSTRY + SENIORITY COMBINATIONS:")
    for i, ((industry, seniority), data) in enumerate(industry_seniority.head(8).iterrows(), 1):
        print(f"   {i}. {industry} | {seniority}")
        print(f"      📊 Open Rate: {data['open_rate']:.1%} | Contacts: {data['contact_count']:,}")
        print()
    
    print("\n🎯 STRATEGIC RECOMMENDATIONS:")
    print("=" * 50)
    
    # Identify patterns
    high_performing_industries = industry_size.groupby('industry_category')['open_rate'].mean().sort_values(ascending=False)
    high_performing_sizes = seniority_size.groupby('company_size')['open_rate'].mean().sort_values(ascending=False)
    high_performing_seniorities = industry_seniority.groupby('seniority_category')['open_rate'].mean().sort_values(ascending=False)
    
    print(f"\n📈 Best Performing Industries: {list(high_performing_industries.head(5).index)}")
    print(f"📈 Best Performing Company Sizes: {list(high_performing_sizes.head(5).index)}")
    print(f"📈 Best Performing Seniority Levels: {list(high_performing_seniorities.head(5).index)}")
    
    print("\n💡 TARGETING STRATEGY:")
    print("1. Focus on segments with 5%+ open rates and 50+ contacts")
    print("2. Prioritize combinations of high-performing industries, sizes, and seniorities")
    print("3. Create targeted campaigns for top 10 segments")
    print("4. Test messaging variations within high-performing segments")
    print("5. Scale successful segment combinations")

def main():
    """Main analysis pipeline"""
    print("=== GRANULAR SEGMENTATION ANALYSIS ===")
    print("🎯 Goal: Identify most responsive segments for Alex's targeting")
    print("=" * 80)
    
    # Load data
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: '{CSV_FILE_PATH}' not found.")
        return
    
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Create segmentation features
    df = create_segmentation_features(df)
    
    # Analyze granular segments
    top_segments, industry_size, seniority_size, industry_seniority = analyze_granular_segments(df)
    
    # Analyze campaign patterns
    analyze_campaign_patterns(df)
    
    # Create visualizations
    create_visualizations(df, top_segments)
    
    # Generate recommendations
    generate_targeting_recommendations(top_segments, industry_size, seniority_size, industry_seniority)
    
    print("\n" + "="*100)
    print("✅ GRANULAR SEGMENTATION ANALYSIS COMPLETE")
    print("📊 Check 'granular_segmentation_analysis.png' for visualizations")
    print("🎯 Use the recommendations above to focus your targeting efforts")
    print("="*100)

if __name__ == "__main__":
    main() 