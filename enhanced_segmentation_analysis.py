"""
Enhanced Granular Segmentation Analysis for B2B Email Targeting
Using additional data columns to improve categorization of "Other" companies

This analysis will help Alex identify the most responsive, granular segments
to focus their targeting efforts on, with improved categorization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def create_enhanced_segmentation_features(df):
    """Create enhanced segmentation features using additional data columns"""
    print("🔧 Creating enhanced segmentation features...")
    
    # 1. Enhanced Company Size Categories
    if 'organization_employees' in df.columns:
        df['company_size'] = pd.cut(
            df['organization_employees'].fillna(-1),
            bins=[-1, 0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['Unknown', '1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
    else:
        df['company_size'] = 'Unknown'
    
    # 2. Enhanced Industry Categories (using multiple data sources)
    df['industry_category'] = 'Other'
    
    # Primary industry categorization from organization_industry
    if 'organization_industry' in df.columns:
        # Enhanced industry mapping with more specific categories
        industry_mapping = {
            'Software': ['Software', 'Technology', 'SaaS', 'IT Services', 'Computer Software', 'information technology & services'],
            'Finance': ['Financial Services', 'Banking', 'Insurance', 'Investment', 'financial services', 'banking', 'insurance'],
            'Healthcare': ['Healthcare', 'Medical', 'Pharmaceuticals', 'Biotechnology', 'hospital & health care', 'pharmaceuticals'],
            'Manufacturing': ['Manufacturing', 'Industrial', 'Automotive', 'Aerospace', 'electrical/electronic manufacturing', 'machinery'],
            'Retail': ['Retail', 'E-commerce', 'Consumer Goods', 'retail'],
            'Consulting': ['Consulting', 'Professional Services', 'Management Consulting', 'management consulting'],
            'Education': ['Education', 'Training', 'E-learning', 'higher education', 'professional training & coaching'],
            'Media': ['Media', 'Entertainment', 'Publishing', 'Marketing', 'marketing & advertising'],
            'Real Estate': ['Real Estate', 'Construction', 'Property', 'real estate', 'construction'],
            'Energy': ['Energy', 'Oil & Gas', 'Utilities', 'oil & energy'],
            'Government': ['Government', 'Public Sector', 'Non-profit', 'government administration', 'nonprofit organization management'],
            'Telecommunications': ['Telecommunications', 'telecommunications'],
            'Mining': ['Mining', 'mining & metals'],
            'Logistics': ['Logistics', 'logistics & supply chain'],
            'Security': ['Security', 'computer & network security'],
            'Staffing': ['Staffing', 'staffing & recruiting'],
            'Venture Capital': ['Venture Capital', 'venture capital & private equity'],
            'Research': ['Research', 'research'],
            'Human Resources': ['Human Resources', 'human resources'],
            'Legal': ['Legal', 'law practice'],
            'Automotive': ['Automotive', 'automotive'],
            'Fitness': ['Fitness', 'health, wellness & fitness'],
            'Investment Management': ['Investment Management', 'investment management'],
            'Accounting': ['Accounting', 'accounting'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        for category, keywords in industry_mapping.items():
            mask = df['organization_industry'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'industry_category'] = category
    
    # 3. Enhanced Industry Categorization using organization_data JSON
    if 'organization_data' in df.columns:
        print("🔍 Using organization_data JSON for enhanced industry categorization...")
        
        # Extract industries from JSON data
        def extract_industries_from_json(json_str):
            try:
                if pd.isna(json_str) or json_str == '{}':
                    return []
                data = json.loads(json_str)
                if 'industries' in data:
                    return data['industries']
                elif 'industry' in data:
                    return [data['industry']]
                return []
            except:
                return []
        
        # Apply enhanced categorization using JSON data
        df['json_industries'] = df['organization_data'].apply(extract_industries_from_json)
        
        # Create additional industry categories based on JSON data
        for idx, row in df.iterrows():
            if row['industry_category'] == 'Other' and row['json_industries']:
                industries = row['json_industries']
                
                # Enhanced mapping for JSON industries
                json_industry_mapping = {
                    'Software': ['computer software', 'software', 'saas', 'information technology'],
                    'Finance': ['financial services', 'banking', 'insurance', 'investment'],
                    'Healthcare': ['healthcare', 'medical', 'pharmaceuticals', 'biotechnology'],
                    'Manufacturing': ['manufacturing', 'industrial', 'automotive', 'aerospace'],
                    'Retail': ['retail', 'e-commerce', 'consumer goods'],
                    'Consulting': ['consulting', 'professional services', 'management consulting'],
                    'Education': ['education', 'training', 'e-learning'],
                    'Media': ['media', 'entertainment', 'publishing', 'marketing'],
                    'Real Estate': ['real estate', 'construction', 'property'],
                    'Energy': ['energy', 'oil & gas', 'utilities'],
                    'Government': ['government', 'public sector', 'non-profit'],
                    'Telecommunications': ['telecommunications'],
                    'Mining': ['mining', 'metals'],
                    'Logistics': ['logistics', 'supply chain'],
                    'Security': ['security', 'cybersecurity'],
                    'Staffing': ['staffing', 'recruiting'],
                    'Venture Capital': ['venture capital', 'private equity'],
                    'Research': ['research'],
                    'Human Resources': ['human resources'],
                    'Legal': ['legal', 'law'],
                    'Automotive': ['automotive'],
                    'Fitness': ['fitness', 'wellness'],
                    'Investment Management': ['investment management'],
                    'Accounting': ['accounting']
                }
                
                for category, keywords in json_industry_mapping.items():
                    if any(keyword in ' '.join(industries).lower() for keyword in keywords):
                        df.at[idx, 'industry_category'] = category
                        break
    
    # 4. Enhanced Seniority Categories (using both seniority and title columns)
    if 'seniority' in df.columns:
        # Enhanced seniority mapping
        seniority_mapping = {
            'C-Level': ['c_suite', 'C-Level', 'CEO', 'CTO', 'CFO', 'COO', 'CMO', 'CIO', 'Founder', 'Owner', 'Chief'],
            'VP/Director': ['vp', 'VP', 'Vice President', 'Director', 'Head of', 'director', 'head'],
            'Manager': ['Manager', 'Lead', 'Senior Manager', 'manager'],
            'Senior': ['Senior', 'Senior Engineer', 'Senior Developer', 'Senior Analyst', 'senior'],
            'Mid-Level': ['Mid', 'Engineer', 'Developer', 'Analyst', 'Specialist', 'entry'],
            'Junior': ['Junior', 'Entry', 'Associate', 'intern'],
            'Other': ['Other', 'Unknown', 'nan', 'partner', 'owner']
        }
        
        df['seniority_category'] = 'Other'
        for category, keywords in seniority_mapping.items():
            mask = df['seniority'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'seniority_category'] = category
    
    # 5. Enhanced Seniority using title column
    if 'title' in df.columns:
        print("🔍 Using title column for enhanced seniority categorization...")
        
        # Enhanced title-based seniority mapping
        title_seniority_mapping = {
            'C-Level': ['CEO', 'CTO', 'CFO', 'COO', 'CMO', 'CIO', 'Chief', 'President', 'Founder', 'Owner'],
            'VP/Director': ['VP', 'Vice President', 'Director', 'Head of', 'Head'],
            'Manager': ['Manager', 'Lead', 'Senior Manager'],
            'Senior': ['Senior', 'Senior Engineer', 'Senior Developer', 'Senior Analyst'],
            'Mid-Level': ['Engineer', 'Developer', 'Analyst', 'Specialist'],
            'Junior': ['Junior', 'Entry', 'Associate', 'Intern'],
            'Other': ['Other', 'Unknown', 'Partner', 'Owner']
        }
        
        # Apply title-based categorization for entries still marked as 'Other'
        for idx, row in df.iterrows():
            if row['seniority_category'] == 'Other' and pd.notna(row['title']):
                title = str(row['title']).lower()
                
                for category, keywords in title_seniority_mapping.items():
                    if any(keyword.lower() in title for keyword in keywords):
                        df.at[idx, 'seniority_category'] = category
                        break
    
    # 6. Company Domain Analysis for Industry Inference
    if 'company_domain' in df.columns:
        print("🔍 Using company domain for additional industry inference...")
        
        # Domain-based industry inference
        domain_industry_mapping = {
            'Software': ['.io', '.ai', '.tech', '.app', '.dev', 'software', 'tech', 'saas'],
            'Finance': ['.bank', '.finance', '.capital', 'bank', 'finance', 'capital'],
            'Healthcare': ['.health', '.medical', '.care', 'health', 'medical', 'care'],
            'Education': ['.edu', '.education', '.learn', 'edu', 'education', 'learn'],
            'Government': ['.gov', '.org', 'gov', 'org'],
            'Media': ['.media', '.tv', '.news', 'media', 'tv', 'news'],
            'Retail': ['.shop', '.store', '.retail', 'shop', 'store', 'retail']
        }
        
        # Apply domain-based categorization for entries still marked as 'Other'
        for idx, row in df.iterrows():
            if row['industry_category'] == 'Other' and pd.notna(row['company_domain']):
                domain = str(row['company_domain']).lower()
                
                for category, keywords in domain_industry_mapping.items():
                    if any(keyword in domain for keyword in keywords):
                        df.at[idx, 'industry_category'] = category
                        break
    
    # 7. Engagement Metrics
    df['opened'] = (df['email_open_count'] > 0).astype(int)
    df['clicked'] = (df['email_click_count'] > 0).astype(int)
    df['replied'] = (df['email_reply_count'] > 0).astype(int)
    
    return df

def analyze_enhanced_segments(df):
    """Analyze engagement patterns by enhanced granular segments"""
    print("\n" + "="*100)
    print("ENHANCED GRANULAR SEGMENTATION ANALYSIS")
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

def analyze_categorization_improvements(df):
    """Analyze improvements in categorization"""
    print("\n\n5. CATEGORIZATION IMPROVEMENTS ANALYSIS:")
    print("-" * 80)
    
    # Count 'Other' categories
    industry_other_count = (df['industry_category'] == 'Other').sum()
    seniority_other_count = (df['seniority_category'] == 'Other').sum()
    
    print(f"📊 Current 'Other' Categories:")
    print(f"   Industry 'Other': {industry_other_count:,} ({industry_other_count/len(df)*100:.1f}%)")
    print(f"   Seniority 'Other': {seniority_other_count:,} ({seniority_other_count/len(df)*100:.1f}%)")
    
    # Show distribution of categories
    print(f"\n📊 Industry Category Distribution:")
    industry_dist = df['industry_category'].value_counts()
    for category, count in industry_dist.items():
        print(f"   {category}: {count:,} ({count/len(df)*100:.1f}%)")
    
    print(f"\n📊 Seniority Category Distribution:")
    seniority_dist = df['seniority_category'].value_counts()
    for category, count in seniority_dist.items():
        print(f"   {category}: {count:,} ({count/len(df)*100:.1f}%)")

def create_enhanced_visualizations(df, top_segments):
    """Create enhanced visualizations for the segmentation analysis"""
    print("\n\n6. CREATING ENHANCED VISUALIZATIONS:")
    print("-" * 80)
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Enhanced Granular Segmentation Analysis - Engagement Patterns', fontsize=16, fontweight='bold')
    
    # Define color palette for deciles (10% segments)
    decile_colors = ['#1e3a8a', '#3b82f6', '#60a5fa', '#dbeafe']  # Dark to light blue
    
    # Calculate overall mean for reference line
    overall_mean = df['opened'].mean()
    
    # 1. Top Segments by Open Rate (sorted by value, colored by sample size)
    top_20_segments = top_segments.head(20)
    
    # Calculate deciles for sample size coloring
    q90_count = top_20_segments['contact_count'].quantile(0.9)  # Top 10%
    q50_count = top_20_segments['contact_count'].quantile(0.5)  # Middle (50%)
    q10_count = top_20_segments['contact_count'].quantile(0.1)  # Bottom 10%
    
    colors = []
    for count in top_20_segments['contact_count']:
        if count >= q90_count:
            colors.append(decile_colors[0])  # Darkest blue (top 10%)
        elif count >= q50_count:
            colors.append(decile_colors[1])  # Medium dark blue (50-90%)
        elif count >= q10_count:
            colors.append(decile_colors[2])  # Medium light blue (10-50%)
        else:
            colors.append(decile_colors[3])  # Lightest blue (bottom 10%)
    
    axes[0, 0].barh(range(len(top_20_segments)), top_20_segments['open_rate'], color=colors)
    axes[0, 0].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes[0, 0].set_yticks(range(len(top_20_segments)))
    axes[0, 0].set_yticklabels([seg[:40] + '...' if len(seg) > 40 else seg for seg in top_20_segments.index])
    axes[0, 0].set_xlabel('Open Rate')
    axes[0, 0].set_title('Top 20 Segments by Open Rate\n(Darker=Higher Sample Size)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add separate legend for top left chart
    top_left_legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[0], label=f'Top 10%: {q90_count:.0f}+ contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[1], label=f'50-90%: {q50_count:.0f}-{q90_count:.0f} contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[2], label=f'10-50%: {q10_count:.0f}-{q50_count:.0f} contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[3], label=f'Bottom 10%: <{q10_count:.0f} contacts')
    ]
    
    # Add legend to the top left chart
    axes[0, 0].legend(handles=top_left_legend_elements, loc='lower right', 
                       title='Top 20 Segments Sample Size', title_fontsize=10, fontsize=9)
    
    # 2. Company Size Performance (sorted by value, colored by sample size)
    size_performance = df.groupby('company_size')['opened'].agg(['count', 'mean']).round(4)
    size_performance = size_performance[size_performance['count'] >= 50].sort_values('mean', ascending=False)
    
    # Calculate deciles for sample size coloring
    q90_count = size_performance['count'].quantile(0.9)  # Top 10%
    q50_count = size_performance['count'].quantile(0.5)  # Middle (50%)
    q10_count = size_performance['count'].quantile(0.1)  # Bottom 10%
    
    colors = []
    for count in size_performance['count']:
        if count >= q90_count:
            colors.append(decile_colors[0])  # Darkest blue (top 10%)
        elif count >= q50_count:
            colors.append(decile_colors[1])  # Medium dark blue (50-90%)
        elif count >= q10_count:
            colors.append(decile_colors[2])  # Medium light blue (10-50%)
        else:
            colors.append(decile_colors[3])  # Lightest blue (bottom 10%)
    
    axes[0, 1].bar(range(len(size_performance)), size_performance['mean'], color=colors)
    axes[0, 1].axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes[0, 1].set_xticks(range(len(size_performance)))
    axes[0, 1].set_xticklabels(size_performance.index, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Open Rate')
    axes[0, 1].set_title('Open Rate by Company Size\n(Darker=Higher Sample Size)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Adjust layout to prevent label overlap
    plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # 3. Industry Performance (sorted by value, colored by sample size)
    industry_performance = df.groupby('industry_category')['opened'].agg(['count', 'mean']).round(4)
    industry_performance = industry_performance[industry_performance['count'] >= 100].sort_values('mean', ascending=False)
    
    # Calculate deciles for sample size coloring
    q90_count = industry_performance['count'].quantile(0.9)  # Top 10%
    q50_count = industry_performance['count'].quantile(0.5)  # Middle (50%)
    q10_count = industry_performance['count'].quantile(0.1)  # Bottom 10%
    
    colors = []
    for count in industry_performance['count']:
        if count >= q90_count:
            colors.append(decile_colors[0])  # Darkest blue (top 10%)
        elif count >= q50_count:
            colors.append(decile_colors[1])  # Medium dark blue (50-90%)
        elif count >= q10_count:
            colors.append(decile_colors[2])  # Medium light blue (10-50%)
        else:
            colors.append(decile_colors[3])  # Lightest blue (bottom 10%)
    
    axes[1, 0].bar(range(len(industry_performance)), industry_performance['mean'], color=colors)
    axes[1, 0].axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes[1, 0].set_xticks(range(len(industry_performance)))
    axes[1, 0].set_xticklabels(industry_performance.index, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Open Rate')
    axes[1, 0].set_title('Open Rate by Industry\n(Darker=Higher Sample Size)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Adjust layout to prevent label overlap
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # 4. Seniority Performance (sorted by value, colored by sample size)
    seniority_performance = df.groupby('seniority_category')['opened'].agg(['count', 'mean']).round(4)
    seniority_performance = seniority_performance[seniority_performance['count'] >= 50].sort_values('mean', ascending=False)
    
    # Calculate deciles for sample size coloring
    q90_count = seniority_performance['count'].quantile(0.9)  # Top 10%
    q50_count = seniority_performance['count'].quantile(0.5)  # Middle (50%)
    q10_count = seniority_performance['count'].quantile(0.1)  # Bottom 10%
    
    colors = []
    for count in seniority_performance['count']:
        if count >= q90_count:
            colors.append(decile_colors[0])  # Darkest blue (top 10%)
        elif count >= q50_count:
            colors.append(decile_colors[1])  # Medium dark blue (50-90%)
        elif count >= q10_count:
            colors.append(decile_colors[2])  # Medium light blue (10-50%)
        else:
            colors.append(decile_colors[3])  # Lightest blue (bottom 10%)
    
    axes[1, 1].bar(range(len(seniority_performance)), seniority_performance['mean'], color=colors)
    axes[1, 1].axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes[1, 1].set_xticks(range(len(seniority_performance)))
    axes[1, 1].set_xticklabels(seniority_performance.index, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Open Rate')
    axes[1, 1].set_title('Open Rate by Seniority\n(Darker=Higher Sample Size)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    # Adjust layout to prevent label overlap
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha='right', fontsize=9)
    
    # Add second legend for the other three charts (Company Size, Industry, Seniority)
    other_charts_legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[0], label=f'Top 10%: {q90_count:.0f}+ contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[1], label=f'50-90%: {q50_count:.0f}-{q90_count:.0f} contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[2], label=f'10-50%: {q10_count:.0f}-{q50_count:.0f} contacts'),
        plt.Rectangle((0,0),1,1, facecolor=decile_colors[3], label=f'Bottom 10%: <{q10_count:.0f} contacts')
    ]
    
    # Add legend to the figure for the other three charts
    fig.legend(handles=other_charts_legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
               title='Other Charts Sample Size', title_fontsize=12, fontsize=10)
    
    # Adjust layout to prevent label overlap
    plt.tight_layout(pad=2.0)
    plt.savefig('enhanced_granular_segmentation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Enhanced visualization saved as 'enhanced_granular_segmentation_analysis.png'")

def main():
    """Main enhanced analysis pipeline"""
    print("=== ENHANCED GRANULAR SEGMENTATION ANALYSIS ===")
    print("🎯 Goal: Identify most responsive segments with improved categorization")
    print("=" * 80)
    
    # Load data
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: '{CSV_FILE_PATH}' not found.")
        return
    
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Create enhanced segmentation features
    df = create_enhanced_segmentation_features(df)
    
    # Analyze categorization improvements
    analyze_categorization_improvements(df)
    
    # Analyze enhanced granular segments
    top_segments, industry_size, seniority_size, industry_seniority = analyze_enhanced_segments(df)
    
    # Create enhanced visualizations
    create_enhanced_visualizations(df, top_segments)
    
    print("\n" + "="*100)
    print("✅ ENHANCED GRANULAR SEGMENTATION ANALYSIS COMPLETE")
    print("📊 Check 'enhanced_granular_segmentation_analysis.png' for visualizations")
    print("🎯 Enhanced categorization provides better insights for targeting")
    print("="*100)

if __name__ == "__main__":
    main() 