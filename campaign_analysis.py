"""
Comprehensive Campaign Analysis for B2B Email Targeting
Analyzing engagement patterns across different campaign_id categories

This analysis will help identify which campaigns perform best for different segments
and provide insights for campaign optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def load_and_prepare_data():
    """Load and prepare data for campaign analysis"""
    print("🔧 Loading and preparing campaign data...")
    
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: '{CSV_FILE_PATH}' not found.")
        return None
    
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Create engagement metrics
    df['opened'] = (df['email_open_count'] > 0).astype(int)
    df['clicked'] = (df['email_click_count'] > 0).astype(int)
    df['replied'] = (df['email_reply_count'] > 0).astype(int)
    
    # Check if campaign_id exists
    if 'campaign_id' not in df.columns:
        print("❌ Error: 'campaign_id' column not found in dataset.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    return df

def analyze_campaign_overview(df):
    """Analyze overall campaign performance and distribution"""
    print("\n" + "="*100)
    print("CAMPAIGN OVERVIEW ANALYSIS")
    print("="*100)
    
    # 1. Campaign Distribution
    print("\n1. CAMPAIGN DISTRIBUTION:")
    print("-" * 80)
    
    campaign_counts = df['campaign_id'].value_counts()
    print(f"Total unique campaigns: {len(campaign_counts)}")
    print(f"Total contacts: {len(df):,}")
    print(f"Average contacts per campaign: {len(df)/len(campaign_counts):.1f}")
    
    print(f"\nTop 20 campaigns by contact count:")
    print(campaign_counts.head(20))
    
    # 2. Campaign Performance Metrics
    print("\n\n2. CAMPAIGN PERFORMANCE METRICS:")
    print("-" * 80)
    
    campaign_performance = df.groupby('campaign_id').agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    campaign_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                  'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for campaigns with sufficient data
    significant_campaigns = campaign_performance[campaign_performance['contact_count'] >= 50]
    significant_campaigns = significant_campaigns.sort_values('open_rate', ascending=False)
    
    print(f"Campaigns with 50+ contacts: {len(significant_campaigns)}")
    print(f"\nTop 20 campaigns by open rate:")
    print(significant_campaigns.head(20))
    
    # 3. Campaign Performance Statistics
    print(f"\n\n3. CAMPAIGN PERFORMANCE STATISTICS:")
    print("-" * 80)
    
    print(f"Overall dataset open rate: {df['opened'].mean():.3f}")
    print(f"Overall dataset click rate: {df['clicked'].mean():.3f}")
    print(f"Overall dataset reply rate: {df['replied'].mean():.3f}")
    
    print(f"\nCampaign performance statistics:")
    print(f"  Average campaign open rate: {significant_campaigns['open_rate'].mean():.3f}")
    print(f"  Average campaign click rate: {significant_campaigns['click_rate'].mean():.3f}")
    print(f"  Average campaign reply rate: {significant_campaigns['reply_rate'].mean():.3f}")
    print(f"  Best campaign open rate: {significant_campaigns['open_rate'].max():.3f}")
    print(f"  Worst campaign open rate: {significant_campaigns['open_rate'].min():.3f}")
    
    return significant_campaigns

def analyze_campaign_segments(df):
    """Analyze campaign performance across different segments using enhanced methodology"""
    print("\n\n4. CAMPAIGN-SEGMENT ANALYSIS:")
    print("-" * 80)
    
    # Create enhanced segments for analysis
    df['company_size'] = 'Unknown'
    if 'organization_employees' in df.columns:
        df['company_size'] = pd.cut(
            df['organization_employees'].fillna(-1),
            bins=[-1, 0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['Unknown', '1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
    
    # Enhanced Industry Categorization (using comprehensive logic from enhanced_segmentation_analysis.py)
    df['industry_category'] = 'Other'
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
        
        # Apply basic industry categorization
        for category, keywords in industry_mapping.items():
            mask = df['organization_industry'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'industry_category'] = category
        
        # Enhanced categorization using organization_data JSON
        if 'organization_data' in df.columns:
            try:
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
            except Exception as e:
                print(f"Warning: Error parsing organization_data: {e}")
        
        # Additional categorization using company_domain
        if 'company_domain' in df.columns:
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
    
    # Enhanced Seniority Categorization (using comprehensive logic from enhanced_segmentation_analysis.py)
    df['seniority_category'] = 'Other'
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
        
        # Apply basic seniority categorization
        for category, keywords in seniority_mapping.items():
            mask = df['seniority'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'seniority_category'] = category
        
        # Enhanced categorization using title column
        if 'title' in df.columns:
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
    
    # 4.1 Campaign + Company Size Analysis
    print("\n4.1 CAMPAIGN + COMPANY SIZE ANALYSIS:")
    print("-" * 50)
    
    campaign_size_performance = df.groupby(['campaign_id', 'company_size']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    campaign_size_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                       'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_campaign_size = campaign_size_performance[
        (campaign_size_performance['contact_count'] >= 20) & 
        (campaign_size_performance['open_rate'] >= 0.05)
    ].sort_values('open_rate', ascending=False)
    
    print(f"Top Campaign + Company Size Combinations (min 20 contacts, 5%+ open rate):")
    print(significant_campaign_size.head(15))
    
    # 4.2 Campaign + Industry Analysis
    print("\n\n4.2 CAMPAIGN + INDUSTRY ANALYSIS:")
    print("-" * 50)
    
    campaign_industry_performance = df.groupby(['campaign_id', 'industry_category']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    campaign_industry_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                           'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_campaign_industry = campaign_industry_performance[
        (campaign_industry_performance['contact_count'] >= 25) & 
        (campaign_industry_performance['open_rate'] >= 0.05)
    ].sort_values('open_rate', ascending=False)
    
    print(f"Top Campaign + Industry Combinations (min 25 contacts, 5%+ open rate):")
    print(significant_campaign_industry.head(15))
    
    # 4.3 Campaign + Seniority Analysis
    print("\n\n4.3 CAMPAIGN + SENIORITY ANALYSIS:")
    print("-" * 50)
    
    campaign_seniority_performance = df.groupby(['campaign_id', 'seniority_category']).agg({
        'opened': ['count', 'sum', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    
    campaign_seniority_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                            'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    
    # Filter for significant combinations
    significant_campaign_seniority = campaign_seniority_performance[
        (campaign_seniority_performance['contact_count'] >= 20) & 
        (campaign_seniority_performance['open_rate'] >= 0.05)
    ].sort_values('open_rate', ascending=False)
    
    print(f"Top Campaign + Seniority Combinations (min 20 contacts, 5%+ open rate):")
    print(significant_campaign_seniority.head(15))
    
    return significant_campaign_size, significant_campaign_industry, significant_campaign_seniority

def analyze_campaign_features(df):
    """Analyze campaign performance by various features"""
    print("\n\n5. CAMPAIGN FEATURE ANALYSIS:")
    print("-" * 80)
    
    # 5.1 Campaign + Title Analysis
    if 'title' in df.columns:
        print("\n5.1 CAMPAIGN + TITLE ANALYSIS:")
        print("-" * 50)
        
        # Create title categories for analysis
        df['title_category'] = 'Other'
        
        title_mapping = {
            'CEO/Founder': ['CEO', 'Chief Executive Officer', 'Founder', 'Co-Founder', 'President', 'Owner'],
            'CTO/Technical': ['CTO', 'Chief Technology Officer', 'Technical Director', 'VP Engineering', 'Head of Engineering'],
            'CFO/Finance': ['CFO', 'Chief Financial Officer', 'Finance Director', 'VP Finance', 'Head of Finance'],
            'CMO/Marketing': ['CMO', 'Chief Marketing Officer', 'Marketing Director', 'VP Marketing', 'Head of Marketing'],
            'Sales': ['Sales Director', 'VP Sales', 'Head of Sales', 'Sales Manager', 'Business Development'],
            'Product': ['Product Manager', 'Product Director', 'VP Product', 'Head of Product'],
            'HR': ['HR Director', 'VP HR', 'Head of HR', 'Human Resources', 'Talent'],
            'Operations': ['Operations Director', 'VP Operations', 'Head of Operations', 'COO'],
            'Consultant': ['Consultant', 'Advisor', 'Partner', 'Principal'],
            'Developer': ['Developer', 'Engineer', 'Software Engineer', 'Programmer'],
            'Analyst': ['Analyst', 'Data Analyst', 'Business Analyst', 'Research Analyst'],
            'Manager': ['Manager', 'Team Lead', 'Supervisor'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        for category, keywords in title_mapping.items():
            mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'title_category'] = category
        
        campaign_title_performance = df.groupby(['campaign_id', 'title_category']).agg({
            'opened': ['count', 'sum', 'mean'],
            'clicked': ['sum', 'mean'],
            'replied': ['sum', 'mean']
        }).round(4)
        
        campaign_title_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                           'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
        
        significant_campaign_title = campaign_title_performance[
            (campaign_title_performance['contact_count'] >= 20) & 
            (campaign_title_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"Top Campaign + Title Combinations (min 20 contacts, 5%+ open rate):")
        print(significant_campaign_title.head(15))
    
    # 5.2 Campaign + Email List Analysis
    if 'email_list' in df.columns:
        print("\n\n5.2 CAMPAIGN + EMAIL LIST ANALYSIS:")
        print("-" * 50)
        
        campaign_list_performance = df.groupby(['campaign_id', 'email_list']).agg({
            'opened': ['count', 'sum', 'mean'],
            'clicked': ['sum', 'mean'],
            'replied': ['sum', 'mean']
        }).round(4)
        
        campaign_list_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                           'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
        
        significant_campaign_list = campaign_list_performance[
            (campaign_list_performance['contact_count'] >= 15) & 
            (campaign_list_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"Top Campaign + Email List Combinations (min 15 contacts, 5%+ open rate):")
        print(significant_campaign_list.head(15))
    
    # 5.3 Campaign + Country Analysis
    if 'country' in df.columns:
        print("\n\n5.3 CAMPAIGN + COUNTRY ANALYSIS:")
        print("-" * 50)
        
        # Focus on top countries
        top_countries = df['country'].value_counts().head(10).index
        df_top_countries = df[df['country'].isin(top_countries)]
        
        campaign_country_performance = df_top_countries.groupby(['campaign_id', 'country']).agg({
            'opened': ['count', 'sum', 'mean'],
            'clicked': ['sum', 'mean'],
            'replied': ['sum', 'mean']
        }).round(4)
        
        campaign_country_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                              'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
        
        significant_campaign_country = campaign_country_performance[
            (campaign_country_performance['contact_count'] >= 20) & 
            (campaign_country_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"Top Campaign + Country Combinations (min 20 contacts, 5%+ open rate):")
        print(significant_campaign_country.head(15))
    
    # 5.4 Campaign + ESP Code Analysis
    if 'esp_code' in df.columns:
        print("\n\n5.4 CAMPAIGN + ESP CODE ANALYSIS:")
        print("-" * 50)
        
        campaign_esp_performance = df.groupby(['campaign_id', 'esp_code']).agg({
            'opened': ['count', 'sum', 'mean'],
            'clicked': ['sum', 'mean'],
            'replied': ['sum', 'mean']
        }).round(4)
        
        campaign_esp_performance.columns = ['contact_count', 'total_opens', 'open_rate', 
                                          'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
        
        significant_campaign_esp = campaign_esp_performance[
            (campaign_esp_performance['contact_count'] >= 20) & 
            (campaign_esp_performance['open_rate'] >= 0.05)
        ].sort_values('open_rate', ascending=False)
        
        print(f"Top Campaign + ESP Code Combinations (min 20 contacts, 5%+ open rate):")
        print(significant_campaign_esp.head(15))

def create_campaign_visualizations(df, significant_campaigns):
    """Create visualizations for campaign analysis"""
    print("\n\n6. CREATING CAMPAIGN VISUALIZATIONS:")
    print("-" * 80)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create two separate figures: one for overall campaign analysis and one for targeting analysis
    # Figure 1: Overall Campaign Analysis (2x2 layout)
    fig1, axes1 = plt.subplots(2, 2, figsize=(20, 16))
    fig1.suptitle('Overall Campaign Performance Analysis', fontsize=16, fontweight='bold')
    
    # Figure 2: Campaign Targeting Analysis (1x3 layout)
    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.suptitle('Campaign Targeting Analysis - Performance by Segment', fontsize=16, fontweight='bold')
    
    # Define color palette
    colors = ['#1e3a8a', '#3b82f6', '#60a5fa', '#dbeafe']  # Blue shades
    
    # Calculate overall mean for reference line
    overall_mean = df['opened'].mean()
    
    # ===== FIGURE 1: OVERALL CAMPAIGN ANALYSIS =====
    
    # 1. Top 20 Campaigns by Open Rate
    top_20_campaigns = significant_campaigns.head(20)
    y_pos = range(len(top_20_campaigns))
    colors_list = []
    for count in top_20_campaigns['contact_count']:
        if count >= 100:
            colors_list.append(colors[0])  # Darkest blue
        elif count >= 50:
            colors_list.append(colors[1])  # Medium dark blue
        elif count >= 20:
            colors_list.append(colors[2])  # Medium light blue
        else:
            colors_list.append(colors[3])  # Lightest blue
    
    axes1[0, 0].barh(y_pos, top_20_campaigns['open_rate'], color=colors_list)
    axes1[0, 0].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes1[0, 0].set_yticks(y_pos)
    axes1[0, 0].set_yticklabels([f"{camp_id[:8]} ({count:.0f})" for camp_id, count in zip(top_20_campaigns.index, top_20_campaigns['contact_count'])])
    axes1[0, 0].set_xlabel('Open Rate')
    axes1[0, 0].set_title('Top 20 Campaigns by Open Rate\n(Darker=Higher Sample Size)')
    axes1[0, 0].grid(True, alpha=0.3)
    axes1[0, 0].legend()
    
    # 2. Distribution of Campaign Open Rates
    all_campaigns = df.groupby('campaign_id')['opened'].agg(['count', 'mean']).round(4)
    all_campaigns.columns = ['contact_count', 'open_rate']
    all_campaigns = all_campaigns[all_campaigns['contact_count'] >= 10]  # Filter for significant campaigns
    
    axes1[0, 1].hist(all_campaigns['open_rate'], bins=30, alpha=0.7, color='#3b82f6', edgecolor='black')
    axes1[0, 1].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes1[0, 1].set_xlabel('Open Rate')
    axes1[0, 1].set_ylabel('Number of Campaigns')
    axes1[0, 1].set_title('Distribution of Campaign Open Rates\n(Min 10 contacts)')
    axes1[0, 1].grid(True, alpha=0.3)
    axes1[0, 1].legend()
    
    # 3. Campaign Size vs Performance
    axes1[1, 0].scatter(all_campaigns['contact_count'], all_campaigns['open_rate'], alpha=0.6, color='#3b82f6')
    axes1[1, 0].axhline(y=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
    axes1[1, 0].set_xlabel('Number of Contacts')
    axes1[1, 0].set_ylabel('Open Rate')
    axes1[1, 0].set_title('Campaign Size vs Performance')
    axes1[1, 0].grid(True, alpha=0.3)
    axes1[1, 0].legend()
    
    # 4. Average Campaign Performance Metrics
    campaign_metrics = df.groupby('campaign_id').agg({
        'opened': ['count', 'mean'],
        'clicked': ['sum', 'mean'],
        'replied': ['sum', 'mean']
    }).round(4)
    campaign_metrics.columns = ['contact_count', 'open_rate', 'total_clicks', 'click_rate', 'total_replies', 'reply_rate']
    campaign_metrics = campaign_metrics[campaign_metrics['contact_count'] >= 10]
    
    metrics_summary = campaign_metrics[['open_rate', 'click_rate', 'reply_rate']].mean()
    
    metrics_names = ['Open Rate', 'Click Rate', 'Reply Rate']
    y_pos = range(len(metrics_names))
    colors_list = ['#1e3a8a', '#3b82f6', '#60a5fa']
    
    axes1[1, 1].bar(y_pos, metrics_summary.values, color=colors_list)
    axes1[1, 1].set_xticks(y_pos)
    axes1[1, 1].set_xticklabels(metrics_names)
    axes1[1, 1].set_ylabel('Average Rate')
    axes1[1, 1].set_title('Average Campaign Performance Metrics\n(Min 10 contacts)')
    axes1[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(metrics_summary.values):
        axes1[1, 1].text(i, v + 0.001, f'{v:.3f}', ha='center', va='bottom')
    
    # Adjust layout for first figure
    plt.figure(fig1.number)
    plt.tight_layout(pad=2.0)
    plt.savefig('campaign_overall_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Overall campaign analysis saved as 'campaign_overall_analysis.png'")
    
    # ===== FIGURE 2: CAMPAIGN TARGETING ANALYSIS =====
    
    # 1. Campaign Performance by Industry (Top 15 campaigns)
    if 'organization_industry' in df.columns:
        # Enhanced industry categorization (using comprehensive logic from enhanced_segmentation_analysis.py)
        df['industry_category'] = 'Other'
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
        
        # Apply basic industry categorization
        for category, keywords in industry_mapping.items():
            mask = df['organization_industry'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'industry_category'] = category
        
        # Enhanced categorization using organization_data JSON
        if 'organization_data' in df.columns:
            try:
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
            except Exception as e:
                print(f"Warning: Error parsing organization_data: {e}")
        
        # Additional categorization using company_domain
        if 'company_domain' in df.columns:
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
        
        # Get top 15 campaigns and their industry performance
        top_15_campaigns = significant_campaigns.head(15).index
        campaign_industry_data = df[df['campaign_id'].isin(top_15_campaigns)].groupby(['campaign_id', 'industry_category'])['opened'].agg(['count', 'mean']).round(4)
        campaign_industry_data = campaign_industry_data[campaign_industry_data['count'] >= 10].sort_values('mean', ascending=False).head(20)
        
        # Create horizontal bar chart
        y_pos = range(len(campaign_industry_data))
        colors_list = []
        for count in campaign_industry_data['count']:
            if count >= 100:
                colors_list.append(colors[0])  # Darkest blue
            elif count >= 50:
                colors_list.append(colors[1])  # Medium dark blue
            elif count >= 20:
                colors_list.append(colors[2])  # Medium light blue
            else:
                colors_list.append(colors[3])  # Lightest blue
        
        axes2[0].barh(y_pos, campaign_industry_data['mean'], color=colors_list)
        axes2[0].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
        axes2[0].set_yticks(y_pos)
        axes2[0].set_yticklabels([f"{camp_id[:8]} + {ind}" for (camp_id, ind) in campaign_industry_data.index])
        axes2[0].set_xlabel('Open Rate')
        axes2[0].set_title('Top Campaign + Industry Combinations\n(Darker=Higher Sample Size)')
        axes2[0].grid(True, alpha=0.3)
        axes2[0].legend()
    
    # 2. Campaign Performance by Company Size (Top 15 campaigns)
    if 'organization_employees' in df.columns:
        # Create company size categories
        df['company_size'] = pd.cut(
            df['organization_employees'].fillna(-1),
            bins=[-1, 0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['Unknown', '1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
        
        # Get top 15 campaigns and their company size performance
        top_15_campaigns = significant_campaigns.head(15).index
        campaign_size_data = df[df['campaign_id'].isin(top_15_campaigns)].groupby(['campaign_id', 'company_size'])['opened'].agg(['count', 'mean']).round(4)
        campaign_size_data = campaign_size_data[campaign_size_data['count'] >= 10].sort_values('mean', ascending=False).head(20)
        
        # Create horizontal bar chart
        y_pos = range(len(campaign_size_data))
        colors_list = []
        for count in campaign_size_data['count']:
            if count >= 100:
                colors_list.append(colors[0])  # Darkest blue
            elif count >= 50:
                colors_list.append(colors[1])  # Medium dark blue
            elif count >= 20:
                colors_list.append(colors[2])  # Medium light blue
            else:
                colors_list.append(colors[3])  # Lightest blue
        
        axes2[1].barh(y_pos, campaign_size_data['mean'], color=colors_list)
        axes2[1].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
        axes2[1].set_yticks(y_pos)
        axes2[1].set_yticklabels([f"{camp_id[:8]} + {size}" for (camp_id, size) in campaign_size_data.index])
        axes2[1].set_xlabel('Open Rate')
        axes2[1].set_title('Top Campaign + Company Size Combinations\n(Darker=Higher Sample Size)')
        axes2[1].grid(True, alpha=0.3)
        axes2[1].legend()
    
    # 3. Campaign Performance by Title (Top 15 campaigns)
    if 'title' in df.columns:
        # Enhanced title categorization using both title and seniority columns
        df['title_category'] = 'Other'
        title_mapping = {
            'CEO/Founder': ['CEO', 'Chief Executive Officer', 'Founder', 'Co-Founder', 'President', 'Owner'],
            'CTO/Technical': ['CTO', 'Chief Technology Officer', 'Technical Director', 'VP Engineering', 'Head of Engineering'],
            'CFO/Finance': ['CFO', 'Chief Financial Officer', 'Finance Director', 'VP Finance', 'Head of Finance'],
            'CMO/Marketing': ['CMO', 'Chief Marketing Officer', 'Marketing Director', 'VP Marketing', 'Head of Marketing'],
            'Sales': ['Sales Director', 'VP Sales', 'Head of Sales', 'Sales Manager', 'Business Development'],
            'Product': ['Product Manager', 'Product Director', 'VP Product', 'Head of Product'],
            'HR': ['HR Director', 'VP HR', 'Head of HR', 'Human Resources', 'Talent'],
            'Operations': ['Operations Director', 'VP Operations', 'Head of Operations', 'COO'],
            'Consultant': ['Consultant', 'Advisor', 'Partner', 'Principal'],
            'Developer': ['Developer', 'Engineer', 'Software Engineer', 'Programmer'],
            'Analyst': ['Analyst', 'Data Analyst', 'Business Analyst', 'Research Analyst'],
            'Manager': ['Manager', 'Team Lead', 'Supervisor'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        # Apply basic title categorization
        for category, keywords in title_mapping.items():
            mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'title_category'] = category
        
        # Enhanced categorization using seniority column
        if 'seniority' in df.columns:
            seniority_title_mapping = {
                'CEO/Founder': ['c_suite', 'C-Level', 'CEO', 'CTO', 'CFO', 'COO', 'CMO', 'CIO', 'Founder', 'Owner', 'Chief'],
                'VP/Director': ['vp', 'VP', 'Vice President', 'Director', 'Head of', 'director', 'head'],
                'Manager': ['Manager', 'Lead', 'Senior Manager', 'manager'],
                'Senior': ['Senior', 'Senior Engineer', 'Senior Developer', 'Senior Analyst', 'senior'],
                'Mid-Level': ['Mid', 'Engineer', 'Developer', 'Analyst', 'Specialist', 'entry'],
                'Junior': ['Junior', 'Entry', 'Associate', 'intern'],
                'Other': ['Other', 'Unknown', 'nan', 'partner', 'owner']
            }
            
            # Apply title-based categorization for entries still marked as 'Other'
            for idx, row in df.iterrows():
                if row['title_category'] == 'Other' and pd.notna(row['seniority']):
                    seniority = str(row['seniority']).lower()
                    
                    for category, keywords in seniority_title_mapping.items():
                        if any(keyword.lower() in seniority for keyword in keywords):
                            df.at[idx, 'title_category'] = category
                            break
        
        # Get top 15 campaigns and their title performance
        top_15_campaigns = significant_campaigns.head(15).index
        campaign_title_data = df[df['campaign_id'].isin(top_15_campaigns)].groupby(['campaign_id', 'title_category'])['opened'].agg(['count', 'mean']).round(4)
        campaign_title_data = campaign_title_data[campaign_title_data['count'] >= 10].sort_values('mean', ascending=False).head(20)
        
        # Create horizontal bar chart
        y_pos = range(len(campaign_title_data))
        colors_list = []
        for count in campaign_title_data['count']:
            if count >= 100:
                colors_list.append(colors[0])  # Darkest blue
            elif count >= 50:
                colors_list.append(colors[1])  # Medium dark blue
            elif count >= 20:
                colors_list.append(colors[2])  # Medium light blue
            else:
                colors_list.append(colors[3])  # Lightest blue
        
        axes2[2].barh(y_pos, campaign_title_data['mean'], color=colors_list)
        axes2[2].axvline(x=overall_mean, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Overall Mean: {overall_mean:.3f}')
        axes2[2].set_yticks(y_pos)
        axes2[2].set_yticklabels([f"{camp_id[:8]} + {title}" for (camp_id, title) in campaign_title_data.index])
        axes2[2].set_xlabel('Open Rate')
        axes2[2].set_title('Top Campaign + Title Combinations\n(Darker=Higher Sample Size)')
        axes2[2].grid(True, alpha=0.3)
        axes2[2].legend()
    
    # Adjust layout for second figure
    plt.figure(fig2.number)
    plt.tight_layout(pad=2.0)
    plt.savefig('campaign_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Campaign targeting analysis saved as 'campaign_analysis_dashboard.png'")

def generate_campaign_recommendations(df, significant_campaigns, campaign_segments):
    """Generate actionable recommendations based on campaign analysis"""
    print("\n\n7. CAMPAIGN TARGETING RECOMMENDATIONS:")
    print("-" * 80)
    
    # Top performing campaigns
    top_campaigns = significant_campaigns.head(10)
    
    print("🎯 TOP PERFORMING CAMPAIGNS:")
    print("-" * 50)
    for idx, (campaign_id, row) in enumerate(top_campaigns.iterrows(), 1):
        print(f"{idx}. Campaign {campaign_id}:")
        print(f"   - Open Rate: {row['open_rate']:.3f} ({row['open_rate']*100:.1f}%)")
        print(f"   - Contact Count: {row['contact_count']:,}")
        print(f"   - Click Rate: {row['click_rate']:.3f} ({row['click_rate']*100:.1f}%)")
        print(f"   - Reply Rate: {row['reply_rate']:.3f} ({row['reply_rate']*100:.1f}%)")
        print()
    
    # Campaign targeting insights
    print("🎯 CAMPAIGN TARGETING INSIGHTS:")
    print("-" * 50)
    
    # Industry targeting insights
    if 'organization_industry' in df.columns:
        print("\n📊 INDUSTRY TARGETING INSIGHTS:")
        print("-" * 30)
        
        # Enhanced industry categorization (using comprehensive logic from enhanced_segmentation_analysis.py)
        df['industry_category'] = 'Other'
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
        
        # Apply basic industry categorization
        for category, keywords in industry_mapping.items():
            mask = df['organization_industry'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'industry_category'] = category
        
        # Enhanced categorization using organization_data JSON
        if 'organization_data' in df.columns:
            try:
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
            except Exception as e:
                print(f"Warning: Error parsing organization_data: {e}")
        
        # Additional categorization using company_domain
        if 'company_domain' in df.columns:
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
        
        # Get top campaigns and their industry targeting
        top_10_campaigns = significant_campaigns.head(10).index
        campaign_industry_targeting = df[df['campaign_id'].isin(top_10_campaigns)].groupby(['campaign_id', 'industry_category'])['opened'].agg(['count', 'mean']).round(4)
        campaign_industry_targeting = campaign_industry_targeting[campaign_industry_targeting['count'] >= 10].sort_values('mean', ascending=False)
        
        print("Top Campaign + Industry Targeting Combinations:")
        for (campaign_id, industry), row in campaign_industry_targeting.head(10).iterrows():
            print(f"   - Campaign {campaign_id[:8]} + {industry}: {row['mean']:.3f} open rate ({row['count']:,} contacts)")
    
    # Company size targeting insights
    if 'organization_employees' in df.columns:
        print("\n🏢 COMPANY SIZE TARGETING INSIGHTS:")
        print("-" * 30)
        
        # Create company size categories
        df['company_size'] = pd.cut(
            df['organization_employees'].fillna(-1),
            bins=[-1, 0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['Unknown', '1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
        
        # Get top campaigns and their company size targeting
        top_10_campaigns = significant_campaigns.head(10).index
        campaign_size_targeting = df[df['campaign_id'].isin(top_10_campaigns)].groupby(['campaign_id', 'company_size'])['opened'].agg(['count', 'mean']).round(4)
        campaign_size_targeting = campaign_size_targeting[campaign_size_targeting['count'] >= 10].sort_values('mean', ascending=False)
        
        print("Top Campaign + Company Size Targeting Combinations:")
        for (campaign_id, size), row in campaign_size_targeting.head(10).iterrows():
            print(f"   - Campaign {campaign_id[:8]} + {size}: {row['mean']:.3f} open rate ({row['count']:,} contacts)")
    
    # Title targeting insights
    if 'title' in df.columns:
        print("\n👥 TITLE TARGETING INSIGHTS:")
        print("-" * 30)
        
        # Enhanced title categorization using both title and seniority columns
        df['title_category'] = 'Other'
        title_mapping = {
            'CEO/Founder': ['CEO', 'Chief Executive Officer', 'Founder', 'Co-Founder', 'President', 'Owner'],
            'CTO/Technical': ['CTO', 'Chief Technology Officer', 'Technical Director', 'VP Engineering', 'Head of Engineering'],
            'CFO/Finance': ['CFO', 'Chief Financial Officer', 'Finance Director', 'VP Finance', 'Head of Finance'],
            'CMO/Marketing': ['CMO', 'Chief Marketing Officer', 'Marketing Director', 'VP Marketing', 'Head of Marketing'],
            'Sales': ['Sales Director', 'VP Sales', 'Head of Sales', 'Sales Manager', 'Business Development'],
            'Product': ['Product Manager', 'Product Director', 'VP Product', 'Head of Product'],
            'HR': ['HR Director', 'VP HR', 'Head of HR', 'Human Resources', 'Talent'],
            'Operations': ['Operations Director', 'VP Operations', 'Head of Operations', 'COO'],
            'Consultant': ['Consultant', 'Advisor', 'Partner', 'Principal'],
            'Developer': ['Developer', 'Engineer', 'Software Engineer', 'Programmer'],
            'Analyst': ['Analyst', 'Data Analyst', 'Business Analyst', 'Research Analyst'],
            'Manager': ['Manager', 'Team Lead', 'Supervisor'],
            'Other': ['Other', 'Unknown', 'nan']
        }
        
        # Apply basic title categorization
        for category, keywords in title_mapping.items():
            mask = df['title'].str.contains('|'.join(keywords), case=False, na=False)
            df.loc[mask, 'title_category'] = category
        
        # Enhanced categorization using seniority column
        if 'seniority' in df.columns:
            seniority_title_mapping = {
                'CEO/Founder': ['c_suite', 'C-Level', 'CEO', 'CTO', 'CFO', 'COO', 'CMO', 'CIO', 'Founder', 'Owner', 'Chief'],
                'VP/Director': ['vp', 'VP', 'Vice President', 'Director', 'Head of', 'director', 'head'],
                'Manager': ['Manager', 'Lead', 'Senior Manager', 'manager'],
                'Senior': ['Senior', 'Senior Engineer', 'Senior Developer', 'Senior Analyst', 'senior'],
                'Mid-Level': ['Mid', 'Engineer', 'Developer', 'Analyst', 'Specialist', 'entry'],
                'Junior': ['Junior', 'Entry', 'Associate', 'intern'],
                'Other': ['Other', 'Unknown', 'nan', 'partner', 'owner']
            }
            
            # Apply title-based categorization for entries still marked as 'Other'
            for idx, row in df.iterrows():
                if row['title_category'] == 'Other' and pd.notna(row['seniority']):
                    seniority = str(row['seniority']).lower()
                    
                    for category, keywords in seniority_title_mapping.items():
                        if any(keyword.lower() in seniority for keyword in keywords):
                            df.at[idx, 'title_category'] = category
                            break
        
        # Get top campaigns and their title targeting
        top_10_campaigns = significant_campaigns.head(10).index
        campaign_title_targeting = df[df['campaign_id'].isin(top_10_campaigns)].groupby(['campaign_id', 'title_category'])['opened'].agg(['count', 'mean']).round(4)
        campaign_title_targeting = campaign_title_targeting[campaign_title_targeting['count'] >= 10].sort_values('mean', ascending=False)
        
        print("Top Campaign + Title Targeting Combinations:")
        for (campaign_id, title), row in campaign_title_targeting.head(10).iterrows():
            print(f"   - Campaign {campaign_id[:8]} + {title}: {row['mean']:.3f} open rate ({row['count']:,} contacts)")
    
    # Strategic recommendations
    print("\n🎯 STRATEGIC RECOMMENDATIONS:")
    print("-" * 50)
    
    print("1. **Scale High-Performing Campaign-Segment Combinations:**")
    print("   - Identify campaigns with 80%+ open rates for specific segments")
    print("   - Increase volume for these proven combinations")
    print("   - Replicate targeting strategies across similar campaigns")
    
    print("\n2. **Optimize Campaign Targeting:**")
    print("   - Use top-performing campaign-segment combinations as templates")
    print("   - A/B test different segment combinations within campaigns")
    print("   - Focus on segments with high sample sizes and performance")
    
    print("\n3. **Campaign-Specific Optimization:**")
    print("   - Analyze which campaigns work best for which industries")
    print("   - Identify optimal company size ranges for each campaign")
    print("   - Determine which titles respond best to specific campaigns")

def main():
    """Main campaign analysis pipeline"""
    print("=== COMPREHENSIVE CAMPAIGN ANALYSIS ===")
    print("🎯 Goal: Analyze engagement patterns across different campaign categories")
    print("=" * 80)
    
    # Load and prepare data
    df = load_and_prepare_data()
    if df is None:
        return
    
    # Analyze campaign overview
    significant_campaigns = analyze_campaign_overview(df)
    
    # Analyze campaign segments
    campaign_segments = analyze_campaign_segments(df)
    
    # Analyze campaign features
    analyze_campaign_features(df)
    
    # Create visualizations
    create_campaign_visualizations(df, significant_campaigns)
    
    # Generate recommendations
    generate_campaign_recommendations(df, significant_campaigns, campaign_segments)
    
    print("\n" + "="*100)
    print("✅ COMPREHENSIVE CAMPAIGN ANALYSIS COMPLETE")
    print("📊 Check 'campaign_analysis_dashboard.png' for visualizations")
    print("🎯 Use recommendations to optimize campaign targeting")
    print("="*100)

if __name__ == "__main__":
    main() 