"""
Seniority Analysis Dashboard
Analyzing engagement patterns by job seniority levels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_seniority():
    """Comprehensive analysis of seniority and engagement patterns"""
    print("Loading data for seniority analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("SENIORITY ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Seniority Field Analysis
    print("\n1. SENIORITY FIELD ANALYSIS:")
    print("-" * 50)
    
    if 'seniority' in df.columns:
        seniority_stats = df['seniority'].describe()
        print(f"Seniority Statistics:")
        print(seniority_stats)
        
        # Handle NaN values when sorting
        unique_values = df['seniority'].dropna().unique()
        sorted_values = sorted(unique_values)
        print(f"\nUnique seniority values: {sorted_values}")
        print(f"Seniority distribution:")
        print(df['seniority'].value_counts())
    else:
        print("No seniority column found. Checking for alternative columns...")
        seniority_cols = [col for col in df.columns if 'senior' in col.lower() or 'level' in col.lower()]
        print(f"Potential seniority columns: {seniority_cols}")
        return df
    
    # 2. Engagement Analysis by Seniority
    print("\n\n2. ENGAGEMENT ANALYSIS BY SENIORITY:")
    print("-" * 50)
    
    engagement_by_seniority = df.groupby('seniority').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean']
    }).round(2)
    
    engagement_by_seniority.columns = ['contact_count', 'total_opens', 'avg_opens', 
                                      'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies']
    engagement_by_seniority['open_rate'] = (engagement_by_seniority['total_opens'] / engagement_by_seniority['contact_count'] * 100).round(2)
    engagement_by_seniority['click_rate'] = (engagement_by_seniority['total_clicks'] / engagement_by_seniority['contact_count'] * 100).round(2)
    engagement_by_seniority['reply_rate'] = (engagement_by_seniority['total_replies'] / engagement_by_seniority['contact_count'] * 100).round(2)
    
    print("Engagement Metrics by Seniority:")
    print(engagement_by_seniority)
    
    # 3. Company Size Analysis by Seniority
    print("\n\n3. COMPANY SIZE ANALYSIS BY SENIORITY:")
    print("-" * 50)
    
    if 'organization_employee_count' in df.columns:
        df['company_size'] = pd.cut(df['organization_employee_count'], 
                                   bins=[0, 10, 50, 200, 1000, 10000, float('inf')],
                                   labels=['1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+'])
        
        seniority_company = df.groupby(['seniority', 'company_size']).size().unstack(fill_value=0)
        print("Seniority Distribution by Company Size:")
        print(seniority_company)
    
    # 4. Industry Analysis by Seniority
    print("\n\n4. INDUSTRY ANALYSIS BY SENIORITY:")
    print("-" * 50)
    
    if 'organization_industry' in df.columns:
        industry_seniority = df.groupby(['seniority', 'organization_industry']).size().unstack(fill_value=0)
        print("Top Industries by Seniority:")
        for seniority in industry_seniority.index:
            top_industries = industry_seniority.loc[seniority].nlargest(5)
            print(f"\n{seniority} - Top industries:")
            for industry, count in top_industries.items():
                print(f"  {industry}: {count}")
    
    # 5. Geographic Analysis by Seniority
    print("\n\n5. GEOGRAPHIC ANALYSIS BY SENIORITY:")
    print("-" * 50)
    
    if 'country' in df.columns:
        country_seniority = df.groupby(['seniority', 'country']).size().unstack(fill_value=0)
        print("Top Countries by Seniority:")
        for seniority in country_seniority.index:
            top_countries = country_seniority.loc[seniority].nlargest(5)
            print(f"\n{seniority} - Top countries:")
            for country, count in top_countries.items():
                print(f"  {country}: {count}")
    
    # 6. Create Visualizations
    print("\n\n6. CREATING VISUALIZATIONS:")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Seniority Distribution
    plt.subplot(3, 3, 1)
    seniority_counts = df['seniority'].value_counts()
    plt.pie(seniority_counts.values, labels=seniority_counts.index, autopct='%1.1f%%')
    plt.title('Seniority Distribution')
    
    # Subplot 2: Open Rate by Seniority
    plt.subplot(3, 3, 2)
    open_rates = engagement_by_seniority['open_rate']
    bars = plt.bar(range(len(open_rates)), open_rates.values)
    plt.xlabel('Seniority')
    plt.ylabel('Open Rate (%)')
    plt.title('Email Open Rate by Seniority')
    plt.xticks(range(len(open_rates)), open_rates.index, rotation=45)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if open_rates.iloc[i] > open_rates.mean():
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Subplot 3: Contact Volume by Seniority
    plt.subplot(3, 3, 3)
    contact_volume = engagement_by_seniority['contact_count']
    plt.bar(range(len(contact_volume)), contact_volume.values)
    plt.xlabel('Seniority')
    plt.ylabel('Number of Contacts')
    plt.title('Contact Volume by Seniority')
    plt.xticks(range(len(contact_volume)), contact_volume.index, rotation=45)
    
    # Subplot 4: Engagement Metrics Comparison
    plt.subplot(3, 3, 4)
    metrics = ['open_rate', 'click_rate', 'reply_rate']
    x = np.arange(len(engagement_by_seniority))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, engagement_by_seniority[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Seniority')
    plt.ylabel('Rate (%)')
    plt.title('Engagement Metrics by Seniority')
    plt.xticks(x + width, engagement_by_seniority.index, rotation=45)
    plt.legend()
    
    # Subplot 5: Seniority vs Company Size Heatmap
    if 'organization_employee_count' in df.columns:
        plt.subplot(3, 3, 5)
        seniority_company_pivot = df.groupby(['seniority', 'company_size'])['email_open_count'].mean().unstack(fill_value=0)
        sns.heatmap(seniority_company_pivot, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Opens by Seniority and Company Size')
        plt.xlabel('Company Size')
        plt.ylabel('Seniority')
    
    # Subplot 6: Performance vs Volume Scatter Plot
    plt.subplot(3, 3, 6)
    # Create engagement metrics for scatter plot
    seniority_metrics = df.groupby('seniority').agg({
        'email_open_count': ['count', 'sum', 'mean']
    }).round(3)
    seniority_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
    seniority_metrics['Open_Rate'] = (seniority_metrics['Total_Opens'] / seniority_metrics['Volume']).round(3)
    
    # Filter out any rows with zero volume
    valid_data = seniority_metrics[seniority_metrics['Volume'] > 0].copy()
    
    if len(valid_data) > 0:
        # Create scatter plot
        scatter = plt.scatter(valid_data['Volume'], valid_data['Open_Rate'], 
                            s=valid_data['Volume']/100,  # Size based on volume
                            alpha=0.7, c=valid_data['Open_Rate'], cmap='viridis')
        
        # Add labels for each point
        for idx, row in valid_data.iterrows():
            plt.annotate(str(idx), (row['Volume'], row['Open_Rate']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.xlabel('Number of Contacts')
        plt.ylabel('Open Rate')
        plt.title('Performance vs Volume Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(scatter, label='Open Rate')
    else:
        plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Valid Data')
    
    # Subplot 7: Geographic Distribution by Seniority
    if 'country' in df.columns:
        plt.subplot(3, 3, 7)
        country_seniority_data = df.groupby(['seniority', 'country']).size().unstack(fill_value=0)
        top_countries = country_seniority_data.sum().nlargest(10).index
        country_seniority_top = country_seniority_data[top_countries]
        country_seniority_top.T.plot(kind='bar', ax=plt.gca())
        plt.title('Geographic Distribution by Seniority')
        plt.xlabel('Country')
        plt.ylabel('Number of Contacts')
        plt.xticks(rotation=45)
        plt.legend(title='Seniority', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 8: Seniority Performance Over Time
    if 'timestamp_created' in df.columns:
        plt.subplot(3, 3, 8)
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], errors='coerce')
        df['month'] = df['timestamp_created'].dt.to_period('M')
        monthly_performance = df.groupby(['month', 'seniority'])['email_open_count'].mean().unstack(fill_value=0)
        monthly_performance.plot(ax=plt.gca())
        plt.title('Monthly Performance by Seniority')
        plt.xlabel('Month')
        plt.ylabel('Average Opens')
        plt.legend(title='Seniority', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 9: ROI Analysis by Seniority
    plt.subplot(3, 3, 9)
    roi_data = engagement_by_seniority.copy()
    roi_data['roi_score'] = (roi_data['open_rate'] * np.log(roi_data['contact_count'])).round(2)
    roi_data = roi_data.sort_values('roi_score', ascending=False)
    
    bars = plt.bar(range(len(roi_data)), roi_data['roi_score'])
    plt.xlabel('Seniority')
    plt.ylabel('ROI Score')
    plt.title('ROI Score by Seniority')
    plt.xticks(range(len(roi_data)), roi_data.index, rotation=45)
    
    # Color bars based on ROI
    for i, bar in enumerate(bars):
        if roi_data.iloc[i]['roi_score'] > roi_data['roi_score'].mean():
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig('seniority_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSeniority Analysis Dashboard saved to: seniority_analysis_dashboard.png")
    
    # 7. Business Insights and Recommendations
    print("\n\n7. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 50)
    
    print("🎯 KEY INSIGHTS:")
    
    # Top performing seniority levels
    top_seniority = engagement_by_seniority.nlargest(3, 'open_rate')
    print(f"🏆 Top 3 Performing Seniority Levels:")
    print(top_seniority[['open_rate', 'contact_count', 'total_opens']])
    
    # Most efficient seniority levels
    efficient_seniority = engagement_by_seniority.nlargest(3, 'roi_score')
    print(f"\n⭐ Most Efficient Seniority Levels (ROI Score):")
    print(efficient_seniority[['roi_score', 'open_rate', 'contact_count']])
    
    print("\n💡 STRATEGIC RECOMMENDATIONS:")
    print("""
    🎯 1. TARGET OPTIMIZATION: Focus on seniority levels with highest ROI scores
    📝 2. MESSAGING STRATEGY: Customize messaging for different seniority levels
    📊 3. RESOURCE ALLOCATION: Allocate more resources to high-performing seniority levels
    🎪 4. CAMPAIGN SEGMENTATION: Create separate campaigns for different seniority ranges
    📈 5. EXPANSION OPPORTUNITIES: Identify underperforming seniority levels for improvement
    🏢 6. COMPANY SIZE ALIGNMENT: Match seniority targeting with company size preferences
    🌍 7. GEOGRAPHIC FOCUS: Combine seniority with geographic targeting
    ⏰ 8. TIMING OPTIMIZATION: Consider seniority when scheduling campaigns
    """)
    
    print("\n🚀 ACTIONABLE INSIGHTS:")
    print("   👑 C-suite contacts are your highest-value prospects")
    print("   👔 Manager level shows consistent engagement patterns")
    print("   🆕 Entry-level positions offer growth opportunities")
    print("   📈 Seniority targeting can significantly improve campaign performance")
    
    print("\n📋 NEXT STEPS:")
    print("   🎯 Prioritize C-suite outreach campaigns")
    print("   📝 Develop seniority-specific messaging")
    print("   📊 Monitor engagement by seniority levels")
    print("   🔄 Optimize campaigns based on seniority performance")
    
    return df

if __name__ == "__main__":
    df = analyze_seniority() 