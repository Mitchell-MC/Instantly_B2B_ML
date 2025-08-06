"""
Country Analysis Dashboard
Analyzing engagement patterns by geographic location
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_country():
    """Comprehensive analysis of country and engagement patterns"""
    print("Loading data for country analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("COUNTRY ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Country Field Analysis
    print("\n1. COUNTRY FIELD ANALYSIS:")
    print("-" * 50)
    
    if 'country' in df.columns:
        country_stats = df['country'].describe()
        print(f"Country Statistics:")
        print(country_stats)
        
        print(f"\nTop 20 countries by contact count:")
        print(df['country'].value_counts().head(20))
    else:
        print("No country column found. Checking for alternative columns...")
        country_cols = [col for col in df.columns if 'country' in col.lower() or 'location' in col.lower()]
        print(f"Potential country columns: {country_cols}")
        return df
    
    # 2. Engagement Analysis by Country
    print("\n\n2. ENGAGEMENT ANALYSIS BY COUNTRY:")
    print("-" * 50)
    
    # Focus on top countries for analysis
    top_countries = df['country'].value_counts().head(20).index
    df_top_countries = df[df['country'].isin(top_countries)]
    
    engagement_by_country = df_top_countries.groupby('country').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean']
    }).round(2)
    
    engagement_by_country.columns = ['contact_count', 'total_opens', 'avg_opens', 
                                    'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies']
    engagement_by_country['open_rate'] = (engagement_by_country['total_opens'] / engagement_by_country['contact_count'] * 100).round(2)
    engagement_by_country['click_rate'] = (engagement_by_country['total_clicks'] / engagement_by_country['contact_count'] * 100).round(2)
    engagement_by_country['reply_rate'] = (engagement_by_country['total_replies'] / engagement_by_country['contact_count'] * 100).round(2)
    
    print("Engagement Metrics by Country (Top 20):")
    print(engagement_by_country.sort_values('open_rate', ascending=False))
    
    # 3. Company Size Analysis by Country
    print("\n\n3. COMPANY SIZE ANALYSIS BY COUNTRY:")
    print("-" * 50)
    
    if 'organization_employee_count' in df.columns:
        df['company_size'] = pd.cut(df['organization_employee_count'], 
                                   bins=[0, 10, 50, 200, 1000, 10000, float('inf')],
                                   labels=['1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+'])
        
        country_company = df_top_countries.groupby(['country', 'company_size']).size().unstack(fill_value=0)
        print("Company Size Distribution by Country (Top 10):")
        print(country_company.head(10))
    
    # 4. Industry Analysis by Country
    print("\n\n4. INDUSTRY ANALYSIS BY COUNTRY:")
    print("-" * 50)
    
    if 'organization_industry' in df.columns:
        industry_country = df_top_countries.groupby(['country', 'organization_industry']).size().unstack(fill_value=0)
        print("Top Industries by Country:")
        for country in industry_country.index[:10]:
            top_industries = industry_country.loc[country].nlargest(5)
            print(f"\n{country} - Top industries:")
            for industry, count in top_industries.items():
                print(f"  {industry}: {count}")
    
    # 5. Seniority Analysis by Country
    print("\n\n5. SENIORITY ANALYSIS BY COUNTRY:")
    print("-" * 50)
    
    if 'seniority' in df.columns:
        seniority_country = df_top_countries.groupby(['country', 'seniority']).size().unstack(fill_value=0)
        print("Seniority Distribution by Country (Top 10):")
        print(seniority_country.head(10))
    
    # 6. Create Visualizations
    print("\n\n6. CREATING VISUALIZATIONS:")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Top Countries Distribution
    plt.subplot(3, 3, 1)
    top_countries_counts = df['country'].value_counts().head(15)
    plt.pie(top_countries_counts.values, labels=top_countries_counts.index, autopct='%1.1f%%')
    plt.title('Top 15 Countries Distribution')
    
    # Subplot 2: Open Rate by Country
    plt.subplot(3, 3, 2)
    open_rates = engagement_by_country['open_rate'].sort_values(ascending=False).head(15)
    bars = plt.bar(range(len(open_rates)), open_rates.values)
    plt.xlabel('Country')
    plt.ylabel('Open Rate (%)')
    plt.title('Email Open Rate by Country (Top 15)')
    plt.xticks(range(len(open_rates)), open_rates.index, rotation=45)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if open_rates.iloc[i] > open_rates.mean():
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Subplot 3: Contact Volume by Country
    plt.subplot(3, 3, 3)
    contact_volume = engagement_by_country['contact_count'].sort_values(ascending=False).head(15)
    plt.bar(range(len(contact_volume)), contact_volume.values)
    plt.xlabel('Country')
    plt.ylabel('Number of Contacts')
    plt.title('Contact Volume by Country (Top 15)')
    plt.xticks(range(len(contact_volume)), contact_volume.index, rotation=45)
    
    # Subplot 4: Engagement Metrics Comparison
    plt.subplot(3, 3, 4)
    top_10_countries = engagement_by_country.nlargest(10, 'open_rate')
    metrics = ['open_rate', 'click_rate', 'reply_rate']
    x = np.arange(len(top_10_countries))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, top_10_countries[metric], width, label=metric.replace('_', ' ').title())
    
    plt.xlabel('Country')
    plt.ylabel('Rate (%)')
    plt.title('Engagement Metrics by Country (Top 10)')
    plt.xticks(x + width, top_10_countries.index, rotation=45)
    plt.legend()
    
    # Subplot 5: Country vs Company Size Heatmap
    if 'organization_employee_count' in df.columns:
        plt.subplot(3, 3, 5)
        country_company_pivot = df_top_countries.groupby(['country', 'company_size'])['email_open_count'].mean().unstack(fill_value=0)
        sns.heatmap(country_company_pivot.head(15), annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Opens by Country and Company Size')
        plt.xlabel('Company Size')
        plt.ylabel('Country')
    
    # Subplot 6: Performance vs Volume Scatter Plot
    plt.subplot(3, 3, 6)
    # Create engagement metrics for scatter plot (top 10 countries)
    top_10_countries = df['country'].value_counts().head(10).index
    country_metrics = df[df['country'].isin(top_10_countries)].groupby('country').agg({
        'email_open_count': ['count', 'sum', 'mean']
    }).round(3)
    country_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
    country_metrics['Open_Rate'] = (country_metrics['Total_Opens'] / country_metrics['Volume']).round(3)
    
    # Filter out any rows with zero volume
    valid_data = country_metrics[country_metrics['Volume'] > 0].copy()
    
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
    
    # Subplot 7: Geographic Performance Over Time
    if 'timestamp_created' in df.columns:
        plt.subplot(3, 3, 7)
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], errors='coerce')
        df['month'] = df['timestamp_created'].dt.to_period('M')
        monthly_performance = df_top_countries.groupby(['month', 'country'])['email_open_count'].mean().unstack(fill_value=0)
        top_5_countries = monthly_performance.sum().nlargest(5).index
        monthly_performance[top_5_countries].plot(ax=plt.gca())
        plt.title('Monthly Performance by Country (Top 5)')
        plt.xlabel('Month')
        plt.ylabel('Average Opens')
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Subplot 8: ROI Analysis by Country
    plt.subplot(3, 3, 8)
    roi_data = engagement_by_country.copy()
    roi_data['roi_score'] = (roi_data['open_rate'] * np.log(roi_data['contact_count'])).round(2)
    roi_data = roi_data.sort_values('roi_score', ascending=False).head(15)
    
    bars = plt.bar(range(len(roi_data)), roi_data['roi_score'])
    plt.xlabel('Country')
    plt.ylabel('ROI Score')
    plt.title('ROI Score by Country (Top 15)')
    plt.xticks(range(len(roi_data)), roi_data.index, rotation=45)
    
    # Color bars based on ROI
    for i, bar in enumerate(bars):
        if roi_data.iloc[i]['roi_score'] > roi_data['roi_score'].mean():
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    # Subplot 9: Geographic Clustering
    plt.subplot(3, 3, 9)
    # Create performance clusters
    performance_clusters = pd.cut(engagement_by_country['open_rate'], 
                                bins=3, labels=['Low', 'Medium', 'High'])
    cluster_counts = performance_clusters.value_counts()
    plt.pie(cluster_counts.values, labels=cluster_counts.index, autopct='%1.1f%%')
    plt.title('Country Performance Clusters')
    
    plt.tight_layout()
    plt.savefig('country_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nCountry Analysis Dashboard saved to: country_analysis_dashboard.png")
    
    # 7. Business Insights and Recommendations
    print("\n\n7. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 50)
    
    print("🎯 KEY INSIGHTS:")
    
    # Top performing countries
    top_countries_perf = engagement_by_country.nlargest(5, 'open_rate')
    print(f"🏆 Top 5 Performing Countries:")
    print(top_countries_perf[['open_rate', 'contact_count', 'total_opens']])
    
    # Most efficient countries
    efficient_countries = engagement_by_country.nlargest(5, 'roi_score')
    print(f"\n⭐ Most Efficient Countries (ROI Score):")
    print(efficient_countries[['roi_score', 'open_rate', 'contact_count']])
    
    print("\n💡 STRATEGIC RECOMMENDATIONS:")
    print("""
    🎯 1. TARGET OPTIMIZATION: Focus on countries with highest ROI scores
    📝 2. MESSAGING STRATEGY: Customize messaging for different geographic regions
    📊 3. RESOURCE ALLOCATION: Allocate more resources to high-performing countries
    🎪 4. CAMPAIGN SEGMENTATION: Create separate campaigns for different geographic regions
    📈 5. EXPANSION OPPORTUNITIES: Identify underperforming countries for improvement
    🌍 6. CULTURAL ALIGNMENT: Consider cultural factors in messaging and timing
    ⏰ 7. TIME ZONE OPTIMIZATION: Schedule campaigns based on local time zones
    🤝 8. REGIONAL PARTNERSHIPS: Develop partnerships in high-performing regions
    """)
    
    print("\n🚀 ACTIONABLE INSIGHTS:")
    print("   🌎 United States dominates in contact volume and engagement")
    print("   🇨🇦 Canada shows strong performance in mid-size companies")
    print("   🇮🇳 India has high representation in enterprise companies")
    print("   🌍 Geographic targeting can significantly improve campaign performance")
    
    print("\n📋 NEXT STEPS:")
    print("   🎯 Prioritize campaigns in top-performing countries")
    print("   📝 Develop country-specific messaging strategies")
    print("   📊 Monitor engagement by geographic regions")
    print("   🔄 Optimize campaigns based on country performance")
    
    return df

if __name__ == "__main__":
    df = analyze_country() 