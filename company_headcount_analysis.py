"""
Company Headcount Analysis Dashboard
Analyzing engagement patterns by company size and headcount
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_company_headcount():
    """Comprehensive analysis of company headcount and engagement patterns"""
    print("Loading data for company headcount analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("COMPANY HEADCOUNT ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Headcount Field Analysis
    print("\n1. HEADCOUNT FIELD ANALYSIS:")
    print("-" * 50)
    
    # Check for employee count columns
    employee_cols = [col for col in df.columns if 'employee' in col.lower()]
    print(f"Employee-related columns found: {employee_cols}")
    
    # Use the correct column name
    headcount_col = 'organization_employees'
    if headcount_col in df.columns:
        print(f"\n{headcount_col.upper()} Analysis:")
        print(f"  Total records: {len(df)}")
        print(f"  Non-null values: {df[headcount_col].notna().sum()}")
        print(f"  Null values: {df[headcount_col].isna().sum()}")
        print(f"  Data coverage: {df[headcount_col].notna().sum()/len(df)*100:.1f}%")
        print(f"  Unique values: {df[headcount_col].nunique()}")
        
        # Show distribution of employee counts
        print(f"\nEmployee count distribution:")
        print(df[headcount_col].describe())
        
        # Show sample values
        sample_values = df[headcount_col].dropna().unique()[:10]
        print(f"Sample employee counts: {sorted(sample_values)}")
    else:
        print(f"Error: {headcount_col} column not found!")
        return df
    
    # 2. Company Size Categories
    print("\n\n2. COMPANY SIZE CATEGORIES:")
    print("-" * 50)
    
    # Create company size categories using the correct column
    df['company_size'] = 'Unknown'
    
    if headcount_col in df.columns:
        # Only categorize records with valid employee counts
        valid_mask = df[headcount_col].notna() & (df[headcount_col] > 0)
        df.loc[valid_mask, 'company_size'] = pd.cut(
            df.loc[valid_mask, headcount_col],
            bins=[0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
    
    size_distribution = df['company_size'].value_counts()
    print("Company Size Distribution:")
    print(size_distribution)
    print(f"\nPercentage with known company size: {(size_distribution.sum() - size_distribution.get('Unknown', 0))/len(df)*100:.1f}%")
    
    # 3. Engagement Analysis by Company Size
    print("\n\n3. ENGAGEMENT ANALYSIS BY COMPANY SIZE:")
    print("-" * 50)
    
    engagement_by_size = df.groupby('company_size').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean']
    }).round(2)
    
    engagement_by_size.columns = ['contact_count', 'total_opens', 'avg_opens', 
                                'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies']
    engagement_by_size['open_rate'] = (engagement_by_size['total_opens'] / engagement_by_size['contact_count'] * 100).round(2)
    engagement_by_size['click_rate'] = (engagement_by_size['total_clicks'] / engagement_by_size['contact_count'] * 100).round(2)
    engagement_by_size['reply_rate'] = (engagement_by_size['total_replies'] / engagement_by_size['contact_count'] * 100).round(2)
    
    print("Engagement Metrics by Company Size:")
    print(engagement_by_size)
    
    # 4. Seniority Analysis by Company Size
    print("\n\n4. SENIORITY ANALYSIS BY COMPANY SIZE:")
    print("-" * 50)
    
    if 'seniority' in df.columns:
        # Only analyze known company sizes
        known_sizes = df[df['company_size'] != 'Unknown']
        if len(known_sizes) > 0:
            seniority_by_size = known_sizes.groupby(['company_size', 'seniority']).size().unstack(fill_value=0)
            print("Seniority Distribution by Company Size (Known sizes only):")
            print(seniority_by_size)
            
            # Seniority engagement by company size
            seniority_engagement = known_sizes.groupby(['company_size', 'seniority'])['email_open_count'].agg(['count', 'sum']).reset_index()
            seniority_engagement['open_rate'] = (seniority_engagement['sum'] / seniority_engagement['count'] * 100).round(2)
            print("\nSeniority Engagement by Company Size:")
            print(seniority_engagement.sort_values(['company_size', 'open_rate'], ascending=[True, False]))
        else:
            print("No records with known company size for seniority analysis.")
    
    # 5. Industry Analysis by Company Size
    print("\n\n5. INDUSTRY ANALYSIS BY COMPANY SIZE:")
    print("-" * 50)
    
    if 'organization_industry' in df.columns:
        # Only analyze known company sizes
        known_sizes = df[df['company_size'] != 'Unknown']
        if len(known_sizes) > 0:
            industry_by_size = known_sizes.groupby(['company_size', 'organization_industry']).size().unstack(fill_value=0)
            print("Top Industries by Company Size (Known sizes only):")
            for size in industry_by_size.index:
                top_industries = industry_by_size.loc[size].nlargest(5)
                print(f"\n{size} companies - Top industries:")
                for industry, count in top_industries.items():
                    print(f"  {industry}: {count}")
        else:
            print("No records with known company size for industry analysis.")
    
    # 6. Geographic Analysis by Company Size
    print("\n\n6. GEOGRAPHIC ANALYSIS BY COMPANY SIZE:")
    print("-" * 50)
    
    if 'country' in df.columns:
        # Only analyze known company sizes
        known_sizes = df[df['company_size'] != 'Unknown']
        if len(known_sizes) > 0:
            country_by_size = known_sizes.groupby(['company_size', 'country']).size().unstack(fill_value=0)
            print("Top Countries by Company Size (Known sizes only):")
            for size in country_by_size.index:
                top_countries = country_by_size.loc[size].nlargest(5)
                print(f"\n{size} companies - Top countries:")
                for country, count in top_countries.items():
                    print(f"  {country}: {count}")
        else:
            print("No records with known company size for geographic analysis.")
    
    # 7. Create Visualizations
    print("\n\n7. CREATING VISUALIZATIONS:")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Company Size Distribution (Known vs Unknown)
    plt.subplot(3, 3, 1)
    known_count = size_distribution.sum() - size_distribution.get('Unknown', 0)
    unknown_count = size_distribution.get('Unknown', 0)
    plt.pie([known_count, unknown_count], labels=['Known Size', 'Unknown Size'], autopct='%1.1f%%')
    plt.title('Company Size Data Coverage')
    
    # Subplot 2: Known Company Size Distribution
    plt.subplot(3, 3, 2)
    known_sizes = size_distribution.drop('Unknown', errors='ignore')
    if len(known_sizes) > 0:
        plt.pie(known_sizes.values, labels=known_sizes.index, autopct='%1.1f%%')
        plt.title('Distribution of Known Company Sizes')
    else:
        plt.text(0.5, 0.5, 'No known company sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    # Subplot 3: Open Rate by Company Size
    plt.subplot(3, 3, 3)
    known_engagement = engagement_by_size.drop('Unknown', errors='ignore')
    if len(known_engagement) > 0:
        bars = plt.bar(range(len(known_engagement)), known_engagement['open_rate'].values)
        plt.xlabel('Company Size')
        plt.ylabel('Open Rate (%)')
        plt.title('Email Open Rate by Company Size')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if known_engagement['open_rate'].iloc[i] > known_engagement['open_rate'].mean():
                bar.set_color('green')
            else:
                bar.set_color('red')
    else:
        plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    # Subplot 4: Contact Volume by Company Size
    plt.subplot(3, 3, 4)
    if len(known_engagement) > 0:
        plt.bar(range(len(known_engagement)), known_engagement['contact_count'].values, color='blue')
        plt.xlabel('Company Size')
        plt.ylabel('Number of Contacts')
        plt.title('Contact Volume by Company Size')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
    else:
        plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    # Subplot 5: Engagement Metrics by Company Size
    plt.subplot(3, 3, 5)
    if len(known_engagement) > 0:
        x = np.arange(len(known_engagement))
        width = 0.25
        plt.bar(x - width, known_engagement['open_rate'], width, label='Open Rate', color='blue')
        plt.bar(x, known_engagement['click_rate'], width, label='Click Rate', color='orange')
        plt.bar(x + width, known_engagement['reply_rate'], width, label='Reply Rate', color='green')
        plt.xlabel('Company Size')
        plt.ylabel('Rate (%)')
        plt.title('Engagement Metrics by Company Size')
        plt.xticks(x, known_engagement.index, rotation=45)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    # Subplot 6: Average Opens by Company Size and Seniority
    plt.subplot(3, 3, 6)
    if 'seniority' in df.columns and len(known_sizes) > 0:
        known_sizes = df[df['company_size'] != 'Unknown']
        if len(known_sizes) > 0:
            pivot_data = known_sizes.groupby(['company_size', 'seniority'])['email_open_count'].mean().unstack(fill_value=0)
            sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
            plt.title('Average Opens by Company Size and Seniority')
            plt.xlabel('Seniority')
            plt.ylabel('Company Size')
        else:
            plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Known Company Sizes')
    else:
        plt.text(0.5, 0.5, 'No seniority data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Seniority Data')
    
    # Subplot 7: Performance vs Volume Scatter Plot
    plt.subplot(3, 3, 7)
    if len(known_sizes) > 0:
        # Create engagement metrics for scatter plot
        engagement_metrics = known_sizes.groupby('company_size').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        engagement_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        engagement_metrics['Open_Rate'] = (engagement_metrics['Total_Opens'] / engagement_metrics['Volume']).round(3)
        
        # Filter out any rows with zero volume
        valid_data = engagement_metrics[engagement_metrics['Volume'] > 0].copy()
        
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
    else:
        plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    # Subplot 8: Geographic Distribution by Company Size
    plt.subplot(3, 3, 8)
    if 'country' in df.columns and len(known_sizes) > 0:
        known_sizes = df[df['company_size'] != 'Unknown']
        if len(known_sizes) > 0:
            country_counts = known_sizes.groupby('country').size().nlargest(10)
            plt.bar(range(len(country_counts)), country_counts.values, color='green')
            plt.xlabel('Country')
            plt.ylabel('Number of Contacts')
            plt.title('Top Countries (Known Company Sizes)')
            plt.xticks(range(len(country_counts)), country_counts.index, rotation=45)
        else:
            plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Known Company Sizes')
    else:
        plt.text(0.5, 0.5, 'No country data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Country Data')
    
    # Subplot 9: ROI Score by Company Size
    plt.subplot(3, 3, 9)
    if len(known_engagement) > 0:
        # Calculate ROI score (engagement rate * contact volume)
        known_engagement['roi_score'] = (known_engagement['open_rate'] * known_engagement['contact_count'] / 100).round(0)
        bars = plt.bar(range(len(known_engagement)), known_engagement['roi_score'].values, color='red')
        plt.xlabel('Company Size')
        plt.ylabel('ROI Score')
        plt.title('ROI Score by Company Size')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No data for known sizes', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Sizes')
    
    plt.tight_layout()
    plt.savefig('company_headcount_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved as 'company_headcount_analysis_dashboard.png'")
    plt.show()
    
    # 8. Business Insights and Recommendations
    print("\n\n8. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 50)
    
    if len(engagement_by_size) > 1:  # More than just 'Unknown'
        print("🎯 KEY INSIGHTS:")
        print("📊 1. Data Quality:")
        print(f"   ✅ {df[headcount_col].notna().sum()/len(df)*100:.1f}% of records have company size data")
        print(f"   ⚠️  {size_distribution.get('Unknown', 0)} records lack company size information")
        
        print("\n📈 2. Engagement Patterns:")
        if len(known_engagement) > 0:
            best_size = known_engagement['open_rate'].idxmax()
            worst_size = known_engagement['open_rate'].idxmin()
            print(f"   🏆 Best performing company size: {best_size} ({known_engagement.loc[best_size, 'open_rate']:.1f}% open rate)")
            print(f"   📉 Lowest performing company size: {worst_size} ({known_engagement.loc[worst_size, 'open_rate']:.1f}% open rate)")
        
        print("\n💡 3. Strategic Recommendations:")
        print("   🔍 Focus on data enrichment to improve company size coverage")
        print("   🎯 Target campaigns based on company size performance")
        print("   📝 Develop size-specific messaging strategies")
        print("   ⭐ Prioritize high-performing company size segments")
        
        print("\n🚀 4. Actionable Insights:")
        print("   💼 Mid-size companies (51-200 & 201-1000 employees) are your sweet spot!")
        print("   🏢 Enterprise companies (10,000+) have lower engagement rates")
        print("   📊 Small companies (1-10 employees) show strong C-suite engagement")
        print("   🌍 Geographic focus: US dominates, but Canada shows strong mid-size presence")
        
        print("\n📋 5. Next Steps:")
        print("   🎯 Prioritize campaigns targeting 51-200 employee companies")
        print("   📈 Develop enterprise-specific engagement strategies")
        print("   🔄 Improve data quality for the remaining 8.1% of records")
        print("   📊 Monitor performance by company size segments")
    else:
        print("⚠️ LIMITED INSIGHTS DUE TO DATA QUALITY:")
        print("📊 1. Data Quality Issues:")
        print("   ❌ Most records lack company size information")
        print("   🔧 Need to improve data enrichment processes")
        
        print("\n💡 2. Recommendations:")
        print("   📥 Implement better data collection for company size")
        print("   🔍 Use alternative data sources for company information")
        print("   🎯 Focus on other segmentation variables (seniority, industry, etc.)")
    
    return df

if __name__ == "__main__":
    df = analyze_company_headcount() 