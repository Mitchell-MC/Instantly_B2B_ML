"""
Founded Year Analysis Dashboard
Analyzing engagement patterns by company age and years since founding
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_founded_year():
    """Comprehensive analysis of organization founded year and years since founding"""
    print("Loading data for founded year analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("FOUNDED YEAR ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Founded Year Field Analysis
    print("\n1. FOUNDED YEAR FIELD ANALYSIS:")
    print("-" * 50)
    
    if 'organization_founded_year' in df.columns:
        print(f"ORGANIZATION_FOUNDED_YEAR Analysis:")
        print(f"  Total records: {len(df)}")
        print(f"  Non-null values: {df['organization_founded_year'].notna().sum()}")
        print(f"  Null values: {df['organization_founded_year'].isna().sum()}")
        print(f"  Data coverage: {df['organization_founded_year'].notna().sum()/len(df)*100:.1f}%")
        print(f"  Unique values: {df['organization_founded_year'].nunique()}")
        
        # Show distribution of founded years
        print(f"\nFounded year distribution:")
        print(df['organization_founded_year'].describe())
        
        # Show sample values
        sample_years = df['organization_founded_year'].dropna().unique()[:10]
        print(f"Sample founded years: {sorted(sample_years)}")
        
        # Calculate years since founding
        current_year = 2024
        df['years_since_founding'] = current_year - df['organization_founded_year']
        
        print(f"\nYears since founding distribution:")
        print(df['years_since_founding'].describe())
        
    else:
        print(f"Error: organization_founded_year column not found!")
        return df
    
    # 2. Company Age Categories
    print("\n\n2. COMPANY AGE CATEGORIES:")
    print("-" * 50)
    
    # Create company age categories
    df['company_age'] = 'Unknown'
    
    if 'years_since_founding' in df.columns:
        # Only categorize records with valid years since founding
        valid_mask = df['years_since_founding'].notna() & (df['years_since_founding'] >= 0)
        df.loc[valid_mask, 'company_age'] = pd.cut(
            df.loc[valid_mask, 'years_since_founding'],
            bins=[0, 5, 10, 20, 50, 100, float('inf')],
            labels=['0-5 years', '6-10 years', '11-20 years', '21-50 years', '51-100 years', '100+ years']
        )
    
    age_distribution = df['company_age'].value_counts()
    print("Company Age Distribution:")
    print(age_distribution)
    print(f"\nPercentage with known company age: {(age_distribution.sum() - age_distribution.get('Unknown', 0))/len(df)*100:.1f}%")
    
    # 3. Engagement Analysis by Company Age
    print("\n\n3. ENGAGEMENT ANALYSIS BY COMPANY AGE:")
    print("-" * 50)
    
    engagement_by_age = df.groupby('company_age').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean']
    }).round(2)
    
    engagement_by_age.columns = ['contact_count', 'total_opens', 'avg_opens', 
                               'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies']
    engagement_by_age['open_rate'] = (engagement_by_age['total_opens'] / engagement_by_age['contact_count'] * 100).round(2)
    engagement_by_age['click_rate'] = (engagement_by_age['total_clicks'] / engagement_by_age['contact_count'] * 100).round(2)
    engagement_by_age['reply_rate'] = (engagement_by_age['total_replies'] / engagement_by_age['contact_count'] * 100).round(2)
    
    print("Engagement Metrics by Company Age:")
    print(engagement_by_age)
    
    # 4. Seniority Analysis by Company Age
    print("\n\n4. SENIORITY ANALYSIS BY COMPANY AGE:")
    print("-" * 50)
    
    if 'seniority' in df.columns:
        # Only analyze known company ages
        known_ages_df = df[df['company_age'] != 'Unknown']
        if len(known_ages_df) > 0:
            seniority_by_age = known_ages_df.groupby(['company_age', 'seniority']).size().unstack(fill_value=0)
            print("Seniority Distribution by Company Age (Known ages only):")
            print(seniority_by_age)
            
            # Seniority engagement by company age
            seniority_engagement = known_ages_df.groupby(['company_age', 'seniority'])['email_open_count'].agg(['count', 'sum']).reset_index()
            seniority_engagement['open_rate'] = (seniority_engagement['sum'] / seniority_engagement['count'] * 100).round(2)
            print("\nSeniority Engagement by Company Age:")
            print(seniority_engagement.sort_values(['company_age', 'open_rate'], ascending=[True, False]))
        else:
            print("No records with known company age for seniority analysis.")
    
    # 5. Industry Analysis by Company Age
    print("\n\n5. INDUSTRY ANALYSIS BY COMPANY AGE:")
    print("-" * 50)
    
    if 'organization_industry' in df.columns:
        # Only analyze known company ages
        known_ages_df = df[df['company_age'] != 'Unknown']
        if len(known_ages_df) > 0:
            industry_by_age = known_ages_df.groupby(['company_age', 'organization_industry']).size().unstack(fill_value=0)
            print("Top Industries by Company Age (Known ages only):")
            for age in industry_by_age.index:
                top_industries = industry_by_age.loc[age].nlargest(5)
                print(f"\n{age} companies - Top industries:")
                for industry, count in top_industries.items():
                    print(f"  {industry}: {count}")
        else:
            print("No records with known company age for industry analysis.")
    
    # 6. Geographic Analysis by Company Age
    print("\n\n6. GEOGRAPHIC ANALYSIS BY COMPANY AGE:")
    print("-" * 50)
    
    if 'country' in df.columns:
        # Only analyze known company ages
        known_ages_df = df[df['company_age'] != 'Unknown']
        if len(known_ages_df) > 0:
            country_by_age = known_ages_df.groupby(['company_age', 'country']).size().unstack(fill_value=0)
            print("Top Countries by Company Age (Known ages only):")
            for age in country_by_age.index:
                top_countries = country_by_age.loc[age].nlargest(5)
                print(f"\n{age} companies - Top countries:")
                for country, count in top_countries.items():
                    print(f"  {country}: {count}")
        else:
            print("No records with known company age for geographic analysis.")
    
    # 7. Create Visualizations
    print("\n\n7. CREATING VISUALIZATIONS:")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Company Age Data Coverage
    plt.subplot(3, 3, 1)
    known_count = age_distribution.sum() - age_distribution.get('Unknown', 0)
    unknown_count = age_distribution.get('Unknown', 0)
    plt.pie([known_count, unknown_count], labels=['Known Age', 'Unknown Age'], autopct='%1.1f%%')
    plt.title('Company Age Data Coverage')
    
    # Subplot 2: Known Company Age Distribution
    plt.subplot(3, 3, 2)
    known_ages = age_distribution.drop('Unknown', errors='ignore')
    if len(known_ages) > 0:
        plt.pie(known_ages.values, labels=known_ages.index, autopct='%1.1f%%')
        plt.title('Distribution of Known Company Ages')
    else:
        plt.text(0.5, 0.5, 'No known company ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    # Subplot 3: Open Rate by Company Age
    plt.subplot(3, 3, 3)
    known_engagement = engagement_by_age.drop('Unknown', errors='ignore')
    if len(known_engagement) > 0:
        bars = plt.bar(range(len(known_engagement)), known_engagement['open_rate'].values)
        plt.xlabel('Company Age')
        plt.ylabel('Open Rate (%)')
        plt.title('Email Open Rate by Company Age')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if known_engagement['open_rate'].iloc[i] > known_engagement['open_rate'].mean():
                bar.set_color('green')
            else:
                bar.set_color('red')
    else:
        plt.text(0.5, 0.5, 'No data for known ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    # Subplot 4: Contact Volume by Company Age
    plt.subplot(3, 3, 4)
    if len(known_engagement) > 0:
        plt.bar(range(len(known_engagement)), known_engagement['contact_count'].values, color='blue')
        plt.xlabel('Company Age')
        plt.ylabel('Number of Contacts')
        plt.title('Contact Volume by Company Age')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
    else:
        plt.text(0.5, 0.5, 'No data for known ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    # Subplot 5: Engagement Metrics by Company Age
    plt.subplot(3, 3, 5)
    if len(known_engagement) > 0:
        x = np.arange(len(known_engagement))
        width = 0.25
        plt.bar(x - width, known_engagement['open_rate'], width, label='Open Rate', color='blue')
        plt.bar(x, known_engagement['click_rate'], width, label='Click Rate', color='orange')
        plt.bar(x + width, known_engagement['reply_rate'], width, label='Reply Rate', color='green')
        plt.xlabel('Company Age')
        plt.ylabel('Rate (%)')
        plt.title('Engagement Metrics by Company Age')
        plt.xticks(x, known_engagement.index, rotation=45)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No data for known ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    # Subplot 6: Performance vs Volume Scatter Plot
    plt.subplot(3, 3, 6)
    # Use the correct DataFrame for known ages
    known_ages_df = df[df['company_age'] != 'Unknown']
    if len(known_ages_df) > 0:
        # Debug: print type and columns before visualization
        print('Type of known_ages_df:', type(known_ages_df))
        print('Columns in known_ages_df before visualization:', known_ages_df.columns.tolist())
        # Use the correct engagement column for scatter plot
        engagement_col = None
        for col in ['email_open_count', 'total_opens', 'opened']:
            if col in known_ages_df.columns:
                engagement_col = col
                break
        if engagement_col is None:
            print('No suitable engagement column found for scatter plot!')
            plt.text(0.5, 0.5, 'No engagement column found', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('No Engagement Column')
        else:
            # Create engagement metrics for scatter plot using manual aggregation for compatibility
            volume = known_ages_df.groupby('company_age')[engagement_col].count()
            total_opens = known_ages_df.groupby('company_age')[engagement_col].sum()
            avg_opens = known_ages_df.groupby('company_age')[engagement_col].mean()
            open_rate = (total_opens / volume).round(3)
            age_metrics = pd.DataFrame({
                'volume': volume,
                'total_opens': total_opens,
                'avg_opens': avg_opens,
                'Open_Rate': open_rate
            })
            # Filter out any rows with zero volume
            valid_data = age_metrics[age_metrics['volume'] > 0].copy()
            if len(valid_data) > 0:
                # Create scatter plot
                scatter = plt.scatter(valid_data['volume'], valid_data['Open_Rate'], 
                                    s=valid_data['volume']/100,  # Size based on volume
                                    alpha=0.7, c=valid_data['Open_Rate'], cmap='viridis')
                # Add labels for each point
                for idx, row in valid_data.iterrows():
                    plt.annotate(str(idx), (row['volume'], row['Open_Rate']), 
                                xytext=(5, 5), textcoords='offset points', fontsize=8)
                plt.xlabel('Number of Contacts')
                plt.ylabel('Open Rate')
                plt.title('Performance vs Volume Analysis')
                plt.grid(True, alpha=0.3)
                plt.colorbar(scatter, label='Open Rate')
            else:
                plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('No Valid Data')
    else:
        plt.text(0.5, 0.5, 'No data for known ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    # Subplot 7: Average Opens by Company Age and Seniority
    plt.subplot(3, 3, 7)
    if 'seniority' in df.columns and len(known_ages_df) > 0:
        pivot_data = known_ages_df.groupby(['company_age', 'seniority'])['email_open_count'].mean().unstack(fill_value=0)
        sns.heatmap(pivot_data, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title('Average Opens by Company Age and Seniority')
        plt.xlabel('Seniority')
        plt.ylabel('Company Age')
    else:
        plt.text(0.5, 0.5, 'No seniority data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Seniority Data')
    
    # Subplot 8: Geographic Distribution by Company Age
    plt.subplot(3, 3, 8)
    if 'country' in df.columns and len(known_ages_df) > 0:
        country_counts = known_ages_df.groupby('country').size().nlargest(10)
        plt.bar(range(len(country_counts)), country_counts.values, color='green')
        plt.xlabel('Country')
        plt.ylabel('Number of Contacts')
        plt.title('Top Countries (Known Company Ages)')
        plt.xticks(range(len(country_counts)), country_counts.index, rotation=45)
    else:
        plt.text(0.5, 0.5, 'No country data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Country Data')
    
    # Subplot 9: ROI Score by Company Age
    plt.subplot(3, 3, 9)
    if len(known_engagement) > 0:
        # Calculate ROI score (engagement rate * contact volume)
        known_engagement['roi_score'] = (known_engagement['open_rate'] * known_engagement['contact_count'] / 100).round(0)
        bars = plt.bar(range(len(known_engagement)), known_engagement['roi_score'].values, color='red')
        plt.xlabel('Company Age')
        plt.ylabel('ROI Score')
        plt.title('ROI Score by Company Age')
        plt.xticks(range(len(known_engagement)), known_engagement.index, rotation=45)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No data for known ages', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('No Known Company Ages')
    
    plt.tight_layout()
    plt.savefig('founded_year_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("Dashboard saved as 'founded_year_analysis_dashboard.png'")
    plt.show()
    
    # 8. Business Insights and Recommendations
    print("\n\n8. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("-" * 50)
    
    if len(engagement_by_age) > 1:  # More than just 'Unknown'
        print("🎯 KEY INSIGHTS:")
        print("📊 1. Data Quality:")
        print(f"   ✅ {df['organization_founded_year'].notna().sum()/len(df)*100:.1f}% of records have founded year data")
        print(f"   ⚠️  {age_distribution.get('Unknown', 0)} records lack company age information")
        
        print("\n📈 2. Engagement Patterns:")
        if len(known_engagement) > 0:
            best_age = known_engagement['open_rate'].idxmax()
            worst_age = known_engagement['open_rate'].idxmin()
            print(f"   🏆 Best performing company age: {best_age} ({known_engagement.loc[best_age, 'open_rate']:.1f}% open rate)")
            print(f"   📉 Lowest performing company age: {worst_age} ({known_engagement.loc[worst_age, 'open_rate']:.1f}% open rate)")
        
        print("\n💡 3. Strategic Recommendations:")
        print("   🔍 Focus on data enrichment to improve company age coverage")
        print("   🎯 Target campaigns based on company age performance")
        print("   📝 Develop age-specific messaging strategies")
        print("   ⭐ Prioritize high-performing company age segments")
        
        print("\n🚀 4. Actionable Insights:")
        print("   🏢 Company age can significantly impact engagement rates")
        print("   📊 Younger companies may have different engagement patterns")
        print("   🌍 Geographic distribution varies by company age")
        print("   👥 Seniority targeting should consider company age")
        
        print("\n📋 5. Next Steps:")
        print("   🎯 Prioritize campaigns targeting high-performing company ages")
        print("   📈 Develop age-specific engagement strategies")
        print("   🔄 Improve data quality for the remaining records")
        print("   📊 Monitor performance by company age segments")
    else:
        print("⚠️ LIMITED INSIGHTS DUE TO DATA QUALITY:")
        print("📊 1. Data Quality Issues:")
        print("   ❌ Most records lack company age information")
        print("   🔧 Need to improve data enrichment processes")
        
        print("\n💡 2. Recommendations:")
        print("   📥 Implement better data collection for company age")
        print("   🔍 Use alternative data sources for company information")
        print("   🎯 Focus on other segmentation variables (seniority, industry, etc.)")
    
    return df

if __name__ == "__main__":
    df = analyze_founded_year() 