"""
Performance vs Volume Analysis
Creates scatter plots showing the relationship between contact volume and engagement performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_performance_vs_volume():
    """Analyze performance vs volume for different categories"""
    print("Loading data for performance vs volume analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    print("\n" + "="*80)
    print("PERFORMANCE VS VOLUME ANALYSIS")
    print("="*80)
    
    # 1. Company Size Performance vs Volume
    print("\n1. COMPANY SIZE PERFORMANCE VS VOLUME:")
    print("-" * 50)
    
    # Create company size categories
    df['company_size'] = 'Unknown'
    if 'organization_employees' in df.columns:
        valid_mask = df['organization_employees'].notna() & (df['organization_employees'] > 0)
        df.loc[valid_mask, 'company_size'] = pd.cut(
            df.loc[valid_mask, 'organization_employees'],
            bins=[0, 10, 50, 200, 1000, 10000, float('inf')],
            labels=['1-10', '11-50', '51-200', '201-1000', '1001-10000', '10000+']
        )
    
    # Calculate metrics by company size
    company_size_metrics = df.groupby('company_size').agg({
        'email_open_count': ['count', 'sum', 'mean']
    }).round(3)
    company_size_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
    company_size_metrics['Open_Rate'] = (company_size_metrics['Total_Opens'] / company_size_metrics['Volume']).round(3)
    
    print("Company Size Metrics:")
    print(company_size_metrics)
    
    # 2. Seniority Performance vs Volume
    print("\n2. SENIORITY PERFORMANCE VS VOLUME:")
    print("-" * 50)
    
    if 'seniority' in df.columns:
        seniority_metrics = df.groupby('seniority').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        seniority_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        seniority_metrics['Open_Rate'] = (seniority_metrics['Total_Opens'] / seniority_metrics['Volume']).round(3)
        
        print("Seniority Metrics:")
        print(seniority_metrics)
    
    # 3. Status X Performance vs Volume
    print("\n3. STATUS X PERFORMANCE VS VOLUME:")
    print("-" * 50)
    
    if 'status_x' in df.columns:
        status_metrics = df.groupby('status_x').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        status_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        status_metrics['Open_Rate'] = (status_metrics['Total_Opens'] / status_metrics['Volume']).round(3)
        
        print("Status X Metrics:")
        print(status_metrics)
    
    # 4. Country Performance vs Volume
    print("\n4. COUNTRY PERFORMANCE VS VOLUME:")
    print("-" * 50)
    
    if 'country' in df.columns:
        # Get top 10 countries by volume
        top_countries = df['country'].value_counts().head(10).index
        country_metrics = df[df['country'].isin(top_countries)].groupby('country').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        country_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        country_metrics['Open_Rate'] = (country_metrics['Total_Opens'] / country_metrics['Volume']).round(3)
        
        print("Top 10 Countries Metrics:")
        print(country_metrics)
    
    # 5. Create Visualizations
    print("\n5. CREATING PERFORMANCE VS VOLUME VISUALIZATIONS:")
    print("-" * 50)
    
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Company Size Performance vs Volume
    plt.subplot(2, 3, 1)
    create_performance_scatter(company_size_metrics, 'Company Size', 'Company Size Performance vs Volume')
    
    # Subplot 2: Seniority Performance vs Volume
    if 'seniority' in df.columns:
        plt.subplot(2, 3, 2)
        create_performance_scatter(seniority_metrics, 'Seniority', 'Seniority Performance vs Volume')
    
    # Subplot 3: Status X Performance vs Volume
    if 'status_x' in df.columns:
        plt.subplot(2, 3, 3)
        create_performance_scatter(status_metrics, 'Status X', 'Status X Performance vs Volume')
    
    # Subplot 4: Country Performance vs Volume
    if 'country' in df.columns:
        plt.subplot(2, 3, 4)
        create_performance_scatter(country_metrics, 'Country', 'Country Performance vs Volume (Top 10)')
    
    # Subplot 5: Industry Performance vs Volume
    if 'organization_industry' in df.columns:
        plt.subplot(2, 3, 5)
        # Get top 10 industries by volume
        top_industries = df['organization_industry'].value_counts().head(10).index
        industry_metrics = df[df['organization_industry'].isin(top_industries)].groupby('organization_industry').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        industry_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        industry_metrics['Open_Rate'] = (industry_metrics['Total_Opens'] / industry_metrics['Volume']).round(3)
        
        create_performance_scatter(industry_metrics, 'Industry', 'Industry Performance vs Volume (Top 10)')
    
    # Subplot 6: ESP Code Performance vs Volume
    if 'esp_code' in df.columns:
        plt.subplot(2, 3, 6)
        # Get top 10 ESP codes by volume
        top_esp_codes = df['esp_code'].value_counts().head(10).index
        esp_metrics = df[df['esp_code'].isin(top_esp_codes)].groupby('esp_code').agg({
            'email_open_count': ['count', 'sum', 'mean']
        }).round(3)
        esp_metrics.columns = ['Volume', 'Total_Opens', 'Avg_Opens']
        esp_metrics['Open_Rate'] = (esp_metrics['Total_Opens'] / esp_metrics['Volume']).round(3)
        
        create_performance_scatter(esp_metrics, 'ESP Code', 'ESP Code Performance vs Volume (Top 10)')
    
    plt.tight_layout()
    plt.savefig('performance_vs_volume_analysis.png', dpi=300, bbox_inches='tight')
    print("Performance vs Volume Analysis saved as 'performance_vs_volume_analysis.png'")
    plt.show()
    
    # 6. Business Insights
    print("\n6. BUSINESS INSIGHTS:")
    print("-" * 50)
    
    print("🎯 KEY FINDINGS:")
    
    # Company Size Insights
    if len(company_size_metrics) > 1:
        best_size = company_size_metrics['Open_Rate'].idxmax()
        worst_size = company_size_metrics['Open_Rate'].idxmin()
        print(f"🏢 Company Size:")
        print(f"   🏆 Best performing: {best_size} ({company_size_metrics.loc[best_size, 'Open_Rate']:.3f} open rate)")
        print(f"   📉 Lowest performing: {worst_size} ({company_size_metrics.loc[worst_size, 'Open_Rate']:.3f} open rate)")
    
    # Seniority Insights
    if 'seniority' in df.columns and len(seniority_metrics) > 1:
        best_seniority = seniority_metrics['Open_Rate'].idxmax()
        worst_seniority = seniority_metrics['Open_Rate'].idxmin()
        print(f"\n👥 Seniority:")
        print(f"   🏆 Best performing: {best_seniority} ({seniority_metrics.loc[best_seniority, 'Open_Rate']:.3f} open rate)")
        print(f"   📉 Lowest performing: {worst_seniority} ({seniority_metrics.loc[worst_seniority, 'Open_Rate']:.3f} open rate)")
    
    # Status X Insights
    if 'status_x' in df.columns and len(status_metrics) > 1:
        best_status = status_metrics['Open_Rate'].idxmax()
        worst_status = status_metrics['Open_Rate'].idxmin()
        print(f"\n📊 Status X:")
        print(f"   🏆 Best performing: Status {best_status} ({status_metrics.loc[best_status, 'Open_Rate']:.3f} open rate)")
        print(f"   📉 Lowest performing: Status {worst_status} ({status_metrics.loc[worst_status, 'Open_Rate']:.3f} open rate)")
    
    print("\n💡 STRATEGIC RECOMMENDATIONS:")
    print("   🎯 Focus on high-volume, high-performance segments")
    print("   📈 Optimize campaigns for segments with good volume but low performance")
    print("   🔍 Investigate low-volume, high-performance segments for expansion")
    print("   ⚠️  Avoid low-volume, low-performance segments")
    
    return df

def create_performance_scatter(metrics_df, category_name, title):
    """Create a performance vs volume scatter plot"""
    # Filter out any rows with zero volume or invalid data
    valid_data = metrics_df[metrics_df['Volume'] > 0].copy()
    
    if len(valid_data) == 0:
        plt.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(title)
        return
    
    # Create scatter plot
    plt.scatter(valid_data['Volume'], valid_data['Open_Rate'], 
                s=valid_data['Volume']/100,  # Size based on volume
                alpha=0.7, c=valid_data['Open_Rate'], cmap='viridis')
    
    # Add labels for each point
    for idx, row in valid_data.iterrows():
        plt.annotate(str(idx), (row['Volume'], row['Open_Rate']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Number of Contacts')
    plt.ylabel('Open Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar
    scatter = plt.scatter(valid_data['Volume'], valid_data['Open_Rate'], 
                         s=valid_data['Volume']/100, alpha=0.7, 
                         c=valid_data['Open_Rate'], cmap='viridis')
    plt.colorbar(scatter, label='Open Rate')

if __name__ == "__main__":
    df = analyze_performance_vs_volume() 