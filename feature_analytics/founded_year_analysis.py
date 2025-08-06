"""
Founded Year Analysis Dashboard
Comprehensive analysis of company founding year and its impact on email performance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")
plt.style.use('seaborn-v0_8')

def analyze_founded_year():
    """Comprehensive founding year analysis with performance correlations"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    if 'timestamp_created_x' in df.columns:
        df['timestamp_created_x'] = pd.to_datetime(df['timestamp_created_x'], utc=True, errors='coerce')
    
    # Identify founding year column
    founded_cols = [col for col in df.columns if any(word in col.lower() for word in ['found', 'year', 'establish', 'start'])]
    print(f"Founded year related columns found: {founded_cols}")
    
    # Use the first founded year column found, or check for common names
    founded_col = None
    for col in ['founded_year', 'year_founded', 'founded', 'establishment_year']:
        if col in df.columns:
            founded_col = col
            break
    
    if not founded_col and founded_cols:
        founded_col = founded_cols[0]
    
    if not founded_col:
        print("❌ No founding year column found. Checking all numeric columns...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Available numeric columns: {list(numeric_cols)}")
        return
    
    print(f"Using founding year column: {founded_col}")
    
    print("\n" + "="*80)
    print("FOUNDED YEAR ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Founded Year Overview
    print("\n1. FOUNDED YEAR OVERVIEW:")
    print("-" * 60)
    
    current_year = datetime.now().year
    total_contacts = len(df)
    valid_founded_data = df[df[founded_col].notna()]
    
    print(f"📊 Total Contacts: {total_contacts:,}")
    print(f"📈 Contacts with Founded Year Data: {len(valid_founded_data):,}")
    print(f"📊 Data Coverage: {len(valid_founded_data)/total_contacts:.1%}")
    
    if len(valid_founded_data) > 0:
        print(f"🏢 Earliest Founded: {int(valid_founded_data[founded_col].min())}")
        print(f"🏢 Latest Founded: {int(valid_founded_data[founded_col].max())}")
        print(f"🏢 Median Founded Year: {int(valid_founded_data[founded_col].median())}")
        print(f"🏢 Mean Founded Year: {valid_founded_data[founded_col].mean():.0f}")
        
        # Calculate company ages
        df['company_age'] = current_year - df[founded_col]
        valid_age_data = df[df['company_age'].notna() & (df['company_age'] >= 0)]
        
        if len(valid_age_data) > 0:
            print(f"🗓️ Youngest Company: {int(valid_age_data['company_age'].min())} years old")
            print(f"🗓️ Oldest Company: {int(valid_age_data['company_age'].max())} years old")
            print(f"🗓️ Median Company Age: {int(valid_age_data['company_age'].median())} years")
            print(f"🗓️ Average Company Age: {valid_age_data['company_age'].mean():.1f} years")
    
    # 2. Company Age Categorization
    print("\n2. COMPANY AGE CATEGORIZATION:")
    print("-" * 60)
    
    def categorize_company_age(founded_year):
        if pd.isna(founded_year):
            return 'Unknown'
        
        age = current_year - founded_year
        
        if age < 0:
            return 'Invalid'
        elif age <= 3:
            return 'Startup (0-3 years)'
        elif age <= 10:
            return 'Young (4-10 years)'
        elif age <= 20:
            return 'Mature (11-20 years)'
        elif age <= 50:
            return 'Established (21-50 years)'
        else:
            return 'Legacy (50+ years)'
    
    df['company_age_category'] = df[founded_col].apply(categorize_company_age)
    
    age_distribution = df['company_age_category'].value_counts()
    print("Company age distribution:")
    for category, count in age_distribution.items():
        percentage = (count / total_contacts) * 100
        print(f"  {category}: {count:,} contacts ({percentage:.1f}%)")
    
    # 3. Performance by Company Age
    print("\n3. PERFORMANCE BY COMPANY AGE:")
    print("-" * 60)
    
    performance_by_age = df.groupby('company_age_category').agg({
        'email_open_count': ['sum', 'mean', 'std'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean'],
        'id': 'count'
    }).round(3)
    
    performance_by_age.columns = [
        'total_opens', 'avg_opens', 'std_opens',
        'total_clicks', 'avg_clicks',
        'total_replies', 'avg_replies',
        'contacts'
    ]
    
    print("Performance metrics by company age:")
    for category, row in performance_by_age.iterrows():
        if row['contacts'] > 0:
            # Calculate engagement rates
            subset = df[df['company_age_category'] == category]
            open_rate = (subset['email_open_count'] > 0).mean()
            click_rate = (subset['email_click_count'] > 0).mean()
            reply_rate = (subset['email_reply_count'] > 0).mean()
            
            print(f"\n🏢 {category}:")
            print(f"  📊 Contacts: {row['contacts']:,}")
            print(f"  📈 Avg Opens: {row['avg_opens']:.3f} | Open Rate: {open_rate:.1%}")
            print(f"  🖱️ Avg Clicks: {row['avg_clicks']:.3f} | Click Rate: {click_rate:.1%}")
            print(f"  💬 Avg Replies: {row['avg_replies']:.3f} | Reply Rate: {reply_rate:.1%}")
            if len(subset) > 0 and 'company_age' in df.columns:
                avg_age = subset['company_age'].median()
                if not pd.isna(avg_age):
                    print(f"  🗓️ Median Age: {avg_age:.0f} years")
    
    # 4. Timeline Analysis by Company Age
    print("\n4. TIMELINE ANALYSIS BY COMPANY AGE:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        monthly_by_age = df.groupby([
            pd.Grouper(key='timestamp_created_x', freq='ME'),
            'company_age_category'
        ]).agg({
            'email_open_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        
        monthly_by_age.columns = ['total_opens', 'avg_opens', 'contacts']
        monthly_by_age = monthly_by_age[monthly_by_age['contacts'] > 0]
        
        print("📅 Monthly Performance by Company Age:")
        
        # Reorganize for display
        timeline_data = {}
        for (month, category), row in monthly_by_age.iterrows():
            month_str = month.strftime('%Y-%m')
            if month_str not in timeline_data:
                timeline_data[month_str] = {}
            
            timeline_data[month_str][category] = {
                'contacts': row['contacts'],
                'total_opens': row['total_opens'],
                'avg_opens': row['avg_opens']
            }
        
        for month_str in sorted(timeline_data.keys()):
            print(f"\n  📅 {month_str}:")
            for category, data in timeline_data[month_str].items():
                print(f"    🏢 {category}: {data['total_opens']:.0f} opens | "
                      f"{data['contacts']:,} contacts | {data['avg_opens']:.3f} avg")
    
    # 5. Founded Year Decade Analysis
    print("\n5. FOUNDED YEAR DECADE ANALYSIS:")
    print("-" * 60)
    
    if len(valid_founded_data) > 0:
        # Group by decades
        df['founded_decade'] = (df[founded_col] // 10) * 10
        decade_performance = df.groupby('founded_decade').agg({
            'email_open_count': ['count', 'sum', 'mean'],
            'email_click_count': 'mean',
            founded_col: 'median'
        }).round(3)
        
        decade_performance.columns = ['contacts', 'total_opens', 'avg_opens', 'avg_clicks', 'median_year']
        decade_performance = decade_performance[decade_performance['contacts'] > 0]
        
        print("Performance by founding decade:")
        for decade, row in decade_performance.iterrows():
            if row['contacts'] >= 10 and decade >= 1900:  # Only show meaningful decades
                print(f"  📅 {decade:.0f}s: {row['avg_opens']:.3f} avg opens | "
                      f"{row['avg_clicks']:.3f} avg clicks | {row['contacts']:,} contacts")
    
    # 6. Company Age Correlation Analysis
    print("\n6. COMPANY AGE CORRELATION ANALYSIS:")
    print("-" * 60)
    
    if 'company_age' in df.columns and len(valid_age_data) > 0:
        # Correlations with performance metrics
        correlations = {}
        correlations['opens'] = valid_age_data['company_age'].corr(valid_age_data['email_open_count'])
        correlations['clicks'] = valid_age_data['company_age'].corr(valid_age_data['email_click_count'])
        correlations['replies'] = valid_age_data['company_age'].corr(valid_age_data['email_reply_count'])
        
        print("Correlation between company age and performance:")
        for metric, corr in correlations.items():
            print(f"  📊 {metric.title()}: {corr:.3f}")
            if abs(corr) < 0.1:
                print(f"    → Very weak correlation")
            elif abs(corr) < 0.3:
                print(f"    → Weak correlation")
            elif abs(corr) < 0.5:
                print(f"    → Moderate correlation")
            else:
                print(f"    → Strong correlation")
    
    # 7. Industry Evolution Analysis
    print("\n7. INDUSTRY EVOLUTION ANALYSIS:")
    print("-" * 60)
    
    if len(valid_founded_data) > 0:
        # Analyze performance by era
        def categorize_business_era(founded_year):
            if pd.isna(founded_year):
                return 'Unknown'
            elif founded_year < 1980:
                return 'Pre-Digital Era'
            elif founded_year < 1995:
                return 'Early Tech Era'
            elif founded_year < 2000:
                return 'Internet Boom'
            elif founded_year < 2008:
                return 'Social Media Era'
            elif founded_year < 2015:
                return 'Mobile Era'
            else:
                return 'Cloud/AI Era'
        
        df['business_era'] = df[founded_col].apply(categorize_business_era)
        
        era_performance = df.groupby('business_era').agg({
            'email_open_count': ['count', 'mean'],
            'email_click_count': 'mean'
        }).round(3)
        
        era_performance.columns = ['contacts', 'avg_opens', 'avg_clicks']
        era_performance = era_performance[era_performance['contacts'] > 0]
        
        print("Performance by business era:")
        for era, row in era_performance.iterrows():
            if row['contacts'] >= 10:
                print(f"  🕰️ {era}: {row['avg_opens']:.3f} avg opens | "
                      f"{row['avg_clicks']:.3f} avg clicks | {row['contacts']:,} contacts")
    
    # 8. Create Comprehensive Dashboard
    print("\n8. CREATING FOUNDED YEAR DASHBOARD:")
    print("-" * 60)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Founded Year Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Company Age Distribution
    ax = axes[0, 0]
    age_counts = df['company_age_category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(age_counts)))
    wedges, texts, autotexts = ax.pie(age_counts.values, labels=age_counts.index, 
                                     autopct='%1.1f%%', colors=colors)
    ax.set_title('Company Age Distribution')
    
    # Plot 2: Founded Year Histogram
    ax = axes[0, 1]
    if len(valid_founded_data) > 0:
        valid_founded_data[founded_col].hist(bins=30, alpha=0.7, color='skyblue', 
                                           edgecolor='black', ax=ax)
        ax.set_title('Founded Year Distribution')
        ax.set_xlabel('Founded Year')
        ax.set_ylabel('Frequency')
    
    # Plot 3: Performance by Company Age
    ax = axes[0, 2]
    if not performance_by_age.empty:
        performance_by_age['avg_opens'].plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Average Opens by Company Age')
        ax.set_ylabel('Average Opens per Contact')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Company Age vs Opens Scatter
    ax = axes[1, 0]
    if 'company_age' in df.columns and len(valid_age_data) > 0:
        # Sample data for better visualization if dataset is large
        sample_size = min(1000, len(valid_age_data))
        sample_data = valid_age_data.sample(sample_size) if len(valid_age_data) > sample_size else valid_age_data
        
        ax.scatter(sample_data['company_age'], sample_data['email_open_count'], 
                  alpha=0.6, color='coral')
        ax.set_xlabel('Company Age (Years)')
        ax.set_ylabel('Email Opens')
        ax.set_title('Company Age vs Email Opens')
    
    # Plot 5: Monthly Timeline by Company Age
    ax = axes[1, 1]
    if 'timestamp_created_x' in df.columns:
        age_categories = ['Startup (0-3 years)', 'Young (4-10 years)', 'Mature (11-20 years)', 
                         'Established (21-50 years)', 'Legacy (50+ years)']
        colors = plt.cm.tab10(np.linspace(0, 1, len(age_categories)))
        
        for i, category in enumerate(age_categories):
            category_data = df[df['company_age_category'] == category]
            if len(category_data) > 0:
                monthly_opens = category_data.groupby(
                    pd.Grouper(key='timestamp_created_x', freq='ME')
                )['email_open_count'].mean()
                
                if len(monthly_opens) > 0:
                    monthly_opens.plot(ax=ax, marker='o', label=category.split('(')[0].strip(), 
                                     color=colors[i], linewidth=2)
        
        ax.set_title('Monthly Average Opens by Company Age')
        ax.set_ylabel('Average Opens per Contact')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 6: Box Plot of Opens by Company Age
    ax = axes[1, 2]
    box_data = []
    box_labels = []
    for category in ['Startup (0-3 years)', 'Young (4-10 years)', 'Mature (11-20 years)', 
                     'Established (21-50 years)', 'Legacy (50+ years)']:
        subset = df[df['company_age_category'] == category]['email_open_count']
        if len(subset) > 0:
            box_data.append(subset)
            box_labels.append(category.split('(')[0].strip())
    
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Average Opens Distribution by Company Age')
        ax.set_ylabel('Average Opens per Contact')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 7: Decade Performance
    ax = axes[2, 0]
    if 'founded_decade' in df.columns and not decade_performance.empty:
        recent_decades = decade_performance[decade_performance.index >= 1980]
        if not recent_decades.empty:
            recent_decades['avg_opens'].plot(kind='bar', ax=ax, color='gold')
            ax.set_title('Performance by Founded Decade')
            ax.set_ylabel('Average Opens')
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 8: Click Rate vs Company Age
    ax = axes[2, 1]
    if not performance_by_age.empty:
        click_rates_by_age = df.groupby('company_age_category').apply(
            lambda x: (x['email_click_count'] > 0).mean()
        )
        click_rates_by_age.plot(kind='bar', ax=ax, color='purple')
        ax.set_title('Click Rate by Company Age')
        ax.set_ylabel('Click Rate')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 9: Business Era Performance
    ax = axes[2, 2]
    if 'business_era' in df.columns and not era_performance.empty:
        era_performance['avg_opens'].plot(kind='bar', ax=ax, color='orange')
        ax.set_title('Performance by Business Era')
        ax.set_ylabel('Average Opens')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save dashboard
    dashboard_filename = 'founded_year_analysis_dashboard.png'
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Founded year dashboard saved as '{dashboard_filename}'")
    
    # 9. Business Insights and Recommendations
    print("\n9. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("="*80)
    
    if not performance_by_age.empty:
        best_performing_age = performance_by_age['avg_opens'].idxmax()
        worst_performing_age = performance_by_age['avg_opens'].idxmin()
        best_performance = performance_by_age.loc[best_performing_age, 'avg_opens']
        worst_performance = performance_by_age.loc[worst_performing_age, 'avg_opens']
        
        print("🔍 KEY FINDINGS:")
        print(f"• Best performing company age: {best_performing_age} ({best_performance:.3f} avg opens)")
        print(f"• Worst performing company age: {worst_performing_age} ({worst_performance:.3f} avg opens)")
        
        if 'company_age' in df.columns and len(valid_age_data) > 0:
            overall_corr = valid_age_data['company_age'].corr(valid_age_data['email_open_count'])
            print(f"• Overall correlation between company age and opens: {overall_corr:.3f}")
        
        print("\n📊 STRATEGIC RECOMMENDATIONS:")
        print(f"• Focus targeting efforts on {best_performing_age} companies for maximum engagement")
        print("• Customize messaging based on company maturity and business lifecycle stage")
        print("• Consider different value propositions for different company ages")
        print("• Monitor performance trends across company ages over time")
        print("• Use company founding year as a key segmentation variable")
        
        if 'company_age' in df.columns and len(valid_age_data) > 0:
            if overall_corr > 0.1:
                print("• Older companies show better engagement - emphasize stability and experience")
            elif overall_corr < -0.1:
                print("• Younger companies show better engagement - focus on innovation and growth")
            else:
                print("• Company age has minimal impact on engagement - focus on other factors")
        
        # Era-specific insights
        if 'business_era' in df.columns and not era_performance.empty:
            print("\n🕰️ ERA-SPECIFIC INSIGHTS:")
            print("• Tailor messaging to reflect the technological context of each era")
            print("• Pre-digital companies may need more education on digital solutions")
            print("• Cloud/AI era companies may be more receptive to cutting-edge technology")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = analyze_founded_year()