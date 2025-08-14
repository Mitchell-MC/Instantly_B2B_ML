"""
Employee Count Analysis Dashboard
Comprehensive analysis of company headcount and its impact on email performance
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

def analyze_employee_count():
    """Comprehensive employee count analysis with performance correlations"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    if 'timestamp_created_x' in df.columns:
        df['timestamp_created_x'] = pd.to_datetime(df['timestamp_created_x'], utc=True, errors='coerce')
    
    # Identify employee count column
    employee_cols = [col for col in df.columns if 'employee' in col.lower() or 'headcount' in col.lower()]
    print(f"Employee-related columns found: {employee_cols}")
    
    # Use the first employee column found, or check for common names
    employee_col = None
    for col in ['employees', 'employee_count', 'headcount', 'company_size']:
        if col in df.columns:
            employee_col = col
            break
    
    if not employee_col and employee_cols:
        employee_col = employee_cols[0]
    
    if not employee_col:
        print("❌ No employee count column found. Checking all numeric columns...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"Available numeric columns: {list(numeric_cols)}")
        return
    
    print(f"Using employee column: {employee_col}")
    
    print("\n" + "="*80)
    print("EMPLOYEE COUNT ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Employee Count Overview
    print("\n1. EMPLOYEE COUNT OVERVIEW:")
    print("-" * 60)
    
    total_contacts = len(df)
    valid_employee_data = df[df[employee_col].notna()]
    
    print(f"📊 Total Contacts: {total_contacts:,}")
    print(f"📈 Contacts with Employee Data: {len(valid_employee_data):,}")
    print(f"📊 Data Coverage: {len(valid_employee_data)/total_contacts:.1%}")
    
    if len(valid_employee_data) > 0:
        print(f"🏢 Min Employees: {valid_employee_data[employee_col].min():,}")
        print(f"🏢 Max Employees: {valid_employee_data[employee_col].max():,}")
        print(f"🏢 Median Employees: {valid_employee_data[employee_col].median():,}")
        print(f"🏢 Mean Employees: {valid_employee_data[employee_col].mean():,.0f}")
    
    # 2. Company Size Categorization
    print("\n2. COMPANY SIZE CATEGORIZATION:")
    print("-" * 60)
    
    def categorize_company_size(employee_count):
        if pd.isna(employee_count):
            return 'Unknown'
        elif employee_count <= 10:
            return 'Micro (1-10)'
        elif employee_count <= 50:
            return 'Small (11-50)'
        elif employee_count <= 200:
            return 'Medium (51-200)'
        elif employee_count <= 1000:
            return 'Large (201-1K)'
        else:
            return 'Enterprise (1K+)'
    
    df['company_size_category'] = df[employee_col].apply(categorize_company_size)
    
    size_distribution = df['company_size_category'].value_counts()
    print("Company size distribution:")
    for category, count in size_distribution.items():
        percentage = (count / total_contacts) * 100
        print(f"  {category}: {count:,} contacts ({percentage:.1f}%)")
    
    # 3. Performance by Company Size
    print("\n3. PERFORMANCE BY COMPANY SIZE:")
    print("-" * 60)
    
    performance_by_size = df.groupby('company_size_category').agg({
        'email_open_count': ['sum', 'mean', 'std'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean'],
        'id': 'count'
    }).round(3)
    
    performance_by_size.columns = [
        'total_opens', 'avg_opens', 'std_opens',
        'total_clicks', 'avg_clicks',
        'total_replies', 'avg_replies',
        'contacts'
    ]
    
    print("Performance metrics by company size:")
    for category, row in performance_by_size.iterrows():
        if row['contacts'] > 0:
            # Calculate engagement rates
            subset = df[df['company_size_category'] == category]
            open_rate = (subset['email_open_count'] > 0).mean()
            click_rate = (subset['email_click_count'] > 0).mean()
            reply_rate = (subset['email_reply_count'] > 0).mean()
            
            print(f"\n🏢 {category}:")
            print(f"  📊 Contacts: {row['contacts']:,}")
            print(f"  📈 Avg Opens: {row['avg_opens']:.3f} | Open Rate: {open_rate:.1%}")
            print(f"  🖱️ Avg Clicks: {row['avg_clicks']:.3f} | Click Rate: {click_rate:.1%}")
            print(f"  💬 Avg Replies: {row['avg_replies']:.3f} | Reply Rate: {reply_rate:.1%}")
            if len(subset) > 0:
                avg_employees = subset[employee_col].median()
                print(f"  👥 Median Employees: {avg_employees:,.0f}")
    
    # 4. Timeline Analysis by Company Size
    print("\n4. TIMELINE ANALYSIS BY COMPANY SIZE:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        monthly_by_size = df.groupby([
            pd.Grouper(key='timestamp_created_x', freq='ME'),
            'company_size_category'
        ]).agg({
            'email_open_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        
        monthly_by_size.columns = ['total_opens', 'avg_opens', 'contacts']
        monthly_by_size = monthly_by_size[monthly_by_size['contacts'] > 0]
        
        print("📅 Monthly Performance by Company Size:")
        
        # Reorganize for display
        timeline_data = {}
        for (month, category), row in monthly_by_size.iterrows():
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
    
    # 5. Employee Count Correlation Analysis
    print("\n5. EMPLOYEE COUNT CORRELATION ANALYSIS:")
    print("-" * 60)
    
    if len(valid_employee_data) > 0:
        # Correlations with performance metrics
        correlations = {}
        correlations['opens'] = valid_employee_data[employee_col].corr(valid_employee_data['email_open_count'])
        correlations['clicks'] = valid_employee_data[employee_col].corr(valid_employee_data['email_click_count'])
        correlations['replies'] = valid_employee_data[employee_col].corr(valid_employee_data['email_reply_count'])
        
        print("Correlation between employee count and performance:")
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
    
    # 6. Employee Range Analysis
    print("\n6. EMPLOYEE RANGE PERFORMANCE ANALYSIS:")
    print("-" * 60)
    
    if len(valid_employee_data) > 0:
        # Create more granular employee ranges
        employee_bins = [0, 5, 10, 25, 50, 100, 250, 500, 1000, 5000, float('inf')]
        employee_labels = ['1-5', '6-10', '11-25', '26-50', '51-100', '101-250', 
                          '251-500', '501-1K', '1K-5K', '5K+']
        
        df['employee_range'] = pd.cut(df[employee_col], bins=employee_bins, 
                                     labels=employee_labels, include_lowest=True)
        
        range_performance = df.groupby('employee_range').agg({
            'email_open_count': ['count', 'sum', 'mean'],
            'email_click_count': 'mean',
            employee_col: 'median'
        }).round(3)
        
        range_performance.columns = ['contacts', 'total_opens', 'avg_opens', 'avg_clicks', 'median_employees']
        range_performance = range_performance[range_performance['contacts'] > 0]
        
        print("Performance by employee range:")
        for emp_range, row in range_performance.iterrows():
            if row['contacts'] >= 10:  # Only show ranges with meaningful sample sizes
                print(f"  👥 {emp_range} employees: {row['avg_opens']:.3f} avg opens | "
                      f"{row['avg_clicks']:.3f} avg clicks | {row['contacts']:,} contacts")
    
    # 7. Create Comprehensive Dashboard
    print("\n7. CREATING EMPLOYEE COUNT DASHBOARD:")
    print("-" * 60)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Employee Count Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Company Size Distribution
    ax = axes[0, 0]
    size_counts = df['company_size_category'].value_counts()
    colors = plt.cm.Set3(np.linspace(0, 1, len(size_counts)))
    wedges, texts, autotexts = ax.pie(size_counts.values, labels=size_counts.index, 
                                     autopct='%1.1f%%', colors=colors)
    ax.set_title('Company Size Distribution')
    
    # Plot 2: Employee Count Histogram
    ax = axes[0, 1]
    if len(valid_employee_data) > 0:
        # Log scale for better visualization
        log_employees = np.log10(valid_employee_data[employee_col] + 1)
        ax.hist(log_employees, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Employee Count Distribution (Log Scale)')
        ax.set_xlabel('Log10(Employees + 1)')
        ax.set_ylabel('Frequency')
    
    # Plot 3: Performance by Company Size
    ax = axes[0, 2]
    if not performance_by_size.empty:
        performance_by_size['avg_opens'].plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_title('Average Opens by Company Size')
        ax.set_ylabel('Average Opens per Contact')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Employee Count vs Opens Scatter
    ax = axes[1, 0]
    if len(valid_employee_data) > 0:
        # Sample data for better visualization if dataset is large
        sample_size = min(1000, len(valid_employee_data))
        sample_data = valid_employee_data.sample(sample_size) if len(valid_employee_data) > sample_size else valid_employee_data
        
        ax.scatter(sample_data[employee_col], sample_data['email_open_count'], 
                  alpha=0.6, color='coral')
        ax.set_xlabel('Number of Employees')
        ax.set_ylabel('Email Opens')
        ax.set_title('Employee Count vs Email Opens')
        ax.set_xscale('log')
    
    # Plot 5: Monthly Timeline by Company Size
    ax = axes[1, 1]
    if 'timestamp_created_x' in df.columns:
        size_categories = ['Micro (1-10)', 'Small (11-50)', 'Medium (51-200)', 'Large (201-1K)', 'Enterprise (1K+)']
        colors = plt.cm.tab10(np.linspace(0, 1, len(size_categories)))
        
        for i, category in enumerate(size_categories):
            category_data = df[df['company_size_category'] == category]
            if len(category_data) > 0:
                monthly_opens = category_data.groupby(
                    pd.Grouper(key='timestamp_created_x', freq='ME')
                )['email_open_count'].mean()
                
                if len(monthly_opens) > 0:
                    monthly_opens.plot(ax=ax, marker='o', label=category, 
                                     color=colors[i], linewidth=2)
        
        ax.set_title('Monthly Average Opens by Company Size')
        ax.set_ylabel('Average Opens per Contact')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 6: Box Plot of Opens by Company Size
    ax = axes[1, 2]
    box_data = []
    box_labels = []
    for category in ['Micro (1-10)', 'Small (11-50)', 'Medium (51-200)', 'Large (201-1K)', 'Enterprise (1K+)']:
        subset = df[df['company_size_category'] == category]['email_open_count']
        if len(subset) > 0:
            box_data.append(subset)
            box_labels.append(category.split('(')[0].strip())
    
    if box_data:
        bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Opens Distribution by Company Size')
        ax.set_ylabel('Number of Opens')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 7: Employee Range Performance
    ax = axes[2, 0]
    if 'employee_range' in df.columns and not range_performance.empty:
        significant_ranges = range_performance[range_performance['contacts'] >= 10]
        if not significant_ranges.empty:
            significant_ranges['avg_opens'].plot(kind='bar', ax=ax, color='gold')
            ax.set_title('Performance by Employee Range')
            ax.set_ylabel('Average Opens')
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 8: Click Rate vs Employee Count
    ax = axes[2, 1]
    # Vectorized: mean of boolean indicates rate
    click_rates_by_size = (df.assign(clicked=(df['email_click_count'] > 0).astype(float))
                             .groupby('company_size_category')['clicked']
                             .mean())
    if not click_rates_by_size.empty:
        click_rates_by_size.plot(kind='bar', ax=ax, color='purple')
        ax.set_title('Click Rate by Company Size')
        ax.set_ylabel('Click Rate')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 9: Correlation Heatmap
    ax = axes[2, 2]
    if len(valid_employee_data) > 0:
        corr_data = valid_employee_data[[employee_col, 'email_open_count', 'email_click_count', 'email_reply_count']].corr()
        im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr_data.columns)))
        ax.set_yticks(range(len(corr_data.columns)))
        ax.set_xticklabels(['Employees', 'Opens', 'Clicks', 'Replies'], rotation=45)
        ax.set_yticklabels(['Employees', 'Opens', 'Clicks', 'Replies'])
        ax.set_title('Performance Correlation Matrix')
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                ax.text(j, i, f'{corr_data.iloc[i, j]:.2f}', ha='center', va='center')
        
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    
    # Save dashboard
    dashboard_filename = 'employee_count_analysis_dashboard.png'
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Employee count dashboard saved as '{dashboard_filename}'")
    
    # 8. Business Insights and Recommendations
    print("\n8. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("="*80)
    
    if not performance_by_size.empty:
        best_performing_size = performance_by_size['avg_opens'].idxmax()
        worst_performing_size = performance_by_size['avg_opens'].idxmin()
        best_performance = performance_by_size.loc[best_performing_size, 'avg_opens']
        worst_performance = performance_by_size.loc[worst_performing_size, 'avg_opens']
        
        print("🔍 KEY FINDINGS:")
        print(f"• Best performing company size: {best_performing_size} ({best_performance:.3f} avg opens)")
        print(f"• Worst performing company size: {worst_performing_size} ({worst_performance:.3f} avg opens)")
        
        if len(valid_employee_data) > 0:
            overall_corr = valid_employee_data[employee_col].corr(valid_employee_data['email_open_count'])
            print(f"• Overall correlation between company size and opens: {overall_corr:.3f}")
        
        print("\n📊 STRATEGIC RECOMMENDATIONS:")
        print(f"• Focus targeting efforts on {best_performing_size} companies for maximum engagement")
        print("• Customize messaging and approach based on company size categories")
        print("• Consider different email sequences for different company sizes")
        print("• Monitor performance trends across company sizes over time")
        print("• Use company size as a key segmentation variable in campaigns")
        
        if len(valid_employee_data) > 0 and overall_corr > 0.1:
            print("• Larger companies show better engagement - prioritize enterprise accounts")
        elif len(valid_employee_data) > 0 and overall_corr < -0.1:
            print("• Smaller companies show better engagement - focus on SMB market")
        else:
            print("• Company size has minimal impact on engagement - focus on other factors")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = analyze_employee_count() 