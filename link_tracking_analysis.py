"""
Link Tracking Analysis Dashboard
Comprehensive analysis of link tracking performance with timeline breakdowns
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

def analyze_link_tracking():
    """Comprehensive link tracking analysis with timeline breakdowns"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    timestamp_cols = ['timestamp_created_x', 'timestamp_last_click']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
    
    print("\n" + "="*80)
    print("LINK TRACKING ANALYSIS DASHBOARD")
    print("="*80)
    
    # 1. Overview Statistics
    print("\n1. LINK TRACKING OVERVIEW:")
    print("-" * 60)
    total_contacts = len(df)
    total_clicks = df['email_click_count'].sum()
    clicking_contacts = (df['email_click_count'] > 0).sum()
    click_rate = clicking_contacts / total_contacts
    avg_clicks_per_contact = total_clicks / total_contacts
    avg_clicks_per_clicker = total_clicks / clicking_contacts if clicking_contacts > 0 else 0
    
    print(f"📊 Total Contacts: {total_contacts:,}")
    print(f"🖱️ Total Clicks: {total_clicks:,}")
    print(f"👆 Clicking Contacts: {clicking_contacts:,}")
    print(f"📈 Click Rate: {click_rate:.1%}")
    print(f"📊 Avg Clicks per Contact: {avg_clicks_per_contact:.3f}")
    print(f"🎯 Avg Clicks per Clicker: {avg_clicks_per_clicker:.3f}")
    
    # 2. Click Distribution Analysis
    print("\n2. CLICK DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    click_dist = df['email_click_count'].value_counts().sort_index()
    print("Click count distribution:")
    for clicks, count in click_dist.head(10).items():
        percentage = (count / total_contacts) * 100
        print(f"  {clicks} clicks: {count:,} contacts ({percentage:.1f}%)")
    
    # 3. Timeline Analysis
    print("\n3. CLICK TIMELINE ANALYSIS:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        # Daily click activity
        df['date'] = df['timestamp_created_x'].dt.date
        daily_clicks = df.groupby('date').agg({
            'email_click_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        daily_clicks.columns = ['total_clicks', 'avg_clicks_per_contact', 'contacts']
        # Calculate click rate separately
        daily_click_rate = df.groupby('date').apply(lambda x: (x['email_click_count'] > 0).sum() / len(x))
        daily_clicks['click_rate'] = daily_click_rate
        
        # Monthly timeline
        monthly_clicks = df.groupby(pd.Grouper(key='timestamp_created_x', freq='M')).agg({
            'email_click_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        monthly_clicks.columns = ['total_clicks', 'avg_clicks_per_contact', 'contacts']
        monthly_clicks = monthly_clicks[monthly_clicks['contacts'] > 0]
        # Calculate monthly click rate as percentage of contacts who clicked
        monthly_click_rate = df.groupby(pd.Grouper(key='timestamp_created_x', freq='M')).apply(
            lambda x: (x['email_click_count'] > 0).sum() / len(x) if len(x) > 0 else 0
        )
        monthly_clicks['click_rate'] = monthly_click_rate[monthly_clicks.index]
        
        print("📅 Monthly Click Timeline:")
        for month, row in monthly_clicks.iterrows():
            month_str = month.strftime('%Y-%m')
            clickers = (df[df['timestamp_created_x'].dt.to_period('M') == pd.Period(month, 'M')]['email_click_count'] > 0).sum()
            click_rate = clickers / row['contacts'] if row['contacts'] > 0 else 0
            print(f"  {month_str}: {row['total_clicks']:.0f} clicks | {clickers:,} clickers | {row['contacts']:,} contacts | {click_rate:.1%} click rate")
    
    # 4. Click Step Analysis
    print("\n4. CLICK STEP ANALYSIS:")
    print("-" * 60)
    
    if 'email_clicked_step' in df.columns:
        step_analysis = df[df['email_clicked_step'].notna()]
        if len(step_analysis) > 0:
            step_dist = step_analysis['email_clicked_step'].value_counts().sort_index()
            print("Clicks by email step:")
            for step, count in step_dist.items():
                percentage = (count / len(step_analysis)) * 100
                avg_clicks = step_analysis[step_analysis['email_clicked_step'] == step]['email_click_count'].mean()
                print(f"  Step {step}: {count:,} contacts ({percentage:.1f}%) | Avg {avg_clicks:.2f} clicks")
    
    # 5. Click Variant Analysis
    print("\n5. CLICK VARIANT ANALYSIS:")
    print("-" * 60)
    
    if 'email_clicked_variant' in df.columns:
        variant_analysis = df[df['email_clicked_variant'].notna()]
        if len(variant_analysis) > 0:
            variant_dist = variant_analysis['email_clicked_variant'].value_counts().sort_index()
            print("Clicks by email variant:")
            for variant, count in variant_dist.items():
                percentage = (count / len(variant_analysis)) * 100
                avg_clicks = variant_analysis[variant_analysis['email_clicked_variant'] == variant]['email_click_count'].mean()
                print(f"  Variant {variant}: {count:,} contacts ({percentage:.1f}%) | Avg {avg_clicks:.2f} clicks")
    
    # 6. Link Tracking Settings Analysis
    print("\n6. LINK TRACKING SETTINGS:")
    print("-" * 60)
    
    if 'link_tracking' in df.columns:
        link_tracking_dist = df['link_tracking'].value_counts()
        print("Link tracking enabled distribution:")
        for setting, count in link_tracking_dist.items():
            percentage = (count / total_contacts) * 100
            subset = df[df['link_tracking'] == setting]
            avg_clicks = subset['email_click_count'].mean()
            clickers = (subset['email_click_count'] > 0).sum()
            click_rate = clickers / count if count > 0 else 0
            print(f"  {setting}: {count:,} contacts ({percentage:.1f}%) | {avg_clicks:.3f} avg clicks | {click_rate:.1%} click rate")
    
    # 7. High-Performance Click Analysis
    print("\n7. HIGH-PERFORMANCE CLICK ANALYSIS:")
    print("-" * 60)
    
    # Top clickers analysis
    high_clickers = df[df['email_click_count'] >= 5]
    if len(high_clickers) > 0:
        print(f"📈 Contacts with 5+ clicks: {len(high_clickers):,} ({len(high_clickers)/total_contacts:.1%})")
        print(f"🎯 Average clicks among high clickers: {high_clickers['email_click_count'].mean():.2f}")
        
        # Analyze characteristics of high clickers
        if 'email_list' in df.columns:
            top_lists = high_clickers['email_list'].value_counts().head(5)
            print("📋 Top email lists for high clickers:")
            for email_list, count in top_lists.items():
                print(f"  {email_list}: {count} high clickers")
    
    # 8. Performance by Email List Size
    print("\n8. PERFORMANCE BY EMAIL LIST SIZE:")
    print("-" * 60)
    
    if 'email_count' in df.columns:
        # Categorize list sizes
        def categorize_list_size(email_count):
            if pd.isna(email_count):
                return 'Unknown'
            elif 1 <= email_count <= 3:
                return 'Small (1-3)'
            elif 4 <= email_count <= 10:
                return 'Medium (4-10)'
            else:
                return 'Large (11+)'
        
        df['list_size_category'] = df['email_count'].apply(categorize_list_size)
        
        size_performance = df.groupby('list_size_category').agg({
            'email_click_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        size_performance.columns = ['total_clicks', 'avg_clicks_per_contact', 'contacts']
        
        print("Click performance by list size:")
        for size_cat, row in size_performance.iterrows():
            subset = df[df['list_size_category'] == size_cat]
            clickers = (subset['email_click_count'] > 0).sum()
            click_rate = clickers / row['contacts'] if row['contacts'] > 0 else 0
            print(f"  {size_cat}: {row['total_clicks']:.0f} clicks | {clickers:,} clickers | {click_rate:.1%} rate | {row['contacts']:,} contacts")
    
    # 9. Weekly Pattern Analysis
    print("\n9. WEEKLY PATTERN ANALYSIS:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        df['weekday'] = df['timestamp_created_x'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekday_performance = df.groupby('weekday').agg({
            'email_click_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        weekday_performance.columns = ['total_clicks', 'avg_clicks_per_contact', 'contacts']
        weekday_performance = weekday_performance.reindex(weekday_order)
        
        print("Click performance by day of week:")
        for day, row in weekday_performance.iterrows():
            if not pd.isna(row['contacts']) and row['contacts'] > 0:
                subset = df[df['weekday'] == day]
                clickers = (subset['email_click_count'] > 0).sum()
                click_rate = clickers / row['contacts']
                print(f"  {day}: {row['total_clicks']:.0f} clicks | {clickers:,} clickers | {click_rate:.1%} rate")
    
    # 10. Create Visualizations
    print("\n10. CREATING LINK TRACKING DASHBOARD:")
    print("-" * 60)
    
    # Set up the figure
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Link Tracking Analysis Dashboard', fontsize=20, fontweight='bold')
    
    # Plot 1: Click Distribution
    ax = axes[0, 0]
    click_counts = df['email_click_count'].value_counts().sort_index().head(10)
    click_counts.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title('Click Count Distribution')
    ax.set_xlabel('Number of Clicks')
    ax.set_ylabel('Number of Contacts')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Monthly Click Timeline
    ax = axes[0, 1]
    if 'timestamp_created_x' in df.columns:
        monthly_clicks['total_clicks'].plot(ax=ax, marker='o', color='green', linewidth=2)
        ax.set_title('Monthly Click Timeline')
        ax.set_ylabel('Total Clicks')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Click Rate by List Size
    ax = axes[0, 2]
    if 'list_size_category' in df.columns:
        size_click_rates = []
        size_labels = []
        for size_cat in ['Small (1-3)', 'Medium (4-10)', 'Large (11+)']:
            subset = df[df['list_size_category'] == size_cat]
            if len(subset) > 0:
                click_rate = (subset['email_click_count'] > 0).mean()
                size_click_rates.append(click_rate)
                size_labels.append(size_cat)
        
        bars = ax.bar(size_labels, size_click_rates, color=['lightblue', 'orange', 'lightgreen'])
        ax.set_title('Click Rate by List Size')
        ax.set_ylabel('Click Rate')
        ax.set_ylim(0, max(size_click_rates) * 1.1 if size_click_rates else 1)
        
        # Add value labels on bars
        for bar, rate in zip(bars, size_click_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{rate:.1%}', ha='center', va='bottom')
    
    # Plot 4: Weekday Performance
    ax = axes[1, 0]
    if 'weekday' in df.columns:
        weekday_clicks = df.groupby('weekday')['email_click_count'].sum().reindex(weekday_order)
        weekday_clicks.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Clicks by Day of Week')
        ax.set_ylabel('Total Clicks')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 5: Click Step Performance
    ax = axes[1, 1]
    if 'email_clicked_step' in df.columns:
        step_data = df[df['email_clicked_step'].notna()]
        if len(step_data) > 0:
            step_performance = step_data.groupby('email_clicked_step')['email_click_count'].mean()
            step_performance.plot(kind='bar', ax=ax, color='purple')
            ax.set_title('Average Clicks by Email Step')
            ax.set_ylabel('Average Clicks')
            ax.set_xlabel('Email Step')
    
    # Plot 6: Link Tracking vs Performance
    ax = axes[1, 2]
    if 'link_tracking' in df.columns:
        tracking_performance = df.groupby('link_tracking').agg({
            'email_click_count': 'mean'
        })
        tracking_performance.plot(kind='bar', ax=ax, color='teal', legend=False)
        ax.set_title('Avg Clicks by Link Tracking Setting')
        ax.set_ylabel('Average Clicks')
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 7: High Clickers Timeline
    ax = axes[2, 0]
    if 'timestamp_created_x' in df.columns:
        high_clickers_monthly = df[df['email_click_count'] >= 3].groupby(
            pd.Grouper(key='timestamp_created_x', freq='M')
        ).size()
        if len(high_clickers_monthly) > 0:
            high_clickers_monthly.plot(ax=ax, marker='s', color='red', linewidth=2)
            ax.set_title('High Clickers Timeline (3+ clicks)')
            ax.set_ylabel('Number of High Clickers')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 8: Click Rate Heatmap by Month and Weekday
    ax = axes[2, 1]
    if 'timestamp_created_x' in df.columns:
        df['month'] = df['timestamp_created_x'].dt.month
        df['weekday_num'] = df['timestamp_created_x'].dt.dayofweek
        
        # Create click rate heatmap data
        heatmap_data = df.groupby(['month', 'weekday_num']).apply(
            lambda x: (x['email_click_count'] > 0).mean()
        ).unstack(fill_value=0)
        
        if not heatmap_data.empty:
            sns.heatmap(heatmap_data, ax=ax, cmap='YlOrRd', annot=True, fmt='.2f')
            ax.set_title('Click Rate Heatmap (Month vs Weekday)')
            ax.set_xlabel('Day of Week (0=Mon, 6=Sun)')
            ax.set_ylabel('Month')
    
    # Plot 9: Cumulative Clicks Over Time
    ax = axes[2, 2]
    if 'timestamp_created_x' in df.columns:
        daily_cumulative = daily_clicks.sort_index()['total_clicks'].cumsum()
        daily_cumulative.plot(ax=ax, color='darkgreen', linewidth=2)
        ax.set_title('Cumulative Clicks Over Time')
        ax.set_ylabel('Cumulative Clicks')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save the dashboard
    dashboard_filename = 'link_tracking_analysis_dashboard.png'
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Link tracking dashboard saved as '{dashboard_filename}'")
    
    # 11. Business Insights and Recommendations
    print("\n11. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("="*80)
    
    print("🔍 KEY INSIGHTS:")
    print(f"• Overall click engagement is relatively low at {click_rate:.1%}")
    print(f"• When contacts do click, they average {avg_clicks_per_clicker:.2f} clicks per person")
    print(f"• High-value clickers (5+ clicks) represent {len(high_clickers)/total_contacts:.1%} of all contacts")
    
    if 'list_size_category' in df.columns:
        best_size = size_performance['avg_clicks_per_contact'].idxmax()
        print(f"• {best_size} email lists show the best click performance")
    
    print("\n📊 STRATEGIC RECOMMENDATIONS:")
    print("• Focus on improving overall click rates through better content and CTAs")
    print("• Analyze high-performing email steps and variants for optimization")
    print("• Consider segmenting campaigns based on list size performance")
    print("• Monitor weekly patterns to optimize send timing")
    print("• Implement A/B testing on link tracking settings")
    print("• Target high-clicker segments for premium campaigns")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = analyze_link_tracking() 