"""
Link Tracking Impact Analysis
Focused analysis of how link_tracking settings influence email open rates
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

def analyze_link_tracking_impact():
    """Analyze how link_tracking settings impact email open rates"""
    print("Loading data...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns to datetime
    if 'timestamp_created_x' in df.columns:
        df['timestamp_created_x'] = pd.to_datetime(df['timestamp_created_x'], utc=True, errors='coerce')
    
    print("\n" + "="*80)
    print("LINK TRACKING IMPACT ON OPEN RATES ANALYSIS")
    print("="*80)
    
    # 1. Link Tracking Setting Overview
    print("\n1. LINK TRACKING SETTINGS OVERVIEW:")
    print("-" * 60)
    
    link_tracking_dist = df['link_tracking'].value_counts()
    total_contacts = len(df)
    
    print("Link tracking distribution:")
    for setting, count in link_tracking_dist.items():
        percentage = (count / total_contacts) * 100
        print(f"  {setting}: {count:,} contacts ({percentage:.1f}%)")
    
    # 2. Open Rate Performance by Link Tracking Setting
    print("\n2. OPEN RATE PERFORMANCE BY LINK TRACKING:")
    print("-" * 60)
    
    performance_comparison = df.groupby('link_tracking').agg({
        'email_open_count': ['sum', 'mean', 'std'],
        'id': 'count'
    }).round(3)
    performance_comparison.columns = ['total_opens', 'avg_opens_per_contact', 'std_opens', 'contacts']
    
    print("Performance comparison:")
    for setting, row in performance_comparison.iterrows():
        # Calculate contacts with opens
        subset = df[df['link_tracking'] == setting]
        contacts_with_opens = (subset['email_open_count'] > 0).sum()
        open_rate = contacts_with_opens / row['contacts'] if row['contacts'] > 0 else 0
        
        print(f"\n🔗 Link Tracking {setting}:")
        print(f"  📊 Contacts: {row['contacts']:,}")
        print(f"  📈 Total Opens: {row['total_opens']:,}")
        print(f"  📊 Avg Opens/Contact: {row['avg_opens_per_contact']:.3f}")
        print(f"  🎯 Open Rate: {open_rate:.1%} ({contacts_with_opens:,} contacts opened)")
        print(f"  📏 Std Deviation: {row['std_opens']:.3f}")
    
    # Calculate performance difference
    if len(performance_comparison) == 2:
        settings = list(performance_comparison.index)
        true_setting = True if True in settings else settings[0]
        false_setting = False if False in settings else settings[1]
        
        if true_setting in performance_comparison.index and false_setting in performance_comparison.index:
            true_opens = performance_comparison.loc[true_setting, 'avg_opens_per_contact']
            false_opens = performance_comparison.loc[false_setting, 'avg_opens_per_contact']
            
            if false_opens > 0:
                improvement = ((true_opens - false_opens) / false_opens) * 100
                print(f"\n📈 PERFORMANCE IMPACT:")
                if improvement > 0:
                    print(f"  ✅ Link tracking IMPROVES opens by {improvement:.1f}%")
                else:
                    print(f"  ❌ Link tracking REDUCES opens by {abs(improvement):.1f}%")
    
    # 3. Timeline Analysis
    print("\n3. TIMELINE ANALYSIS OF LINK TRACKING IMPACT:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        # Monthly timeline by link tracking setting
        monthly_analysis = df.groupby([
            pd.Grouper(key='timestamp_created_x', freq='ME'),
            'link_tracking'
        ]).agg({
            'email_open_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        
        monthly_analysis.columns = ['total_opens', 'avg_opens_per_contact', 'contacts']
        monthly_analysis = monthly_analysis[monthly_analysis['contacts'] > 0]
        
        print("📅 Monthly Timeline by Link Tracking Setting:")
        
        # Reorganize for better display
        timeline_data = {}
        for (month, setting), row in monthly_analysis.iterrows():
            month_str = month.strftime('%Y-%m')
            if month_str not in timeline_data:
                timeline_data[month_str] = {}
            
            # Calculate open rate
            month_subset = df[
                (df['timestamp_created_x'].dt.to_period('M') == pd.Period(month, 'M')) &
                (df['link_tracking'] == setting)
            ]
            contacts_with_opens = (month_subset['email_open_count'] > 0).sum()
            open_rate = contacts_with_opens / row['contacts'] if row['contacts'] > 0 else 0
            
            timeline_data[month_str][setting] = {
                'contacts': row['contacts'],
                'total_opens': row['total_opens'],
                'avg_opens': row['avg_opens_per_contact'],
                'open_rate': open_rate,
                'contacts_with_opens': contacts_with_opens
            }
        
        for month_str in sorted(timeline_data.keys()):
            print(f"\n  📅 {month_str}:")
            for setting, data in timeline_data[month_str].items():
                print(f"    🔗 Link Tracking {setting}: {data['total_opens']:.0f} opens | "
                      f"{data['contacts_with_opens']:,} openers | {data['contacts']:,} contacts | "
                      f"{data['open_rate']:.1%} open rate | {data['avg_opens']:.3f} avg")
    
    # 4. Weekly Pattern Analysis
    print("\n4. WEEKLY PATTERNS BY LINK TRACKING:")
    print("-" * 60)
    
    if 'timestamp_created_x' in df.columns:
        df['weekday'] = df['timestamp_created_x'].dt.day_name()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        weekly_analysis = df.groupby(['weekday', 'link_tracking']).agg({
            'email_open_count': ['sum', 'mean'],
            'id': 'count'
        }).round(3)
        weekly_analysis.columns = ['total_opens', 'avg_opens_per_contact', 'contacts']
        
        print("Weekly performance by link tracking setting:")
        for day in weekday_order:
            if day in weekly_analysis.index.get_level_values(0):
                print(f"\n  📅 {day}:")
                day_data = weekly_analysis.loc[day]
                for setting, row in day_data.iterrows():
                    if row['contacts'] > 0:
                        day_subset = df[(df['weekday'] == day) & (df['link_tracking'] == setting)]
                        contacts_with_opens = (day_subset['email_open_count'] > 0).sum()
                        open_rate = contacts_with_opens / row['contacts']
                        print(f"    🔗 {setting}: {row['total_opens']:.0f} opens | "
                              f"{contacts_with_opens:,} openers | {open_rate:.1%} rate")
    
    # 5. Statistical Significance Test
    print("\n5. STATISTICAL ANALYSIS:")
    print("-" * 60)
    
    # Separate groups for statistical testing
    tracking_enabled = df[df['link_tracking'] == True]['email_open_count']
    tracking_disabled = df[df['link_tracking'] == False]['email_open_count']
    
    if len(tracking_enabled) > 0 and len(tracking_disabled) > 0:
        from scipy import stats
        
        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(tracking_enabled, tracking_disabled)
        
        print(f"📊 T-test Results:")
        print(f"  T-statistic: {t_stat:.4f}")
        print(f"  P-value: {p_value:.6f}")
        
        if p_value < 0.05:
            print(f"  ✅ STATISTICALLY SIGNIFICANT difference (p < 0.05)")
        else:
            print(f"  ❌ NOT statistically significant (p >= 0.05)")
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(tracking_enabled) - 1) * tracking_enabled.var() + 
                             (len(tracking_disabled) - 1) * tracking_disabled.var()) / 
                            (len(tracking_enabled) + len(tracking_disabled) - 2))
        cohens_d = (tracking_enabled.mean() - tracking_disabled.mean()) / pooled_std
        
        print(f"  📏 Effect size (Cohen's d): {cohens_d:.4f}")
        if abs(cohens_d) < 0.2:
            print(f"  📊 Small effect size")
        elif abs(cohens_d) < 0.5:
            print(f"  📊 Medium effect size")
        else:
            print(f"  📊 Large effect size")
    
    # 6. Create Visualizations
    print("\n6. CREATING LINK TRACKING IMPACT DASHBOARD:")
    print("-" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Link Tracking Impact on Open Rates Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Emails Sent Timeline by Link Tracking
    ax = axes[0, 0]
    if 'timestamp_created_x' in df.columns:
        for setting in [True, False]:
            setting_data = df[df['link_tracking'] == setting]
            if len(setting_data) > 0:
                monthly_emails = setting_data.groupby(
                    pd.Grouper(key='timestamp_created_x', freq='ME')
                ).size()
                
                if len(monthly_emails) > 0:
                    monthly_emails.plot(ax=ax, marker='o', label=f'Tracking {setting}', linewidth=2)
        
        ax.set_title('Emails Sent Timeline by Link Tracking')
        ax.set_ylabel('Number of Emails Sent')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Enhanced Box and Whisker Plot
    ax = axes[0, 1]
    tracking_data = []
    labels = []
    colors = ['lightcoral', 'lightblue']
    
    for i, setting in enumerate([True, False]):
        subset = df[df['link_tracking'] == setting]['email_open_count']
        if len(subset) > 0:
            tracking_data.append(subset)
            labels.append(f'Tracking {setting}')
    
    if tracking_data:
        bp = ax.boxplot(tracking_data, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Email Opens Distribution Comparison')
        ax.set_ylabel('Number of Opens')
        ax.grid(True, alpha=0.3)
        
        # Add mean markers
        for i, data in enumerate(tracking_data):
            mean_val = data.mean()
            ax.plot(i+1, mean_val, marker='D', color='red', markersize=8, label='Mean' if i == 0 else "")
        
        if len(tracking_data) > 0:
            ax.legend()
    
    # Plot 3: Monthly Timeline
    ax = axes[0, 2]
    if 'timestamp_created_x' in df.columns:
        for setting in [True, False]:
            setting_data = df[df['link_tracking'] == setting]
            if len(setting_data) > 0:
                monthly_opens = setting_data.groupby(
                    pd.Grouper(key='timestamp_created_x', freq='ME')
                )['email_open_count'].mean()
                
                if len(monthly_opens) > 0:
                    monthly_opens.plot(ax=ax, marker='o', label=f'Tracking {setting}', linewidth=2)
        
        ax.set_title('Monthly Average Opens Timeline')
        ax.set_ylabel('Average Opens per Contact')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Plot 4: Weekly Patterns
    ax = axes[1, 0]
    if 'weekday' in df.columns:
        weekly_data = df.groupby(['weekday', 'link_tracking'])['email_open_count'].mean().unstack()
        if not weekly_data.empty:
            weekly_data.reindex(weekday_order).plot(kind='bar', ax=ax, width=0.8)
            ax.set_title('Weekly Pattern by Link Tracking')
            ax.set_ylabel('Average Opens')
            ax.legend(title='Link Tracking')
            ax.tick_params(axis='x', rotation=45)
    
    # Plot 5: Distribution Comparison
    ax = axes[1, 1]
    for setting in [True, False]:
        subset = df[df['link_tracking'] == setting]['email_open_count']
        if len(subset) > 0:
            subset.hist(ax=ax, alpha=0.7, bins=20, label=f'Tracking {setting}', density=True)
    
    ax.set_title('Opens Distribution by Link Tracking')
    ax.set_xlabel('Number of Opens')
    ax.set_ylabel('Density')
    ax.legend()
    
    # Plot 6: Contact Volume Comparison
    ax = axes[1, 2]
    if 'timestamp_created_x' in df.columns:
        volume_data = []
        volume_labels = []
        volume_colors = ['lightcoral', 'lightblue']
        
        for setting in [True, False]:
            subset = df[df['link_tracking'] == setting]
            if len(subset) > 0:
                volume_data.append(len(subset))
                volume_labels.append(f'Tracking {setting}')
        
        if volume_data:
            bars = ax.bar(volume_labels, volume_data, color=volume_colors)
            ax.set_title('Total Contact Volume by Link Tracking')
            ax.set_ylabel('Number of Contacts')
            
            # Add value labels
            for bar, volume in zip(bars, volume_data):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(volume_data)*0.01,
                       f'{volume:,}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Timestamp data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Contact Volume by Link Tracking')
    
    plt.tight_layout()
    
    # Save dashboard
    dashboard_filename = 'link_tracking_impact_dashboard.png'
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"✅ Link tracking impact dashboard saved as '{dashboard_filename}'")
    
    # 7. Business Insights and Recommendations
    print("\n7. BUSINESS INSIGHTS AND RECOMMENDATIONS:")
    print("="*80)
    
    tracking_true_avg = df[df['link_tracking'] == True]['email_open_count'].mean()
    tracking_false_avg = df[df['link_tracking'] == False]['email_open_count'].mean()
    
    tracking_true_rate = (df[df['link_tracking'] == True]['email_open_count'] > 0).mean()
    tracking_false_rate = (df[df['link_tracking'] == False]['email_open_count'] > 0).mean()
    
    print("🔍 KEY FINDINGS:")
    print(f"• Link tracking enabled: {tracking_true_avg:.3f} avg opens, {tracking_true_rate:.1%} open rate")
    print(f"• Link tracking disabled: {tracking_false_avg:.3f} avg opens, {tracking_false_rate:.1%} open rate")
    
    if tracking_true_avg > tracking_false_avg:
        print("• ✅ Link tracking appears to IMPROVE email open performance")
    else:
        print("• ❌ Link tracking appears to REDUCE email open performance")
    
    print("\n📊 STRATEGIC RECOMMENDATIONS:")
    if tracking_true_avg > tracking_false_avg:
        print("• ✅ ENABLE link tracking for all campaigns to maximize open rates")
        print("• 📈 Link tracking likely provides valuable analytics without hurting performance")
        print("• 🎯 Use link tracking data to optimize email content and timing")
    else:
        print("• ⚠️ CONSIDER DISABLING link tracking if open rates are priority")
        print("• 🔍 Test with/without link tracking on small segments first")
        print("• ⚖️ Weigh tracking benefits vs. potential open rate impact")
    
    print("• 📊 Monitor performance continuously as tracking technology evolves")
    print("• 🧪 A/B test link tracking settings with different email types")
    print("• 📈 Combine with other performance metrics for comprehensive analysis")
    
    plt.show()
    return df

if __name__ == "__main__":
    df = analyze_link_tracking_impact() 