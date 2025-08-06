"""
Comprehensive Daily Limit Code Analysis
Analyze daily sending limits and their impact on email campaigns
Includes timeline analysis, box plots, bucketed analysis, and dashboard visualizations
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_daily_limits():
    """Comprehensive daily limit analysis with timeline and performance insights"""
    print("Loading data for daily limit analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns
    timestamp_cols = ['timestamp_created_x', 'timestamp_created_y', 'timestamp_last_contact', 
                     'timestamp_last_open', 'timestamp_last_click', 'retrieval_timestamp']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DAILY LIMIT ANALYSIS")
    print("="*80)
    
    # 1. Basic Daily Limit Statistics
    print("\n1. DAILY LIMIT BASIC STATISTICS:")
    print("-" * 60)
    
    if 'daily_limit' not in df.columns:
        print("❌ No 'daily_limit' column found in dataset")
        return
    
    # Handle missing values and basic stats
    daily_limit_stats = df['daily_limit'].describe()
    missing_values = df['daily_limit'].isnull().sum()
    missing_pct = (missing_values / len(df)) * 100
    
    print("📊 Basic Statistics:")
    print(f"  Data type: {df['daily_limit'].dtype}")
    print(f"  Missing values: {missing_values:,} ({missing_pct:.1f}%)")
    print(f"  Unique values: {df['daily_limit'].nunique():,}")
    print(f"  Min: {daily_limit_stats['min']}")
    print(f"  25th percentile: {daily_limit_stats['25%']}")
    print(f"  Median: {daily_limit_stats['50%']}")
    print(f"  75th percentile: {daily_limit_stats['75%']}")
    print(f"  Max: {daily_limit_stats['max']}")
    print(f"  Mean: {daily_limit_stats['mean']:.2f}")
    print(f"  Std: {daily_limit_stats['std']:.2f}")
    
    # 2. Daily Limit Distribution Analysis
    print("\n2. DAILY LIMIT DISTRIBUTION ANALYSIS:")
    print("-" * 60)
    
    # Most common daily limits
    limit_counts = df['daily_limit'].value_counts().head(20)
    print("📈 Top 20 Daily Limits by Frequency:")
    for limit, count in limit_counts.items():
        percentage = (count / len(df[df['daily_limit'].notna()])) * 100
        print(f"  {limit}: {count:,} contacts ({percentage:.1f}%)")
    
    # 3. Create Daily Limit Buckets for Analysis
    print("\n3. DAILY LIMIT BUCKET ANALYSIS:")
    print("-" * 60)
    
    # Create meaningful buckets based on distribution
    def categorize_daily_limit(limit):
        """Categorize daily limits into meaningful buckets based on actual data distribution"""
        if pd.isna(limit):
            return 'Unknown'
        elif limit <= 100:
            return 'Low Volume (≤100)'
        elif limit <= 250:
            return 'Medium-Low (101-250)'
        elif limit <= 350:
            return 'Medium (251-350)'
        elif limit <= 450:
            return 'Medium-High (351-450)'
        elif limit <= 650:
            return 'High (451-650)'
        else:
            return 'Very High (651+)'
    
    df['daily_limit_bucket'] = df['daily_limit'].apply(categorize_daily_limit)
    
    # Analyze performance by bucket (focus on opens only)
    bucket_analysis = df.groupby('daily_limit_bucket').agg({
        'email_open_count': ['count', 'sum', 'mean', 'std'],
        'daily_limit': ['mean', 'min', 'max']
    }).round(3)
    
    # Flatten column names
    bucket_analysis.columns = ['_'.join(col).strip() for col in bucket_analysis.columns]
    bucket_analysis.columns = ['contacts', 'total_opens', 'avg_opens', 'opens_std',
                              'avg_limit', 'min_limit', 'max_limit']
    
    # Calculate rates
    bucket_analysis['open_rate'] = bucket_analysis['total_opens'] / bucket_analysis['contacts']
    
    # Sort buckets in logical order
    bucket_order = ['Low Volume (≤100)', 'Medium-Low (101-250)', 'Medium (251-350)', 
                   'Medium-High (351-450)', 'High (451-650)', 'Very High (651+)', 'Unknown']
    bucket_analysis = bucket_analysis.reindex([b for b in bucket_order if b in bucket_analysis.index])
    
    print("📊 Performance by Daily Limit Bucket:")
    for bucket, row in bucket_analysis.iterrows():
        print(f"\n  {bucket}:")
        print(f"    📧 {row['contacts']:,} contacts | Avg limit: {row['avg_limit']:.0f}")
        print(f"    📈 Open rate: {row['open_rate']:.3f} | Avg opens: {row['avg_opens']:.3f}")
        print(f"    📊 Total opens: {row['total_opens']:.0f} | Opens std: {row['opens_std']:.3f}")
    
    # Find optimal bucket
    optimal_bucket = bucket_analysis['open_rate'].idxmax()
    worst_bucket = bucket_analysis['open_rate'].idxmin()
    print(f"\n🏆 Best performing bucket: {optimal_bucket} ({bucket_analysis.loc[optimal_bucket, 'open_rate']:.3f} open rate)")
    print(f"📉 Worst performing bucket: {worst_bucket} ({bucket_analysis.loc[worst_bucket, 'open_rate']:.3f} open rate)")
    
    # 4. Timeline Analysis
    print("\n4. DAILY LIMIT TIMELINE ANALYSIS:")
    print("-" * 60)
    
    # Use the most appropriate timestamp - let's check which has the most reasonable date range
    timestamp_candidates = ['timestamp_last_contact', 'timestamp_created_x', 'timestamp_created_y']
    main_timestamp_col = None
    
    for col in timestamp_candidates:
        if col in df.columns and df[col].notna().sum() > 1000:  # Need sufficient data
            # Filter to realistic dates - from last November to now (no future dates)
            current_date = pd.Timestamp.now(tz='UTC')
            
            # Set range from last November to now
            if current_date.month >= 11:  # If we're past November this year
                past_cutoff = current_date.replace(month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
            else:  # If we're before November, go to last year's November
                past_cutoff = current_date.replace(year=current_date.year-1, month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
            
            future_cutoff = current_date  # Only up to now, no future dates
            
            valid_dates = df[col][(df[col] >= past_cutoff) & (df[col] <= future_cutoff)].notna().sum()
            if valid_dates > 1000:
                main_timestamp_col = col
                print(f"📅 Using {col} for timeline analysis ({valid_dates} valid entries from {past_cutoff.strftime('%Y-%m-%d')} to {future_cutoff.strftime('%Y-%m-%d')})")
                break
    
    if main_timestamp_col and main_timestamp_col in df.columns and df[main_timestamp_col].notna().sum() > 0:
        # Filter to realistic date range for analysis - from last November to now
        current_date = pd.Timestamp.now(tz='UTC')
        
        # Set range from last November to now
        if current_date.month >= 11:  # If we're past November this year
            past_cutoff = current_date.replace(month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
        else:  # If we're before November, go to last year's November
            past_cutoff = current_date.replace(year=current_date.year-1, month=11, day=1, hour=0, minute=0, second=0, microsecond=0)
        
        future_cutoff = current_date  # Only up to now, no future dates
        
        # Create a filtered dataset for timeline analysis
        timeline_mask = (df[main_timestamp_col] >= past_cutoff) & (df[main_timestamp_col] <= future_cutoff)
        timeline_df = df[timeline_mask].copy()
        
        print(f"📊 Filtered {len(timeline_df)} records with realistic timestamps out of {len(df)} total")
        print(f"📅 Analysis period: {past_cutoff.strftime('%Y-%m-%d')} to {future_cutoff.strftime('%Y-%m-%d')}")
        
        # Create time-based features using filtered data
        timeline_df['date'] = timeline_df[main_timestamp_col].dt.date
        timeline_df['week'] = timeline_df[main_timestamp_col].dt.isocalendar().week
        timeline_df['month'] = timeline_df[main_timestamp_col].dt.month
        timeline_df['year'] = timeline_df[main_timestamp_col].dt.year
        timeline_df['month_year'] = timeline_df[main_timestamp_col].dt.strftime('%Y-%m')
        
        # Daily timeline analysis (focus on opens only) using filtered data
        daily_timeline = timeline_df.groupby('date').agg({
            'daily_limit': ['count', 'mean', 'median', 'std', 'min', 'max'],
            'email_open_count': ['sum', 'mean'],
            'daily_limit_bucket': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        daily_timeline.columns = ['contacts', 'avg_limit', 'median_limit', 'limit_std', 
                                 'min_limit', 'max_limit', 'total_opens', 'avg_opens', 
                                 'bucket_distribution']
        
        # Calculate performance metrics
        daily_timeline['open_rate'] = daily_timeline['total_opens'] / daily_timeline['contacts']
        
        print(f"📅 Timeline covers {len(daily_timeline)} days")
        print(f"📊 Date range: {daily_timeline.index[0]} to {daily_timeline.index[-1]}")
        print(f"📈 Average daily limit across time: {daily_timeline['avg_limit'].mean():.1f}")
        print(f"📊 Daily limit volatility: {daily_timeline['avg_limit'].std():.1f}")
        
        # Show trends
        early_period = daily_timeline.head(7)['avg_limit'].mean()
        recent_period = daily_timeline.tail(7)['avg_limit'].mean()
        limit_trend = recent_period - early_period
        print(f"📈 Limit trend (recent vs early): {limit_trend:+.1f}")
        
        # Weekly analysis (focus on opens only) using filtered data
        weekly_timeline = timeline_df.groupby(timeline_df[main_timestamp_col].dt.strftime('%Y-W%U')).agg({
            'daily_limit': ['count', 'mean', 'std'],
            'email_open_count': ['sum', 'mean']
        }).round(2)
        
        weekly_timeline.columns = ['contacts', 'avg_limit', 'limit_std', 'total_opens', 'avg_opens']
        weekly_timeline['open_rate'] = weekly_timeline['total_opens'] / weekly_timeline['contacts']
        
        print("\n📅 WEEKLY TIMELINE SUMMARY:")
        for week, row in weekly_timeline.head(10).iterrows():
            print(f"  {week}: {row['contacts']} contacts | Avg limit: {row['avg_limit']:.0f} | Open rate: {row['open_rate']:.3f}")
        
        # Monthly milestones (focus on opens only) using filtered data
        monthly_timeline = timeline_df.groupby('month_year').agg({
            'daily_limit': ['count', 'mean', 'std', 'min', 'max'],
            'email_open_count': ['sum', 'mean'],
            'daily_limit_bucket': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        monthly_timeline.columns = ['contacts', 'avg_limit', 'limit_std', 'min_limit', 'max_limit',
                                   'total_opens', 'avg_opens', 'bucket_distribution']
        monthly_timeline['open_rate'] = monthly_timeline['total_opens'] / monthly_timeline['contacts']
        
        print("\n📅 MONTHLY MILESTONES:")
        for month, row in monthly_timeline.iterrows():
            top_bucket = max(row['bucket_distribution'].items(), key=lambda x: x[1]) if isinstance(row['bucket_distribution'], dict) else ('Unknown', 0)
            print(f"  {month}: {row['contacts']:,} contacts | Avg limit: {row['avg_limit']:.0f} | Open rate: {row['open_rate']:.3f} | Top bucket: {top_bucket[0]} ({top_bucket[1]})")
    
    else:
        print("⚠️ No suitable timestamp data available for timeline analysis")
        print("Available timestamp columns and their coverage:")
        for col in ['timestamp_created_x', 'timestamp_created_y', 'timestamp_last_contact', 'timestamp_last_open']:
            if col in df.columns:
                valid_count = df[col].notna().sum()
                print(f"  {col}: {valid_count} valid entries")
        
        # Create dummy time features
        df['date'] = pd.to_datetime('2024-01-01').date()
        df['month_year'] = '2024-01'
        timeline_df = df.copy()  # Use all data if no good timestamps
    
    # 5. Daily Limit vs Performance Correlation
    print("\n5. DAILY LIMIT CORRELATION ANALYSIS:")
    print("-" * 60)
    
    # Calculate correlations (focus on opens only)
    valid_data = df[df['daily_limit'].notna()]
    if len(valid_data) > 0:
        open_correlation = valid_data['daily_limit'].corr(valid_data['email_open_count'])
        
        print("📊 Correlation Analysis:")
        print(f"  Daily Limit vs Opens: {open_correlation:.3f}")
        
        # Interpret correlation
        def interpret_correlation(corr, metric):
            if abs(corr) < 0.1:
                return f"No correlation with {metric}"
            elif abs(corr) < 0.3:
                return f"Weak correlation with {metric}"
            elif abs(corr) < 0.5:
                return f"Moderate correlation with {metric}"
            else:
                return f"Strong correlation with {metric}"
        
        print("\n📈 Correlation Insights:")
        print(f"  • {interpret_correlation(open_correlation, 'email opens')}")
    
    # 6. Daily Limit Optimization Analysis
    print("\n6. DAILY LIMIT OPTIMIZATION ANALYSIS:")
    print("-" * 60)
    
    # Find optimal daily limit ranges (focus on opens only)
    limit_performance = df[df['daily_limit'].notna()].groupby('daily_limit').agg({
        'email_open_count': ['count', 'sum', 'mean', 'std']
    }).round(3)
    
    limit_performance.columns = ['contacts', 'total_opens', 'avg_opens', 'opens_std']
    limit_performance['open_rate'] = limit_performance['total_opens'] / limit_performance['contacts']
    
    # Filter for statistically significant samples (10+ contacts)
    significant_limits = limit_performance[limit_performance['contacts'] >= 10].copy()
    significant_limits = significant_limits.sort_values('open_rate', ascending=False)
    
    print("📊 Top 10 Daily Limits by Performance (10+ contacts):")
    for limit, row in significant_limits.head(10).iterrows():
        print(f"  Limit {limit}: {row['open_rate']:.3f} open rate ({row['contacts']} contacts)")
    
    print("\n📉 Bottom 5 Daily Limits by Performance:")
    for limit, row in significant_limits.tail(5).iterrows():
        print(f"  Limit {limit}: {row['open_rate']:.3f} open rate ({row['contacts']} contacts)")
    
    # 7. Advanced Analytics - Daily Limit Patterns
    print("\n7. ADVANCED DAILY LIMIT PATTERNS:")
    print("-" * 60)
    
    # Analyze daily limit by other features
    if 'upload_method' in df.columns:
        method_limits = df.groupby('upload_method')['daily_limit'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        print("📊 Daily Limits by Upload Method:")
        for method, row in method_limits.iterrows():
            print(f"  {method.title()}: Avg {row['mean']:.0f} | Range {row['min']:.0f}-{row['max']:.0f} | Std {row['std']:.1f}")
    
    if 'esp_code' in df.columns:
        esp_limits = df.groupby('esp_code')['daily_limit'].agg(['count', 'mean']).round(2)
        esp_limits = esp_limits[esp_limits['count'] >= 100].sort_values('mean', ascending=False)
        print("\n📊 Daily Limits by ESP Code (100+ contacts):")
        for esp, row in esp_limits.head(10).iterrows():
            print(f"  ESP {esp}: Avg {row['mean']:.0f} ({row['count']} contacts)")
    
    if 'organization_employees' in df.columns:
        # Company size vs daily limit
        df['company_size_bucket'] = pd.cut(df['organization_employees'], 
                                          bins=[0, 10, 50, 200, 1000, float('inf')], 
                                          labels=['Micro (1-10)', 'Small (11-50)', 'Medium (51-200)', 'Large (201-1000)', 'Enterprise (1000+)'])
        
        size_limits = df.groupby('company_size_bucket')['daily_limit'].agg(['count', 'mean', 'std']).round(2)
        print("\n📊 Daily Limits by Company Size:")
        for size, row in size_limits.iterrows():
            if pd.notna(row['mean']):
                print(f"  {size}: Avg {row['mean']:.0f} | Std {row['std']:.1f} ({row['count']} contacts)")
    
    # 8. Create Comprehensive Dashboard
    print("\n8. CREATING COMPREHENSIVE DASHBOARD:")
    print("-" * 60)
    
    # Set up the dashboard
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.6, wspace=0.4)
    
    # Plot 1: Daily Limit Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    df['daily_limit'].hist(bins=50, ax=ax1, color='skyblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Daily Limit Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Daily Limit')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot - Daily Limits by Performance Bucket
    ax2 = fig.add_subplot(gs[0, 1])
    # Create performance buckets for box plot
    df['performance_bucket'] = pd.cut(df['email_open_count'], 
                                    bins=[-0.1, 0, 1, 2, 5, float('inf')], 
                                    labels=['No Opens', 'Low (1)', 'Medium (2)', 'High (3-5)', 'Very High (5+)'])
    
    valid_box_data = df[df['daily_limit'].notna() & df['performance_bucket'].notna()]
    if len(valid_box_data) > 0:
        box_data = [valid_box_data[valid_box_data['performance_bucket'] == cat]['daily_limit'].values 
                   for cat in valid_box_data['performance_bucket'].cat.categories]
        ax2.boxplot(box_data, labels=valid_box_data['performance_bucket'].cat.categories)
        ax2.set_title('Daily Limits by Email Open Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Open Performance Bucket')
        ax2.set_ylabel('Daily Limit')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Daily Limit vs Opens Scatter
    ax3 = fig.add_subplot(gs[0, 2])
    sample_data = df[df['daily_limit'].notna()].sample(min(1000, len(df)))
    ax3.scatter(sample_data['daily_limit'], sample_data['email_open_count'], alpha=0.5, color='green')
    ax3.set_title('Daily Limit vs Email Opens', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Daily Limit')
    ax3.set_ylabel('Email Opens')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Timeline - Average Daily Limit
    ax4 = fig.add_subplot(gs[1, 0])
    if main_timestamp_col and len(daily_timeline) > 1:
        daily_timeline.reset_index()['avg_limit'].plot(ax=ax4, color='blue', linewidth=2)
        ax4.set_title('Daily Limit Timeline (Daily Average)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Daily Limit')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax4.text(0.5, 0.5, 'No suitable\ntimeline data\navailable', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Daily Limit Timeline')
    
    # Plot 5: Timeline Scatter - Daily Limits Only
    ax5 = fig.add_subplot(gs[1, 1])
    if main_timestamp_col and len(timeline_df) > 0:
        # Sample data to avoid overcrowding (max 500 points)
        sample_size = min(500, len(timeline_df))
        sample_df = timeline_df.sample(sample_size) if len(timeline_df) > sample_size else timeline_df
        
        # Plot daily limits as blue dots
        ax5.scatter(sample_df[main_timestamp_col], sample_df['daily_limit'], 
                    alpha=0.6, s=30, c='blue', label='Daily Limits')
        ax5.set_ylabel('Daily Limit', color='blue')
        ax5.tick_params(axis='y', labelcolor='blue')
        ax5.set_title('Timeline: Daily Limits (Dots)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax5.text(0.5, 0.5, 'No timeline\ndata available', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Timeline: Daily Limits')
    

    
    # Plot 6: Correlation Over Time - Daily Limits vs Opens
    ax6 = fig.add_subplot(gs[1, 2])
    if main_timestamp_col and len(timeline_df) > 0:
        # Calculate rolling correlation over time
        timeline_df_sorted = timeline_df.sort_values(main_timestamp_col)
        timeline_df_sorted['month_year'] = timeline_df_sorted[main_timestamp_col].dt.strftime('%Y-%m')
        
        # Calculate monthly correlations
        monthly_correlations = []
        months = []
        
        for month, group in timeline_df_sorted.groupby('month_year'):
            if len(group) >= 10:  # Need at least 10 data points for meaningful correlation
                correlation = group['daily_limit'].corr(group['email_open_count'])
                if not pd.isna(correlation):
                    monthly_correlations.append(correlation)
                    months.append(month)
        
        if len(monthly_correlations) > 1:
            # Plot correlation over time
            month_dates = [pd.to_datetime(month + '-01') for month in months]
            ax6.plot(month_dates, monthly_correlations, marker='o', linewidth=2, markersize=6, color='purple')
            ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax6.set_ylabel('Correlation (Daily Limit vs Opens)')
            ax6.set_title('Daily Limit-Opens Correlation Over Time', fontsize=14, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            ax6.set_ylim(-1, 1)
            plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45)
            
            # Add trend annotation
            if len(monthly_correlations) > 2:
                trend = monthly_correlations[-1] - monthly_correlations[0]
                trend_text = f"Trend: {trend:+.3f}"
                ax6.text(0.02, 0.98, trend_text, transform=ax6.transAxes, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax6.text(0.5, 0.5, 'Insufficient data\nfor correlation\nanalysis', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Daily Limit-Opens Correlation Over Time')
    else:
        ax6.text(0.5, 0.5, 'No timeline\ndata available', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Daily Limit-Opens Correlation Over Time')
    
    # Plot 7: Box and Whisker - Opens by Daily Limit Bucket (detailed view)
    ax7 = fig.add_subplot(gs[2, 0])
    valid_data = df[df['daily_limit_bucket'] != 'Unknown']
    if len(valid_data) > 0:
        bucket_categories = ['Low Volume (≤100)', 'Medium-Low (101-250)', 'Medium (251-350)', 'Medium-High (351-450)', 'High (451-650)', 'Very High (651+)']
        open_data = [valid_data[valid_data['daily_limit_bucket'] == cat]['email_open_count'].values 
                     for cat in bucket_categories if cat in valid_data['daily_limit_bucket'].values]
        categories = [cat for cat in bucket_categories if cat in valid_data['daily_limit_bucket'].values]
        
        if open_data:
            ax7.boxplot(open_data, labels=categories)
            ax7.set_title('Email Opens Distribution by Daily Limit Bucket', fontsize=14, fontweight='bold')
            ax7.set_xlabel('Daily Limit Bucket')
            ax7.set_ylabel('Email Opens')
            ax7.tick_params(axis='x', rotation=45)
    

    
    # Plot 9: Daily Limit vs Average Opens scatter
    ax9 = fig.add_subplot(gs[2, 2])
    if len(significant_limits) > 5:
        # Create scatter plot of daily limit vs average opens
        limits = significant_limits.index.values
        opens = significant_limits['avg_opens'].values
        ax9.scatter(limits, opens, alpha=0.7, s=50, c='green')
        ax9.set_title('Daily Limit vs Average Opens', fontsize=14, fontweight='bold')
        ax9.set_xlabel('Daily Limit')
        ax9.set_ylabel('Average Opens')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'Insufficient data\nfor comparison', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Daily Limit vs Opens')
    
    # Plot 10: Monthly Timeline - Average Daily Limit
    ax10 = fig.add_subplot(gs[3, 0])
    if main_timestamp_col and len(monthly_timeline) > 1:
        monthly_timeline['avg_limit'].plot(kind='bar', ax=ax10, color='purple')
        ax10.set_title('Monthly Average Daily Limit', fontsize=14, fontweight='bold')
        ax10.set_ylabel('Average Daily Limit')
        ax10.tick_params(axis='x', rotation=45)
    else:
        ax10.text(0.5, 0.5, 'No suitable\nmonthly data\navailable', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Monthly Average Daily Limit')
    

    
    # Plot 12: Histogram - Daily Limit Distribution by Bucket
    ax12 = fig.add_subplot(gs[3, 2])
    if len(valid_data) > 0:
        bucket_counts = valid_data['daily_limit_bucket'].value_counts()
        bucket_counts.plot(kind='bar', ax=ax12, color='lightcoral')
        ax12.set_title('Contact Count by Daily Limit Bucket', fontsize=14, fontweight='bold')
        ax12.set_xlabel('Daily Limit Bucket')
        ax12.set_ylabel('Contact Count')
        ax12.tick_params(axis='x', rotation=45)
    

    
    plt.suptitle('Comprehensive Daily Limit Analysis Dashboard', fontsize=20, fontweight='bold', y=0.96)
    
    # Use gridspec spacing (remove conflicting subplots_adjust)
    # You can now adjust hspace and wspace in the gridspec line above (line 331)
    
    plt.savefig('daily_limit_analysis_dashboard.png', dpi=300)
    print("✅ Comprehensive dashboard saved as 'daily_limit_analysis_dashboard.png'")
    plt.show()
    
    # 9. Strategic Insights and Recommendations
    print("\n9. STRATEGIC INSIGHTS AND RECOMMENDATIONS:")
    print("=" * 80)
    
    print("🎯 OPTIMAL DAILY LIMIT STRATEGY:")
    if len(significant_limits) > 0:
        best_limit = significant_limits.index[0]
        best_performance = significant_limits.iloc[0]['open_rate']
        print(f"  • Best performing daily limit: {best_limit} ({best_performance:.3f} open rate)")
    
    if len(bucket_analysis) > 0:
        best_bucket = bucket_analysis['open_rate'].idxmax()
        best_bucket_performance = bucket_analysis.loc[best_bucket, 'open_rate']
        print(f"  • Optimal daily limit range: {best_bucket} ({best_bucket_performance:.3f} open rate)")
    
    print(f"\n📊 PERFORMANCE INSIGHTS:")
    if 'open_correlation' in locals():
        if abs(open_correlation) > 0.1:
            direction = "positively" if open_correlation > 0 else "negatively"
            strength = "strongly" if abs(open_correlation) > 0.3 else "moderately" if abs(open_correlation) > 0.1 else "weakly"
            print(f"  • Daily limits are {strength} {direction} correlated with email opens ({open_correlation:.3f})")
        else:
            print(f"  • Daily limits show no significant correlation with email opens")
    
    # Volume vs Performance analysis
    total_contacts = len(df[df['daily_limit'].notna()])
    if len(bucket_analysis) > 0:
        high_perf_buckets = bucket_analysis[bucket_analysis['open_rate'] > bucket_analysis['open_rate'].median()]
        high_perf_contacts = high_perf_buckets['contacts'].sum()
        high_perf_pct = (high_perf_contacts / total_contacts) * 100
        print(f"  • {high_perf_pct:.1f}% of contacts are in high-performing daily limit buckets")
    
    print(f"\n⚡ EFFICIENCY INSIGHTS:")
    if 'efficiency_by_bucket' in locals() and len(efficiency_by_bucket) > 0:
        most_efficient = efficiency_by_bucket.idxmax()
        efficiency_score = efficiency_by_bucket.max()
        print(f"  • Most efficient daily limit bucket: {most_efficient} ({efficiency_score:.3f} opens per unit limit)")
    
    print(f"\n📈 TREND INSIGHTS:")
    if 'limit_trend' in locals():
        if abs(limit_trend) > 5:
            direction = "increasing" if limit_trend > 0 else "decreasing"
            print(f"  • Daily limits are {direction} over time (trend: {limit_trend:+.1f})")
        else:
            print(f"  • Daily limits remain stable over time")
    
    print(f"\n🎯 ACTIONABLE RECOMMENDATIONS:")
    print(f"  • Focus campaigns on the optimal daily limit range identified")
    print(f"  • Monitor performance degradation in extreme daily limit buckets")
    print(f"  • Consider adjusting daily limits based on historical performance data")
    if 'upload_method' in df.columns:
        print(f"  • Align daily limit strategy with upload method performance")
    print(f"  • Use efficiency metrics to optimize resource allocation")
    
    print(f"\n💡 DATA QUALITY OPPORTUNITIES:")
    if missing_pct > 5:
        print(f"  • {missing_pct:.1f}% of contacts missing daily limit data - investigate data collection")
    print(f"  • Consider A/B testing different daily limit ranges")
    print(f"  • Implement monitoring for daily limit performance trends")
    
    print("\n" + "="*80)
    print("✅ COMPREHENSIVE DAILY LIMIT ANALYSIS COMPLETE!")
    print("📊 Generated detailed dashboard with timeline, box plots, and bucketed analysis")
    print("🎯 Strategic recommendations provided for optimization")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = analyze_daily_limits()