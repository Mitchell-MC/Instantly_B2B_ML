"""
Advanced Email List Analysis with Time Series and Comprehensive Insights
Includes temporal patterns, cohort analysis, list lifecycle, and predictive analytics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def analyze_email_lists():
    """Comprehensive email list analysis with time series and advanced insights"""
    print("Loading data for advanced email list analysis...")
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Data shape: {df.shape}")
    
    # Convert timestamp columns (check for actual column names)
    timestamp_cols = ['timestamp_created_x', 'timestamp_created_y', 'timestamp_last_contact', 
                     'timestamp_last_open', 'timestamp_last_click', 'retrieval_timestamp']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print("\n" + "="*80)
    print("ADVANCED EMAIL LIST ANALYSIS WITH TIME SERIES")
    print("="*80)
    
    # 1. Basic Statistics and Data Quality
    print("\n1. DATA QUALITY AND BASIC STATISTICS:")
    print("-" * 60)
    print(f"Data type: {df['email_list'].dtype}")
    print(f"Missing values: {df['email_list'].isnull().sum()} ({df['email_list'].isnull().sum()/len(df)*100:.1f}%)")
    print(f"Unique email lists: {df['email_list'].nunique()}")
    # Use the main timestamp column (timestamp_created_x)
    main_timestamp_col = 'timestamp_created_x'
    if main_timestamp_col in df.columns:
        print(f"Date range: {df[main_timestamp_col].min()} to {df[main_timestamp_col].max()}")
        print(f"Analysis period: {(df[main_timestamp_col].max() - df[main_timestamp_col].min()).days} days")
    else:
        print("No main timestamp column found - using available timestamp data")
    
    # 2. Time Series Preparation
    print("\n2. TIME SERIES DATA PREPARATION:")
    print("-" * 60)
    
    # Create time-based features using the main timestamp column
    if main_timestamp_col in df.columns and df[main_timestamp_col].notna().sum() > 0:
        df['date'] = df[main_timestamp_col].dt.date
        df['week'] = df[main_timestamp_col].dt.isocalendar().week
        df['month'] = df[main_timestamp_col].dt.month
        df['quarter'] = df[main_timestamp_col].dt.quarter
        df['year'] = df[main_timestamp_col].dt.year
        df['day_of_week'] = df[main_timestamp_col].dt.day_name()
        df['hour'] = df[main_timestamp_col].dt.hour
        df['is_weekend'] = df[main_timestamp_col].dt.weekday >= 5
        df['days_since_start'] = (df[main_timestamp_col] - df[main_timestamp_col].min()).dt.days
    else:
        # Create dummy time features if no timestamp available
        df['date'] = pd.to_datetime('2024-01-01').date()
        df['week'] = 1
        df['month'] = 1
        df['quarter'] = 1
        df['year'] = 2024
        df['day_of_week'] = 'Monday'
        df['hour'] = 12
        df['is_weekend'] = False
        df['days_since_start'] = 0
        print("⚠️ No timestamp data available - using dummy values for time analysis")
    
    # Extract list characteristics
    def extract_list_features(email_list_str):
        """Extract comprehensive features from email list"""
        if pd.isna(email_list_str):
            return {'email_count': 0, 'unique_domains': 0, 'list_diversity': 0, 'list_length': 0}
        
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', str(email_list_str))
        domains = [email.split('@')[1] for email in emails if '@' in email]
        unique_domains = len(set(domains))
        
        return {
            'email_count': len(emails),
            'unique_domains': unique_domains,
            'list_diversity': unique_domains / max(len(emails), 1),
            'list_length': len(str(email_list_str))
        }
    
    # Apply feature extraction
    list_features = df['email_list'].apply(extract_list_features)
    feature_df = pd.DataFrame(list_features.tolist())
    df = pd.concat([df, feature_df], axis=1)
    
    print(f"✅ Enhanced dataset with temporal and list features")
    print(f"📊 Email count range: {df['email_count'].min()}-{df['email_count'].max()}")
    print(f"🌐 Domain diversity range: {df['list_diversity'].min():.2f}-{df['list_diversity'].max():.2f}")
    
    # 3. Time Series Analysis - Daily Trends
    print("\n3. DAILY TIME SERIES ANALYSIS:")
    print("-" * 60)
    
    daily_metrics = df.groupby('date').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean'],
        'email_list': 'nunique',
        'list_diversity': 'mean'
    }).round(3)
    
    daily_metrics.columns = ['contacts', 'total_opens', 'avg_opens', 'total_clicks', 
                           'avg_clicks', 'total_replies', 'avg_replies', 'unique_lists', 'avg_diversity']
    daily_metrics['open_rate'] = daily_metrics['total_opens'] / daily_metrics['contacts']
    daily_metrics['click_rate'] = daily_metrics['total_clicks'] / daily_metrics['contacts']
    
    print(f"📈 Daily metrics calculated for {len(daily_metrics)} days")
    print(f"📊 Average daily contacts: {daily_metrics['contacts'].mean():.0f}")
    print(f"🎯 Average daily open rate: {daily_metrics['open_rate'].mean():.3f}")
    print(f"📉 Open rate volatility (std): {daily_metrics['open_rate'].std():.3f}")
    
    # Trend analysis
    daily_metrics['open_rate_7d_avg'] = daily_metrics['open_rate'].rolling(window=7, center=True).mean()
    daily_metrics['open_rate_30d_avg'] = daily_metrics['open_rate'].rolling(window=30, center=True).mean()
    
    recent_trend = daily_metrics['open_rate'].tail(7).mean() - daily_metrics['open_rate'].head(7).mean()
    print(f"📈 Recent trend (last 7 vs first 7 days): {recent_trend:+.3f} open rate change")
    
    # 4. Weekly and Monthly Patterns
    print("\n4. TEMPORAL PATTERN ANALYSIS:")
    print("-" * 60)
    
    # Day of week analysis
    dow_analysis = df.groupby('day_of_week').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'list_diversity': 'mean'
    }).round(3)
    dow_analysis.columns = ['contacts', 'total_opens', 'avg_opens', 'avg_diversity']
    dow_analysis['open_rate'] = dow_analysis['total_opens'] / dow_analysis['contacts']
    
    # Order by weekday
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis = dow_analysis.reindex(day_order)
    
    print("📅 Day of Week Performance:")
    for day, row in dow_analysis.iterrows():
        print(f"  {day:<10}: {row['open_rate']:.3f} open rate ({row['contacts']:,} contacts)")
    
    best_day = dow_analysis['open_rate'].idxmax()
    worst_day = dow_analysis['open_rate'].idxmin()
    print(f"🏆 Best day: {best_day} ({dow_analysis.loc[best_day, 'open_rate']:.3f})")
    print(f"📉 Worst day: {worst_day} ({dow_analysis.loc[worst_day, 'open_rate']:.3f})")
    
    # Monthly analysis
    monthly_analysis = df.groupby('month').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'list_diversity': 'mean'
    }).round(3)
    monthly_analysis.columns = ['contacts', 'total_opens', 'avg_opens', 'avg_diversity']
    monthly_analysis['open_rate'] = monthly_analysis['total_opens'] / monthly_analysis['contacts']
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"\n📅 Monthly Performance:")
    for month, row in monthly_analysis.iterrows():
        if pd.notna(month) and isinstance(month, (int, float)) and 1 <= month <= 12:
            month_name = month_names[int(month)-1]
            print(f"  {month_name:<4}: {row['open_rate']:.3f} open rate ({row['contacts']:,} contacts)")
        else:
            print(f"  Unknown: {row['open_rate']:.3f} open rate ({row['contacts']:,} contacts)")
    
    # 5. Cohort Analysis by List First Appearance
    print("\n5. COHORT ANALYSIS BY LIST INTRODUCTION:")
    print("-" * 60)
    
    # Find first appearance of each email list
    if main_timestamp_col in df.columns and df[main_timestamp_col].notna().sum() > 0:
        list_first_seen = df.groupby('email_list')[main_timestamp_col].min().reset_index()
        list_first_seen.columns = ['email_list', 'first_seen']
        list_first_seen['cohort_week'] = list_first_seen['first_seen'].dt.isocalendar().week
        list_first_seen['cohort_month'] = list_first_seen['first_seen'].dt.to_period('M')
        
        # Merge back with main data
        df = df.merge(list_first_seen[['email_list', 'first_seen', 'cohort_month']], on='email_list', how='left')
        df['weeks_since_list_intro'] = ((df[main_timestamp_col] - df['first_seen']).dt.days / 7).round(0)
    else:
        # Create dummy cohort data
        df['first_seen'] = pd.to_datetime('2024-01-01')
        df['cohort_month'] = pd.to_datetime('2024-01-01').to_period('M')
        df['weeks_since_list_intro'] = 0
    
    # Cohort performance analysis
    cohort_analysis = df.groupby(['cohort_month', 'weeks_since_list_intro']).agg({
        'email_open_count': ['count', 'sum', 'mean']
    }).round(3)
    cohort_analysis.columns = ['contacts', 'total_opens', 'avg_opens']
    cohort_analysis['open_rate'] = cohort_analysis['total_opens'] / cohort_analysis['contacts']
    
    print(f"📊 Cohort analysis covering {df['cohort_month'].nunique()} monthly cohorts")
    print(f"⏰ Maximum weeks tracked: {df['weeks_since_list_intro'].max():.0f}")
    
    # Show cohort degradation
    avg_performance_by_week = df.groupby('weeks_since_list_intro')['email_open_count'].mean()
    if len(avg_performance_by_week) > 1:
        week0_performance = avg_performance_by_week.iloc[0]
        week1_performance = avg_performance_by_week.iloc[1] if len(avg_performance_by_week) > 1 else week0_performance
        degradation = (week0_performance - week1_performance) / week0_performance * 100
        print(f"📉 Week 0 to Week 1 performance change: {degradation:+.1f}%")
    
    # 6. List Lifecycle Analysis
    print("\n6. EMAIL LIST LIFECYCLE ANALYSIS:")
    print("-" * 60)
    
    # Analyze list performance over time
    list_lifecycle = df.groupby(['email_list', 'weeks_since_list_intro']).agg({
        'email_open_count': ['count', 'sum', 'mean']
    }).round(3)
    list_lifecycle.columns = ['contacts', 'total_opens', 'avg_opens']
    list_lifecycle['open_rate'] = list_lifecycle['total_opens'] / list_lifecycle['contacts']
    
    # Find lists with significant lifecycle data
    lists_with_history = list_lifecycle.groupby('email_list').size()
    mature_lists = lists_with_history[lists_with_history >= 3].index  # Lists with 3+ weeks of data
    
    print(f"📈 {len(mature_lists)} lists have 3+ weeks of performance data")
    
    if len(mature_lists) > 0:
        # Analyze performance degradation
        degradation_analysis = []
        for email_list in mature_lists[:10]:  # Top 10 for detailed analysis
            list_data = list_lifecycle.loc[email_list].sort_index()
            if len(list_data) >= 2:
                initial_rate = list_data['open_rate'].iloc[0]
                final_rate = list_data['open_rate'].iloc[-1]
                degradation = (initial_rate - final_rate) / initial_rate * 100 if initial_rate > 0 else 0
                degradation_analysis.append({
                    'email_list': email_list,
                    'initial_rate': initial_rate,
                    'final_rate': final_rate,
                    'degradation_pct': degradation,
                    'weeks_tracked': len(list_data)
                })
        
        if degradation_analysis:
            degradation_df = pd.DataFrame(degradation_analysis)
            avg_degradation = degradation_df['degradation_pct'].mean()
            print(f"📉 Average list performance degradation: {avg_degradation:.1f}%")
            
            print(f"\n🔍 Top 5 Most Degraded Lists:")
            top_degraded = degradation_df.nlargest(5, 'degradation_pct')
            for idx, row in top_degraded.iterrows():
                print(f"  {row['email_list'][:50]}...")
                print(f"    {row['initial_rate']:.3f} → {row['final_rate']:.3f} ({row['degradation_pct']:+.1f}%)")
    
    # 7. List Fatigue Analysis
    print("\n7. LIST FATIGUE AND SATURATION ANALYSIS:")
    print("-" * 60)
    
    # Analyze performance by contact frequency
    contact_frequency = df.groupby('email_list').size()
    df['list_contact_frequency'] = df['email_list'].map(contact_frequency)
    
    # Create fatigue categories
    fatigue_bins = [0, 10, 50, 100, 500, float('inf')]
    fatigue_labels = ['Low (1-10)', 'Medium (11-50)', 'High (51-100)', 'Very High (101-500)', 'Extreme (500+)']
    df['fatigue_category'] = pd.cut(df['list_contact_frequency'], bins=fatigue_bins, labels=fatigue_labels)
    
    fatigue_analysis = df.groupby('fatigue_category').agg({
        'email_open_count': ['count', 'sum', 'mean'],
        'email_list': 'nunique',
        'list_diversity': 'mean'
    }).round(3)
    fatigue_analysis.columns = ['contacts', 'total_opens', 'avg_opens', 'unique_lists', 'avg_diversity']
    fatigue_analysis['open_rate'] = fatigue_analysis['total_opens'] / fatigue_analysis['contacts']
    
    print("📊 Performance by Contact Frequency (Fatigue Analysis):")
    for category, row in fatigue_analysis.iterrows():
        print(f"  {category:<20}: {row['open_rate']:.3f} open rate")
        print(f"    {row['contacts']:,} contacts, {row['unique_lists']} unique lists")
    
    # 8. Advanced Performance Metrics
    print("\n8. ADVANCED PERFORMANCE METRICS:")
    print("-" * 60)
    
    # Calculate advanced metrics
    list_performance = df.groupby('email_list').agg({
        'email_open_count': ['count', 'sum', 'mean', 'std'],
        'email_click_count': ['sum', 'mean'],
        'email_reply_count': ['sum', 'mean'],
        'list_diversity': 'first',
        'email_count': 'first',
        'days_since_start': ['min', 'max']
    }).round(3)
    
    # Flatten column names
    list_performance.columns = ['_'.join(col).strip() for col in list_performance.columns]
    list_performance.columns = ['contacts', 'total_opens', 'avg_opens', 'opens_std',
                               'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies',
                               'diversity', 'email_count', 'first_day', 'last_day']
    
    # Calculate performance metrics
    list_performance['open_rate'] = list_performance['total_opens'] / list_performance['contacts']
    list_performance['click_rate'] = list_performance['total_clicks'] / list_performance['contacts']
    list_performance['reply_rate'] = list_performance['total_replies'] / list_performance['contacts']
    list_performance['consistency'] = 1 - (list_performance['opens_std'] / list_performance['avg_opens']).fillna(0)
    list_performance['longevity_days'] = list_performance['last_day'] - list_performance['first_day']
    list_performance['performance_score'] = (
        list_performance['open_rate'] * 0.5 + 
        list_performance['click_rate'] * 0.3 + 
        list_performance['reply_rate'] * 0.2
    )
    
    # Filter for lists with significant data
    significant_lists = list_performance[list_performance['contacts'] >= 10].copy()
    significant_lists = significant_lists.sort_values('performance_score', ascending=False)
    
    print(f"📊 {len(significant_lists)} lists with 10+ contacts analyzed")
    print(f"🏆 Top Performance Score: {significant_lists['performance_score'].max():.3f}")
    print(f"📈 Average Performance Score: {significant_lists['performance_score'].mean():.3f}")
    
    print(f"\n🏆 TOP 10 PERFORMING EMAIL LISTS:")
    for idx, (email_list, row) in enumerate(significant_lists.head(10).iterrows(), 1):
        print(f"{idx:2d}. {email_list[:60]}...")
        print(f"    Score: {row['performance_score']:.3f} | Open: {row['open_rate']:.3f} | Click: {row['click_rate']:.3f}")
        print(f"    Contacts: {row['contacts']:.0f} | Diversity: {row['diversity']:.2f} | Days: {row['longevity_days']:.0f}")
    
    # 9. Predictive Analytics
    print("\n9. PREDICTIVE ANALYTICS AND FORECASTING:")
    print("-" * 60)
    
    # Simple trend analysis for forecasting
    if len(daily_metrics) > 7:
        recent_data = daily_metrics.tail(14)  # Last 2 weeks
        trend_slope = np.polyfit(range(len(recent_data)), recent_data['open_rate'], 1)[0]
        
        print(f"📈 Recent trend slope: {trend_slope:+.6f} open rate change per day")
        
        # 7-day forecast
        forecast_days = 7
        last_rate = recent_data['open_rate'].iloc[-1]
        forecasted_rate = last_rate + (trend_slope * forecast_days)
        
        print(f"🔮 7-day forecast: {forecasted_rate:.3f} open rate")
        print(f"📊 Expected change: {(forecasted_rate - last_rate) * 100:+.1f}%")
        
        # Volatility analysis
        volatility = daily_metrics['open_rate'].std()
        print(f"⚡ Open rate volatility: {volatility:.3f} (daily standard deviation)")
    
    # 10. Comprehensive Calendar Timeline Analysis
    print("\n10. COMPREHENSIVE CALENDAR TIMELINE ANALYSIS:")
    print("=" * 80)
    
    if main_timestamp_col in df.columns and df[main_timestamp_col].notna().sum() > 0:
        # Create comprehensive timeline
        timeline_df = df[df[main_timestamp_col].notna()].copy()
        timeline_df['date'] = timeline_df[main_timestamp_col].dt.date
        timeline_df['week'] = timeline_df[main_timestamp_col].dt.isocalendar().week
        timeline_df['month_year'] = timeline_df[main_timestamp_col].dt.strftime('%Y-%m')
        
        # Daily activity summary
        daily_activity = timeline_df.groupby('date').agg({
            'id': 'count',
            'email_open_count': ['sum', 'mean'],
            'email_click_count': ['sum', 'mean'],
            'upload_method': lambda x: x.value_counts().to_dict() if 'upload_method' in timeline_df.columns else {},
            'email_list': 'nunique',
            'organization_employees': 'mean',
            'esp_code': lambda x: x.value_counts().head(3).to_dict()
        }).round(3)
        
        daily_activity.columns = ['contacts', 'total_opens', 'avg_opens', 'total_clicks', 'avg_clicks', 
                                 'upload_methods', 'unique_lists', 'avg_company_size', 'top_esp_codes']
        
        # Sort by date
        daily_activity = daily_activity.sort_index()
        
        print(f"📅 DAILY TIMELINE ({daily_activity.index[0]} to {daily_activity.index[-1]}):")
        print(f"Total analysis period: {(daily_activity.index[-1] - daily_activity.index[0]).days} days")
        print("-" * 80)
        
        # Show significant days (high activity or major changes)
        significant_days = []
        
        # Find days with high activity
        high_activity_threshold = daily_activity['contacts'].quantile(0.8)
        high_activity_days = daily_activity[daily_activity['contacts'] >= high_activity_threshold]
        
        # Find first and last days
        first_day = daily_activity.index[0]
        last_day = daily_activity.index[-1]
        
        # Find days with method changes (if upload_method exists)
        method_change_days = []
        if 'upload_method' in timeline_df.columns:
            prev_methods = set()
            for date, row in daily_activity.iterrows():
                current_methods = set(row['upload_methods'].keys()) if isinstance(row['upload_methods'], dict) else set()
                if current_methods != prev_methods and len(current_methods) > 0:
                    method_change_days.append(date)
                prev_methods = current_methods
        
        # Create timeline events
        timeline_events = []
        
        # First day event
        first_row = daily_activity.loc[first_day]
        timeline_events.append({
            'date': first_day,
            'event': '🚀 Campaign Start',
            'details': f"{first_row['contacts']} contacts, {first_row['unique_lists']} lists, {first_row['avg_opens']:.2f} avg opens"
        })
        
        # High activity days
        for date in high_activity_days.index[:5]:  # Top 5 highest activity days
            row = daily_activity.loc[date]
            timeline_events.append({
                'date': date,
                'event': '📈 High Activity Day',
                'details': f"{row['contacts']} contacts ({row['total_opens']} opens, {row['total_clicks']} clicks)"
            })
        
        # Method change days
        for date in method_change_days[:3]:  # First 3 method changes
            row = daily_activity.loc[date]
            methods = row['upload_methods'] if isinstance(row['upload_methods'], dict) else {}
            method_str = ', '.join([f"{k}: {v}" for k, v in methods.items()])
            timeline_events.append({
                'date': date,
                'event': '🔄 Upload Method Change',
                'details': f"Methods: {method_str}"
            })
        
        # Last day event
        last_row = daily_activity.loc[last_day]
        timeline_events.append({
            'date': last_day,
            'event': '🏁 Latest Activity',
            'details': f"{last_row['contacts']} contacts, {last_row['unique_lists']} lists, {last_row['avg_opens']:.2f} avg opens"
        })
        
        # Sort timeline events by date
        timeline_events.sort(key=lambda x: x['date'])
        
        print("🗓️ KEY TIMELINE EVENTS:")
        for event in timeline_events:
            print(f"  {event['date'].strftime('%Y-%m-%d (%A)')}: {event['event']}")
            print(f"    └─ {event['details']}")
        
        # Weekly summary
        print(f"\n📊 WEEKLY TIMELINE SUMMARY:")
        weekly_summary = timeline_df.groupby(timeline_df[main_timestamp_col].dt.strftime('%Y-W%U')).agg({
            'id': 'count',
            'email_open_count': ['sum', 'mean'],
            'email_list': 'nunique',
            'upload_method': lambda x: x.value_counts().to_dict() if 'upload_method' in timeline_df.columns else {}
        }).round(2)
        
        weekly_summary.columns = ['contacts', 'total_opens', 'avg_opens', 'unique_lists', 'upload_methods']
        
        for week, row in weekly_summary.head(10).iterrows():  # Show first 10 weeks
            methods_str = ""
            if isinstance(row['upload_methods'], dict) and len(row['upload_methods']) > 0:
                methods_str = f" | Methods: {', '.join([f'{k}({v})' for k, v in row['upload_methods'].items()])}"
            print(f"  {week}: {row['contacts']} contacts, {row['total_opens']} opens, {row['unique_lists']} lists{methods_str}")
        
        # Monthly milestones
        print(f"\n📅 MONTHLY MILESTONES:")
        monthly_summary = timeline_df.groupby('month_year').agg({
            'id': 'count',
            'email_open_count': ['sum', 'mean'],
            'email_list': 'nunique',
            'upload_method': lambda x: x.value_counts().to_dict() if 'upload_method' in timeline_df.columns else {},
            'organization_employees': 'mean'
        }).round(2)
        
        monthly_summary.columns = ['contacts', 'total_opens', 'avg_opens', 'unique_lists', 'upload_methods', 'avg_company_size']
        
        for month, row in monthly_summary.iterrows():
            methods_str = ""
            if isinstance(row['upload_methods'], dict) and len(row['upload_methods']) > 0:
                methods_str = f" | Methods: {', '.join([f'{k}({v})' for k, v in row['upload_methods'].items()])}"
            print(f"  {month}: {row['contacts']:,} contacts | {row['total_opens']:.0f} opens | {row['unique_lists']} lists | Avg company: {row['avg_company_size']:.0f} employees{methods_str}")
        
        # Performance timeline
        print(f"\n📈 PERFORMANCE EVOLUTION TIMELINE:")
        performance_timeline = daily_activity.rolling(window=7, center=True).agg({
            'avg_opens': 'mean',
            'avg_clicks': 'mean',
            'contacts': 'mean'
        }).dropna().round(3)
        
        # Show key performance periods
        if len(performance_timeline) > 0:
            best_period = performance_timeline['avg_opens'].idxmax()
            worst_period = performance_timeline['avg_opens'].idxmin()
            
            print(f"  🏆 Best 7-day period: {best_period} (avg {performance_timeline.loc[best_period, 'avg_opens']:.3f} opens)")
            print(f"  📉 Worst 7-day period: {worst_period} (avg {performance_timeline.loc[worst_period, 'avg_opens']:.3f} opens)")
            
            # Show trend
            early_performance = performance_timeline.head(7)['avg_opens'].mean()
            recent_performance = performance_timeline.tail(7)['avg_opens'].mean()
            trend = ((recent_performance - early_performance) / early_performance * 100) if early_performance > 0 else 0
            
            print(f"  📊 Overall trend: {trend:+.1f}% change from start to recent period")
    
    else:
        print("⚠️ No timestamp data available for timeline analysis")
    
    # 11. Upload Method Temporal Analysis
    print("\n\n11. UPLOAD METHOD TEMPORAL ANALYSIS:")
    print("-" * 60)
    
    if 'upload_method' in df.columns and main_timestamp_col in df.columns:
        # Overall method distribution
        method_distribution = df['upload_method'].value_counts()
        total_contacts = len(df)
        
        print("📊 Overall Upload Method Distribution:")
        for method, count in method_distribution.items():
            percentage = (count / total_contacts) * 100
            print(f"  {method.title():<10}: {count:,} contacts ({percentage:.1f}%)")
        
        # Daily method trends
        daily_methods = df.groupby([df[main_timestamp_col].dt.date, 'upload_method']).size().unstack(fill_value=0)
        daily_methods.index = pd.to_datetime(daily_methods.index)  # Ensure datetime index
        daily_methods = daily_methods.resample('D').sum()
        
        # Calculate daily proportions
        daily_proportions = daily_methods.div(daily_methods.sum(axis=1), axis=0)
        
        print(f"\n📈 Daily Upload Method Timeline:")
        if len(daily_proportions) > 0:
            # Find first and last API usage dates
            api_days = daily_proportions[daily_proportions.get('api', 0) > 0]
            if len(api_days) > 0:
                first_api_date = api_days.index[0].strftime('%Y-%m-%d')
                last_api_date = api_days.index[-1].strftime('%Y-%m-%d')
                peak_api_date = api_days['api'].idxmax().strftime('%Y-%m-%d')
                peak_api_pct = api_days['api'].max()
                
                print(f"  🚀 First API usage: {first_api_date}")
                print(f"  📈 Peak API usage: {peak_api_date} ({peak_api_pct:.1%})")
                print(f"  📉 Last API usage: {last_api_date}")
                
                # Calculate API usage duration
                duration = (api_days.index[-1] - api_days.index[0]).days
                print(f"  ⏱️  API usage period: {duration} days")
            else:
                print(f"  ⚠️ No API usage detected in daily data")
            
            # Recent trends (last 30 days)
            recent_30_days = daily_proportions.tail(30)
            if len(recent_30_days) > 0:
                print(f"\n📊 Recent 30-day Summary:")
                for method in daily_methods.columns:
                    if method in recent_30_days.columns:
                        avg_proportion = recent_30_days[method].mean()
                        recent_trend = recent_30_days[method].iloc[-7:].mean() - recent_30_days[method].iloc[:7].mean()
                        print(f"  {method.title():<10}: {avg_proportion:.1%} average, {recent_trend:+.1%} recent change")
        
        # Weekly patterns
        df['weekday'] = df[main_timestamp_col].dt.day_name()
        weekly_methods = df.groupby(['weekday', 'upload_method']).size().unstack(fill_value=0)
        weekly_proportions = weekly_methods.div(weekly_methods.sum(axis=1), axis=0)
        
        print(f"\n📅 Weekly Upload Method Patterns:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in weekly_proportions.index:
                row = weekly_proportions.loc[day]
                api_pct = row.get('api', 0) * 100
                manual_pct = row.get('manual', 0) * 100
                print(f"  {day:<10}: Manual {manual_pct:.1f}%, API {api_pct:.1f}%")
        
        # Monthly evolution
        monthly_methods = df.groupby([df[main_timestamp_col].dt.to_period('M'), 'upload_method']).size().unstack(fill_value=0)
        monthly_proportions = monthly_methods.div(monthly_methods.sum(axis=1), axis=0)
        
        print(f"\n📆 Upload Method Timeline (Monthly Evolution):")
        timeline_events = []
        for month, row in monthly_proportions.iterrows():
            api_pct = row.get('api', 0) * 100
            manual_pct = row.get('manual', 0) * 100
            total_contacts = monthly_methods.loc[month].sum()
            
            # Convert period to readable date
            month_str = str(month)
            print(f"  {month_str}: Manual {manual_pct:.1f}%, API {api_pct:.1f}% ({total_contacts:,} total)")
            
            # Track significant events
            if api_pct > 5 and len(timeline_events) == 0:
                timeline_events.append(f"🚀 {month_str}: API adoption begins ({api_pct:.1f}%)")
            elif api_pct > 30:
                timeline_events.append(f"📈 {month_str}: Peak API usage ({api_pct:.1f}%)")
            elif api_pct < 5 and len([e for e in timeline_events if "begins" in e]) > 0:
                timeline_events.append(f"📉 {month_str}: API usage drops ({api_pct:.1f}%)")
        
        # Print timeline events
        if timeline_events:
            print(f"\n🕒 Key Timeline Events:")
            for event in timeline_events:
                print(f"  {event}")
        
        # Performance by upload method
        method_performance = df.groupby('upload_method').agg({
            'email_open_count': ['count', 'sum', 'mean'],
            'email_click_count': ['sum', 'mean'],
            'email_reply_count': ['sum', 'mean']
        }).round(3)
        
        method_performance.columns = ['contacts', 'total_opens', 'avg_opens', 'total_clicks', 'avg_clicks', 'total_replies', 'avg_replies']
        method_performance['open_rate'] = method_performance['total_opens'] / method_performance['contacts']
        method_performance['click_rate'] = method_performance['total_clicks'] / method_performance['contacts']
        method_performance['reply_rate'] = method_performance['total_replies'] / method_performance['contacts']
        
        print(f"\n⚡ Performance by Upload Method:")
        for method, row in method_performance.iterrows():
            print(f"  {method.title():<10}:")
            print(f"    Open Rate:  {row['open_rate']:.3f} ({row['total_opens']:.0f} opens/{row['contacts']:.0f} contacts)")
            print(f"    Click Rate: {row['click_rate']:.3f} ({row['total_clicks']:.0f} clicks/{row['contacts']:.0f} contacts)")
            print(f"    Reply Rate: {row['reply_rate']:.3f} ({row['total_replies']:.0f} replies/{row['contacts']:.0f} contacts)")
        
        # API adoption trends
        if 'api' in daily_proportions.columns and len(daily_proportions) > 14:
            api_adoption = daily_proportions['api'].rolling(7).mean()
            early_adoption = api_adoption.head(7).mean()
            recent_adoption = api_adoption.tail(7).mean()
            adoption_growth = (recent_adoption - early_adoption) / early_adoption * 100 if early_adoption > 0 else 0
            
            print(f"\n🚀 API Adoption Insights:")
            print(f"  Early period API usage: {early_adoption:.1%}")
            print(f"  Recent period API usage: {recent_adoption:.1%}")
            print(f"  Adoption growth rate: {adoption_growth:+.1f}%")
            
            if adoption_growth > 10:
                print(f"  📈 Strong API adoption trend detected")
            elif adoption_growth < -10:
                print(f"  📉 Declining API usage trend detected")
            else:
                print(f"  📊 Stable API usage pattern")
    
        # Timeline Breakdowns by List Size and Domain
        print(f"\n🔍 TIMELINE BREAKDOWN BY EMAIL LIST SIZE:")
        print("-" * 60)
        
        # Categorize lists using simplified ranges
        def categorize_list_size(email_count):
            if pd.isna(email_count):
                return 'Unknown'
            elif 1 <= email_count <= 3:
                return 'Small Lists (1-3)'
            elif 4 <= email_count <= 10:
                return 'Medium Lists (4-10)'
            else:
                return 'Large Lists (11+)'
        
        df['list_size_category'] = df['email_count'].apply(categorize_list_size)
        
        # Timeline breakdown by simplified list size categories
        size_categories = ['Small Lists (1-3)', 'Medium Lists (4-10)', 'Large Lists (11+)']
        
        for size_cat in size_categories:
            size_data = df[df['list_size_category'] == size_cat]
            
            if len(size_data) > 0 and main_timestamp_col in size_data.columns:
                # Monthly activity summary
                size_monthly = size_data.groupby(size_data[main_timestamp_col].dt.to_period('M')).agg({
                    'id': 'count',
                    'email_open_count': 'sum',
                    'email_list': 'nunique'
                }).round(1)
                
                size_monthly = size_monthly[size_monthly['id'] > 0]
                size_monthly['opens_per_contact'] = (size_monthly['email_open_count'] / size_monthly['id']).round(2)
                
                if len(size_monthly) > 0:
                    print(f"\n  📊 {size_cat} Timeline:")
                    for month, row in size_monthly.iterrows():
                        print(f"    {month}: {row['id']:,} contacts | {row['email_open_count']:.0f} opens | {row['email_list']:.0f} lists | Avg: {row['opens_per_contact']:.2f} opens/contact")
        
        print(f"\n🌐 TIMELINE BREAKDOWN BY DOMAIN TYPE:")
        print("-" * 60)
        
        # Categorize by domain type
        def categorize_domain(email_list_str):
            if pd.isna(email_list_str):
                return 'Unknown'
            email_list_lower = str(email_list_str).lower()
            if 'wecloud' in email_list_lower:
                return 'WeCloud Domain'
            elif 'beam' in email_list_lower:
                return 'Beam Data Domain'
            else:
                return 'Other Domain'
        
        df_with_domains = df.copy()
        df_with_domains['domain_category'] = df_with_domains['email_list'].apply(categorize_domain)
        
        domain_distribution = df_with_domains['domain_category'].value_counts()
        print("📈 Domain Distribution:")
        for domain, count in domain_distribution.items():
            print(f"  {domain}: {count:,} contacts ({count/len(df)*100:.1f}%)")
        
        # Create timeline by domain type
        for domain_cat in ['WeCloud Domain', 'Beam Data Domain', 'Other Domain', 'Unknown']:
            domain_data = df_with_domains[df_with_domains['domain_category'] == domain_cat]
            
            if len(domain_data) > 0 and main_timestamp_col in domain_data.columns:
                # Monthly activity summary
                domain_monthly = domain_data.groupby(domain_data[main_timestamp_col].dt.to_period('M')).agg({
                    'id': 'count',
                    'email_open_count': ['sum', 'mean'],
                    'email_click_count': ['sum', 'mean'],
                    'email_list': 'nunique',
                    'organization_employees': 'mean'
                }).round(2)
                
                domain_monthly.columns = ['contacts', 'total_opens', 'avg_opens', 'total_clicks', 'avg_clicks', 'unique_lists', 'avg_company_size']
                
                if len(domain_monthly) > 0:
                    print(f"\n  🎯 {domain_cat} Timeline:")
                    for month, row in domain_monthly.iterrows():
                        print(f"    {month}: {row['contacts']:,} contacts | {row['total_opens']:.0f} opens | {row['unique_lists']:.0f} lists | Avg: {row['avg_opens']:.2f} opens/contact | Company size: {row['avg_company_size']:.0f}")
                    
                    # Performance summary
                    overall_performance = domain_data['email_open_count'].mean()
                    total_contacts = len(domain_data)
                    print(f"    Overall Performance: {overall_performance:.3f} opens/contact ({total_contacts:,} total contacts)")
    
    else:
        print("⚠️ Upload method or timestamp data not available for temporal analysis")
    
    # 12. Seasonal Analysis
    print("\n12. SEASONAL AND CYCLICAL ANALYSIS:")
    print("-" * 60)
    
    # Weekend vs weekday analysis
    weekend_performance = df.groupby('is_weekend')['email_open_count'].mean()
    weekday_rate = weekend_performance.get(False, 0)
    weekend_rate = weekend_performance.get(True, 0)
    
    if weekday_rate > 0 and weekend_rate > 0:
        weekend_effect = (weekend_rate - weekday_rate) / weekday_rate * 100
    else:
        weekend_effect = 0
    
    print(f"📅 Weekday vs Weekend Analysis:")
    print(f"  Weekday average: {weekday_rate:.3f} opens/contact")
    print(f"  Weekend average: {weekend_rate:.3f} opens/contact")
    print(f"  Weekend effect: {weekend_effect:+.1f}%")
    
    # Hour of day analysis (if timestamp has time)
    if 'hour' in df.columns and df['hour'].notna().sum() > 0:
        hourly_performance = df.groupby('hour')['email_open_count'].mean()
        best_hour = hourly_performance.idxmax()
        worst_hour = hourly_performance.idxmin()
        print(f"\n⏰ Best hour for engagement: {best_hour}:00 ({hourly_performance[best_hour]:.3f})")
        print(f"⏰ Worst hour for engagement: {worst_hour}:00 ({hourly_performance[worst_hour]:.3f})")
    
    # 13. Create Comprehensive Visualizations
    print("\n13. CREATING COMPREHENSIVE VISUALIZATIONS:")
    print("-" * 60)
    
    # Create a comprehensive dashboard
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle('Comprehensive Email List Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Daily time series
    ax = axes[0, 0]
    daily_metrics.reset_index()['open_rate'].plot(ax=ax, color='blue', alpha=0.7)
    daily_metrics.reset_index()['open_rate_7d_avg'].plot(ax=ax, color='red', linewidth=2)
    ax.set_title('Daily Open Rate Time Series')
    ax.set_ylabel('Open Rate')
    ax.legend(['Daily', '7-day Average'])
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Day of week performance
    ax = axes[0, 1]
    dow_analysis['open_rate'].plot(kind='bar', ax=ax, color='green')
    ax.set_title('Performance by Day of Week')
    ax.set_ylabel('Open Rate')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Monthly performance
    ax = axes[0, 2]
    monthly_analysis['open_rate'].plot(kind='bar', ax=ax, color='orange')
    ax.set_title('Performance by Month')
    ax.set_ylabel('Open Rate')
    ax.set_xlabel('Month')
    
    # Plot 4: List diversity vs performance
    ax = axes[1, 0]
    scatter_data = significant_lists.sample(min(100, len(significant_lists)))
    ax.scatter(scatter_data['diversity'], scatter_data['open_rate'], alpha=0.6, color='purple')
    ax.set_xlabel('List Diversity')
    ax.set_ylabel('Open Rate')
    ax.set_title('Diversity vs Performance')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Contact frequency vs performance (fatigue analysis)
    ax = axes[1, 1]
    fatigue_analysis['open_rate'].plot(kind='bar', ax=ax, color='red')
    ax.set_title('Performance by Contact Frequency')
    ax.set_ylabel('Open Rate')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 6: Performance distribution
    ax = axes[1, 2]
    significant_lists['performance_score'].hist(bins=20, ax=ax, color='skyblue', alpha=0.7)
    ax.set_title('Performance Score Distribution')
    ax.set_xlabel('Performance Score')
    ax.set_ylabel('Frequency')
    
    # Plot 7: List size vs performance
    ax = axes[2, 0]
    size_performance = df.groupby('email_count')['email_open_count'].mean()
    size_performance = size_performance[size_performance.index <= 20]  # Limit to reasonable sizes
    size_performance.plot(ax=ax, color='brown', marker='o')
    ax.set_title('List Size vs Performance')
    ax.set_xlabel('Number of Emails in List')
    ax.set_ylabel('Average Opens')
    ax.grid(True, alpha=0.3)
    
    # Plot 8: Cohort heatmap (if enough data)
    ax = axes[2, 1]
    if len(df['cohort_month'].unique()) > 1 and len(df['weeks_since_list_intro'].unique()) > 1:
        cohort_pivot = df.groupby(['cohort_month', 'weeks_since_list_intro'])['email_open_count'].mean().unstack()
        cohort_pivot = cohort_pivot.iloc[:6, :8]  # Limit size for readability
        sns.heatmap(cohort_pivot, annot=True, fmt='.2f', ax=ax, cmap='YlOrRd')
        ax.set_title('Cohort Analysis Heatmap')
        ax.set_ylabel('Cohort Month')
        ax.set_xlabel('Weeks Since List Introduction')
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor cohort analysis', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cohort Analysis Heatmap')
    
    # Plot 9: Hour of day performance (if available)
    ax = axes[2, 2]
    if 'hour' in df.columns and df['hour'].notna().sum() > 10:
        hourly_performance.plot(kind='bar', ax=ax, color='teal')
        ax.set_title('Performance by Hour of Day')
        ax.set_ylabel('Average Opens')
        ax.set_xlabel('Hour')
    else:
        ax.text(0.5, 0.5, 'Hour data not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance by Hour of Day')
    
    # Plot 10: Contact Volume Timeline by List Size (Original Categorization)
    ax = axes[3, 0]
    if main_timestamp_col in df.columns:
        # Use the simplified categorization
        df['list_size_category'] = df['email_count'].apply(lambda x: 
            'Small Lists (1-3)' if 1 <= x <= 3 else
            'Medium Lists (4-10)' if 4 <= x <= 10 else
            'Large Lists (11+)' if x >= 11 else 'Unknown')
        
        # Create monthly data for each size category
        colors = ['lightblue', 'orange', 'lightgreen']
        size_categories = ['Small Lists (1-3)', 'Medium Lists (4-10)', 'Large Lists (11+)']
        
        for i, size_cat in enumerate(size_categories):
            size_df = df[df['list_size_category'] == size_cat]
            
            if len(size_df) > 0:
                size_monthly = size_df.groupby(size_df[main_timestamp_col].dt.to_period('M')).agg({
                    'id': 'count'
                })
                
                if len(size_monthly) > 0:
                    dates = [pd.to_datetime(str(period)) for period in size_monthly.index]
                    contacts = size_monthly['id'].values
                    ax.plot(dates, contacts, marker='o', label=f'{size_cat}', 
                           color=colors[i], linewidth=2, markersize=4)
        
        ax.set_title('Contact Volume Timeline by Email List Size')
        ax.set_ylabel('Number of Contacts')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No timestamp\ndata available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Contact Volume by Email List Size')
    
    # Plot 11: Contact Volume Timeline by Domain Type
    ax = axes[3, 1]
    if main_timestamp_col in df.columns:
        # Use existing domain categorization
        df_with_domains = df.copy()
        df_with_domains['domain_category'] = df_with_domains['email_list'].apply(categorize_domain)
        
        # Create monthly data for each domain category
        colors = ['purple', 'green', 'gray', 'red']
        domain_categories = ['WeCloud Domain', 'Beam Data Domain', 'Other Domain', 'Unknown']
        
        for i, domain_cat in enumerate(domain_categories):
            domain_df = df_with_domains[df_with_domains['domain_category'] == domain_cat]
            
            if len(domain_df) > 0:
                domain_monthly = domain_df.groupby(domain_df[main_timestamp_col].dt.to_period('M')).agg({
                    'id': 'count'
                })
                
                if len(domain_monthly) > 0:
                    dates = [pd.to_datetime(str(period)) for period in domain_monthly.index]
                    contacts = domain_monthly['id'].values
                    label = domain_cat.replace(' Domain', '') if 'Domain' in domain_cat else domain_cat
                    ax.plot(dates, contacts, marker='s', label=label, 
                           color=colors[i], linewidth=2, markersize=5)
        
        ax.set_title('Contact Volume Timeline by Domain Type')
        ax.set_ylabel('Number of Contacts')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No timestamp\ndata available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Contact Volume by Domain Type')
    
    # Plot 12: Performance Timeline by Domain Type
    ax = axes[3, 2]
    if main_timestamp_col in df.columns:
        # Use existing domain categorization
        df_with_domains = df.copy()
        df_with_domains['domain_category'] = df_with_domains['email_list'].apply(categorize_domain)
        
        # Create monthly performance data for each domain category
        colors = ['purple', 'green', 'gray', 'red']
        
        for i, domain_cat in enumerate(['WeCloud Domain', 'Beam Data Domain', 'Other Domain', 'Unknown']):
            domain_df = df_with_domains[df_with_domains['domain_category'] == domain_cat]
            
            if len(domain_df) > 0:
                domain_monthly = domain_df.groupby(domain_df[main_timestamp_col].dt.to_period('M')).agg({
                    'email_open_count': 'mean'
                })
                
                if len(domain_monthly) > 0:
                    dates = [pd.to_datetime(str(period)) for period in domain_monthly.index]
                    performance = domain_monthly['email_open_count'].values
                    label = domain_cat.replace(' Domain', '') if 'Domain' in domain_cat else domain_cat
                    ax.plot(dates, performance, marker='o', label=label, 
                           color=colors[i], linewidth=2, markersize=4)
        
        ax.set_title('Performance Timeline by Domain Type')
        ax.set_ylabel('Average Opens per Contact')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.text(0.5, 0.5, 'No timestamp\ndata available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Performance by Domain Type')
    
    plt.tight_layout()
    plt.savefig('email_list_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive dashboard saved as 'email_list_analysis_dashboard.png'")
    plt.show()
    
    # 14. Strategic Recommendations
    print("\n14. STRATEGIC RECOMMENDATIONS AND INSIGHTS:")
    print("=" * 60)
    
    print("🎯 TARGETING STRATEGY:")
    print(f"  • Focus on lists with diversity score > {significant_lists['diversity'].quantile(0.75):.2f}")
    print(f"  • Prioritize lists with performance score > {significant_lists['performance_score'].quantile(0.8):.3f}")
    if weekend_effect > 0:
        print(f"  • Weekend campaigns show {weekend_effect:.1f}% better performance")
    else:
        print(f"  • Weekday campaigns show {abs(weekend_effect):.1f}% better performance")
    
    print(f"\n⏰ TIMING OPTIMIZATION:")
    print(f"  • Best day of week: {best_day}")
    print(f"  • Worst day of week: {worst_day}")
    if 'hour' in df.columns and df['hour'].notna().sum() > 0:
        print(f"  • Best hour: {best_hour}:00")
        print(f"  • Worst hour: {worst_hour}:00")
    
    print(f"\n📉 LIST FATIGUE MANAGEMENT:")
    optimal_frequency = fatigue_analysis['open_rate'].idxmax()
    print(f"  • Optimal contact frequency: {optimal_frequency}")
    print(f"  • Monitor lists with 500+ contacts for fatigue")
    if 'degradation_analysis' in locals() and len(degradation_analysis) > 0:
        print(f"  • Average list degradation: {avg_degradation:.1f}% over time")
    
    print(f"\n🔮 PREDICTIVE INSIGHTS:")
    if 'trend_slope' in locals():
        if trend_slope > 0:
            print(f"  • Positive trend: {trend_slope*365:.3f} annual open rate increase")
        else:
            print(f"  • Declining trend: {abs(trend_slope*365):.3f} annual open rate decrease")
    
    print(f"\n📤 UPLOAD METHOD OPTIMIZATION:")
    if 'upload_method' in df.columns:
        method_performance = df.groupby('upload_method')['email_open_count'].mean()
        if 'api' in method_performance.index and 'manual' in method_performance.index:
            api_performance = method_performance['api']
            manual_performance = method_performance['manual']
            if api_performance > manual_performance:
                improvement = (api_performance - manual_performance) / manual_performance * 100
                print(f"  • API uploads perform {improvement:.1f}% better than manual uploads")
                print(f"  • Consider increasing API adoption for better engagement")
            elif manual_performance > api_performance:
                improvement = (manual_performance - api_performance) / api_performance * 100
                print(f"  • Manual uploads perform {improvement:.1f}% better than API uploads")
                print(f"  • Focus on manual upload quality and processes")
            else:
                print(f"  • API and manual uploads show similar performance")
        
        api_percentage = (df['upload_method'] == 'api').mean() * 100
        print(f"  • Current API adoption: {api_percentage:.1f}% of all uploads")
        if api_percentage < 20:
            print(f"  • Low API adoption - opportunity for automation")
        elif api_percentage > 80:
            print(f"  • High API adoption - monitor quality consistency")
    
    print(f"\n💡 DATA QUALITY OPPORTUNITIES:")
    print(f"  • {df['email_list'].isnull().sum()} contacts missing email list data")
    print(f"  • {len(df) - len(mature_lists)} lists need lifecycle tracking")
    print(f"  • Focus on improving {len(significant_lists[significant_lists['diversity'] < 0.5])} low-diversity lists")
    
    print("\n" + "="*80)
    print("✅ ADVANCED EMAIL LIST ANALYSIS COMPLETE!")
    print("📊 Generated comprehensive dashboard with time series insights")
    print("🎯 Strategic recommendations provided for optimization")
    print("="*80)
    
    return df

if __name__ == "__main__":
    df = analyze_email_lists() 