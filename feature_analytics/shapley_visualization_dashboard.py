"""
Comprehensive Visualization Dashboard for Shapley Analysis Insights
Creates charts for all high-impact features from the analysis breakdown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_esp_code_charts():
    """Create charts for ESP Code analysis"""
    print("Creating ESP Code visualizations...")
    
    # ESP Code data from analysis
    esp_data = {
        'ESP_Code': [1, 2, 5, 8, 999, 10, 12],
        'Provider': ['Gmail/Google Workspace', 'Microsoft Outlook/Office 365', 'Apple iCloud Mail', 
                    'Zoho Mail', 'Custom/Internal Servers', 'GMX Mail', 'Custom/Unknown'],
        'Open_Rate': [1.042, 1.914, 4.500, 1.582, 0.464, 0.000, 0.000],
        'Contacts': [7492, 10332, 2, 371, 5718, 0, 0],
        'Percentage': [28.2, 38.8, 0.01, 1.4, 21.5, 0, 0],
        'Category': ['Major', 'Major', 'Niche', 'Business', 'Custom', 'Niche', 'Custom']
    }
    
    df_esp = pd.DataFrame(esp_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ESP Code Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Open Rate by ESP Code
    ax1 = axes[0, 0]
    bars = ax1.bar(df_esp['Provider'], df_esp['Open_Rate'], color=['#2E8B57', '#4169E1', '#FFD700', '#32CD32', '#DC143C', '#696969', '#A9A9A9'])
    ax1.set_title('Open Rate by Email Service Provider', fontweight='bold')
    ax1.set_ylabel('Opens per Contact')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, df_esp['Open_Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Contact Distribution
    ax2 = axes[0, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    wedges, texts, autotexts = ax2.pie(df_esp['Contacts'], labels=df_esp['Provider'], 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Contact Distribution by Provider', fontweight='bold')
    
    # 3. Performance vs Volume Scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df_esp['Contacts'], df_esp['Open_Rate'], 
                          s=df_esp['Contacts']/100, alpha=0.7, c=range(len(df_esp)), cmap='viridis')
    ax3.set_xlabel('Number of Contacts')
    ax3.set_ylabel('Open Rate (Opens per Contact)')
    ax3.set_title('Performance vs Volume Analysis', fontweight='bold')
    
    # Add labels for points
    for i, (x, y, provider) in enumerate(zip(df_esp['Contacts'], df_esp['Open_Rate'], df_esp['Provider'])):
        ax3.annotate(provider, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Category Performance
    ax4 = axes[1, 1]
    category_perf = df_esp.groupby('Category')['Open_Rate'].mean().sort_values(ascending=False)
    bars = ax4.bar(category_perf.index, category_perf.values, 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#DC143C', '#696969'])
    ax4.set_title('Average Performance by Provider Category', fontweight='bold')
    ax4.set_ylabel('Average Open Rate')
    
    # Add value labels
    for bar, rate in zip(bars, category_perf.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('esp_code_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_email_list_charts():
    """Create charts for Email List analysis"""
    print("Creating Email List visualizations...")
    
    # Email List data from analysis
    list_data = {
        'List_Type': ['BeamData', 'WeCloudData', 'Small Lists (1-5)', 'Medium Lists (6-10)', 'Large Lists (11-20)'],
        'Open_Rate': [2.117, 1.235, 0.357, 1.757, 0.956],
        'Contacts': [2753, 21257, 5000, 11990, 5000],
        'Performance_Score': [100, 58, 17, 83, 45],
        'List_Size_Category': ['Medium', 'Large', 'Small', 'Medium', 'Large']
    }
    
    df_list = pd.DataFrame(list_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Email List Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Open Rate by List Type
    ax1 = axes[0, 0]
    bars = ax1.bar(df_list['List_Type'], df_list['Open_Rate'], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax1.set_title('Open Rate by List Type', fontweight='bold')
    ax1.set_ylabel('Opens per Contact')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, df_list['Open_Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance vs Volume
    ax2 = axes[0, 1]
    scatter = ax2.scatter(df_list['Contacts'], df_list['Open_Rate'], 
                          s=df_list['Contacts']/100, alpha=0.7, c=range(len(df_list)), cmap='viridis')
    ax2.set_xlabel('Number of Contacts')
    ax2.set_ylabel('Open Rate')
    ax2.set_title('Performance vs Volume Analysis', fontweight='bold')
    
    # Add labels
    for i, (x, y, list_type) in enumerate(zip(df_list['Contacts'], df_list['Open_Rate'], df_list['List_Type'])):
        ax2.annotate(list_type, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Performance Score
    ax3 = axes[1, 0]
    bars = ax3.bar(df_list['List_Type'], df_list['Performance_Score'], 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
    ax3.set_title('Performance Score by List Type', fontweight='bold')
    ax3.set_ylabel('Performance Score (0-100)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, df_list['Performance_Score']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # 4. List Size Category Performance
    ax4 = axes[1, 1]
    size_perf = df_list.groupby('List_Size_Category')['Open_Rate'].mean().sort_values(ascending=False)
    bars = ax4.bar(size_perf.index, size_perf.values, 
                   color=['#96CEB4', '#4ECDC4', '#FF6B6B'])
    ax4.set_title('Average Performance by List Size Category', fontweight='bold')
    ax4.set_ylabel('Average Open Rate')
    
    # Add value labels
    for bar, rate in zip(bars, size_perf.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('email_list_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_page_retrieved_charts():
    """Create charts for page_retrieved analysis"""
    print("Creating page_retrieved visualizations...")
    
    # Page Retrieved data from analysis
    page_data = {
        'Page_Range': ['Pages 1-50', 'Pages 51-100', 'Pages 101-150', 'Pages 151-200', 'Pages 201-266'],
        'Open_Rate': [1.392, 1.266, 1.172, 1.119, 1.110],
        'Performance_Rank': [1, 2, 3, 4, 5],
        'Quality_Score': [100, 85, 75, 65, 60],
        'Contact_Count': [5000, 5000, 5000, 5000, 6600]
    }
    
    df_page = pd.DataFrame(page_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Page Retrieved Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Open Rate by Page Range
    ax1 = axes[0, 0]
    bars = ax1.bar(df_page['Page_Range'], df_page['Open_Rate'], 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#DC143C'])
    ax1.set_title('Open Rate by Page Range', fontweight='bold')
    ax1.set_ylabel('Opens per Contact')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, df_page['Open_Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Performance Trend
    ax2 = axes[0, 1]
    ax2.plot(df_page['Performance_Rank'], df_page['Open_Rate'], 'o-', linewidth=3, markersize=8)
    ax2.set_xlabel('Performance Rank (1=Best)')
    ax2.set_ylabel('Open Rate')
    ax2.set_title('Performance Trend by Page Rank', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add labels
    for i, (rank, rate, page_range) in enumerate(zip(df_page['Performance_Rank'], df_page['Open_Rate'], df_page['Page_Range'])):
        ax2.annotate(page_range, (rank, rate), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 3. Quality Score
    ax3 = axes[1, 0]
    bars = ax3.bar(df_page['Page_Range'], df_page['Quality_Score'], 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#DC143C'])
    ax3.set_title('Quality Score by Page Range', fontweight='bold')
    ax3.set_ylabel('Quality Score (0-100)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, df_page['Quality_Score']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Performance vs Volume
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_page['Contact_Count'], df_page['Open_Rate'], 
                          s=df_page['Contact_Count']/100, alpha=0.7, c=range(len(df_page)), cmap='viridis')
    ax4.set_xlabel('Number of Contacts')
    ax4.set_ylabel('Open Rate')
    ax4.set_title('Performance vs Volume Analysis', fontweight='bold')
    
    # Add labels
    for i, (x, y, page_range) in enumerate(zip(df_page['Contact_Count'], df_page['Open_Rate'], df_page['Page_Range'])):
        ax4.annotate(page_range, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('page_retrieved_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_status_charts():
    """Create charts for Status X and Status Y analysis"""
    print("Creating Status field visualizations...")
    
    # Status data from analysis
    status_data = {
        'Status_X': [3, 1, -1, -3, 0],
        'Status_Y': [0.0, 1.0, 1.0, 0.0, 0.0],
        'Open_Rate': [0.472, 0.461, 0.455, 0.429, 0.364],
        'Contacts': [18015, 8441, 110, 21, 11],
        'Percentage': [67.7, 31.7, 0.4, 0.1, 0.04],
        'Status_Label': ['Verified/Valid', 'Unverified/Uncertain', 'Invalid/Rejected', 'Blocked/Banned', 'Pending/Unknown']
    }
    
    df_status = pd.DataFrame(status_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Status X and Status Y Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Open Rate by Status X
    ax1 = axes[0, 0]
    bars = ax1.bar(df_status['Status_Label'], df_status['Open_Rate'], 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#DC143C'])
    ax1.set_title('Open Rate by Status X', fontweight='bold')
    ax1.set_ylabel('Open Rate')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, df_status['Open_Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Contact Distribution
    ax2 = axes[0, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    wedges, texts, autotexts = ax2.pie(df_status['Contacts'], labels=df_status['Status_Label'], 
                                       autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title('Contact Distribution by Status X', fontweight='bold')
    
    # 3. Status X vs Status Y Correlation
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df_status['Status_X'], df_status['Status_Y'], 
                          s=df_status['Contacts']/100, alpha=0.7, c=df_status['Open_Rate'], cmap='viridis')
    ax3.set_xlabel('Status X')
    ax3.set_ylabel('Status Y')
    ax3.set_title('Status X vs Status Y Correlation', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Open Rate')
    
    # Add labels
    for i, (x, y, label) in enumerate(zip(df_status['Status_X'], df_status['Status_Y'], df_status['Status_Label'])):
        ax3.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Performance vs Volume
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df_status['Contacts'], df_status['Open_Rate'], 
                          s=df_status['Contacts']/100, alpha=0.7, c=range(len(df_status)), cmap='viridis')
    ax4.set_xlabel('Number of Contacts')
    ax4.set_ylabel('Open Rate')
    ax4.set_title('Performance vs Volume Analysis', fontweight='bold')
    
    # Add labels
    for i, (x, y, label) in enumerate(zip(df_status['Contacts'], df_status['Open_Rate'], df_status['Status_Label'])):
        ax4.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('status_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_summary():
    """Create a comprehensive summary chart"""
    print("Creating comprehensive summary visualization...")
    
    # Summary data for all features
    summary_data = {
        'Feature': ['ESP Code (Outlook)', 'ESP Code (Gmail)', 'Email List (BeamData)', 'Email List (WeCloudData)', 
                   'Page Retrieved (1-50)', 'Page Retrieved (200+)', 'Status X (3)', 'Status X (1)'],
        'Open_Rate': [1.914, 1.042, 2.117, 1.235, 1.392, 1.110, 0.472, 0.461],
        'Impact_Score': [100, 85, 95, 70, 90, 60, 88, 75],
        'Category': ['Provider', 'Provider', 'List', 'List', 'Quality', 'Quality', 'Status', 'Status']
    }
    
    df_summary = pd.DataFrame(summary_data)
    
    # Create comprehensive chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Comprehensive Feature Impact Analysis', fontsize=16, fontweight='bold')
    
    # 1. Open Rate Comparison
    bars = ax1.bar(df_summary['Feature'], df_summary['Open_Rate'], 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#32CD32', '#DC143C', '#9370DB', '#20B2AA'])
    ax1.set_title('Open Rate by Feature', fontweight='bold')
    ax1.set_ylabel('Opens per Contact')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, rate in zip(bars, df_summary['Open_Rate']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Impact Score
    bars = ax2.bar(df_summary['Feature'], df_summary['Impact_Score'], 
                   color=['#2E8B57', '#4169E1', '#FFD700', '#FF6347', '#32CD32', '#DC143C', '#9370DB', '#20B2AA'])
    ax2.set_title('Impact Score by Feature', fontweight='bold')
    ax2.set_ylabel('Impact Score (0-100)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, df_summary['Impact_Score']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{score}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_feature_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to create all visualizations"""
    print("Creating comprehensive visualization dashboard for Shapley analysis insights...")
    
    # Create all charts
    create_esp_code_charts()
    create_email_list_charts()
    create_page_retrieved_charts()
    create_status_charts()
    create_comprehensive_summary()
    
    print("\n✅ All visualizations completed!")
    print("Generated files:")
    print("- esp_code_analysis_dashboard.png")
    print("- email_list_analysis_dashboard.png")
    print("- page_retrieved_analysis_dashboard.png")
    print("- status_analysis_dashboard.png")
    print("- comprehensive_feature_analysis.png")

if __name__ == "__main__":
    main() 