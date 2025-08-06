"""
Dataset Exploration for Better Categorization
Exploring additional data columns to improve categorization of "Other" companies
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
CSV_FILE_PATH = Path("merged_contacts.csv")

def explore_dataset_columns():
    """Explore all columns in the dataset to understand available data"""
    print("🔍 EXPLORING DATASET COLUMNS")
    print("=" * 80)
    
    # Load data
    if not CSV_FILE_PATH.exists():
        print(f"❌ Error: '{CSV_FILE_PATH}' not found.")
        return
    
    df = pd.read_csv(CSV_FILE_PATH, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    print(f"📊 Total columns: {len(df.columns)}")
    
    # Display all columns
    print("\n📋 ALL AVAILABLE COLUMNS:")
    print("-" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")
    
    return df

def analyze_organization_data(df):
    """Analyze organization-related columns to find additional categorization data"""
    print("\n\n🏢 ORGANIZATION DATA ANALYSIS")
    print("=" * 80)
    
    # Find all organization-related columns
    org_columns = [col for col in df.columns if 'organization' in col.lower() or 'company' in col.lower()]
    print(f"📋 Organization-related columns found: {len(org_columns)}")
    
    for col in org_columns:
        print(f"\n🔍 Analyzing: {col}")
        print("-" * 50)
        
        # Basic stats
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        
        print(f"   Non-null values: {non_null:,}")
        print(f"   Null values: {null_count:,}")
        print(f"   Unique values: {unique_count:,}")
        
        # Show sample values
        if unique_count <= 20:
            print(f"   All unique values:")
            for val in df[col].dropna().unique():
                print(f"     - {val}")
        else:
            print(f"   Sample values (first 10):")
            for val in df[col].dropna().unique()[:10]:
                print(f"     - {val}")
            print(f"     ... and {unique_count - 10} more")

def analyze_job_title_data(df):
    """Analyze job title and role-related columns"""
    print("\n\n👔 JOB TITLE DATA ANALYSIS")
    print("=" * 80)
    
    # Find job-related columns
    job_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                   ['title', 'role', 'position', 'job', 'seniority', 'level'])]
    
    print(f"📋 Job-related columns found: {len(job_columns)}")
    
    for col in job_columns:
        print(f"\n🔍 Analyzing: {col}")
        print("-" * 50)
        
        # Basic stats
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        
        print(f"   Non-null values: {non_null:,}")
        print(f"   Null values: {null_count:,}")
        print(f"   Unique values: {unique_count:,}")
        
        # Show sample values
        if unique_count <= 30:
            print(f"   All unique values:")
            for val in sorted(df[col].dropna().unique()):
                print(f"     - {val}")
        else:
            print(f"   Sample values (first 15):")
            for val in sorted(df[col].dropna().unique())[:15]:
                print(f"     - {val}")
            print(f"     ... and {unique_count - 15} more")

def analyze_industry_breakdown(df):
    """Analyze current industry categorization to see what's in 'Other'"""
    print("\n\n🏭 INDUSTRY CATEGORIZATION ANALYSIS")
    print("=" * 80)
    
    if 'organization_industry' in df.columns:
        print("📊 Current Industry Distribution:")
        industry_counts = df['organization_industry'].value_counts()
        print(f"   Total unique industries: {len(industry_counts)}")
        print(f"   Top 20 industries:")
        for i, (industry, count) in enumerate(industry_counts.head(20).items(), 1):
            print(f"   {i:2d}. {industry}: {count:,}")
        
        # Check for industries that might be miscategorized
        print(f"\n🔍 Industries containing 'Other' or 'Unknown':")
        other_industries = df[df['organization_industry'].str.contains('other|unknown|nan', case=False, na=False)]
        print(f"   Count: {len(other_industries):,}")
        
        # Show unique values that might be miscategorized
        print(f"\n🔍 Industries that might be miscategorized:")
        potential_miscategorized = df[
            ~df['organization_industry'].str.contains('other|unknown|nan', case=False, na=False) &
            df['organization_industry'].notna()
        ]['organization_industry'].value_counts()
        
        print(f"   Top 30 non-'Other' industries:")
        for i, (industry, count) in enumerate(potential_miscategorized.head(30).items(), 1):
            print(f"   {i:2d}. {industry}: {count:,}")

def analyze_seniority_breakdown(df):
    """Analyze current seniority categorization to see what's in 'Other'"""
    print("\n\n👤 SENIORITY CATEGORIZATION ANALYSIS")
    print("=" * 80)
    
    if 'seniority' in df.columns:
        print("📊 Current Seniority Distribution:")
        seniority_counts = df['seniority'].value_counts()
        print(f"   Total unique seniority levels: {len(seniority_counts)}")
        print(f"   Top 20 seniority levels:")
        for i, (seniority, count) in enumerate(seniority_counts.head(20).items(), 1):
            print(f"   {i:2d}. {seniority}: {count:,}")
        
        # Check for seniority levels that might be miscategorized
        print(f"\n🔍 Seniority levels containing 'Other' or 'Unknown':")
        other_seniorities = df[df['seniority'].str.contains('other|unknown|nan', case=False, na=False)]
        print(f"   Count: {len(other_seniorities):,}")
        
        # Show unique values that might be miscategorized
        print(f"\n🔍 Seniority levels that might be miscategorized:")
        potential_miscategorized = df[
            ~df['seniority'].str.contains('other|unknown|nan', case=False, na=False) &
            df['seniority'].notna()
        ]['seniority'].value_counts()
        
        print(f"   Top 30 non-'Other' seniority levels:")
        for i, (seniority, count) in enumerate(potential_miscategorized.head(30).items(), 1):
            print(f"   {i:2d}. {seniority}: {count:,}")

def analyze_additional_categorization_columns(df):
    """Analyze any additional columns that could help with categorization"""
    print("\n\n🔍 ADDITIONAL CATEGORIZATION COLUMNS")
    print("=" * 80)
    
    # Look for columns that might contain categorization data
    potential_categorization_cols = []
    
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['type', 'category', 'sector', 'domain', 'field', 'area']):
            potential_categorization_cols.append(col)
    
    print(f"📋 Potential categorization columns found: {len(potential_categorization_cols)}")
    
    for col in potential_categorization_cols:
        print(f"\n🔍 Analyzing: {col}")
        print("-" * 50)
        
        # Basic stats
        non_null = df[col].notna().sum()
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()
        
        print(f"   Non-null values: {non_null:,}")
        print(f"   Null values: {null_count:,}")
        print(f"   Unique values: {unique_count:,}")
        
        # Show sample values
        if unique_count <= 20:
            print(f"   All unique values:")
            for val in sorted(df[col].dropna().unique()):
                print(f"     - {val}")
        else:
            print(f"   Sample values (first 10):")
            for val in sorted(df[col].dropna().unique())[:10]:
                print(f"     - {val}")
            print(f"     ... and {unique_count - 10} more")

def main():
    """Main exploration pipeline"""
    print("=== DATASET EXPLORATION FOR BETTER CATEGORIZATION ===")
    print("🎯 Goal: Find additional data to improve 'Other' categorization")
    print("=" * 80)
    
    # Load and explore data
    df = explore_dataset_columns()
    
    if df is not None:
        # Analyze different aspects of the data
        analyze_organization_data(df)
        analyze_job_title_data(df)
        analyze_industry_breakdown(df)
        analyze_seniority_breakdown(df)
        analyze_additional_categorization_columns(df)
        
        print("\n" + "="*100)
        print("✅ DATASET EXPLORATION COMPLETE")
        print("📊 Review the findings above to identify additional categorization opportunities")
        print("="*100)

if __name__ == "__main__":
    main() 