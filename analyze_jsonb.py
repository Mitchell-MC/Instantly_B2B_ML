"""
Analyze JSONB columns in the database to understand their structure and content.
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def analyze_jsonb_columns():
    """Analyze JSONB columns in the merged_contacts.csv file."""
    print("🔍 Analyzing JSONB columns in database...")
    print("=" * 60)
    
    # Load sample data
    df = pd.read_csv('merged_contacts.csv', nrows=1000)
    
    # JSONB columns to analyze
    jsonb_cols = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']
    
    for col in jsonb_cols:
        if col in df.columns:
            print(f"\n📊 Analyzing {col}:")
            print("-" * 40)
            
            # Basic statistics
            non_null_count = df[col].notna().sum()
            null_count = df[col].isna().sum()
            total_count = len(df)
            
            print(f"Total records: {total_count}")
            print(f"Non-null records: {non_null_count} ({non_null_count/total_count*100:.1f}%)")
            print(f"Null records: {null_count} ({null_count/total_count*100:.1f}%)")
            
            # Sample non-null values
            non_null_values = df[col].dropna().head(3)
            print(f"\nSample values:")
            for i, val in enumerate(non_null_values):
                print(f"  {i+1}. {val[:200]}{'...' if len(str(val)) > 200 else ''}")
            
            # Try to parse JSON and analyze structure
            print(f"\nJSON Structure Analysis:")
            try:
                parsed_samples = []
                for val in non_null_values:
                    try:
                        parsed = json.loads(val)
                        parsed_samples.append(parsed)
                    except:
                        continue
                
                if parsed_samples:
                    # Analyze structure
                    if isinstance(parsed_samples[0], list):
                        print(f"  Type: Array/List")
                        print(f"  Average length: {np.mean([len(x) for x in parsed_samples]):.1f}")
                        if parsed_samples[0]:
                            print(f"  Sample item keys: {list(parsed_samples[0][0].keys()) if isinstance(parsed_samples[0][0], dict) else 'Not dict'}")
                    elif isinstance(parsed_samples[0], dict):
                        print(f"  Type: Object/Dictionary")
                        all_keys = set()
                        for sample in parsed_samples:
                            all_keys.update(sample.keys())
                        print(f"  Common keys: {list(all_keys)[:10]}")
                    else:
                        print(f"  Type: {type(parsed_samples[0])}")
                else:
                    print("  Could not parse any JSON samples")
                    
            except Exception as e:
                print(f"  Error analyzing structure: {e}")
            
            # Length analysis
            lengths = df[col].astype(str).str.len()
            print(f"\nLength Analysis:")
            print(f"  Min length: {lengths.min()}")
            print(f"  Max length: {lengths.max()}")
            print(f"  Mean length: {lengths.mean():.1f}")
            print(f"  Median length: {lengths.median():.1f}")
            
        else:
            print(f"\n❌ Column '{col}' not found in dataset")
    
    print("\n" + "=" * 60)
    print("✅ JSONB Analysis Complete")

if __name__ == "__main__":
    analyze_jsonb_columns() 