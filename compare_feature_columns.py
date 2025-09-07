#!/usr/bin/env python3
"""
Compare feature columns between the original comprehensive_preprocessing_pipeline.py 
and our new bronze_to_silver_pipeline.py
"""

import pandas as pd
import numpy as np

def get_original_pipeline_features():
    """Features created by the original comprehensive_preprocessing_pipeline.py"""
    
    # From the original script analysis
    original_features = {
        # Text features
        'combined_text',
        'combined_text_length',    # -> text_length in new pipeline
        'combined_text_word_count', # -> text_word_count in new pipeline
        'has_numbers_in_text',
        'has_email_in_text',
        'has_url_in_text',
        'text_quality_score',
        
        # Timestamp features from timestamp_created
        'created_day_of_week',
        'created_month', 
        'created_hour',
        'created_quarter',
        'created_is_weekend',
        'created_is_business_hours',
        'created_is_morning',
        'created_season',
        'days_since_creation',
        'weeks_since_creation',
        
        # Time differences (from TIMESTAMP_COLS)
        'days_between_last_contact_and_created',
        'days_between_retrieval_timestamp_and_created', # -> days_between_retrieval_and_created
        'days_between_enriched_at_and_created',        # -> days_between_enriched_and_created
        'days_between_inserted_at_and_created',        # -> days_between_inserted_and_created
        'days_between_last_contacted_from_and_created',
        
        # Absolute time differences
        'days_between_last_contact_and_created_abs',
        'days_between_retrieval_timestamp_and_created_abs',
        'days_between_enriched_at_and_created_abs',
        'days_between_inserted_at_and_created_abs',
        'days_between_last_contacted_from_and_created_abs',
        
        # Has timestamp indicators
        'has_last_contact',
        'has_retrieval_timestamp',  # -> has_retrieval
        'has_enriched_at',          # -> has_enriched
        'has_inserted_at',          # -> has_inserted
        'has_last_contacted_from',
        
        # Interaction features
        'industry_seniority_interaction',    # -> apollo_seniority_industry
        'geo_industry_interaction',          # -> apollo_geo_industry
        'title_industry_interaction',        # Similar to org_title_interaction
        
        # JSONB features
        'has_employment_history',
        'has_organization_data',
        'has_account_data',
        'has_api_response_raw',
        'enrichment_completeness',
        'enrichment_completeness_pct',
    }
    
    return original_features

def get_new_pipeline_features():
    """Features created by our new bronze_to_silver_pipeline.py"""
    
    new_features = {
        # Text features
        'combined_text',
        'text_length',              # was combined_text_length
        'text_word_count',          # was combined_text_word_count
        'text_char_count',          # NEW
        'text_avg_word_length',     # NEW
        'text_uppercase_ratio',     # NEW
        'text_digit_ratio',         # NEW
        'text_punctuation_ratio',   # NEW
        
        # Timestamp features
        'created_day_of_week',
        'created_month',
        'created_hour',
        'days_since_creation',
        'weeks_since_creation',
        
        # Time differences (updated names)
        'days_between_updated_and_created',
        'days_between_last_contact_and_created',
        'days_between_last_touch_and_created',
        'days_between_last_open_and_created',
        'days_between_last_click_and_created', 
        'days_between_last_reply_and_created',
        'days_between_last_interest_change_and_created',
        'days_between_retrieval_and_created',
        'days_between_inserted_and_created',
        'days_between_enriched_and_created',
        
        # Has timestamp indicators
        'has_updated',
        'has_last_contact',
        'has_last_touch',
        'has_last_open',
        'has_last_click',
        'has_last_reply',
        'has_last_interest_change',
        'has_retrieval',
        'has_inserted',
        'has_enriched',
        
        # Interaction features
        'org_title_interaction',
        'status_method_interaction',
        'apollo_seniority_industry',
        'apollo_dept_function',
        'apollo_geo_industry',
        
        # Apollo enrichment features (NEW - major addition)
        'has_apollo_enrichment',
        'apollo_api_success',
        'company_size_category',
        'company_size_log',
        'company_age_years',
        'company_age_category',
        'apollo_enrichment_cost',
        'days_since_apollo_enrichment',
        'apollo_data_freshness',
        'apollo_data_completeness_pct',
        'is_high_value_title',
        'is_high_value_seniority',
        'is_tech_department',
        
        # Quality scores (NEW)
        'data_quality_score',
        'feature_completeness_score',
    }
    
    return new_features

def compare_pipelines():
    """Compare the two feature sets"""
    
    original = get_original_pipeline_features()
    new = get_new_pipeline_features()
    
    print("🔍 FEATURE COMPARISON: Original vs New Pipeline")
    print("=" * 60)
    
    # Features in both
    common = original & new
    print(f"\n✅ COMMON FEATURES ({len(common)}):")
    for feature in sorted(common):
        print(f"   {feature}")
    
    # Features only in original
    only_original = original - new
    print(f"\n❌ MISSING FROM NEW PIPELINE ({len(only_original)}):")
    for feature in sorted(only_original):
        print(f"   {feature}")
    
    # Features only in new
    only_new = new - original  
    print(f"\n🆕 NEW FEATURES ADDED ({len(only_new)}):")
    for feature in sorted(only_new):
        print(f"   {feature}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Original pipeline: {len(original)} features")
    print(f"   New pipeline: {len(new)} features")
    print(f"   Common: {len(common)} features")
    print(f"   Missing: {len(only_original)} features")
    print(f"   Added: {len(only_new)} features")
    
    coverage = len(common) / len(original) * 100
    print(f"   Coverage: {coverage:.1f}%")
    
    return {
        'common': common,
        'missing': only_original,
        'added': only_new,
        'coverage': coverage
    }

if __name__ == "__main__":
    results = compare_pipelines()
    
    if results['missing']:
        print(f"\n🔧 RECOMMENDED ACTIONS:")
        print("   1. Add missing features to bronze_to_silver_pipeline.py")
        print("   2. Update silver table schema to include missing columns")
        print("   3. Test pipeline with complete feature set")

