#!/usr/bin/env python3
"""
Check column mismatch between silver and gold schemas
"""

import yaml
import pandas as pd
from sqlalchemy import create_engine

# Load config
with open('config/silver_layer_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

db_config = config['database']
connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
engine = create_engine(connection_string)

# Get actual silver columns
df = pd.read_sql('SELECT * FROM leads.silver_ml_features LIMIT 1', engine)
silver_cols = set(df.columns)
print(f"Silver layer has {len(silver_cols)} columns")

# Expected gold columns from our corrected schema
gold_cols_corrected = {
    'id', 'email', 'campaign', 'engagement_level', 'email_open_count', 'email_click_count', 'email_reply_count',
    'created_day_of_week', 'created_month', 'created_hour', 'days_since_creation', 'weeks_since_creation',
    'days_between_updated_and_created', 'days_between_last_contact_and_created', 'days_between_last_touch_and_created',
    'days_between_last_open_and_created', 'days_between_last_click_and_created', 'days_between_last_reply_and_created',
    'days_between_last_interest_change_and_created', 'days_between_retrieval_and_created', 'days_between_inserted_and_created',
    'days_between_enriched_and_created', 'has_updated', 'has_last_contact', 'has_last_touch', 'has_last_open',
    'has_last_click', 'has_last_reply', 'has_last_interest_change', 'has_retrieval', 'has_inserted', 'has_enriched',
    'combined_text', 'text_length', 'text_word_count', 'text_char_count', 'text_avg_word_length', 'text_uppercase_ratio',
    'text_digit_ratio', 'text_punctuation_ratio', 'org_title_interaction', 'status_method_interaction',
    'apollo_seniority_industry', 'apollo_dept_function', 'apollo_geo_industry', 'has_apollo_enrichment',
    'apollo_api_success', 'company_size_category', 'company_size_log', 'company_age_years', 'company_age_category',
    'apollo_enrichment_cost', 'days_since_apollo_enrichment', 'apollo_data_freshness', 'apollo_data_completeness_pct',
    'is_high_value_title', 'is_high_value_seniority', 'is_tech_department', 'data_quality_score', 'feature_completeness_score',
    'timestamp_last_contact', 'timestamp_created', 'timestamp_updated', 'processed_timestamp'
}

print(f"Expected gold columns: {len(gold_cols_corrected)}")
print(f"Missing in silver: {gold_cols_corrected - silver_cols}")
print(f"Extra in silver: {silver_cols - gold_cols_corrected}")

# Check what the actual silver columns are
print("\nActual silver columns:")
for col in sorted(silver_cols):
    print(f"  {col}")
