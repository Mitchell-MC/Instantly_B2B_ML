-- Update silver_ml_features table to include all engineered features
-- This matches the actual features created by the bronze_to_silver_pipeline.py

-- Drop existing table and recreate with full schema
DROP TABLE IF EXISTS leads.silver_ml_features CASCADE;

-- Create comprehensive silver layer table
CREATE TABLE leads.silver_ml_features (
    -- Primary identifiers
    id UUID PRIMARY KEY,
    email VARCHAR(255),
    campaign UUID,
    
    -- Target variable
    engagement_level INTEGER CHECK (engagement_level IN (0, 1, 2)),
    
    -- Processing metadata
    processed_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Core engagement metrics (for reference, not features)
    email_open_count INTEGER,
    email_click_count INTEGER,
    email_reply_count INTEGER,
    
    -- === TIMESTAMP FEATURES ===
    created_day_of_week INTEGER,
    created_month INTEGER,
    created_hour INTEGER,
    days_since_creation INTEGER,
    weeks_since_creation INTEGER,
    
    -- Time differences from creation
    days_between_updated_and_created INTEGER,
    days_between_last_contact_and_created INTEGER,
    days_between_last_touch_and_created INTEGER,
    days_between_last_open_and_created INTEGER,
    days_between_last_click_and_created INTEGER,
    days_between_last_reply_and_created INTEGER,
    days_between_last_interest_change_and_created INTEGER,
    days_between_retrieval_and_created INTEGER,
    days_between_inserted_and_created INTEGER,
    days_between_enriched_and_created INTEGER,
    
    -- Timestamp presence indicators
    has_updated INTEGER,
    has_last_contact INTEGER,
    has_last_touch INTEGER,
    has_last_open INTEGER,
    has_last_click INTEGER,
    has_last_reply INTEGER,
    has_last_interest_change INTEGER,
    has_retrieval INTEGER,
    has_inserted INTEGER,
    has_enriched INTEGER,
    
    -- === TEXT FEATURES ===
    combined_text TEXT,
    text_length INTEGER,
    text_word_count INTEGER,
    text_char_count INTEGER,
    text_avg_word_length FLOAT,
    text_uppercase_ratio FLOAT,
    text_digit_ratio FLOAT,
    text_punctuation_ratio FLOAT,
    
    -- === CATEGORICAL INTERACTION FEATURES ===
    org_title_interaction VARCHAR(500),
    status_method_interaction VARCHAR(200),
    apollo_seniority_industry VARCHAR(300),
    apollo_dept_function VARCHAR(300),
    apollo_geo_industry VARCHAR(300),
    
    -- === APOLLO ENRICHMENT FEATURES ===
    has_apollo_enrichment INTEGER,
    apollo_api_success INTEGER,
    
    -- Company size features
    company_size_category VARCHAR(20) CHECK (company_size_category IN ('Startup', 'Small', 'Medium', 'Large', 'Enterprise', 'nan')),
    company_size_log FLOAT,
    
    -- Company age features
    company_age_years INTEGER,
    company_age_category VARCHAR(20) CHECK (company_age_category IN ('New', 'Growing', 'Established', 'Mature', 'nan')),
    
    -- Apollo quality and cost
    apollo_enrichment_cost INTEGER,
    days_since_apollo_enrichment INTEGER,
    apollo_data_freshness VARCHAR(20) CHECK (apollo_data_freshness IN ('Fresh', 'Recent', 'Stale', 'Old', 'nan')),
    apollo_data_completeness_pct FLOAT,
    
    -- High-value prospect indicators
    is_high_value_title INTEGER,
    is_high_value_seniority INTEGER,
    is_tech_department INTEGER,
    
    -- === QUALITY SCORES ===
    data_quality_score FLOAT,
    feature_completeness_score FLOAT
);

-- Create indexes for performance
CREATE INDEX idx_silver_ml_features_engagement ON leads.silver_ml_features(engagement_level);
CREATE INDEX idx_silver_ml_features_campaign ON leads.silver_ml_features(campaign);
CREATE INDEX idx_silver_ml_features_processed ON leads.silver_ml_features(processed_timestamp);
CREATE INDEX idx_silver_ml_features_apollo ON leads.silver_ml_features(has_apollo_enrichment);
CREATE INDEX idx_silver_ml_features_quality ON leads.silver_ml_features(data_quality_score);
CREATE INDEX idx_silver_ml_features_company_size ON leads.silver_ml_features(company_size_category);
CREATE INDEX idx_silver_ml_features_high_value ON leads.silver_ml_features(is_high_value_title, is_high_value_seniority);

-- Create view for ML training
CREATE OR REPLACE VIEW leads.v_ml_training_features AS
SELECT 
    id,
    email,
    campaign,
    engagement_level,
    
    -- Timestamp features
    created_day_of_week,
    created_month,
    created_hour,
    days_since_creation,
    weeks_since_creation,
    
    -- Time difference features
    days_between_updated_and_created,
    days_between_last_contact_and_created,
    days_between_enriched_and_created,
    
    -- Presence indicators
    has_updated,
    has_last_contact,
    has_enriched,
    
    -- Text features
    text_length,
    text_word_count,
    text_avg_word_length,
    text_uppercase_ratio,
    
    -- Apollo features
    has_apollo_enrichment,
    apollo_api_success,
    company_size_log,
    company_age_years,
    apollo_data_completeness_pct,
    is_high_value_title,
    is_high_value_seniority,
    is_tech_department,
    
    -- Quality scores
    data_quality_score,
    feature_completeness_score,
    
    -- Categorical features (for encoding)
    company_size_category,
    company_age_category,
    apollo_data_freshness,
    org_title_interaction,
    apollo_seniority_industry,
    apollo_geo_industry
    
FROM leads.silver_ml_features
WHERE engagement_level IS NOT NULL;

-- Grant permissions
GRANT SELECT ON leads.silver_ml_features TO PUBLIC;
GRANT SELECT ON leads.v_ml_training_features TO PUBLIC;
