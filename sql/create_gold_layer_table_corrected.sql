-- Gold Layer Table for ML Training Data (CORRECTED - matches actual silver columns)
-- Contains mature leads (30+ days from last contact) ready for model training

-- Drop existing table if it exists
DROP TABLE IF EXISTS leads.gold_ml_features CASCADE;

-- Create gold layer table for mature ML training data
CREATE TABLE leads.gold_ml_features (
    -- Copy ALL actual silver layer columns as they exist
    
    -- Primary identifiers (from actual silver)
    id TEXT PRIMARY KEY,
    email VARCHAR(255),
    campaign VARCHAR(255),
    
    -- Target variable (from actual silver)
    engagement_level INTEGER CHECK (engagement_level IN (0, 1, 2)),
    
    -- LEAKY VARIABLES (retained for analysis/debugging in gold layer)
    email_open_count INTEGER,
    email_click_count INTEGER,
    email_reply_count INTEGER,
    
    -- Core timestamp features (from actual silver)
    created_day_of_week INTEGER,
    created_month INTEGER,
    created_hour INTEGER,
    days_since_creation INTEGER,
    weeks_since_creation DECIMAL(5,2),
    
    -- Time differences (from actual silver - exact column names)
    days_between_updated_and_created DECIMAL(10,2),
    days_between_last_contact_and_created DECIMAL(10,2),
    days_between_last_touch_and_created DECIMAL(10,2),
    days_between_last_open_and_created DECIMAL(10,2),
    days_between_last_click_and_created DECIMAL(10,2),
    days_between_last_reply_and_created DECIMAL(10,2),
    days_between_last_interest_change_and_created DECIMAL(10,2),
    days_between_retrieval_and_created DECIMAL(10,2),
    days_between_inserted_and_created DECIMAL(10,2),
    days_between_enriched_and_created DECIMAL(10,2),
    
    -- Has indicators for timestamps (from actual silver)
    has_updated DECIMAL(3,2),
    has_last_contact DECIMAL(3,2),
    has_last_touch DECIMAL(3,2),
    has_last_open DECIMAL(3,2),
    has_last_click DECIMAL(3,2),
    has_last_reply DECIMAL(3,2),
    has_last_interest_change DECIMAL(3,2),
    has_retrieval DECIMAL(3,2),
    has_inserted DECIMAL(3,2),
    has_enriched DECIMAL(3,2),
    
    -- Text features (from actual silver - exact column names)
    combined_text TEXT,
    text_length DECIMAL(10,2),
    text_word_count DECIMAL(10,2),
    text_char_count DECIMAL(10,2),
    text_avg_word_length DECIMAL(10,4),
    text_uppercase_ratio DECIMAL(10,4),
    text_digit_ratio DECIMAL(10,4),
    text_punctuation_ratio DECIMAL(10,4),
    
    -- Interaction features (from actual silver)
    org_title_interaction TEXT,
    status_method_interaction TEXT,
    apollo_seniority_industry TEXT,
    apollo_dept_function TEXT,
    apollo_geo_industry TEXT,
    
    -- Apollo features (from actual silver)
    has_apollo_enrichment INTEGER,
    apollo_api_success DECIMAL(3,2),
    company_size_category VARCHAR(50),
    company_size_log DECIMAL(10,6),
    company_age_years DECIMAL(10,2),
    company_age_category VARCHAR(50),
    apollo_enrichment_cost DECIMAL(10,2),
    days_since_apollo_enrichment DECIMAL(10,2),
    apollo_data_freshness VARCHAR(50),
    apollo_data_completeness_pct DECIMAL(5,4),
    is_high_value_title DECIMAL(3,2),
    is_high_value_seniority DECIMAL(3,2),
    is_tech_department DECIMAL(3,2),
    
    -- Quality scores (from actual silver)
    data_quality_score DECIMAL(5,4),
    feature_completeness_score DECIMAL(5,4),
    
    -- KEY TIMESTAMPS (from actual silver - needed for gold layer maturity filtering)
    timestamp_last_contact TIMESTAMPTZ,
    timestamp_created TIMESTAMPTZ,
    timestamp_updated TIMESTAMPTZ,
    
    -- Silver layer metadata (inherited)
    processed_timestamp TIMESTAMPTZ,
    
    -- GOLD LAYER SPECIFIC METADATA
    lead_maturity_days INTEGER NOT NULL CHECK (lead_maturity_days >= 30),
    is_mature_lead INTEGER NOT NULL DEFAULT 1 CHECK (is_mature_lead = 1),
    gold_processed_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance (in leads schema)
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_engagement_level ON leads.gold_ml_features(engagement_level);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_maturity_days ON leads.gold_ml_features(lead_maturity_days);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_gold_processed ON leads.gold_ml_features(gold_processed_timestamp);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_timestamp_last_contact ON leads.gold_ml_features(timestamp_last_contact);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_data_quality ON leads.gold_ml_features(data_quality_score);

-- Comments for documentation
COMMENT ON TABLE leads.gold_ml_features IS 'Gold layer ML training data - mature leads (30+ days from last contact) with all actual silver features';
COMMENT ON COLUMN leads.gold_ml_features.lead_maturity_days IS 'Days since last contact - must be >= 30 for gold layer';
COMMENT ON COLUMN leads.gold_ml_features.is_mature_lead IS 'Always 1 in gold layer - indicates lead maturity';
COMMENT ON COLUMN leads.gold_ml_features.timestamp_last_contact IS 'Last contact timestamp - used for maturity filtering';
COMMENT ON COLUMN leads.gold_ml_features.data_quality_score IS 'Overall data quality score from silver layer';
