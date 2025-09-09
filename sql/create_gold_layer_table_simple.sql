-- Gold Layer Table for ML Training Data (Simple Version)
-- Contains mature leads (30+ days from last contact) ready for model training

-- Create gold layer table for mature ML training data
CREATE TABLE IF NOT EXISTS leads.gold_ml_features (
    -- Copy all silver layer columns plus gold-specific metadata
    
    -- Primary identifiers (from silver)
    id TEXT PRIMARY KEY,
    email VARCHAR(255),
    campaign VARCHAR(255),
    
    -- Target variable (from silver)
    engagement_level INTEGER CHECK (engagement_level IN (0, 1, 2)),
    
    -- LEAKY VARIABLES (retained for analysis/debugging in gold layer)
    email_open_count INTEGER,
    email_click_count INTEGER,
    email_reply_count INTEGER,
    
    -- Core timestamp features (from silver)
    created_day_of_week INTEGER,
    created_month INTEGER,
    created_hour INTEGER,
    created_quarter INTEGER,
    created_is_weekend INTEGER,
    created_is_business_hours INTEGER,
    created_is_morning INTEGER,
    created_season INTEGER,
    days_since_creation INTEGER,
    weeks_since_creation INTEGER,
    
    -- Time differences (from silver)
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
    
    -- Time differences (absolute values, from silver)
    days_between_updated_and_created_abs INTEGER,
    days_between_last_contact_and_created_abs INTEGER,
    days_between_last_touch_and_created_abs INTEGER,
    days_between_last_open_and_created_abs INTEGER,
    days_between_last_click_and_created_abs INTEGER,
    days_between_last_reply_and_created_abs INTEGER,
    days_between_last_interest_change_and_created_abs INTEGER,
    days_between_retrieval_and_created_abs INTEGER,
    days_between_inserted_and_created_abs INTEGER,
    days_between_enriched_and_created_abs INTEGER,
    
    -- Has indicators for timestamps (from silver)
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
    
    -- Text features (from silver)
    combined_text TEXT,
    combined_text_length INTEGER,
    combined_text_word_count INTEGER,
    has_numbers_in_text INTEGER,
    has_email_in_text INTEGER,
    has_url_in_text INTEGER,
    text_quality_score DECIMAL(3,2),
    
    -- Interaction features (from silver)
    industry_seniority_interaction TEXT,
    geo_industry_interaction TEXT,
    title_industry_interaction TEXT,
    
    -- JSONB features (from silver)
    has_employment_history INTEGER,
    has_organization_data INTEGER,
    has_account_data INTEGER,
    has_api_response_raw INTEGER,
    enrichment_completeness INTEGER,
    enrichment_completeness_pct DECIMAL(5,4),
    
    -- ADVANCED JSONB features (from silver - NEW nested features)
    -- Employment history features
    job_count INTEGER,
    current_job_tenure_years INTEGER,
    avg_job_tenure_years DECIMAL(5,2),
    total_career_years INTEGER,
    job_progression_score DECIMAL(5,4),
    has_management_experience INTEGER,
    industry_stability DECIMAL(5,4),
    education_level INTEGER,
    
    -- Organization features
    company_size_bucket VARCHAR(20),
    has_public_trading INTEGER,
    alexa_rank_tier VARCHAR(20),
    has_social_presence INTEGER,
    industry_keywords_count INTEGER,
    headcount_growth_6m DECIMAL(8,4),
    headcount_growth_12m DECIMAL(8,4),
    headcount_growth_24m DECIMAL(8,4),
    is_tech_company INTEGER,
    is_enterprise INTEGER,
    market_cap_tier VARCHAR(20),
    
    -- Account features
    account_age_days INTEGER,
    has_crm_integration INTEGER,
    label_count INTEGER,
    account_stage_maturity INTEGER,
    phone_status_quality INTEGER,
    existence_level_score INTEGER,
    
    -- API response features
    email_confidence DECIMAL(5,4),
    intent_strength_score INTEGER,
    functions_count INTEGER,
    departments_count INTEGER,
    is_decision_maker INTEGER,
    seniority_level INTEGER,
    has_personal_emails INTEGER,
    revealed_for_team INTEGER,
    
    -- Categorical features (from silver)
    title TEXT,
    seniority TEXT,
    organization_industry TEXT,
    country TEXT,
    city TEXT,
    enrichment_status TEXT,
    upload_method TEXT,
    api_status TEXT,
    state TEXT,
    
    -- KEY TIMESTAMPS (needed for gold layer maturity filtering)
    timestamp_last_contact TIMESTAMPTZ,
    timestamp_created TIMESTAMPTZ,
    timestamp_updated TIMESTAMPTZ,
    
    -- GOLD LAYER SPECIFIC METADATA
    lead_maturity_days INTEGER NOT NULL CHECK (lead_maturity_days >= 30),
    is_mature_lead INTEGER NOT NULL DEFAULT 1 CHECK (is_mature_lead = 1),
    gold_processed_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- Silver layer metadata (inherited)
    processed_timestamp TIMESTAMPTZ,
    pipeline_version VARCHAR(50),
    
    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance (in leads schema)
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_engagement_level ON leads.gold_ml_features(engagement_level);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_maturity_days ON leads.gold_ml_features(lead_maturity_days);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_gold_processed ON leads.gold_ml_features(gold_processed_timestamp);
CREATE INDEX IF NOT EXISTS idx_gold_ml_features_timestamp_last_contact ON leads.gold_ml_features(timestamp_last_contact);

-- Comments for documentation
COMMENT ON TABLE leads.gold_ml_features IS 'Gold layer ML training data - mature leads (30+ days from last contact) with all engineered features';
COMMENT ON COLUMN leads.gold_ml_features.lead_maturity_days IS 'Days since last contact - must be >= 30 for gold layer';
COMMENT ON COLUMN leads.gold_ml_features.is_mature_lead IS 'Always 1 in gold layer - indicates lead maturity';
COMMENT ON COLUMN leads.gold_ml_features.timestamp_last_contact IS 'Last contact timestamp - used for maturity filtering';

