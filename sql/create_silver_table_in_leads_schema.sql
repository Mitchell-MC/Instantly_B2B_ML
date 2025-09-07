-- Silver Layer Table in Existing LEADS Schema
-- Creates the silver_ml_features table in your existing leads schema

-- Create silver layer table for ML features in leads schema
CREATE TABLE IF NOT EXISTS leads.silver_ml_features (
    -- Primary identifiers
    id TEXT PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    campaign VARCHAR(255),
    
    -- Target variable
    engagement_level INTEGER NOT NULL CHECK (engagement_level IN (0, 1, 2)),
    
    -- Timestamp features
    created_day_of_week INTEGER CHECK (created_day_of_week BETWEEN 0 AND 6),
    created_month INTEGER CHECK (created_month BETWEEN 1 AND 12),
    created_hour INTEGER CHECK (created_hour BETWEEN 0 AND 23),
    created_quarter INTEGER CHECK (created_quarter BETWEEN 1 AND 4),
    created_is_weekend INTEGER CHECK (created_is_weekend IN (0, 1)),
    created_is_business_hours INTEGER CHECK (created_is_business_hours IN (0, 1)),
    days_since_creation INTEGER,
    weeks_since_creation INTEGER,
    
    -- Dynamic timestamp features
    days_between_updated_and_created INTEGER,
    days_between_last_contact_and_created INTEGER,
    has_updated INTEGER CHECK (has_updated IN (0, 1)),
    has_last_contact INTEGER CHECK (has_last_contact IN (0, 1)),
    
    -- Text features
    text_length INTEGER CHECK (text_length >= 0),
    text_word_count INTEGER CHECK (text_word_count >= 0),
    has_numbers_in_text INTEGER CHECK (has_numbers_in_text IN (0, 1)),
    text_quality_score DECIMAL(3,2) CHECK (text_quality_score BETWEEN 0 AND 3),
    
    -- Categorical interaction features
    org_title_interaction TEXT,
    status_method_interaction TEXT,
    apollo_seniority_industry TEXT,
    apollo_dept_function TEXT,
    apollo_geo_industry TEXT,
    
    -- Grouped categorical features
    title_grouped TEXT,
    organization_grouped TEXT,
    status_grouped TEXT,
    upload_method_grouped TEXT,
    a_seniority_grouped TEXT,
    a_city_grouped TEXT,
    a_country_grouped TEXT,
    
    -- Apollo enrichment features
    has_apollo_enrichment INTEGER CHECK (has_apollo_enrichment IN (0, 1)),
    apollo_api_success INTEGER CHECK (apollo_api_success IN (0, 1)),
    company_size_category TEXT CHECK (company_size_category IN ('Startup', 'Small', 'Medium', 'Large', 'Enterprise', 'nan')),
    company_size_log DECIMAL(10,4),
    company_age_years INTEGER,
    company_age_category TEXT CHECK (company_age_category IN ('New', 'Growing', 'Established', 'Mature', 'nan')),
    apollo_enrichment_cost DECIMAL(8,2),
    days_since_apollo_enrichment INTEGER,
    apollo_data_freshness TEXT CHECK (apollo_data_freshness IN ('Fresh', 'Recent', 'Stale', 'Old', 'nan')),
    apollo_data_completeness INTEGER CHECK (apollo_data_completeness >= 0),
    apollo_data_completeness_pct DECIMAL(5,4) CHECK (apollo_data_completeness_pct BETWEEN 0 AND 1),
    is_high_value_title INTEGER CHECK (is_high_value_title IN (0, 1)),
    is_high_value_seniority INTEGER CHECK (is_high_value_seniority IN (0, 1)),
    is_tech_department INTEGER CHECK (is_tech_department IN (0, 1)),
    
    -- Engagement features
    campaign_size INTEGER CHECK (campaign_size > 0),
    campaign_duration_days INTEGER CHECK (campaign_duration_days >= 0),
    created_hour_category TEXT CHECK (created_hour_category IN ('Night', 'Morning', 'Afternoon', 'Evening')),
    
    -- Quality metrics
    data_quality_score DECIMAL(5,4) CHECK (data_quality_score BETWEEN 0 AND 1),
    
    -- Pipeline metadata
    processed_timestamp TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    pipeline_version VARCHAR(50) NOT NULL DEFAULT '1.0',
    feature_engineering_date DATE NOT NULL DEFAULT CURRENT_DATE,
    
    -- Audit fields
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_email ON leads.silver_ml_features(email);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_campaign ON leads.silver_ml_features(campaign);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_engagement_level ON leads.silver_ml_features(engagement_level);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_processed_timestamp ON leads.silver_ml_features(processed_timestamp);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_data_quality ON leads.silver_ml_features(data_quality_score);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_composite ON leads.silver_ml_features(email, campaign, processed_timestamp);

-- Apollo-specific indexes for B2B analysis
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_apollo_enrichment ON leads.silver_ml_features(has_apollo_enrichment);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_company_size ON leads.silver_ml_features(company_size_category);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_company_age ON leads.silver_ml_features(company_age_category);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_high_value ON leads.silver_ml_features(is_high_value_title, is_high_value_seniority);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_apollo_quality ON leads.silver_ml_features(apollo_data_completeness_pct);
CREATE INDEX IF NOT EXISTS idx_leads_silver_ml_features_apollo_freshness ON leads.silver_ml_features(apollo_data_freshness);

-- Create trigger for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_leads_silver_ml_features_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_leads_silver_ml_features_timestamp
    BEFORE UPDATE ON leads.silver_ml_features
    FOR EACH ROW EXECUTE FUNCTION update_leads_silver_ml_features_timestamp();

-- Create view for ML training data (excludes identifiers)
CREATE OR REPLACE VIEW leads.v_ml_training_features AS
SELECT 
    -- Target variable
    engagement_level,
    
    -- Timestamp features
    created_day_of_week,
    created_month,
    created_hour,
    created_quarter,
    created_is_weekend,
    created_is_business_hours,
    days_since_creation,
    weeks_since_creation,
    days_between_updated_and_created,
    days_between_last_contact_and_created,
    has_updated,
    has_last_contact,
    
    -- Text features
    text_length,
    text_word_count,
    has_numbers_in_text,
    text_quality_score,
    
    -- Categorical features (for encoding)
    org_title_interaction,
    status_method_interaction,
    apollo_seniority_industry,
    apollo_dept_function,
    apollo_geo_industry,
    title_grouped,
    organization_grouped,
    status_grouped,
    upload_method_grouped,
    a_seniority_grouped,
    a_city_grouped,
    a_country_grouped,
    
    -- Apollo enrichment features
    has_apollo_enrichment,
    apollo_api_success,
    company_size_category,
    company_size_log,
    company_age_years,
    company_age_category,
    apollo_enrichment_cost,
    days_since_apollo_enrichment,
    apollo_data_freshness,
    apollo_data_completeness,
    apollo_data_completeness_pct,
    is_high_value_title,
    is_high_value_seniority,
    is_tech_department,
    
    -- Engagement features
    campaign_size,
    campaign_duration_days,
    created_hour_category,
    
    -- Quality metrics
    data_quality_score,
    
    -- Metadata for filtering
    processed_timestamp,
    pipeline_version
FROM leads.silver_ml_features
WHERE data_quality_score >= 0.5  -- Filter low-quality records
  AND engagement_level IS NOT NULL;

-- Comments for documentation
COMMENT ON TABLE leads.silver_ml_features IS 'Silver layer ML-ready features derived from bronze layer data - stored in existing leads schema';
COMMENT ON COLUMN leads.silver_ml_features.engagement_level IS 'Target variable: 0=no engagement, 1=opened, 2=clicked/replied';
COMMENT ON COLUMN leads.silver_ml_features.data_quality_score IS 'Overall data quality score (0-1, higher is better)';
COMMENT ON VIEW leads.v_ml_training_features IS 'Training-ready features excluding PII and identifiers';

-- Show success message
SELECT 'Silver layer table created successfully in leads schema!' as status;
