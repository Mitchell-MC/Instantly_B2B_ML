-- ML Lead Scoring Database Schema
-- Three-layer data architecture: Bronze (raw), Silver (processed), Gold (mature)

-- Create ML schema
CREATE SCHEMA IF NOT EXISTS ml_lead_scoring;

-- Bronze Layer: Raw data from APIs
CREATE TABLE ml_lead_scoring.bronze_instantly_leads (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(255) UNIQUE NOT NULL,
    campaign_id VARCHAR(255),
    email VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    company VARCHAR(255),
    title VARCHAR(255),
    industry VARCHAR(255),
    lead_status VARCHAR(100),
    open_rate DECIMAL(5,4),
    click_rate DECIMAL(5,4),
    reply_rate DECIMAL(5,4),
    bounce_rate DECIMAL(5,4),
    last_activity_date TIMESTAMP,
    created_date TIMESTAMP,
    raw_data JSONB,
    ingestion_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ml_lead_scoring.bronze_apollo_enrichment (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(255) REFERENCES ml_lead_scoring.bronze_instantly_leads(lead_id),
    company_size VARCHAR(100),
    company_revenue VARCHAR(100),
    company_industry VARCHAR(255),
    company_location VARCHAR(255),
    employee_count INTEGER,
    technologies JSONB,
    social_media_profiles JSONB,
    company_description TEXT,
    funding_info JSONB,
    raw_apollo_data JSONB,
    enrichment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    apollo_credits_used INTEGER DEFAULT 1
);

-- Silver Layer: Preprocessed ML-ready features
CREATE TABLE ml_lead_scoring.silver_ml_features (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Engagement Features
    avg_open_rate DECIMAL(5,4),
    avg_click_rate DECIMAL(5,4),
    avg_reply_rate DECIMAL(5,4),
    engagement_trend VARCHAR(50), -- 'increasing', 'decreasing', 'stable'
    days_since_last_activity INTEGER,
    total_emails_sent INTEGER,
    
    -- Company Features (from Apollo)
    company_size_category VARCHAR(50), -- 'startup', 'small', 'medium', 'large', 'enterprise'
    revenue_category VARCHAR(50),
    industry_category VARCHAR(100),
    employee_count_bucket VARCHAR(50),
    has_technology_stack BOOLEAN,
    social_media_presence_score DECIMAL(3,2),
    
    -- Derived Features
    lead_age_days INTEGER,
    response_velocity DECIMAL(5,4), -- responses per day
    email_frequency DECIMAL(5,4), -- emails per day
    engagement_score DECIMAL(5,4), -- composite engagement metric
    
    -- Target Variables
    is_qualified_lead BOOLEAN,
    lead_quality_score DECIMAL(5,4), -- 0-1 scale
    
    feature_vector JSONB, -- Serialized feature vector for ML
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gold Layer: Mature data for model training (1+ month old)
CREATE TABLE ml_lead_scoring.gold_training_data (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(255),
    feature_vector JSONB,
    target_label INTEGER, -- 0: not qualified, 1: qualified, 2: high-value
    lead_outcome VARCHAR(100), -- 'converted', 'disqualified', 'ongoing', 'cold'
    final_engagement_score DECIMAL(5,4),
    conversion_timestamp TIMESTAMP,
    training_eligible BOOLEAN DEFAULT FALSE,
    data_maturity_days INTEGER,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model tracking and performance
CREATE TABLE ml_lead_scoring.model_versions (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255),
    version VARCHAR(50),
    model_path VARCHAR(500),
    training_data_size INTEGER,
    features_used JSONB,
    performance_metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ml_lead_scoring.lead_predictions (
    id SERIAL PRIMARY KEY,
    lead_id VARCHAR(255),
    model_version_id INTEGER REFERENCES ml_lead_scoring.model_versions(id),
    predicted_score DECIMAL(5,4),
    confidence_level DECIMAL(5,4),
    feature_importance JSONB,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking (for Apollo credit management)
CREATE TABLE ml_lead_scoring.api_usage_log (
    id SERIAL PRIMARY KEY,
    api_source VARCHAR(100), -- 'apollo', 'instantly'
    endpoint VARCHAR(255),
    credits_used INTEGER,
    request_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    monthly_total INTEGER,
    remaining_credits INTEGER
);

-- Indexes for performance
CREATE INDEX idx_bronze_instantly_lead_id ON ml_lead_scoring.bronze_instantly_leads(lead_id);
CREATE INDEX idx_bronze_instantly_updated ON ml_lead_scoring.bronze_instantly_leads(updated_timestamp);
CREATE INDEX idx_silver_lead_id ON ml_lead_scoring.silver_ml_features(lead_id);
CREATE INDEX idx_silver_updated ON ml_lead_scoring.silver_ml_features(updated_timestamp);
CREATE INDEX idx_gold_training_eligible ON ml_lead_scoring.gold_training_data(training_eligible);
CREATE INDEX idx_gold_maturity ON ml_lead_scoring.gold_training_data(data_maturity_days);
CREATE INDEX idx_predictions_lead_id ON ml_lead_scoring.lead_predictions(lead_id);
CREATE INDEX idx_api_usage_timestamp ON ml_lead_scoring.api_usage_log(request_timestamp);

-- Triggers for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_timestamp = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_bronze_instantly_timestamp
    BEFORE UPDATE ON ml_lead_scoring.bronze_instantly_leads
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

CREATE TRIGGER update_silver_timestamp
    BEFORE UPDATE ON ml_lead_scoring.silver_ml_features
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();
