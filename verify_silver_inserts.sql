-- Comprehensive verification queries for silver_ml_features table
-- Run these to verify the ETL pipeline inserts

-- 1. Basic record count and freshness
SELECT 
    COUNT(*) as total_records,
    MIN(processed_timestamp) as first_insert,
    MAX(processed_timestamp) as last_insert,
    COUNT(DISTINCT DATE(processed_timestamp)) as insert_days
FROM leads.silver_ml_features;

-- 2. Target variable distribution
SELECT 
    engagement_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM leads.silver_ml_features
GROUP BY engagement_level
ORDER BY engagement_level;

-- 3. Feature completeness check
SELECT 
    -- Core features
    COUNT(CASE WHEN text_length IS NOT NULL THEN 1 END) as has_text_features,
    COUNT(CASE WHEN created_day_of_week IS NOT NULL THEN 1 END) as has_timestamp_features,
    COUNT(CASE WHEN has_apollo_enrichment IS NOT NULL THEN 1 END) as has_apollo_features,
    
    -- Apollo enrichment stats
    SUM(has_apollo_enrichment) as apollo_enriched_count,
    ROUND(AVG(has_apollo_enrichment) * 100, 2) as apollo_enrichment_rate,
    
    -- Data quality scores
    ROUND(AVG(data_quality_score), 3) as avg_data_quality,
    ROUND(AVG(feature_completeness_score), 3) as avg_feature_completeness,
    
    -- Text analysis
    ROUND(AVG(text_length), 0) as avg_text_length,
    ROUND(AVG(text_word_count), 1) as avg_word_count
FROM leads.silver_ml_features;

-- 4. Apollo enrichment analysis
SELECT 
    company_size_category,
    COUNT(*) as count,
    ROUND(AVG(is_high_value_title), 2) as high_value_title_rate,
    ROUND(AVG(is_high_value_seniority), 2) as high_value_seniority_rate,
    ROUND(AVG(is_tech_department), 2) as tech_department_rate
FROM leads.silver_ml_features
WHERE has_apollo_enrichment = 1
GROUP BY company_size_category
ORDER BY count DESC;

-- 5. Recent processing activity (last 24 hours)
SELECT 
    DATE_TRUNC('hour', processed_timestamp) as processing_hour,
    COUNT(*) as records_processed
FROM leads.silver_ml_features
WHERE processed_timestamp >= NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', processed_timestamp)
ORDER BY processing_hour DESC;

-- 6. Feature value distributions (check for nulls/issues)
SELECT 
    'created_day_of_week' as feature,
    COUNT(*) as total,
    COUNT(created_day_of_week) as non_null,
    MIN(created_day_of_week) as min_val,
    MAX(created_day_of_week) as max_val
FROM leads.silver_ml_features
UNION ALL
SELECT 
    'days_since_creation',
    COUNT(*),
    COUNT(days_since_creation),
    MIN(days_since_creation),
    MAX(days_since_creation)
FROM leads.silver_ml_features
UNION ALL
SELECT 
    'text_length',
    COUNT(*),
    COUNT(text_length),
    MIN(text_length),
    MAX(text_length)
FROM leads.silver_ml_features;

-- 7. Sample records with all key features
SELECT 
    id,
    email,
    engagement_level,
    has_apollo_enrichment,
    company_size_category,
    text_length,
    created_day_of_week,
    days_since_creation,
    is_high_value_title,
    data_quality_score,
    processed_timestamp
FROM leads.silver_ml_features
ORDER BY processed_timestamp DESC
LIMIT 5;

-- 8. Interaction features verification
SELECT 
    'org_title_interaction' as interaction_type,
    COUNT(DISTINCT org_title_interaction) as unique_values,
    COUNT(CASE WHEN org_title_interaction IS NOT NULL THEN 1 END) as non_null_count
FROM leads.silver_ml_features
UNION ALL
SELECT 
    'apollo_seniority_industry',
    COUNT(DISTINCT apollo_seniority_industry),
    COUNT(CASE WHEN apollo_seniority_industry IS NOT NULL THEN 1 END)
FROM leads.silver_ml_features
UNION ALL
SELECT 
    'apollo_geo_industry',
    COUNT(DISTINCT apollo_geo_industry),
    COUNT(CASE WHEN apollo_geo_industry IS NOT NULL THEN 1 END)
FROM leads.silver_ml_features;

-- 9. Training view verification
SELECT 
    COUNT(*) as training_ready_records,
    COUNT(CASE WHEN engagement_level IS NOT NULL THEN 1 END) as with_target,
    MIN(processed_timestamp) as oldest_training_data,
    MAX(processed_timestamp) as newest_training_data
FROM leads.v_ml_training_features;

-- 10. Check for duplicates
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT id) as unique_ids,
    COUNT(*) - COUNT(DISTINCT id) as duplicate_count
FROM leads.silver_ml_features;

