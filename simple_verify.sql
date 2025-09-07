-- Simple verification queries for silver_ml_features table

-- 1. Basic record count
SELECT COUNT(*) as total_records FROM leads.silver_ml_features;

-- 2. Check if table exists and has correct structure
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_schema = 'leads' 
AND table_name = 'silver_ml_features'
ORDER BY ordinal_position
LIMIT 10;

-- 3. If records exist, show target distribution
SELECT 
    engagement_level,
    COUNT(*) as count
FROM leads.silver_ml_features
GROUP BY engagement_level
ORDER BY engagement_level;

-- 4. Sample of latest records (if any)
SELECT 
    id,
    email,
    engagement_level,
    processed_timestamp
FROM leads.silver_ml_features
ORDER BY processed_timestamp DESC
LIMIT 5;
