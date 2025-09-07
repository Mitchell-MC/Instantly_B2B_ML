-- Create the ml_lead_scoring schema first
-- This must be run before creating the silver layer tables

-- Create the schema
CREATE SCHEMA IF NOT EXISTS ml_lead_scoring;

-- Grant permissions (adjust as needed for your environment)
GRANT USAGE ON SCHEMA ml_lead_scoring TO postgres;
GRANT CREATE ON SCHEMA ml_lead_scoring TO postgres;

-- Verify schema creation
SELECT schema_name 
FROM information_schema.schemata 
WHERE schema_name = 'ml_lead_scoring';

-- Show confirmation
\echo 'ml_lead_scoring schema created successfully!'

