-- Add timestamp columns to silver layer for gold layer maturity filtering
-- These columns are needed for the gold transformation pipeline

-- Add key timestamp columns if they don't exist
DO $$ 
BEGIN
    -- Add timestamp_last_contact if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'leads' 
        AND table_name = 'silver_ml_features' 
        AND column_name = 'timestamp_last_contact'
    ) THEN
        ALTER TABLE leads.silver_ml_features 
        ADD COLUMN timestamp_last_contact TIMESTAMPTZ;
        
        COMMENT ON COLUMN leads.silver_ml_features.timestamp_last_contact 
        IS 'Last contact timestamp - used for gold layer maturity filtering (30+ days)';
    END IF;
    
    -- Add timestamp_created if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'leads' 
        AND table_name = 'silver_ml_features' 
        AND column_name = 'timestamp_created'
    ) THEN
        ALTER TABLE leads.silver_ml_features 
        ADD COLUMN timestamp_created TIMESTAMPTZ;
        
        COMMENT ON COLUMN leads.silver_ml_features.timestamp_created 
        IS 'Record creation timestamp from bronze layer';
    END IF;
    
    -- Add timestamp_updated if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_schema = 'leads' 
        AND table_name = 'silver_ml_features' 
        AND column_name = 'timestamp_updated'
    ) THEN
        ALTER TABLE leads.silver_ml_features 
        ADD COLUMN timestamp_updated TIMESTAMPTZ;
        
        COMMENT ON COLUMN leads.silver_ml_features.timestamp_updated 
        IS 'Record update timestamp from bronze layer';
    END IF;
END $$;

-- Create index on timestamp_last_contact for gold layer filtering performance
CREATE INDEX IF NOT EXISTS idx_silver_ml_features_last_contact 
ON leads.silver_ml_features(timestamp_last_contact) 
WHERE timestamp_last_contact IS NOT NULL;

-- Create index on timestamp_created for general timestamp queries
CREATE INDEX IF NOT EXISTS idx_silver_ml_features_created 
ON leads.silver_ml_features(timestamp_created) 
WHERE timestamp_created IS NOT NULL;

-- Update any existing records that might have these timestamps from bronze layer
-- (This will only work if the bronze data has these columns)
-- Note: This is a one-time backfill - future records will get these through the pipeline

UPDATE leads.silver_ml_features 
SET timestamp_last_contact = (
    SELECT ec.timestamp_last_contact 
    FROM leads.enriched_contacts ec 
    WHERE ec.id = leads.silver_ml_features.id 
    AND ec.timestamp_last_contact IS NOT NULL
    LIMIT 1
)
WHERE timestamp_last_contact IS NULL;

UPDATE leads.silver_ml_features 
SET timestamp_created = (
    SELECT ec.timestamp_created 
    FROM leads.enriched_contacts ec 
    WHERE ec.id = leads.silver_ml_features.id 
    AND ec.timestamp_created IS NOT NULL
    LIMIT 1
)
WHERE timestamp_created IS NULL;

UPDATE leads.silver_ml_features 
SET timestamp_updated = (
    SELECT ec.timestamp_updated 
    FROM leads.enriched_contacts ec 
    WHERE ec.id = leads.silver_ml_features.id 
    AND ec.timestamp_updated IS NOT NULL
    LIMIT 1
)
WHERE timestamp_updated IS NULL;

-- Report results
DO $$
DECLARE
    total_records INTEGER;
    records_with_last_contact INTEGER;
    records_with_created INTEGER;
    records_with_updated INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_records FROM leads.silver_ml_features;
    
    SELECT COUNT(*) INTO records_with_last_contact 
    FROM leads.silver_ml_features 
    WHERE timestamp_last_contact IS NOT NULL;
    
    SELECT COUNT(*) INTO records_with_created 
    FROM leads.silver_ml_features 
    WHERE timestamp_created IS NOT NULL;
    
    SELECT COUNT(*) INTO records_with_updated 
    FROM leads.silver_ml_features 
    WHERE timestamp_updated IS NOT NULL;
    
    RAISE NOTICE 'Silver layer timestamp column update complete:';
    RAISE NOTICE 'Total records: %', total_records;
    RAISE NOTICE 'Records with timestamp_last_contact: % (%.1f%%)', 
        records_with_last_contact, 
        CASE WHEN total_records > 0 THEN (records_with_last_contact::FLOAT / total_records * 100) ELSE 0 END;
    RAISE NOTICE 'Records with timestamp_created: % (%.1f%%)', 
        records_with_created,
        CASE WHEN total_records > 0 THEN (records_with_created::FLOAT / total_records * 100) ELSE 0 END;
    RAISE NOTICE 'Records with timestamp_updated: % (%.1f%%)', 
        records_with_updated,
        CASE WHEN total_records > 0 THEN (records_with_updated::FLOAT / total_records * 100) ELSE 0 END;
END $$;
