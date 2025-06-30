-- Main table for campaigns, removing the deeply nested 'sequences' column.
CREATE TABLE IF NOT EXISTS leads.instantly_campaigns (
    id TEXT PRIMARY KEY,
    name TEXT,
    status INTEGER,
    organization TEXT,
    timestamp_created TIMESTAMPTZ,
    timestamp_updated TIMESTAMPTZ,
    daily_limit INTEGER,
    stop_on_reply BOOLEAN,
    link_tracking BOOLEAN,
    open_tracking BOOLEAN,
    stop_on_auto_reply BOOLEAN,
    prioritize_new_leads BOOLEAN,
    stop_for_company BOOLEAN,
    insert_unsubscribe_header BOOLEAN,
    email_gap INTEGER,
    -- Other top-level fields from your schema can be added here
    campaign_schedule JSONB, -- Storing schedule as JSONB is still efficient
    email_list JSONB
);

-- Table for the individual steps in a campaign sequence
CREATE TABLE IF NOT EXISTS leads.instantly_campaign_steps (
    step_id SERIAL PRIMARY KEY, -- A unique ID for each step
    campaign_id TEXT REFERENCES leads.instantly_campaigns(id),
    step_index INTEGER, -- The order of the step in the sequence (0, 1, 2...)
    type TEXT,
    delay INTEGER,
    UNIQUE(campaign_id, step_index) -- Ensure a campaign can't have two steps with the same index
);

-- Table for the A/B test variants within each step
CREATE TABLE IF NOT EXISTS leads.instantly_step_variants (
    variant_id SERIAL PRIMARY KEY,
    step_id INTEGER REFERENCES leads.instantly_campaign_steps(step_id),
    variant_index INTEGER, -- The order of the variant (0, 1, 2...)
    subject TEXT,
    body TEXT,
    UNIQUE(step_id, variant_index)
);