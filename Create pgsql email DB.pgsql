CREATE SCHEMA IF NOT EXISTS leads;

CREATE TABLE leads.instantly_campaigns (
    id SERIAL PRIMARY KEY,
    email_subj TEXT,
    email_bodi TEXT,
    auto_varia1 TEXT,
    auto_varia2 TEXT,
    campaign_campaign TEXT,
    campaign_daily_limit INTEGER,
    email_gap INTEGER,
    email_list TEXT,
    email_tag_insert_uns TEXT,
    link_trackii TEXT,
    name TEXT,
    open_tracl TEXT,
    organizatic TEXT,
    prioritize_r TEXT,
    sequences TEXT,
    status TEXT,
    stop_for_c TEXT,
    stop_on_al TEXT,
    stop_on_re TEXT,
    text_only TEXT,
    timestamp TIMESTAMP,
    timestamp_updated TIMESTAMP
    -- Add/adjust columns and types as needed to match your CSV
);