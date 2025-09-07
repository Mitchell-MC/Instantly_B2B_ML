# 🚀 Bronze to Silver Layer ETL Pipeline - Deployment Guide

## 📋 Overview

This guide walks you through deploying the production-ready Bronze to Silver Layer ETL pipeline that transforms your raw B2B lead data into ML-ready features.

## 🏗️ Architecture Overview

```
Bronze Layer (Raw Data)     →     Silver Layer (ML Features)
┌─────────────────────┐           ┌──────────────────────────┐
│ leads.instantly_    │           │ ml_lead_scoring.         │
│ enriched_contacts   │    ETL    │ silver_ml_features       │
│                     │  ──────→  │                          │
│ - Raw email data    │           │ - Engineered features    │
│ - Campaign info     │           │ - Quality scores         │
│ - Engagement metrics│           │ - Target variables       │
└─────────────────────┘           └──────────────────────────┘
```

## 🔧 Prerequisites

### System Requirements
- **Python**: 3.9+
- **PostgreSQL**: 12+
- **Memory**: 8GB+ recommended
- **Storage**: 50GB+ for data and logs

### Database Setup
1. **Ensure your bronze layer exists**:
   ```sql
   -- Your existing table should be accessible
   SELECT COUNT(*) FROM leads.instantly_enriched_contacts;
   ```

2. **Create silver layer schema**:
   ```bash
   psql -h your_host -U your_user -d B2B_Leads_DB -f sql/create_silver_layer_tables.sql
   ```

## 📦 Installation

### 1. Install Dependencies
```bash
# Install required packages
pip install -r requirements_silver_layer.txt

# Or using conda
conda install --file requirements_silver_layer.txt
```

### 2. Environment Configuration
```bash
# Create environment file
cp config/silver_layer_config.yaml config/production_config.yaml

# Set environment variables
export DB_HOST=your_postgres_host
export DB_PORT=5432
export DB_NAME=B2B_Leads_DB
export DB_USER=your_username
export DB_PASSWORD=your_password
```

### 3. Create Directory Structure
```bash
mkdir -p logs
mkdir -p sql
mkdir -p config
```

## ⚙️ Configuration

### Database Configuration
Update `config/production_config.yaml`:
```yaml
database:
  host: your_postgres_host
  port: 5432
  database: B2B_Leads_DB
  user: your_username
  password: your_password
  schema_bronze: leads
  schema_silver: ml_lead_scoring
```

### Processing Configuration
```yaml
processing:
  batch_size: 1000          # Adjust based on memory
  incremental_processing: true
  default_lookback_days: 7  # Process last 7 days
```

## ✅ Pre-Deployment Validation

### Schema Validation (CRITICAL)
Before deploying, validate that your database schema matches the pipeline expectations:

```bash
# Run comprehensive schema validation
python validate_schema_alignment.py

# This will check:
# - Database connectivity
# - Column availability and types
# - Apollo enrichment data presence
# - Silver layer schema compatibility
# - Target variable creation logic
```

**Expected Output:**
```
🔍 BRONZE TO SILVER PIPELINE - SCHEMA VALIDATION REPORT
================================================================================
📊 Overall Status: ✅ PASSED
📋 Bronze Schema Validation: ✅ Schema accessible with 45+ columns
🗂️  Column Mapping Validation: 📈 Coverage: 95.2% (40/42 columns)
🧪 Sample Data Validation: ✅ Successfully extracted 5 sample rows
🎯 Target Variable Validation: ✅ Target variable creation successful
🥈 Silver Layer Validation: ✅ Silver layer schema compatible
```

### Manual Data Quality Check
```sql
-- Verify your bronze layer data
SELECT 
    COUNT(*) as total_records,
    COUNT(DISTINCT email) as unique_emails,
    COUNT(DISTINCT campaign) as unique_campaigns,
    AVG(CASE WHEN a_apollo_id IS NOT NULL THEN 1 ELSE 0 END) as apollo_enrichment_rate,
    MIN(timestamp_created) as oldest_record,
    MAX(timestamp_created) as newest_record
FROM leads.instantly_enriched_contacts;
```

## 🚀 Deployment Options

### Option 1: Manual Execution

#### Run Incremental Pipeline
```bash
# Process last 7 days of data
python pipeline_runner.py --mode incremental

# Process last 30 days
python pipeline_runner.py --mode incremental --lookback-days 30
```

#### Run Full Refresh
```bash
# Process all historical data
python pipeline_runner.py --mode full-refresh
```

### Option 2: Scheduled Execution
```bash
# Run scheduler (recommended for production)
python pipeline_runner.py --mode scheduler

# This will run:
# - Incremental processing every 6 hours
# - Full refresh weekly on Sunday 2 AM
# - Monitoring checks every 15 minutes
```

### Option 3: Direct Pipeline Usage
```python
from bronze_to_silver_pipeline import BronzeToSilverPipeline

# Initialize pipeline
pipeline = BronzeToSilverPipeline('config/production_config.yaml')

# Run incremental processing
result = pipeline.run_pipeline(incremental=True, lookback_days=7)
print(f"Processed {len(result)} records")
```

## 📊 Data Flow & Features

### Input Data (Bronze Layer)
Your `leads.instantly_enriched_contacts` table with comprehensive schema:

#### **Core Lead Data**
- **Identifiers**: `id`, `email`, `campaign`, `list_id`
- **Engagement Metrics**: `email_open_count`, `email_click_count`, `email_reply_count`
- **Engagement Details**: `email_opened_step`, `email_clicked_step`, `email_replied_step`
- **Timestamps**: `timestamp_created`, `timestamp_updated`, `timestamp_last_contact`, `timestamp_last_touch`
- **Campaign Data**: `personalization`, `payload`, `status`, `upload_method`, `assigned_to`

#### **Apollo Enrichment Data (A_* columns)**
- **Personal Info**: `a_first_name`, `a_last_name`, `a_title`, `a_headline`
- **Company Info**: `a_organization_name`, `a_organization_industry`, `a_organization_employees`, `a_organization_founded_year`
- **Location**: `a_city`, `a_state`, `a_country`
- **Professional**: `a_seniority`, `a_departments`, `a_functions`
- **Enrichment Meta**: `a_apollo_id`, `a_credits_consumed`, `a_enriched_at`, `a_api_status`

### Output Features (Silver Layer)
The pipeline creates `ml_lead_scoring.silver_ml_features` with:

#### **Target Variable**
- `engagement_level`: 0 (no engagement), 1 (opened), 2 (clicked/replied)

#### **Timestamp Features**
- `created_day_of_week`, `created_month`, `created_hour`, `created_quarter`
- `created_is_weekend`, `created_is_business_hours`
- `days_since_creation`, `weeks_since_creation`

#### **Text Features**
- `text_length`, `text_word_count`, `has_numbers_in_text`
- `text_quality_score` (0-3 scale)

#### **Categorical Features**
- `org_title_interaction`, `status_method_interaction`
- `apollo_seniority_industry`, `apollo_dept_function`, `apollo_geo_industry`
- Grouped high-cardinality categories for all categorical fields

#### **Apollo B2B Features**
- `has_apollo_enrichment`, `apollo_api_success` - Enrichment availability
- `company_size_category`, `company_size_log` - Company size indicators
- `company_age_years`, `company_age_category` - Company maturity
- `apollo_data_completeness_pct` - Enrichment quality score
- `is_high_value_title`, `is_high_value_seniority` - Decision maker indicators
- `is_tech_department` - Technology focus indicator
- `apollo_data_freshness` - Enrichment recency

#### **Engagement Features**
- `campaign_size`, `campaign_duration_days`
- `created_hour_category` (Night/Morning/Afternoon/Evening)

#### **Quality Metrics**
- `data_quality_score` (0-1 scale)
- `apollo_data_completeness_pct` (Apollo-specific quality)

## 🔍 Monitoring & Validation

### Check Pipeline Status
```sql
-- View recent processing results
SELECT 
    processed_timestamp,
    COUNT(*) as records_processed,
    AVG(data_quality_score) as avg_quality
FROM ml_lead_scoring.silver_ml_features 
WHERE processed_timestamp > NOW() - INTERVAL '24 hours'
GROUP BY processed_timestamp
ORDER BY processed_timestamp DESC;
```

### Data Quality Metrics
```sql
-- Check target variable distribution
SELECT 
    engagement_level,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM ml_lead_scoring.silver_ml_features
GROUP BY engagement_level
ORDER BY engagement_level;
```

### Feature Statistics
```sql
-- View feature engineering summary
SELECT 
    AVG(text_quality_score) as avg_text_quality,
    AVG(data_quality_score) as avg_data_quality,
    AVG(campaign_size) as avg_campaign_size,
    COUNT(DISTINCT org_title_interaction) as unique_interactions
FROM ml_lead_scoring.silver_ml_features;
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Test database connectivity
python -c "
from bronze_to_silver_pipeline import BronzeToSilverPipeline
pipeline = BronzeToSilverPipeline()
print('Database connection successful!')
"
```

#### 2. Memory Issues
- Reduce `batch_size` in configuration
- Increase system memory or use smaller `lookback_days`

#### 3. Data Quality Issues
```sql
-- Check for data issues
SELECT 
    COUNT(*) as total_records,
    COUNT(email) as valid_emails,
    COUNT(campaign) as valid_campaigns,
    AVG(CASE WHEN data_quality_score < 0.5 THEN 1 ELSE 0 END) as low_quality_rate
FROM ml_lead_scoring.silver_ml_features;
```

### Log Analysis
```bash
# View recent logs
tail -f logs/silver_layer_pipeline.log

# Search for errors
grep -i error logs/silver_layer_pipeline.log
```

## 📈 Performance Optimization

### Database Optimization
```sql
-- Analyze table statistics
ANALYZE ml_lead_scoring.silver_ml_features;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch 
FROM pg_stat_user_indexes 
WHERE schemaname = 'ml_lead_scoring';
```

### Pipeline Tuning
- **Batch Size**: Start with 1000, increase if memory allows
- **Parallel Processing**: Enable for large datasets
- **Incremental Processing**: Use for regular updates

## 🔒 Security & Best Practices

### Environment Variables
```bash
# Use environment variables for sensitive data
export DB_PASSWORD=your_secure_password
export DB_HOST=your_db_host
```

### Data Privacy
- PII columns (`email`, `first_name`, `last_name`) are excluded from ML features
- Use secure database connections (`ssl_mode: require`)

### Monitoring
- Set up alerts for pipeline failures
- Monitor data quality metrics
- Track processing performance

## 🚀 Next Steps

### 1. ML Model Training
```python
# Use the silver layer for ML training
from sqlalchemy import create_engine
import pandas as pd

engine = create_engine('your_connection_string')
df = pd.read_sql('SELECT * FROM ml_lead_scoring.v_ml_training_features', engine)

# Your ML training code here...
```

### 2. Automated Deployment
- Set up CI/CD pipeline
- Use Docker for containerization
- Implement blue-green deployments

### 3. Advanced Features
- Feature store integration
- Real-time feature serving
- A/B testing framework

## 📞 Support

### Logs Location
- Pipeline logs: `logs/silver_layer_pipeline.log`
- Error logs: `logs/bronze_to_silver_pipeline.log`

### Key Metrics to Monitor
- **Processing Time**: Should be < 2 hours for incremental runs
- **Data Quality Score**: Should be > 0.8 average
- **Record Count**: Should match expected volume
- **Error Rate**: Should be < 5%

---

## 🎯 Quick Start Checklist

- [ ] Install dependencies (`pip install -r requirements_silver_layer.txt`)
- [ ] Configure database connection in `config/production_config.yaml`
- [ ] Create silver layer tables (`psql -f sql/create_silver_layer_tables.sql`)
- [ ] Test pipeline (`python pipeline_runner.py --mode incremental`)
- [ ] Verify results (`SELECT COUNT(*) FROM ml_lead_scoring.silver_ml_features`)
- [ ] Set up scheduling (`python pipeline_runner.py --mode scheduler`)
- [ ] Monitor logs (`tail -f logs/silver_layer_pipeline.log`)

Your bronze-to-silver ETL pipeline is now ready for production! 🎉
