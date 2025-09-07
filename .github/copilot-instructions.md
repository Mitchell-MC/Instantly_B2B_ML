# Instantly B2B ML Lead Scoring System - AI Coding Agent Instructions

## System Architecture Overview

This is a production-ready ML lead scoring system with a **three-layer data architecture** (Bronze → Silver → Gold) that integrates N8N automation, Apollo enrichment, and Instantly data sources. The system uses containerized microservices with cost-optimized EC2 management.

### Key Components
- **Bronze Layer**: Raw data from Instantly/Apollo APIs stored in PostgreSQL
- **Silver Layer**: Processed ML-ready features via `data_processing_service.py`
- **Gold Layer**: Mature data (1+ month old) for model training
- **N8N Workflows**: Automated data ingestion (`n8n_apollo_enrichment_workflow.json`, `n8n_instantly_ingestion_workflow.json`)
- **ML Service**: Flask API at port 5001 (`ml_model_service.py`)
- **Monitoring**: Grafana/Prometheus stack with custom dashboards

## Critical Entry Points

### Unified Pipeline Runner
**Always start with `run_ml_pipeline.py`** - this is the main orchestrator:
```bash
# Complete pipeline (first-time setup)
python run_ml_pipeline.py --all

# Production API serving
python run_ml_pipeline.py --serve --host 0.0.0.0 --port 8000

# Docker deployment
python run_ml_pipeline.py --docker start --profile production
```

### Service Architecture
- **Port 5000**: Data Processing Service
- **Port 5001**: ML Model Service 
- **Port 5002**: Monitoring Dashboard
- **Port 5678**: N8N Workflow Engine
- **Port 3000**: Grafana
- **Port 5432**: PostgreSQL

## Database Schema Patterns

The system uses **schema-first design** with `ml_lead_scoring_schema.sql`. Key tables follow this pattern:
- **Bronze tables**: `bronze_instantly_*`, `bronze_apollo_*` (raw API data)
- **Silver tables**: `silver_*` (processed features)
- **Gold tables**: `gold_*` (training-ready datasets)

Lead maturity is determined by `lead_maturity_config.py` - default 30 days for production.

## Docker & Environment Management

### Multi-Profile Docker Setup
The system uses **Docker Compose profiles** for different environments:
```bash
# Profiles: production, development, analysis
docker-compose --profile production up -d
```

### EC2 Cost Optimization
The `ec2_management.py` module implements **automatic instance lifecycle management**:
- Auto-start for ML tasks
- Auto-stop after completion (configurable via `AUTO_STOP_AFTER_TASK`)
- Cost estimation based on task type and data size

## Data Processing Workflows

### N8N Integration Points
N8N workflows run on 6-hour schedules and trigger data processing:
1. **Apollo Enrichment**: Enriches new leads not in `bronze_apollo_enrichment`
2. **Instantly Ingestion**: Pulls campaign data and lead interactions
3. **Prediction Pipeline**: Scores leads using trained models

### Feature Engineering Pipeline
The `unified_data_pipeline.py` combines noise reduction + feature engineering:
1. **Phase 1**: Comprehensive noise reduction and outlier handling
2. **Phase 2**: Domain-specific B2B feature engineering
3. **Phase 3**: ML pipeline creation with ensemble methods

## Model Training & Serving

### Model Types
- **Primary**: XGBoost with hyperparameter tuning
- **Secondary**: Random Forest for ensemble voting
- **Target Variables**: Binary (`opened`) or multi-class (`engagement_level`)

### API Endpoints
The ML service exposes these critical endpoints:
- `POST /api/train-model` - Background model training
- `POST /api/predict` - Single/batch predictions
- `GET /api/model-performance` - Performance metrics

## Configuration Management

### Environment Configuration
Uses `config/integration_config.yaml` for centralized settings:
- Development vs Production maturity windows
- API rate limits for Apollo (200/day default)
- Cost optimization settings
- Security configurations

**Environment file location**: `C:\Users\mccal\Downloads\Instantly B2B Main\Instantly.env`

### Key Environment Variables
```bash
DB_PASSWORD=ml_password_123
N8N_USER=admin
N8N_PASSWORD=n8n_admin_123
AUTO_STOP_AFTER_TASK=true
```

## Development Workflows

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start development services
python run_ml_pipeline.py --serve --reload

# Run monitoring
python run_ml_pipeline.py --monitor
```

### Testing & Validation
The system includes comprehensive monitoring:
- **Data Quality**: Automated quality checks with alerting
- **Model Performance**: Drift detection and degradation monitoring
- **CI/CD Pipeline**: Automated testing and validation stages

## Integration Patterns

### API Integration
- **Apollo**: Rate-limited enrichment with error handling
- **Instantly**: Campaign and lead data ingestion
- **Both APIs**: Use retry logic and exponential backoff

### Database Operations
- Use PostgreSQL connection pooling
- Implement proper transaction handling for Bronze→Silver→Gold transformations
- Follow the schema naming conventions for new tables

## Common Debugging Patterns

### Service Health Checks
```bash
# Check all service health
curl http://localhost:5000/health  # Data processor
curl http://localhost:5001/health  # ML service
curl http://localhost:5002/health  # Monitoring
```

### Log Locations
- Pipeline logs: `logs/pipeline_runner.log`
- Service logs: `logs/` directory
- Docker logs: `docker-compose logs -f [service]`

## Important Notes

- **Always use absolute paths** when referencing files in the codebase
- **The system auto-creates required directories** (`logs`, `models`, `data`, etc.)
- **Environment detection**: Code checks for EC2 environment vs local development
- **Cost optimization is enabled by default** - instances auto-stop after tasks
- **Database schema is version-controlled** - use the SQL file for reference
