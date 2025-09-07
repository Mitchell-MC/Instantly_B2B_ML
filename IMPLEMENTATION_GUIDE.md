# ML-Driven Lead Scoring Implementation Guide

## Overview
This guide provides step-by-step instructions for implementing the ML-driven lead scoring system using N8N, Apollo, and Instantly data sources. Based on the architecture discussion from the August 27 meeting transcript.

## 🎯 Meeting Requirements Implementation

### ✅ **Key Requirements Addressed**
- **Two Main Pipelines**: Instantly data ingestion + Apollo enrichment
- **Single EC2 Instance**: Initially use one instance (can split later)  
- **Three-Layer Structure**: Bronze (raw) → Silver (preprocessed) → Gold (mature)
- **Cost Optimization**: Apollo credit management (20K/month), Instantly prioritization
- **1-Month Maturity**: Gold layer waits 1 month before considering leads for training
- **Class Balance**: Handles 46-47% open rate and 12% third class imbalance

## Architecture Summary

### Data Flow
1. **Instantly API** → **Bronze Layer** (Raw data ingestion)
2. **Apollo API** → **Bronze Layer** (Lead enrichment)
3. **Bronze Layer** → **Silver Layer** (Feature engineering)
4. **Silver Layer** → **Gold Layer** (Mature data for training)
5. **Gold Layer** → **ML Models** → **Lead Predictions**

### Three-Layer Data Structure
- **Bronze**: Raw data from APIs
- **Silver**: Processed features ready for ML
- **Gold**: Mature data (1+ month old) for model training

## Prerequisites

### Infrastructure Requirements
- AWS EC2 instance (t3.large or larger recommended)
- Ubuntu 20.04 LTS
- Docker and Docker Compose
- PostgreSQL 13+
- Python 3.9+

### API Access
- Instantly.ai API credentials
- Apollo.io API credentials (20K credits/month)
- SMTP server for alerts (optional)

## Step 1: EC2 Setup and Access

### 1.1 EC2 Instance Configuration
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Reboot to apply changes
sudo reboot
```

### 1.2 Clone Repository and Setup
```bash
# Clone your repository (replace with actual repo URL)
git clone <your-repo-url> ml-lead-scoring
cd ml-lead-scoring

# Create environment file
cp .env.example .env
# Edit .env with your configurations
```

## Step 2: Environment Configuration

### 2.1 Create .env File
```bash
# Database Configuration
DB_PASSWORD=your_secure_password_here

# N8N Configuration
N8N_USER=admin
N8N_PASSWORD=your_n8n_password_here
N8N_HOST=your-ec2-public-ip

# API Credentials
INSTANTLY_API_KEY=your_instantly_api_key
APOLLO_API_KEY=your_apollo_api_key

# Monitoring & Alerts
GRAFANA_PASSWORD=your_grafana_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SENDER_EMAIL=your_alert_email@domain.com
SENDER_PASSWORD=your_email_password
ALERT_RECIPIENTS=recipient1@domain.com,recipient2@domain.com
```

### 2.2 Security Group Configuration
Configure your EC2 security group to allow:
- Port 80 (HTTP)
- Port 443 (HTTPS)
- Port 5678 (N8N)
- Port 3000 (Grafana)
- Port 5432 (PostgreSQL - for development only)

## Step 3: Database Setup

### 3.1 Deploy Services
```bash
# Start PostgreSQL first
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
sleep 30

# Create database schema
docker-compose exec postgres psql -U postgres -d ml_lead_scoring -f /docker-entrypoint-initdb.d/01-schema.sql
```

### 3.2 Verify Database Setup
```bash
# Connect to database
docker-compose exec postgres psql -U postgres -d ml_lead_scoring

# Verify tables exist
\dt ml_lead_scoring.*

# Exit
\q
```

## Step 4: Deploy Full Stack

### 4.1 Start All Services
```bash
# Deploy complete stack
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 4.2 Verify Service Health
```bash
# Check individual service health
curl http://localhost:5000/health  # Data processor
curl http://localhost:5001/health  # ML service
curl http://localhost:5002/health  # Monitoring

# Check N8N
curl http://localhost:5678/healthz
```

## Step 5: N8N Workflow Configuration

### 5.1 Access N8N Interface
1. Open `http://your-ec2-ip:5678` in browser
2. Login with credentials from .env file
3. Import workflow files:
   - `n8n_instantly_ingestion_workflow.json`
   - `n8n_apollo_enrichment_workflow.json`

### 5.2 Configure N8N Credentials
1. **Instantly API Credential**:
   - Type: HTTP Header Auth
   - Name: `instantlyApi`
   - Header: `Authorization`
   - Value: `Bearer your_instantly_api_key`

2. **Apollo API Credential**:
   - Type: HTTP Header Auth
   - Name: `apolloApi`
   - Header: `X-Api-Key`
   - Value: `your_apollo_api_key`

3. **PostgreSQL Credential**:
   - Host: `postgres`
   - Port: `5432`
   - Database: `ml_lead_scoring`
   - User: `postgres`
   - Password: `your_db_password`

### 5.3 Test Workflows
1. **Test Instantly Ingestion**:
   - Open Instantly workflow
   - Click "Execute Workflow"
   - Verify data appears in `bronze_instantly_leads` table

2. **Test Apollo Enrichment**:
   - Ensure you have leads in bronze layer
   - Execute Apollo workflow
   - Verify enrichment data in `bronze_apollo_enrichment` table

## Step 6: ML Model Training

### 6.1 Initial Data Processing
```bash
# Trigger silver layer processing for existing data
curl -X POST http://localhost:5000/api/process-silver-layer \
  -H "Content-Type: application/json" \
  -d '{"lead_ids": []}'

# Wait for gold layer maturity (or use historical data)
# For testing, you can reduce the maturity requirement in the code
```

### 6.2 Train Initial Model
```bash
# Train first model
curl -X POST http://localhost:5001/api/train-model \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest"}'

# Check training progress in logs
docker-compose logs -f ml_service
```

### 6.3 Verify Model Performance
```bash
# Get model performance metrics
curl http://localhost:5001/api/model-performance

# Get feature importance
curl http://localhost:5001/api/feature-importance
```

## Step 7: Monitoring Setup

### 7.1 Access Monitoring Dashboards
1. **Grafana**: `http://your-ec2-ip:3000`
   - Username: `admin`
   - Password: From .env file

2. **Custom Monitoring**: `http://your-ec2-ip:5002`
   - Real-time system metrics
   - Data quality scores
   - Active alerts

3. **Prometheus**: `http://your-ec2-ip:9090`
   - Raw metrics for advanced users

### 7.2 Configure Grafana Dashboards
1. Import PostgreSQL data source
2. Create dashboards for:
   - Lead ingestion rates
   - Model performance over time
   - Apollo credit usage
   - Data quality trends

## Step 8: Production Optimization

### 8.1 Schedule Regular Operations
N8N workflows are configured with:
- **Instantly Ingestion**: Daily at midnight
- **Apollo Enrichment**: Every 6 hours
- **Model Retraining**: Daily (automatic)
- **Gold Layer Processing**: Every 6 hours

### 8.2 Performance Tuning
```bash
# Optimize PostgreSQL
docker-compose exec postgres psql -U postgres -d ml_lead_scoring -c "
ANALYZE ml_lead_scoring.bronze_instantly_leads;
ANALYZE ml_lead_scoring.silver_ml_features;
ANALYZE ml_lead_scoring.gold_training_data;
"

# Check query performance
docker-compose exec postgres psql -U postgres -d ml_lead_scoring -c "
SELECT schemaname, tablename, attname, n_distinct, correlation 
FROM pg_stats 
WHERE schemaname = 'ml_lead_scoring';
"
```

### 8.3 Backup Strategy
```bash
# Create backup script
cat > backup_database.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U postgres ml_lead_scoring | gzip > "backup_${DATE}.sql.gz"
# Upload to S3 or your preferred backup location
EOF

chmod +x backup_database.sh

# Add to crontab for daily backups
crontab -e
# Add: 0 2 * * * /path/to/backup_database.sh
```

## Step 9: Troubleshooting

### 9.1 Common Issues

**Issue**: N8N workflows failing with database connection errors
**Solution**: 
```bash
# Check PostgreSQL connectivity
docker-compose exec n8n nc -zv postgres 5432

# Restart N8N if needed
docker-compose restart n8n
```

**Issue**: Apollo API rate limiting
**Solution**: 
- Check current usage: `curl http://localhost:5002/api/metrics`
- Reduce enrichment frequency in N8N workflow
- Monitor credit usage in Grafana

**Issue**: Model training failing due to insufficient data
**Solution**:
```bash
# Check gold layer data
docker-compose exec postgres psql -U postgres -d ml_lead_scoring -c "
SELECT COUNT(*) FROM ml_lead_scoring.gold_training_data WHERE training_eligible = true;
"

# Reduce maturity requirements temporarily for testing
```

### 9.2 Log Analysis
```bash
# View all service logs
docker-compose logs --tail=100 -f

# View specific service logs
docker-compose logs data_processor
docker-compose logs ml_service
docker-compose logs n8n
```

### 9.3 Health Checks
```bash
# Run comprehensive health check
./health_check.sh

# Or manually check each service
for service in postgres n8n data_processor ml_service; do
  echo "Checking $service..."
  docker-compose exec $service curl -f http://localhost:5000/health 2>/dev/null || echo "$service: UNHEALTHY"
done
```

## Step 10: Scaling Considerations

### 10.1 Horizontal Scaling
When ready to scale beyond a single EC2 instance:

1. **Separate ML Processing**:
   - Deploy ML service on dedicated GPU instance
   - Use RDS for PostgreSQL
   - Implement Redis for caching

2. **Load Balancing**:
   - Use Application Load Balancer
   - Deploy multiple N8N instances
   - Implement queue-based processing

### 10.2 Performance Monitoring
Key metrics to monitor for scaling decisions:
- CPU/Memory usage on EC2 instance
- Database query performance
- API response times
- Model training duration
- Data processing throughput

## Security Considerations

### 10.1 Production Security
1. **API Keys**: Use AWS Secrets Manager
2. **Database**: Enable SSL, restrict access
3. **N8N**: Use HTTPS, enable authentication
4. **Monitoring**: Secure Grafana access
5. **Backups**: Encrypt backup files

### 10.2 Network Security
1. Use VPC with private subnets
2. Implement WAF for public endpoints
3. Regular security updates
4. Monitor access logs

## Maintenance Schedule

### Daily
- Review monitoring alerts
- Check data ingestion status
- Verify Apollo credit usage

### Weekly
- Review model performance metrics
- Analyze data quality trends
- Check system resource usage

### Monthly
- Update dependencies
- Review and optimize queries
- Backup validation
- Security audit

## Support and Troubleshooting

For issues not covered in this guide:
1. Check service logs: `docker-compose logs [service_name]`
2. Review monitoring dashboard for alerts
3. Verify API credentials and limits
4. Check database connectivity and performance
5. Ensure sufficient disk space and memory

## Next Steps

After successful implementation:
1. **Feature Enhancement**: Add more data sources
2. **Model Improvement**: Experiment with advanced algorithms
3. **Integration**: Connect with CRM systems
4. **Automation**: Implement automated lead actions
5. **Analytics**: Build business intelligence dashboards
