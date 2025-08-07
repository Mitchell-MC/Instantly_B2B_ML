# RDS Database Scripts for Email Engagement Prediction

This directory contains scripts that connect directly to the RDS database for training, feature engineering, and prediction of email engagement models.

## Prerequisites

### 1. Database Connection Setup

Before running any RDS scripts, you need to establish a connection to the RDS database through SSH tunnel:

```bash
# Run the port forwarding script (make sure Session Manager Plugin is installed)
./ssm_port_forwarding_rds.sh
```

### 2. Required Dependencies

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn xgboost psycopg2-binary pyyaml joblib
```

## Scripts Overview

### 1. `test_rds_connection.py`
**Purpose**: Test database connectivity before running main scripts

**Usage**:
```bash
python test_rds_connection.py
```

**What it does**:
- Tests connection to RDS database
- Verifies table access
- Loads sample data
- Reports database statistics

### 2. `src/train_rds_advanced.py`
**Purpose**: Train advanced ensemble models using RDS data

**Usage**:
```bash
python src/train_rds_advanced.py
```

**Features**:
- Loads data directly from RDS database
- Applies advanced feature engineering
- Creates ensemble models (Random Forest, XGBoost, Gradient Boosting, Logistic Regression)
- Performs cross-validation
- Saves trained models and artifacts

**Output**:
- `models/email_engagement_predictor_rds_v1.0.joblib` - Trained model
- `models/feature_names_rds_v1.0.json` - Feature names used in training
- `models/performance_metrics_rds_v1.0.json` - Performance metrics

### 3. `src/feature_engineering_rds.py`
**Purpose**: Advanced feature engineering optimized for RDS data

**Usage**:
```bash
python src/feature_engineering_rds.py
```

**Features**:
- Enhanced text preprocessing
- Advanced timestamp features
- JSONB data extraction
- Organization features
- Engagement-based features
- Categorical encoding
- Outlier handling

**Test Function**:
```python
from src.feature_engineering_rds import test_rds_feature_engineering
test_rds_feature_engineering()
```

### 4. `src/predict_rds.py`
**Purpose**: Make predictions on RDS data using trained models

**Usage**:
```bash
python src/predict_rds.py
```

**Features**:
- Loads trained models
- Applies feature engineering to prediction data
- Makes predictions with confidence scores
- Saves results to CSV
- Analyzes prediction performance

**Output**:
- `prediction_results_rds.csv` - Prediction results with confidence scores

## Database Configuration

All scripts use the following database configuration:

```python
DB_CONFIG = {
    'host': 'localhost',  # Localhost through SSH tunnel
    'database': 'postgres',
    'user': 'mitchell',
    'password': 'CTej3Ba8uBrx6o',
    'port': 5431  # Port forwarded through SSH tunnel
}
```

## Data Schema

The scripts expect data from the `leads.enriched_contacts` table with the following key columns:

### Required Columns:
- `email_open_count` - Number of email opens
- `email_click_count` - Number of email clicks  
- `email_reply_count` - Number of email replies
- `timestamp_created` - Contact creation timestamp

### Optional Columns:
- `organization_employee_count` - Company size
- `organization_industry` - Industry
- `country`, `state`, `city` - Location data
- `title`, `seniority` - Professional data
- `employment_history`, `organization_data` - JSONB data

## Target Variable

The scripts create a 3-level engagement classification:

- **Level 0**: No opens (0 opens)
- **Level 1**: Moderate engagement (1-2 opens, no clicks/replies)
- **Level 2**: High engagement (3+ opens OR any opens + click/reply)

## Workflow

### Step 1: Test Connection
```bash
python test_rds_connection.py
```

### Step 2: Train Model
```bash
python src/train_rds_advanced.py
```

### Step 3: Make Predictions
```bash
python src/predict_rds.py
```

## Advanced Usage

### Custom Data Loading
```python
from src.train_rds_advanced import load_data_from_rds

# Load custom amount of data
df = load_data_from_rds(limit=50000)
```

### Single Contact Prediction
```python
from src.predict_rds import predict_single_contact

contact_data = {
    'email_open_count': 2,
    'organization_employee_count': 100,
    'country': 'United States',
    # ... other features
}

result = predict_single_contact(contact_data)
print(f"Predicted engagement: {result['predicted_engagement_level']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Feature Engineering Testing
```python
from src.feature_engineering_rds import test_rds_feature_engineering

# Test with sample data
df_engineered = test_rds_feature_engineering()
```

## Troubleshooting

### Connection Issues
1. Ensure SSH tunnel is active: `./ssm_port_forwarding_rds.sh`
2. Check port forwarding: `netstat -an | grep 5431`
3. Test connection: `python test_rds_connection.py`

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Memory Issues
- Reduce data limit in scripts (default: 100,000 records)
- Use smaller sample sizes for testing
- Monitor memory usage during training

### Model Loading Issues
- Ensure models are trained before running prediction scripts
- Check file paths in `models/` directory
- Verify feature names match between training and prediction

## Performance Tips

1. **Data Loading**: Use appropriate LIMIT clauses for testing
2. **Feature Engineering**: Test with small samples first
3. **Model Training**: Start with smaller datasets for validation
4. **Predictions**: Process data in batches for large datasets

## Output Files

### Training Outputs:
- `models/email_engagement_predictor_rds_v1.0.joblib` - Trained ensemble model
- `models/feature_names_rds_v1.0.json` - Feature names for consistency
- `models/performance_metrics_rds_v1.0.json` - Training performance metrics

### Prediction Outputs:
- `prediction_results_rds.csv` - Predictions with confidence scores
- Console output with analysis and statistics

## Security Notes

- Database credentials are hardcoded for development
- Use environment variables for production
- Ensure SSH tunnel is secure
- Don't commit credentials to version control

## Support

For issues with:
- **Database Connection**: Check SSH tunnel and credentials
- **Feature Engineering**: Test with `test_rds_feature_engineering()`
- **Model Training**: Verify data quality and feature availability
- **Predictions**: Ensure model files exist and feature names match
