# CI/CD Pipeline for ML Models

## Overview

The CI/CD Pipeline provides automated testing, validation, and deployment of machine learning models in your MLOps pipeline. It ensures model quality, security, and reliability through comprehensive validation stages before deployment.

## Features

### 🧪 **Automated Testing**
- **Unit Tests**: Model file validation and loading tests
- **Integration Tests**: Prediction interface validation
- **Performance Tests**: Loading and prediction performance
- **Coverage Reporting**: Test success rate tracking

### ✅ **Model Validation**
- Performance validation using ModelPerformanceTracker
- Data quality validation using DataQualityMonitor
- Drift detection validation
- Configurable validation thresholds

### 🔒 **Security Scanning**
- File permission checks
- Hardcoded secret detection
- Vulnerability assessment
- Risk classification (Low/Medium/High)

### 🚀 **Automated Deployment**
- Environment management (Staging/Production)
- Health checks post-deployment
- Rollback mechanisms on failure
- Model version backup management

### 📊 **Monitoring & Reporting**
- Complete pipeline execution history
- Performance metrics and execution time tracking
- Detailed error analysis and reporting
- Webhook integration for external notifications

### 🔗 **External Integrations**
- GitHub Actions integration
- Jenkins integration
- GitLab CI integration
- Azure DevOps integration

## Installation

The CI/CD Pipeline is included in the MLOps pipeline. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Configuration

The system uses `config/main_config.yaml` for configuration:

```yaml
cicd:
  pipeline:
    stages: ['test', 'validate', 'security_scan', 'deploy']
    timeout_minutes: 30
    max_retries: 3
    parallel_execution: true
  testing:
    unit_test_threshold: 0.90
    integration_test_threshold: 0.85
    performance_test_threshold: 0.80
    data_quality_threshold: 0.85
  validation:
    model_performance_threshold: 0.75
    data_drift_threshold: 0.15
    feature_importance_stability: 0.80
  security:
    vulnerability_scan: true
    dependency_check: true
    secrets_scan: true
    max_severity: 'medium'
  deployment:
    environments: ['staging', 'production']
    auto_approval: false
    rollback_threshold: 0.70
    health_check_timeout: 300
  integrations:
    github_actions: false
    jenkins: false
    gitlab_ci: false
    azure_devops: false
    webhook_urls: []
```

### Pipeline Stages

| Stage | Description | Threshold |
|-------|-------------|-----------|
| `test` | Automated testing | 90% unit, 85% integration, 80% performance |
| `validate` | Model and data validation | 75% performance, 85% quality |
| `security_scan` | Security validation | Medium risk tolerance |
| `deploy` | Model deployment | 70% health check threshold |

### Testing Thresholds

| Test Type | Default | Description |
|-----------|---------|-------------|
| `unit_test_threshold` | 0.90 | Minimum unit test pass rate |
| `integration_test_threshold` | 0.85 | Minimum integration test pass rate |
| `performance_test_threshold` | 0.80 | Minimum performance test pass rate |
| `data_quality_threshold` | 0.85 | Minimum data quality score |

### Security Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `vulnerability_scan` | true | Enable vulnerability scanning |
| `dependency_check` | true | Check for dependency issues |
| `secrets_scan` | true | Scan for hardcoded secrets |
| `max_severity` | 'medium' | Maximum allowed security risk |

### Deployment Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `environments` | ['staging', 'production'] | Available deployment environments |
| `auto_approval` | false | Enable automatic deployment approval |
| `rollback_threshold` | 0.70 | Health check threshold for rollback |
| `health_check_timeout` | 300 | Health check timeout in seconds |

## Usage

### Basic Usage

```python
from src.cicd_pipeline import CICDPipeline

# Initialize the pipeline
pipeline = CICDPipeline()

# Run pipeline with testing only
pipeline_result = pipeline.run_pipeline(
    model_path='path/to/model.joblib',
    environment='staging',
    auto_deploy=False
)

# Run full pipeline with deployment
full_result = pipeline.run_pipeline(
    model_path='path/to/model.joblib',
    environment='production',
    auto_deploy=True
)
```

### Advanced Usage

```python
# Check pipeline status
status = pipeline.get_pipeline_status()

# Get specific pipeline details
pipeline_details = pipeline.get_pipeline_status('pipeline_20250101_120000')

# Rollback deployment
rollback_result = pipeline.rollback_deployment('production')

# Check configuration
config = pipeline.config
print(f"Pipeline stages: {config['pipeline']['stages']}")
```

### Integration with MLOps Pipeline

```python
# In your automated retraining pipeline
from src.auto_retrain import AutomatedRetraining
from src.cicd_pipeline import CICDPipeline

retrainer = AutomatedRetraining()
pipeline = CICDPipeline()

# After retraining, validate and deploy
if retraining_successful:
    pipeline_result = pipeline.run_pipeline(
        model_path='new_model.joblib',
        environment='staging',
        auto_deploy=True
    )
    
    if pipeline_result['overall_status'] == 'deployed':
        print("✅ New model deployed successfully")
    else:
        print("❌ Deployment failed")
```

## Pipeline Stages Explained

### 1. Testing Stage

The testing stage validates the model file and basic functionality.

#### Unit Tests
- **File Existence**: Checks if model file exists
- **File Size**: Validates minimum file size (1KB)
- **File Format**: Validates supported formats (.joblib, .pkl, .h5, .onnx)
- **Model Loading**: Tests if model can be loaded successfully

#### Integration Tests
- **Prediction Interface**: Tests `predict()` method
- **Probability Interface**: Tests `predict_proba()` method if available
- **Data Compatibility**: Tests with sample data

#### Performance Tests
- **Loading Performance**: Model should load in < 5 seconds
- **Prediction Performance**: 100 predictions should complete in < 1 second

### 2. Validation Stage

The validation stage ensures model quality and performance.

#### Model Performance Validation
- Uses ModelPerformanceTracker to validate performance
- Creates dummy validation data for testing
- Ensures performance metrics are within acceptable ranges

#### Data Quality Validation
- Uses DataQualityMonitor to validate data quality
- Creates sample datasets for quality testing
- Ensures quality metrics meet thresholds

#### Overall Validation
- Combines performance and quality validation results
- Sets overall validation status based on combined scores

### 3. Security Stage

The security stage identifies potential security issues.

#### File Permission Checks
- **World Readable**: Checks if model file is world readable
- **World Writable**: Checks if model file is world writable
- **Owner Permissions**: Validates appropriate ownership

#### Secret Detection
- Scans for hardcoded secrets in model files
- Common patterns: password, secret, key, token, api_key
- Flags potential security risks

#### Risk Classification
- **Low Risk**: No security issues detected
- **Medium Risk**: Potential secrets found
- **High Risk**: Permission vulnerabilities detected

### 4. Deployment Stage

The deployment stage manages model deployment and health checks.

#### Environment Management
- Creates deployment directories for each environment
- Manages model file placement and organization
- Handles environment-specific configurations

#### Backup Management
- Automatically backs up existing models before deployment
- Creates timestamped backup files
- Enables rollback capability

#### Health Checks
- **File Existence**: Verifies model file is deployed
- **Model Loading**: Tests if deployed model can be loaded
- **Prediction Testing**: Validates prediction functionality
- **Overall Health**: Determines deployment success

#### Rollback Mechanism
- Automatically rolls back on health check failure
- Restores previous model version
- Maintains system stability

## Production Workflow

### 1. Model Development
```python
# Train and save model
model = train_model(training_data)
joblib.dump(model, 'models/candidate_model.joblib')

# Run CI/CD pipeline
pipeline = CICDPipeline()
result = pipeline.run_pipeline(
    model_path='models/candidate_model.joblib',
    environment='staging',
    auto_deploy=True
)
```

### 2. Staging Validation
```python
# Model is automatically deployed to staging
# Run additional validation tests
staging_validation = run_staging_tests()

if staging_validation['passed']:
    # Promote to production
    production_result = pipeline.run_pipeline(
        model_path='models/candidate_model.joblib',
        environment='production',
        auto_deploy=True
    )
else:
    # Fix issues and retry
    print("Staging validation failed")
```

### 3. Production Deployment
```python
# Model is automatically deployed to production
# Monitor health and performance
health_status = monitor_production_health()

if health_status['degraded']:
    # Trigger rollback
    rollback_result = pipeline.rollback_deployment('production')
    print(f"Rollback result: {rollback_result['status']}")
```

### 4. Continuous Monitoring
```python
# Set up automated monitoring
import schedule
import time

def check_production_health():
    health = monitor_production_health()
    if health['status'] == 'unhealthy':
        # Trigger rollback
        pipeline.rollback_deployment('production')

# Check every 5 minutes
schedule.every(5).minutes.do(check_production_health)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Integration Examples

### With GitHub Actions
```yaml
# .github/workflows/ml-deploy.yml
name: ML Model Deployment

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    - name: Run CI/CD Pipeline
      run: |
        python -c "
        from src.cicd_pipeline import CICDPipeline
        pipeline = CICDPipeline()
        result = pipeline.run_pipeline('models/model.joblib', 'staging', False)
        if result['overall_status'] != 'ready_for_deployment':
          exit(1)
        "
```

### With Jenkins
```groovy
// Jenkinsfile
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'python -c "from src.cicd_pipeline import CICDPipeline; pipeline = CICDPipeline(); result = pipeline.run_pipeline(\'models/model.joblib\', \'staging\', false); exit(0 if result[\'overall_status\'] == \'ready_for_deployment\' else 1)"'
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'python -c "from src.cicd_pipeline import CICDPipeline; pipeline = CICDPipeline(); pipeline.run_pipeline(\'models/model.joblib\', \'production\', true)"'
            }
        }
    }
}
```

### With External Webhooks
```python
# Configure webhooks in config/main_config.yaml
cicd:
  integrations:
    webhook_urls:
      - 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
      - 'https://api.telegram.org/botYOUR_BOT_TOKEN/sendMessage'

# Pipeline will automatically send notifications
pipeline_result = pipeline.run_pipeline('model.joblib', 'production')

# Webhook payload includes:
# - pipeline_id
# - status
# - environment
# - execution_time
# - timestamp
```

## Testing

### Run the Demo
```bash
python cicd_pipeline_demo.py
```

### Unit Tests
```python
# Test pipeline execution
def test_pipeline_execution():
    pipeline = CICDPipeline()
    
    # Test with valid model
    result = pipeline.run_pipeline(
        'test_model.joblib',
        'staging',
        auto_deploy=False
    )
    
    assert result['overall_status'] in ['ready_for_deployment', 'deployed']
    assert 'stages' in result

# Test rollback functionality
def test_rollback():
    pipeline = CICDPipeline()
    
    # Test rollback
    rollback_result = pipeline.rollback_deployment('staging')
    
    assert rollback_result['status'] in ['success', 'failed']
```

### Integration Tests
```python
# Test with actual model files
def test_full_pipeline():
    pipeline = CICDPipeline()
    
    # Create test model
    model = create_test_model()
    model_path = 'test_model.joblib'
    joblib.dump(model, model_path)
    
    # Run full pipeline
    result = pipeline.run_pipeline(
        model_path,
        'staging',
        auto_deploy=True
    )
    
    # Verify deployment
    assert result['overall_status'] == 'deployed'
    assert os.path.exists('models/staging/test_model.joblib')
    
    # Cleanup
    os.remove(model_path)
    shutil.rmtree('models/staging')
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   ```python
   # Check model file format
   if not model_path.endswith(('.joblib', '.pkl', '.h5', '.onnx')):
       print("Unsupported model format")
   
   # Check file size
   if os.path.getsize(model_path) < 1000:
       print("Model file too small")
   ```

2. **Permission Errors**
   ```python
   # Check file permissions
   import stat
   file_stat = os.stat(model_path)
   if file_stat.st_mode & stat.S_IROTH:
       print("Model file is world readable")
   ```

3. **Health Check Failures**
   ```python
   # Verify model can be loaded
   try:
       model = joblib.load(model_path)
       print("Model loaded successfully")
   except Exception as e:
       print(f"Model loading failed: {e}")
   
   # Test predictions
   try:
       predictions = model.predict(test_data)
       print("Predictions working")
   except Exception as e:
       print(f"Predictions failed: {e}")
   ```

### Performance Optimization

1. **Large Models**
   ```python
   # Increase timeout for large models
   pipeline.config['pipeline']['timeout_minutes'] = 60
   
   # Use parallel execution
   pipeline.config['pipeline']['parallel_execution'] = True
   ```

2. **Frequent Deployments**
   ```python
   # Reduce health check timeout
   pipeline.config['deployment']['health_check_timeout'] = 60
   
   # Enable auto-approval for staging
   pipeline.config['deployment']['auto_approval'] = True
   ```

3. **External Integrations**
   ```python
   # Configure webhook timeouts
   # Add to your webhook configuration
   webhook_config = {
       'timeout': 10,
       'retries': 3,
       'backoff': 'exponential'
   }
   ```

## Best Practices

### 1. **Environment Strategy**
- Use staging for validation and testing
- Promote to production only after staging validation
- Maintain separate configurations per environment

### 2. **Testing Strategy**
- Set realistic test thresholds
- Include performance testing in CI/CD
- Test with representative data

### 3. **Security Strategy**
- Regular security scans
- Monitor file permissions
- Scan for secrets in model files

### 4. **Deployment Strategy**
- Always backup before deployment
- Use health checks for validation
- Implement automatic rollback

### 5. **Monitoring Strategy**
- Monitor pipeline execution times
- Track deployment success rates
- Set up alerts for failures

## API Reference

### CICDPipeline Class

#### Methods

- `run_pipeline(model_path, environment, auto_deploy)` - Run CI/CD pipeline
- `get_pipeline_status(pipeline_id)` - Get pipeline status
- `rollback_deployment(environment, backup_path)` - Rollback deployment

#### Properties

- `pipeline_history` - List of pipeline executions
- `deployment_history` - List of deployments
- `test_results` - Test execution results
- `config` - Configuration settings

### Pipeline Result Structure

```python
{
    'pipeline_id': 'pipeline_20250101_120000',
    'start_time': '2025-01-01T12:00:00',
    'model_path': 'path/to/model.joblib',
    'environment': 'staging',
    'stages': {
        'testing': {
            'status': 'passed',
            'tests_run': 10,
            'tests_passed': 10,
            'coverage': 1.0
        },
        'validation': {
            'status': 'passed',
            'model_performance': {...},
            'data_quality': {...}
        },
        'security': {
            'status': 'passed',
            'overall_risk': 'low'
        },
        'deployment': {
            'status': 'deployed',
            'environment': 'staging',
            'health_check': {...}
        }
    },
    'overall_status': 'deployed',
    'execution_time_seconds': 45.2,
    'errors': [],
    'warnings': []
}
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the demo script for usage examples
3. Check the configuration file for settings
4. Review the logs for detailed error information

## Contributing

To contribute to the CI/CD Pipeline:

1. Follow the existing code style
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure integration with existing MLOps components

---

**CI/CD Pipeline** - Part of the comprehensive MLOps pipeline for production-ready machine learning systems.
