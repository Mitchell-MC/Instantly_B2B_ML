# Data Quality Monitoring System

## Overview

The Data Quality Monitoring System provides comprehensive monitoring and validation of data quality metrics in your MLOps pipeline. It automatically detects quality degradation, generates alerts, and integrates seamlessly with other MLOps components.

## Features

### 🔍 **Comprehensive Quality Metrics**
- **Completeness**: Missing value detection and analysis
- **Type Consistency**: Data type validation and consistency checks
- **Range Validity**: Statistical range validation for numeric data
- **Uniqueness**: Duplicate detection and uniqueness analysis
- **Freshness**: Timestamp validation and data recency checks
- **Anomaly Detection**: Statistical outlier detection using IQR method

### 🚨 **Automated Alerting**
- Quality degradation detection with severity classification
- Configurable quality thresholds
- Automatic retraining triggers
- Alert cooldown and deduplication

### 📈 **Trend Analysis**
- Quality trend direction (improving, stable, degrading)
- Statistical trend analysis
- Historical quality comparison
- Baseline establishment and monitoring

### 📄 **Reporting & Visualization**
- Comprehensive quality reports
- Quality trend analysis
- Degradation analysis charts
- JSON export for external analysis

### 🔗 **MLOps Integration**
- Production monitoring integration
- Drift detection correlation
- Automated retraining triggers
- CI/CD pipeline validation
- Champion/Challenger quality comparison

## Installation

The Data Quality Monitor is included in the MLOps pipeline. Ensure you have the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Configuration

The system uses `config/main_config.yaml` for configuration:

```yaml
data_quality:
  quality_thresholds:
    completeness_min: 0.95        # 95% completeness required
    accuracy_min: 0.90            # 90% accuracy required
    consistency_min: 0.85         # 85% consistency required
    validity_min: 0.90            # 90% validity required
    uniqueness_min: 0.95          # 95% uniqueness required
    timeliness_hours: 24          # Data should be within 24 hours
    anomaly_threshold: 0.10       # 10% anomaly threshold
  monitoring_window_days: 30      # Days to consider for trend analysis
  min_samples_for_analysis: 100   # Minimum samples for reliable analysis
  alert_cooldown_hours: 12        # Hours between similar alerts
  auto_retraining_threshold: 0.20 # 20% quality degradation triggers retraining
```

### Quality Thresholds

| Metric | Default | Description |
|--------|---------|-------------|
| `completeness_min` | 0.95 | Minimum required data completeness |
| `accuracy_min` | 0.90 | Minimum required data accuracy |
| `consistency_min` | 0.85 | Minimum required type consistency |
| `validity_min` | 0.90 | Minimum required range validity |
| `uniqueness_min` | 0.95 | Minimum required uniqueness |
| `timeliness_hours` | 24 | Maximum allowed data age in hours |
| `anomaly_threshold` | 0.10 | Maximum allowed anomaly ratio |

## Usage

### Basic Usage

```python
from src.data_quality_monitor import DataQualityMonitor

# Initialize the monitor
monitor = DataQualityMonitor()

# Monitor data quality
quality_result = monitor.monitor_data_quality(
    data=dataframe,
    metadata={'source': 'production', 'version': '1.0'}
)

# Check quality score
quality_score = quality_result['quality_score']
print(f"Data quality score: {quality_score:.3f}")
```

### Advanced Usage

```python
# Get quality summary for the last 30 days
summary = monitor.get_quality_summary(days=30)

# Generate quality report
report = monitor.generate_quality_report(
    output_path="quality_report.json",
    days=30
)

# Get current monitoring status
status = monitor.get_quality_status()
```

### Integration with MLOps Pipeline

```python
# In your production monitoring
from src.production_monitor import ProductionMonitor

monitor = ProductionMonitor()

# Data quality is automatically monitored
# when you call track_model_performance
performance_result = monitor.track_model_performance(
    y_true, y_pred, y_pred_proba,
    metadata={'data_source': 'production'}
)
```

## Quality Metrics Explained

### 1. Completeness
Measures the percentage of non-missing values in your dataset.

```python
completeness = 1 - (missing_values / total_values)
```

**When to worry**: Completeness < 0.95 (5% missing data)

### 2. Type Consistency
Validates that data types are consistent across columns and samples.

**Checks**:
- String columns have reasonable unique value ratios
- Numeric columns maintain consistent data types
- Categorical columns don't have too many unique values

**When to worry**: Type consistency < 0.85

### 3. Range Validity
For numeric data, ensures values fall within reasonable statistical ranges.

**Method**: Uses 3-standard deviation rule (99.7% of data should fall within range)

**When to worry**: Range validity < 0.90

### 4. Uniqueness
Balances between too few and too many unique values.

**For categorical data**: Optimal unique ratio < 0.5
**For numeric data**: Optimal unique ratio > 0.1

**When to worry**: Uniqueness < 0.95

### 5. Freshness
Ensures data is recent and up-to-date.

**Default threshold**: Data should be within 24 hours
**Scoring**: Decays over time (1 week = 0 score)

**When to worry**: Freshness < 0.8

### 6. Anomaly Detection
Uses Interquartile Range (IQR) method to detect statistical outliers.

**Method**: 
- Q1 = 25th percentile, Q3 = 75th percentile
- IQR = Q3 - Q1
- Outliers: < Q1 - 1.5*IQR or > Q3 + 1.5*IQR

**When to worry**: Anomaly ratio > 0.10 (10% outliers)

## Alerting System

### Alert Types

1. **Quality Degradation Alert**
   - Triggered when quality drops below baseline
   - Severity: minor, moderate, critical
   - Includes retraining triggers

2. **Low Quality Alert**
   - Triggered when overall quality < 0.7
   - Severity: high
   - Immediate attention required

3. **Low Completeness Alert**
   - Triggered when completeness < 0.8
   - Severity: medium
   - Data ingestion issues

### Alert Configuration

```yaml
data_quality:
  alert_cooldown_hours: 12        # Prevent alert spam
  auto_retraining_threshold: 0.20 # Trigger retraining at 20% degradation
```

## Production Workflow

### 1. Data Ingestion
```python
# Monitor quality after each data ingestion
new_data = load_new_data()
quality_result = monitor.monitor_data_quality(new_data, metadata)

if quality_result['quality_score'] < 0.8:
    # Trigger alert
    send_alert("Low data quality detected")
    
if quality_result['degradation_analysis']['degradation_detected']:
    # Consider retraining
    if quality_result['degradation_analysis']['degradation_severity'] == 'critical':
        trigger_retraining()
```

### 2. Automated Monitoring
```python
# Set up scheduled quality checks
import schedule
import time

def check_data_quality():
    recent_data = get_recent_data()
    monitor.monitor_data_quality(recent_data)

# Run every hour
schedule.every().hour.do(check_data_quality)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 3. Quality Reports
```python
# Generate daily quality reports
daily_report = monitor.generate_quality_report(
    output_path=f"reports/quality_{datetime.now().strftime('%Y%m%d')}.json",
    days=1
)

# Generate weekly summaries
weekly_summary = monitor.get_quality_summary(days=7)
```

## Integration Examples

### With Drift Detection
```python
from src.advanced_drift_detection import AdvancedDriftDetector

# Quality degradation can indicate data drift
if quality_result['degradation_analysis']['degradation_detected']:
    drift_detector = AdvancedDriftDetector()
    drift_result = drift_detector.detect_drift(baseline_data, new_data)
    
    if drift_result['drift_detected']:
        # Handle both quality and drift issues
        handle_data_issues(quality_result, drift_result)
```

### With Automated Retraining
```python
from src.auto_retrain import AutomatedRetraining

# Quality degradation triggers retraining
if quality_result['degradation_analysis']['overall_degradation_score'] > 0.20:
    retrainer = AutomatedRetraining()
    retrainer.execute_retraining(
        training_data=baseline_data,
        new_data=new_data,
        retraining_strategy='weighted'
    )
```

### With CI/CD Pipeline
```python
from src.cicd_pipeline import CICDPipeline

# Quality validation in deployment pipeline
pipeline = CICDPipeline()
pipeline_result = pipeline.run_pipeline(
    model_path='model.joblib',
    environment='staging'
)

# Quality checks are automatically included in validation stage
```

## Testing

### Run the Demo
```bash
python data_quality_demo.py
```

### Unit Tests
```python
# Test quality monitoring
def test_quality_monitoring():
    monitor = DataQualityMonitor()
    
    # Test with high quality data
    high_quality_data = create_test_data(quality='high')
    result = monitor.monitor_data_quality(high_quality_data)
    
    assert result['quality_score'] > 0.8
    assert not result['degradation_analysis']['degradation_detected']

# Test with low quality data
def test_low_quality_detection():
    monitor = DataQualityMonitor()
    
    low_quality_data = create_test_data(quality='low')
    result = monitor.monitor_data_quality(low_quality_data)
    
    assert result['quality_score'] < 0.7
    assert result['degradation_analysis']['degradation_detected']
```

## Troubleshooting

### Common Issues

1. **Configuration Not Loading**
   ```python
   # Check if config file exists
   import os
   config_path = "config/main_config.yaml"
   if not os.path.exists(config_path):
       print("Config file not found")
   ```

2. **Quality Metrics Calculation Errors**
   ```python
   # Check data types
   print(data.dtypes)
   print(data.isnull().sum())
   
   # Ensure numeric columns are numeric
   data['numeric_column'] = pd.to_numeric(data['numeric_column'], errors='coerce')
   ```

3. **Alert Notifications Not Working**
   ```python
   # Check alert configuration
   print(monitor.config.get('alert_channels', []))
   
   # Verify storage directories exist
   print(os.path.exists("data/quality"))
   ```

### Performance Optimization

1. **Large Datasets**
   ```python
   # Sample data for quality monitoring
   sample_data = data.sample(n=10000, random_state=42)
   quality_result = monitor.monitor_data_quality(sample_data)
   ```

2. **Frequent Monitoring**
   ```python
   # Use alert cooldown to prevent spam
   monitor.config['alert_cooldown_hours'] = 24
   ```

3. **Batch Processing**
   ```python
   # Process data in batches
   for batch in data_batches:
       quality_result = monitor.monitor_data_quality(batch)
       # Process results
   ```

## Best Practices

### 1. **Set Realistic Thresholds**
- Start with conservative thresholds
- Adjust based on your data characteristics
- Consider business impact of false positives

### 2. **Monitor Quality Trends**
- Track quality over time
- Set up alerts for degradation patterns
- Use quality metrics for data pipeline optimization

### 3. **Integrate with Data Pipeline**
- Monitor quality at each pipeline stage
- Use quality gates for data promotion
- Automate quality-based decisions

### 4. **Document Quality Issues**
- Maintain quality issue logs
- Track resolution times
- Use for continuous improvement

### 5. **Regular Quality Reviews**
- Schedule weekly quality reviews
- Analyze quality trends
- Update thresholds based on findings

## API Reference

### DataQualityMonitor Class

#### Methods

- `monitor_data_quality(data, metadata)` - Monitor data quality
- `get_quality_summary(days)` - Get quality summary
- `generate_quality_report(output_path, days)` - Generate quality report
- `get_quality_status()` - Get current monitoring status

#### Properties

- `quality_history` - List of quality records
- `alert_history` - List of quality alerts
- `baseline_quality` - Baseline quality metrics
- `config` - Configuration settings

### Quality Record Structure

```python
{
    'timestamp': '2025-01-01T12:00:00',
    'metrics': {
        'completeness': 0.95,
        'type_consistency': 0.90,
        'range_validity': 0.88,
        'uniqueness': 0.92,
        'freshness': 1.0,
        'anomaly_free': 0.95,
        'overall_accuracy': 0.93
    },
    'metadata': {'source': 'production'},
    'sample_count': 1000,
    'columns_count': 10,
    'quality_score': 0.93,
    'degradation_analysis': {
        'degradation_detected': False,
        'degradation_severity': 'none',
        'overall_degradation_score': 0.0
    }
}
```

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the demo script for usage examples
3. Check the configuration file for settings
4. Review the logs for detailed error information

## Contributing

To contribute to the Data Quality Monitoring system:

1. Follow the existing code style
2. Add comprehensive tests for new features
3. Update documentation for any changes
4. Ensure integration with existing MLOps components

---

**Data Quality Monitoring System** - Part of the comprehensive MLOps pipeline for production-ready machine learning systems.
