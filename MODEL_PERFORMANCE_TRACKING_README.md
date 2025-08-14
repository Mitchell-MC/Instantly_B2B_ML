# 📊 Model Performance Tracking Implementation Guide

## 🎯 Overview

This guide explains how to implement comprehensive model performance tracking in your MLOps pipeline. The `ModelPerformanceTracker` class provides real-time monitoring, degradation detection, and automated alerting for model performance issues.

## 🚀 Key Features

### ✅ **Comprehensive Metrics Tracking**
- **Classification Metrics**: Accuracy, Precision, Recall, F1, ROC AUC, Log Loss
- **Regression Metrics**: MSE, RMSE, MAE, R², MAPE, SMAPE
- **Common Metrics**: Sample count, prediction variance

### ✅ **Automated Degradation Detection**
- Performance degradation monitoring with configurable thresholds
- Severity classification (minor, moderate, critical)
- Automatic retraining triggers based on degradation levels

### ✅ **Historical Performance Analysis**
- Performance trend analysis over time
- Baseline establishment and monitoring
- Statistical trend calculations

### ✅ **Automated Alerting**
- Real-time performance alerts
- Configurable alert thresholds
- Multiple alert channels (log, file, email)

### ✅ **Reporting & Visualization**
- Comprehensive performance reports
- Performance trend plots
- JSON export for external analysis

## 🔧 Implementation

### 1. **Installation & Setup**

The performance tracker is already integrated into your MLOps pipeline. No additional installation is required.

### 2. **Configuration**

Add the following section to your `config/main_config.yaml`:

```yaml
# Model Performance Tracking Configuration
model_performance:
  degradation_thresholds:
    accuracy_drop: 0.05           # 5% accuracy drop triggers alert
    auc_drop: 0.03                # 3% AUC drop triggers alert
    f1_drop: 0.05                 # 5% F1 drop triggers alert
    precision_drop: 0.05          # 5% precision drop triggers alert
    recall_drop: 0.05             # 5% recall drop triggers alert
    mse_increase: 0.10            # 10% MSE increase triggers alert
    mae_increase: 0.10            # 10% MAE increase triggers alert
  performance_window_days: 30     # Days to consider for trend analysis
  min_samples_for_trend: 10      # Minimum samples for reliable trend
  alert_cooldown_hours: 24       # Hours between similar alerts
  auto_retraining_threshold: 0.15 # 15% degradation triggers retraining
  metrics_to_track:
    - accuracy
    - precision_macro
    - recall_macro
    - f1_macro
    - roc_auc
    - log_loss
    - mse
    - rmse
    - mae
    - r2
  alert_channels: ['log', 'file', 'email']
  performance_storage:
    history_file: "data/performance/performance_history.json"
    alerts_file: "data/performance/alerts.json"
    reports_dir: "data/performance/reports"
```

### 3. **Basic Usage**

#### **Initialize the Tracker**

```python
from src.model_performance_tracker import ModelPerformanceTracker

# Initialize with default config
tracker = ModelPerformanceTracker()

# Or with custom config path
tracker = ModelPerformanceTracker("config/custom_config.yaml")
```

#### **Track Model Performance**

```python
# For classification models
performance_result = tracker.track_performance(
    y_true=y_true,           # True labels
    y_pred=y_pred,           # Predicted labels
    y_pred_proba=y_pred_proba,  # Predicted probabilities
    metadata={
        'model_version': 'v1.0',
        'data_source': 'production',
        'batch_id': 'batch_001'
    }
)

# For regression models
performance_result = tracker.track_performance(
    y_true=y_true,           # True values
    y_pred=y_pred,           # Predicted values
    metadata={
        'model_version': 'v1.0',
        'data_source': 'production'
    }
)
```

#### **Get Performance Summary**

```python
# Get summary for last 30 days
summary = tracker.get_performance_summary(days=30)

# Get summary for last week
weekly_summary = tracker.get_performance_summary(days=7)

print(f"Performance Trend: {summary['performance_trend']['trend_direction']}")
print(f"Total Alerts: {summary['degradation_summary']['total_alerts']}")
```

#### **Generate Reports**

```python
# Generate comprehensive performance report
report = tracker.generate_performance_report(
    output_path="data/performance/performance_report.json",
    days=30
)

# Create performance trend visualization
tracker.plot_performance_trends(
    output_path="data/performance/performance_trends.png",
    days=30
)
```

### 4. **Integration with MLOps Pipeline**

#### **Production Monitoring Integration**

```python
from src.production_monitor import ProductionMonitor

monitor = ProductionMonitor()

# Track performance through production monitor
performance_result = monitor.track_model_performance(
    y_true, y_pred, y_pred_proba, metadata
)
```

#### **Automated Retraining Integration**

The performance tracker automatically triggers retraining when degradation exceeds the `auto_retraining_threshold` (default: 15%).

#### **Drift Detection Correlation**

Performance degradation can be correlated with data drift to provide comprehensive model health assessment.

## 📊 **Performance Metrics Explained**

### **Classification Metrics**

| Metric | Description | Range | Good Value |
|--------|-------------|-------|-------------|
| **Accuracy** | Overall correct predictions | 0-1 | >0.8 |
| **Precision** | True positives / (True + False positives) | 0-1 | >0.8 |
| **Recall** | True positives / (True + False negatives) | 0-1 | >0.8 |
| **F1 Score** | Harmonic mean of precision and recall | 0-1 | >0.8 |
| **ROC AUC** | Area under ROC curve | 0-1 | >0.8 |
| **Log Loss** | Logarithmic loss (lower is better) | 0-∞ | <0.3 |

### **Regression Metrics**

| Metric | Description | Range | Good Value |
|--------|-------------|-------|-------------|
| **MSE** | Mean squared error | 0-∞ | Low |
| **RMSE** | Root mean squared error | 0-∞ | Low |
| **MAE** | Mean absolute error | 0-∞ | Low |
| **R²** | Coefficient of determination | -∞ to 1 | >0.7 |
| **MAPE** | Mean absolute percentage error | 0-∞ | <10% |
| **SMAPE** | Symmetric mean absolute percentage error | 0-∞ | <10% |

## 🚨 **Alerting System**

### **Degradation Levels**

1. **Minor Degradation** (5-10% drop)
   - Warning alerts
   - Monitor closely
   - Consider investigation

2. **Moderate Degradation** (10-20% drop)
   - Warning alerts
   - Schedule retraining
   - Investigate root cause

3. **Critical Degradation** (>20% drop)
   - Critical alerts
   - Immediate retraining required
   - Emergency investigation

### **Alert Types**

- **Performance Degradation**: Metrics below thresholds
- **Low Performance**: Overall performance score <0.5
- **Retraining Trigger**: Degradation exceeds auto-retraining threshold

## 📈 **Trend Analysis**

### **Trend Directions**

- **Improving**: Performance getting better over time
- **Stable**: Performance consistent within acceptable range
- **Degrading**: Performance declining over time

### **Statistical Analysis**

- Linear trend calculation using polynomial fitting
- Confidence intervals for trend reliability
- Minimum sample requirements for reliable analysis

## 🔄 **Production Workflow**

### **1. Daily Performance Tracking**

```python
# After each prediction batch
def track_daily_performance(model, X_batch, y_true_batch):
    # Make predictions
    y_pred = model.predict(X_batch)
    y_pred_proba = model.predict_proba(X_batch) if hasattr(model, 'predict_proba') else None
    
    # Track performance
    metadata = {
        'model_version': model.version,
        'data_source': 'production',
        'batch_timestamp': datetime.now().isoformat()
    }
    
    performance_result = tracker.track_performance(
        y_true_batch, y_pred, y_pred_proba, metadata
    )
    
    return performance_result
```

### **2. Weekly Performance Review**

```python
def weekly_performance_review():
    # Get weekly summary
    weekly_summary = tracker.get_performance_summary(days=7)
    
    # Generate weekly report
    weekly_report = tracker.generate_performance_report(
        output_path=f"data/performance/weekly_report_{datetime.now().strftime('%Y%m%d')}.json",
        days=7
    )
    
    # Create visualizations
    tracker.plot_performance_trends(
        output_path=f"data/performance/weekly_trends_{datetime.now().strftime('%Y%m%d')}.png",
        days=7
    )
    
    return weekly_summary, weekly_report
```

### **3. Automated Alerting Setup**

```python
def setup_performance_alerts():
    # Configure alert thresholds
    alert_config = {
        'email_alerts': True,
        'slack_alerts': True,
        'critical_threshold': 0.20,  # 20% degradation
        'warning_threshold': 0.10,   # 10% degradation
        'alert_cooldown_hours': 24
    }
    
    # Set up alert channels
    # (Implementation depends on your notification system)
    
    return alert_config
```

## 🧪 **Testing & Validation**

### **Unit Testing**

```python
import unittest
import numpy as np
from src.model_performance_tracker import ModelPerformanceTracker

class TestModelPerformanceTracker(unittest.TestCase):
    
    def setUp(self):
        self.tracker = ModelPerformanceTracker()
        self.y_true = np.array([0, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 1, 0])
        self.y_pred_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])
    
    def test_performance_tracking(self):
        result = self.tracker.track_performance(
            self.y_true, self.y_pred, self.y_pred_proba
        )
        
        self.assertIsNotNone(result)
        self.assertIn('performance_score', result)
        self.assertIn('metrics', result)
        self.assertEqual(result['sample_count'], 5)
    
    def test_performance_summary(self):
        # Track some performance first
        self.tracker.track_performance(self.y_true, self.y_pred, self.y_pred_proba)
        
        summary = self.tracker.get_performance_summary(days=1)
        self.assertIsNotNone(summary)
        self.assertIn('total_records', summary)

if __name__ == '__main__':
    unittest.main()
```

### **Integration Testing**

```python
def test_mlops_integration():
    """Test performance tracking integration with MLOps pipeline."""
    
    # Initialize components
    tracker = ModelPerformanceTracker()
    monitor = ProductionMonitor()
    
    # Simulate performance tracking
    y_true = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    y_pred = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    y_pred_proba = np.random.random((100, 2))
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Test direct tracking
    direct_result = tracker.track_performance(y_true, y_pred, y_pred_proba)
    
    # Test through production monitor
    monitor_result = monitor.track_model_performance(y_true, y_pred, y_pred_proba)
    
    # Verify results are consistent
    assert direct_result['performance_score'] == monitor_result['performance_score']
    
    print("✅ MLOps integration test passed")
```

## 📋 **Best Practices**

### **1. Performance Tracking Frequency**

- **High-frequency models**: Track after each batch (real-time)
- **Batch models**: Track after each batch completion
- **Scheduled models**: Track after each scheduled run

### **2. Metadata Management**

Always include relevant metadata:
- Model version
- Data source
- Batch/timestamp information
- Environment (dev/staging/prod)

### **3. Threshold Configuration**

- Start with conservative thresholds
- Adjust based on business requirements
- Consider model-specific characteristics
- Regular threshold review and updates

### **4. Alert Management**

- Avoid alert fatigue with cooldown periods
- Escalate critical alerts immediately
- Maintain alert history for analysis
- Regular alert effectiveness review

### **5. Performance Storage**

- Regular backup of performance data
- Archive old performance records
- Monitor storage usage
- Implement data retention policies

## 🚀 **Advanced Features**

### **Custom Metrics**

```python
class CustomPerformanceTracker(ModelPerformanceTracker):
    
    def _calculate_custom_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate custom business-specific metrics."""
        custom_metrics = {}
        
        # Example: Business-specific accuracy
        business_accuracy = self._calculate_business_accuracy(y_true, y_pred)
        custom_metrics['business_accuracy'] = business_accuracy
        
        # Example: Cost-based metric
        cost_metric = self._calculate_cost_metric(y_true, y_pred)
        custom_metrics['cost_metric'] = cost_metric
        
        return custom_metrics
    
    def _calculate_business_accuracy(self, y_true, y_pred):
        """Calculate accuracy weighted by business importance."""
        # Implementation depends on business logic
        pass
    
    def _calculate_cost_metric(self, y_true, y_pred):
        """Calculate cost-based performance metric."""
        # Implementation depends on cost structure
        pass
```

### **Real-time Streaming**

```python
def stream_performance_tracking():
    """Real-time performance tracking for streaming data."""
    
    tracker = ModelPerformanceTracker()
    
    # Set up streaming data source
    for batch in streaming_data_source:
        # Process batch
        y_true, y_pred, y_pred_proba = process_batch(batch)
        
        # Track performance in real-time
        performance_result = tracker.track_performance(
            y_true, y_pred, y_pred_proba,
            metadata={'streaming': True, 'batch_id': batch.id}
        )
        
        # Handle alerts immediately
        if performance_result.get('degradation_analysis', {}).get('degradation_detected'):
            handle_immediate_alert(performance_result)
        
        # Yield result for downstream processing
        yield performance_result
```

## 🔍 **Troubleshooting**

### **Common Issues**

1. **Import Errors**
   ```bash
   # Ensure src directory is in Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Configuration Loading Issues**
   ```python
   # Check config file path
   tracker = ModelPerformanceTracker("config/main_config.yaml")
   
   # Verify config structure
   print(tracker.config)
   ```

3. **Performance Calculation Errors**
   ```python
   # Check data types and shapes
   print(f"y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
   print(f"y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
   
   # Ensure no NaN values
   print(f"NaN in y_true: {np.isnan(y_true).sum()}")
   print(f"NaN in y_pred: {np.isnan(y_pred).sum()}")
   ```

4. **Storage Issues**
   ```python
   # Check directory permissions
   import os
   os.makedirs("data/performance", exist_ok=True)
   
   # Verify file write permissions
   test_file = "data/performance/test.json"
   with open(test_file, 'w') as f:
       f.write('{"test": "data"}')
   os.remove(test_file)
   ```

### **Debug Mode**

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Initialize tracker with debug info
tracker = ModelPerformanceTracker()
print(f"Tracker config: {tracker.config}")
print(f"Performance history: {len(tracker.performance_history)} records")
```

## 📚 **Additional Resources**

### **Related Documentation**
- [MLOps Implementation README](MLOPS_IMPLEMENTATION_README.md)
- [Production Monitor Documentation](src/production_monitor.py)
- [Configuration Guide](config/main_config.yaml)

### **External Resources**
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
- [MLOps Best Practices](https://mlops.community/)
- [Model Monitoring Guidelines](https://www.kaggle.com/docs/kaggle-ml)

## 🎯 **Next Steps**

1. **Configure Performance Thresholds**: Update `config/main_config.yaml` with your specific requirements
2. **Integrate with Prediction Pipeline**: Add performance tracking calls after model predictions
3. **Set Up Automated Monitoring**: Configure scheduled performance checks
4. **Implement Alert Notifications**: Set up email/Slack alerts for performance issues
5. **Create Performance Dashboards**: Build custom visualizations for stakeholders

---

**🎉 Your model performance tracking system is now ready for production use!**

For questions or support, refer to the troubleshooting section or check the demo script (`model_performance_demo.py`) for working examples.
