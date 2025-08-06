"""
Model Monitoring for Email Opening Prediction
Production monitoring for data drift and model performance degradation.
"""

import sys
import os
import yaml
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from feature_engineering import create_xgboost_optimized_features, encode_categorical_features

def load_config(config_path="config/main_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_model_artifacts(model_path):
    """Load model artifacts from saved file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifacts not found at {model_path}")
    
    artifacts = joblib.load(model_path)
    return artifacts

def detect_data_drift(training_data, new_data, config):
    """
    Detect data drift by comparing feature distributions.
    
    Args:
        training_data (pd.DataFrame): Original training data
        new_data (pd.DataFrame): New data to compare against
        config (dict): Configuration dictionary
        
    Returns:
        dict: Drift detection results
    """
    print("🔍 Detecting data drift...")
    
    drift_results = {}
    
    # Apply same feature engineering to both datasets
    training_processed = create_xgboost_optimized_features(training_data.copy())
    new_processed = create_xgboost_optimized_features(new_data.copy())
    
    # Compare key numerical features
    numerical_features = ['organization_employees', 'daily_limit', 'esp_code']
    
    for feature in numerical_features:
        if feature in training_processed.columns and feature in new_processed.columns:
            train_mean = training_processed[feature].mean()
            new_mean = new_processed[feature].mean()
            
            # Calculate drift as percentage change
            drift_pct = abs(new_mean - train_mean) / train_mean if train_mean != 0 else 0
            
            drift_results[feature] = {
                'train_mean': train_mean,
                'new_mean': new_mean,
                'drift_pct': drift_pct,
                'drift_detected': drift_pct > config['monitoring']['drift_threshold']
            }
    
    # Overall drift assessment
    drift_detected = any(result['drift_detected'] for result in drift_results.values())
    
    print(f"✅ Data drift analysis complete:")
    for feature, result in drift_results.items():
        status = "⚠️  DRIFT" if result['drift_detected'] else "✅ OK"
        print(f"  {feature}: {status} ({result['drift_pct']*100:.1f}% change)")
    
    return {
        'drift_detected': drift_detected,
        'feature_drift': drift_results,
        'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def monitor_model_performance(actual_outcomes, predictions, probabilities, config):
    """
    Monitor model performance on new data with known outcomes.
    
    Args:
        actual_outcomes (pd.Series): Actual email open outcomes
        predictions (np.ndarray): Model predictions
        probabilities (np.ndarray): Model probabilities
        config (dict): Configuration dictionary
        
    Returns:
        dict: Performance monitoring results
    """
    print("📊 Monitoring model performance...")
    
    # Calculate performance metrics
    accuracy = accuracy_score(actual_outcomes, predictions)
    auc = roc_auc_score(actual_outcomes, probabilities)
    
    # Check against thresholds
    accuracy_threshold = config['monitoring']['retrain_trigger_accuracy']
    auc_threshold = config['monitoring']['retrain_trigger_auc']
    
    performance_degraded = (accuracy < accuracy_threshold) or (auc < auc_threshold)
    
    results = {
        'accuracy': accuracy,
        'auc': auc,
        'accuracy_threshold': accuracy_threshold,
        'auc_threshold': auc_threshold,
        'performance_degraded': performance_degraded,
        'monitoring_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"✅ Performance monitoring complete:")
    print(f"  🎯 Accuracy: {accuracy:.4f} (Threshold: {accuracy_threshold:.2f})")
    print(f"  📈 AUC: {auc:.4f} (Threshold: {auc_threshold:.2f})")
    
    if performance_degraded:
        print("⚠️  PERFORMANCE DEGRADATION DETECTED!")
        print("   Consider retraining the model.")
    else:
        print("✅ Model performance is within acceptable range.")
    
    return results

def generate_monitoring_report(drift_results, performance_results, config):
    """
    Generate a comprehensive monitoring report.
    
    Args:
        drift_results (dict): Data drift detection results
        performance_results (dict): Model performance results
        config (dict): Configuration dictionary
        
    Returns:
        dict: Monitoring report
    """
    report = {
        'monitoring_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_drift': drift_results,
        'model_performance': performance_results,
        'recommendations': []
    }
    
    # Generate recommendations
    if drift_results['drift_detected']:
        report['recommendations'].append("Data drift detected - consider retraining model")
    
    if performance_results['performance_degraded']:
        report['recommendations'].append("Model performance degraded - retrain model immediately")
    
    if not drift_results['drift_detected'] and not performance_results['performance_degraded']:
        report['recommendations'].append("Model performing well - continue monitoring")
    
    return report

def save_monitoring_report(report, output_path):
    """
    Save monitoring report to file.
    
    Args:
        report (dict): Monitoring report
        output_path (str): Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as JSON
    import json
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Monitoring report saved to {output_path}")

def main():
    """Main monitoring pipeline."""
    print("=== MODEL MONITORING PIPELINE: EMAIL OPENING PREDICTION ===")
    print("🔍 Monitoring for data drift and performance degradation")
    print("="*65)
    
    # Load configuration
    config = load_config()
    
    # Load model artifacts
    model_path = config['paths']['model_artifact']
    artifacts = load_model_artifacts(model_path)
    
    # Example: Load training data (for drift comparison)
    training_data_path = config['data']['input_file']
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found at {training_data_path}")
        return
    
    training_data = pd.read_csv(training_data_path, low_memory=False)
    print(f"✅ Training data loaded. Shape: {training_data.shape}")
    
    # Example: Load new data with known outcomes (for performance monitoring)
    # In production, this would be recent data where you know the actual outcomes
    new_data_path = "data/recent_leads_with_outcomes.csv"  # Example path
    
    if not os.path.exists(new_data_path):
        print(f"⚠️  Example: No recent data with outcomes found at {new_data_path}")
        print(f"   This is expected in this demo. In production, you would:")
        print(f"   1. Load recent data where outcomes are known")
        print(f"   2. Apply the same preprocessing pipeline")
        print(f"   3. Make predictions")
        print(f"   4. Compare predictions with actual outcomes")
        print(f"   5. Generate monitoring report")
        return
    
    # Load new data
    new_data = pd.read_csv(new_data_path, low_memory=False)
    print(f"✅ New data loaded. Shape: {new_data.shape}")
    
    # Detect data drift
    drift_results = detect_data_drift(training_data, new_data, config)
    
    # Monitor model performance (if outcomes are available)
    if 'opened' in new_data.columns:
        # Prepare new data for prediction
        from predict import prepare_new_data
        X_prepared = prepare_new_data(new_data, artifacts, config)
        
        # Make predictions
        model = artifacts['model']
        predictions = model.predict(X_prepared)
        probabilities = model.predict_proba(X_prepared)[:, 1]
        
        # Monitor performance
        performance_results = monitor_model_performance(
            new_data['opened'], predictions, probabilities, config
        )
    else:
        print("⚠️  No outcome column found in new data - skipping performance monitoring")
        performance_results = None
    
    # Generate monitoring report
    report = generate_monitoring_report(drift_results, performance_results, config)
    
    # Save report
    report_path = f"data/monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_monitoring_report(report, report_path)
    
    # Print summary
    print(f"\n" + "="*60)
    print("📊 MONITORING SUMMARY")
    print("="*60)
    print(f"📅 Monitoring Date: {report['monitoring_date']}")
    print(f"🔍 Data Drift: {'⚠️  DETECTED' if drift_results['drift_detected'] else '✅ None'}")
    
    if performance_results:
        print(f"📈 Performance: {'⚠️  DEGRADED' if performance_results['performance_degraded'] else '✅ Good'}")
    
    print(f"")
    print(f"💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("="*60)
    
    return report

if __name__ == "__main__":
    monitoring_report = main() 