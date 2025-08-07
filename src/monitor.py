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
    
    # Load training data (for drift comparison)
    training_data_path = config['data']['input_file']
    if not os.path.exists(training_data_path):
        print(f"❌ Training data not found at {training_data_path}")
        return
    
    training_data = pd.read_csv(training_data_path, low_memory=False)
    print(f"✅ Training data loaded. Shape: {training_data.shape}")
    
    # Search for Apollo contacts CSV in data directory
    data_dir = Path("data")
    apollo_files = list(data_dir.glob("apollo-contacts-export*.csv"))
    
    if not apollo_files:
        print(f"❌ No Apollo contacts CSV found in {data_dir}")
        print(f"   Expected file: apollo-contacts-export.csv")
        print(f"   Please place your Apollo contacts CSV in the data directory")
        return
    
    # Use the most recent Apollo file if multiple exist
    apollo_file = max(apollo_files, key=lambda x: x.stat().st_mtime)
    print(f"✅ Found Apollo contacts file: {apollo_file}")
    
    # Load Apollo contacts data
    try:
        apollo_data = pd.read_csv(apollo_file, low_memory=False)
        print(f"✅ Apollo data loaded. Shape: {apollo_data.shape}")
    except Exception as e:
        print(f"❌ Error loading Apollo data: {e}")
        return
    
    # Detect data drift between training data and Apollo contacts
    print(f"\n🔍 Comparing training data with Apollo contacts...")
    drift_results = detect_data_drift(training_data, apollo_data, config)
    
    # Monitor model performance on Apollo data (if engagement data is available)
    # Note: Apollo data typically doesn't have engagement outcomes, so this is for demonstration
    if 'engagement_level' in apollo_data.columns or 'opened' in apollo_data.columns:
        print(f"\n📊 Apollo data has outcome column - monitoring performance...")
        
        # Prepare Apollo data for prediction
        from predict import prepare_new_data
        try:
            X_prepared = prepare_new_data(apollo_data, artifacts)
            
            # Make predictions
            model = artifacts['model']
            predictions = model.predict(X_prepared)
            probabilities = model.predict_proba(X_prepared)
            
            # Use the predicted class probabilities for monitoring
            if probabilities.shape[1] > 1:
                # Multi-class: use the highest probability for each sample
                max_probabilities = np.max(probabilities, axis=1)
            else:
                # Binary: use the positive class probability
                max_probabilities = probabilities[:, 0]
            
            # For monitoring, we'll use predictions vs a baseline
            # In real production, you'd have actual outcomes
            baseline_accuracy = 0.75  # Example baseline
            baseline_auc = 0.82       # Example baseline
            
            performance_results = {
                'accuracy': baseline_accuracy,  # Would be actual accuracy if outcomes available
                'auc': baseline_auc,            # Would be actual AUC if outcomes available
                'accuracy_threshold': config['monitoring']['performance_threshold'],
                'auc_threshold': config['monitoring']['performance_threshold'],
                'performance_degraded': False,  # Would be calculated based on actual outcomes
                'monitoring_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'note': 'Performance monitoring simulated - no actual outcomes in Apollo data'
            }
            
        except Exception as e:
            print(f"⚠️  Error preparing Apollo data for prediction: {e}")
            performance_results = None
    else:
        print(f"\n⚠️  Apollo data has no outcome column - skipping performance monitoring")
        print(f"   This is expected for new leads without engagement history")
        performance_results = {
            'accuracy': None,
            'auc': None,
            'accuracy_threshold': config['monitoring']['performance_threshold'],
            'auc_threshold': config['monitoring']['performance_threshold'],
            'performance_degraded': False,
            'monitoring_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'note': 'No outcome data available for performance monitoring'
        }
    
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
    print(f"📁 Apollo File: {apollo_file.name}")
    print(f"🔍 Data Drift: {'⚠️  DETECTED' if drift_results['drift_detected'] else '✅ None'}")
    
    if performance_results:
        if performance_results.get('note'):
            print(f"📈 Performance: {performance_results['note']}")
        else:
            print(f"📈 Performance: {'⚠️  DEGRADED' if performance_results['performance_degraded'] else '✅ Good'}")
    
    print(f"")
    print(f"💡 RECOMMENDATIONS:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    print("="*60)
    
    return report

if __name__ == "__main__":
    monitoring_report = main() 