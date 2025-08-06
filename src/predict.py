"""
Enhanced Inference Pipeline for Email Opening Prediction
Production-ready inference with advanced preprocessing support.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import (
    enhanced_text_preprocessing, advanced_timestamp_features, 
    create_interaction_features, create_jsonb_features, handle_outliers,
    create_xgboost_optimized_features, prepare_features_for_model
)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_model_artifacts(config):
    """Load trained model artifacts."""
    print("📦 Loading model artifacts...")
    
    model_path = Path(config['paths']['model_artifact'])
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifacts not found: {model_path}")
    
    artifacts = joblib.load(model_path)
    
    print(f"✅ Model artifacts loaded:")
    print(f"  Model version: {artifacts.get('model_version', 'Unknown')}")
    print(f"  Training shape: {artifacts.get('training_shape', 'Unknown')}")
    print(f"  Features: {len(artifacts.get('feature_names', []))}")
    
    return artifacts

def prepare_new_data(df, config):
    """Prepare new data with enhanced feature engineering."""
    print("🔧 Preparing new data with enhanced feature engineering...")
    
    # Apply the same feature engineering pipeline as training
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_jsonb_features(df)
    df = create_xgboost_optimized_features(df)
    df = handle_outliers(df)
    
    # Prepare features for model
    X, _, selected_features = prepare_features_for_model(
        df, 
        target_variable=config['data']['target_variable'],
        cols_to_drop=config['features']['cols_to_drop']
    )
    
    print(f"✅ Feature engineering complete. Shape: {X.shape}")
    return X, selected_features

def predict_new_leads(model, preprocessor, X_new, config):
    """Make predictions on new data."""
    print("🔮 Making predictions...")
    
    # Transform features using the preprocessor
    X_transformed = preprocessor.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_transformed)
    prediction_probas = model.predict_proba(X_transformed)
    
    # Get prediction probabilities for the positive class (binary) or all classes (multiclass)
    if len(model.classes_) == 2:
        # Binary classification
        positive_proba = prediction_probas[:, 1]
        prediction_dict = {
            'prediction': predictions,
            'probability': positive_proba,
            'confidence': np.max(prediction_probas, axis=1)
        }
    else:
        # Multi-class classification
        prediction_dict = {
            'prediction': predictions,
            'probability': prediction_probas,
            'confidence': np.max(prediction_probas, axis=1)
        }
    
    print(f"✅ Predictions made for {len(predictions)} leads")
    return prediction_dict

def add_prediction_columns(df, predictions, config):
    """Add prediction columns to the dataframe."""
    print("📊 Adding prediction columns...")
    
    df_result = df.copy()
    
    # Add prediction columns
    df_result['predicted_opened'] = predictions['prediction']
    df_result['prediction_confidence'] = predictions['confidence']
    
    if len(predictions['probability'].shape) == 1:
        # Binary classification
        df_result['opened_probability'] = predictions['probability']
    else:
        # Multi-class classification
        for i, class_name in enumerate(['no_engagement', 'opened', 'engaged']):
            df_result[f'{class_name}_probability'] = predictions['probability'][:, i]
    
    # Add prediction metadata
    df_result['prediction_timestamp'] = datetime.now()
    df_result['model_version'] = config['model']['version']
    
    return df_result

def save_predictions(df_result, config):
    """Save predictions to CSV file."""
    print("💾 Saving predictions...")
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"predictions_{timestamp}.csv"
    output_path = Path("data") / output_filename
    
    # Ensure data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df_result.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to: {output_path}")
    
    return output_path

def create_prediction_summary(df_result, predictions, config):
    """Create a summary of predictions."""
    print("📈 Creating prediction summary...")
    
    total_leads = len(df_result)
    predicted_positive = (df_result['predicted_opened'] == 1).sum()
    positive_rate = predicted_positive / total_leads * 100
    
    # Confidence statistics
    avg_confidence = df_result['prediction_confidence'].mean()
    high_confidence = (df_result['prediction_confidence'] > 0.8).sum()
    high_confidence_rate = high_confidence / total_leads * 100
    
    summary = {
        'total_leads': total_leads,
        'predicted_positive': predicted_positive,
        'positive_rate': positive_rate,
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_confidence,
        'high_confidence_rate': high_confidence_rate,
        'model_version': config['model']['version'],
        'prediction_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return summary

def print_prediction_summary(summary):
    """Print prediction summary."""
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"📊 Total leads processed: {summary['total_leads']:,}")
    print(f"🎯 Predicted to open: {summary['predicted_positive']:,} ({summary['positive_rate']:.1f}%)")
    print(f"📈 Average confidence: {summary['avg_confidence']:.3f}")
    print(f"🔝 High confidence predictions: {summary['high_confidence_count']:,} ({summary['high_confidence_rate']:.1f}%)")
    print(f"🏷️ Model version: {summary['model_version']}")
    print(f"⏰ Prediction timestamp: {summary['prediction_timestamp']}")
    print("="*60)

def handle_missing_features(X_new, training_features, config):
    """Handle missing features in new data."""
    print("🔧 Handling missing features...")
    
    missing_features = set(training_features) - set(X_new.columns)
    extra_features = set(X_new.columns) - set(training_features)
    
    if missing_features:
        print(f"⚠️ Missing {len(missing_features)} features in new data")
        for feature in list(missing_features)[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(missing_features) > 5:
            print(f"  ... and {len(missing_features) - 5} more")
        
        # Add missing features with default values
        for feature in missing_features:
            X_new[feature] = 0
    
    if extra_features:
        print(f"⚠️ Extra {len(extra_features)} features in new data (will be dropped)")
        for feature in list(extra_features)[:5]:  # Show first 5
            print(f"  - {feature}")
        if len(extra_features) > 5:
            print(f"  ... and {len(extra_features) - 5} more")
    
    # Ensure correct column order
    X_new = X_new.reindex(columns=training_features, fill_value=0)
    
    return X_new

def main():
    """Main inference function."""
    print("🚀 Starting Enhanced Inference Pipeline")
    print("="*50)
    
    try:
        # 1. Load configuration
        config = load_config()
        
        # 2. Load model artifacts
        artifacts = load_model_artifacts(config)
        model = artifacts['model']
        preprocessor = artifacts['preprocessor']
        training_features = artifacts['feature_names']
        
        # 3. Load new data (example - replace with your data loading logic)
        print("📊 Loading new data...")
        
        # For demonstration, we'll create sample data
        # In production, replace this with your actual data loading
        sample_data = {
            'email': ['test1@example.com', 'test2@example.com'],
            'first_name': ['John', 'Jane'],
            'last_name': ['Doe', 'Smith'],
            'organization_employees': [500, 1000],
            'daily_limit': [200, 150],
            'country': ['United States', 'Canada'],
            'title': ['Manager', 'Director'],
            'organization_industry': ['Technology', 'Finance'],
            'esp_code': [8.0, 11.0],
            'timestamp_created': ['2024-01-01', '2024-01-02'],
            'campaign_id': ['campaign_1', 'campaign_2'],
            'email_subjects': ['Test Subject 1', 'Test Subject 2'],
            'email_bodies': ['Test body 1', 'Test body 2']
        }
        
        df_new = pd.DataFrame(sample_data)
        print(f"✅ New data loaded. Shape: {df_new.shape}")
        
        # 4. Prepare new data with feature engineering
        X_new, selected_features = prepare_new_data(df_new, config)
        
        # 5. Handle missing features
        X_new = handle_missing_features(X_new, training_features, config)
        
        # 6. Make predictions
        predictions = predict_new_leads(model, preprocessor, X_new, config)
        
        # 7. Add prediction columns to dataframe
        df_result = add_prediction_columns(df_new, predictions, config)
        
        # 8. Save predictions
        output_path = save_predictions(df_result, config)
        
        # 9. Create and print summary
        summary = create_prediction_summary(df_result, predictions, config)
        print_prediction_summary(summary)
        
        print("\n🎉 Inference pipeline completed successfully!")
        print(f"Results saved to: {output_path}")
        
        return df_result
        
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 