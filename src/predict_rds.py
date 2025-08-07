"""
RDS Prediction Pipeline for Email Engagement Prediction
Load trained models and make predictions on RDS database data.
"""

import pandas as pd
import numpy as np
import joblib
import json
import psycopg2
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import RDS feature engineering
from feature_engineering_rds import apply_rds_feature_engineering, connect_to_database

# --- Database Configuration ---
DB_CONFIG = {
    'host': 'localhost',  # Use localhost since we're connecting through SSH tunnel
    'database': 'postgres',
    'user': 'mitchell',
    'password': 'CTej3Ba8uBrx6o',
    'port': 5431  # Local port forwarded through SSH tunnel
}

def load_trained_model(model_path="models/email_engagement_predictor_rds_v1.0.joblib"):
    """Load the trained model from disk."""
    print(f"📥 Loading model from {model_path}...")
    
    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        raise
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise

def load_feature_names(feature_path="models/feature_names_rds_v1.0.json"):
    """Load the feature names used during training."""
    print(f"📥 Loading feature names from {feature_path}...")
    
    try:
        with open(feature_path, 'r') as f:
            feature_names = json.load(f)
        print(f"✅ Feature names loaded. Count: {len(feature_names)}")
        return feature_names
    except FileNotFoundError:
        print(f"❌ Feature names file not found: {feature_path}")
        return None
    except Exception as e:
        print(f"❌ Error loading feature names: {e}")
        return None

def load_prediction_data_from_rds(limit=10000):
    """Load data from RDS for prediction."""
    print("📊 Loading prediction data from RDS...")
    
    conn = connect_to_database()
    
    try:
        # Load data for prediction (without target variable)
        query = f"""
        SELECT * FROM leads.enriched_contacts
        WHERE email_open_count IS NOT NULL
        AND email_open_count >= 0
        LIMIT {limit}
        """
        
        df = pd.read_sql(query, conn)
        print(f"✅ Prediction data loaded. Shape: {df.shape}")
        
        conn.close()
        return df
        
    except Exception as e:
        print(f"❌ Error loading prediction data: {e}")
        conn.close()
        raise

def prepare_data_for_prediction(df, feature_names):
    """Prepare data for prediction by applying feature engineering and selecting features."""
    print("🔧 Preparing data for prediction...")
    
    # Apply feature engineering
    df_engineered = apply_rds_feature_engineering(df)
    
    # Select only the features used during training
    if feature_names:
        available_features = [f for f in feature_names if f in df_engineered.columns]
        missing_features = [f for f in feature_names if f not in df_engineered.columns]
        
        if missing_features:
            print(f"⚠️ Warning: {len(missing_features)} features missing from prediction data")
            print(f"Missing features: {missing_features[:5]}...")
        
        df_prediction = df_engineered[available_features]
        
        # Add missing features with default values
        for feature in missing_features:
            df_prediction[feature] = 0
        
        # Ensure correct column order
        df_prediction = df_prediction[feature_names]
        
    else:
        df_prediction = df_engineered
    
    print(f"✅ Prediction data prepared. Shape: {df_prediction.shape}")
    return df_prediction

def make_predictions(model, df_prediction):
    """Make predictions using the trained model."""
    print("🎯 Making predictions...")
    
    try:
        # Make predictions
        predictions = model.predict(df_prediction)
        probabilities = model.predict_proba(df_prediction)
        
        print(f"✅ Predictions made. Shape: {predictions.shape}")
        
        return predictions, probabilities
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        raise

def create_prediction_results(df_original, predictions, probabilities):
    """Create a results dataframe with predictions and original data."""
    print("📊 Creating prediction results...")
    
    # Create results dataframe
    results_df = df_original.copy()
    results_df['predicted_engagement_level'] = predictions
    
    # Add probability columns for each class
    if probabilities.shape[1] == 3:  # 3-class problem
        results_df['prob_no_engagement'] = probabilities[:, 0]
        results_df['prob_moderate_engagement'] = probabilities[:, 1]
        results_df['prob_high_engagement'] = probabilities[:, 2]
    else:  # Binary or other
        results_df['prob_negative'] = probabilities[:, 0]
        results_df['prob_positive'] = probabilities[:, 1]
    
    # Add confidence score (max probability)
    results_df['prediction_confidence'] = np.max(probabilities, axis=1)
    
    print(f"✅ Results created. Shape: {results_df.shape}")
    return results_df

def save_prediction_results(results_df, output_path="prediction_results_rds.csv"):
    """Save prediction results to CSV."""
    print(f"💾 Saving results to {output_path}...")
    
    try:
        results_df.to_csv(output_path, index=False)
        print(f"✅ Results saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
        raise

def analyze_predictions(results_df):
    """Analyze prediction results and provide insights."""
    print("📈 Analyzing predictions...")
    
    # Prediction distribution
    print("\nPrediction Distribution:")
    print(results_df['predicted_engagement_level'].value_counts().sort_index())
    
    # Confidence analysis
    print(f"\nConfidence Statistics:")
    print(f"Mean confidence: {results_df['prediction_confidence'].mean():.3f}")
    print(f"Std confidence: {results_df['prediction_confidence'].std():.3f}")
    print(f"High confidence predictions (>0.8): {(results_df['prediction_confidence'] > 0.8).sum()}")
    
    # Top predictions by confidence
    print(f"\nTop 5 High-Confidence Predictions:")
    top_confident = results_df.nlargest(5, 'prediction_confidence')
    for idx, row in top_confident.iterrows():
        print(f"ID: {row.get('id', 'N/A')}, "
              f"Predicted: {row['predicted_engagement_level']}, "
              f"Confidence: {row['prediction_confidence']:.3f}")
    
    # If we have actual engagement data, compare
    if 'email_open_count' in results_df.columns:
        print(f"\nPrediction vs Actual Analysis:")
        
        # Create actual engagement levels for comparison
        actual_engagement = np.zeros(len(results_df))
        actual_engagement[(results_df['email_open_count'] >= 1) & (results_df['email_open_count'] <= 2)] = 1
        actual_engagement[results_df['email_open_count'] >= 3] = 2
        
        results_df['actual_engagement_level'] = actual_engagement
        
        # Calculate accuracy
        accuracy = accuracy_score(actual_engagement, results_df['predicted_engagement_level'])
        print(f"Accuracy: {accuracy:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(actual_engagement, results_df['predicted_engagement_level']))
        
        # Confusion matrix
        cm = confusion_matrix(actual_engagement, results_df['predicted_engagement_level'])
        print(f"\nConfusion Matrix:\n{cm}")

def main():
    """Main prediction pipeline."""
    print("🚀 Starting RDS Prediction Pipeline")
    print("=" * 50)
    
    try:
        # Load trained model
        model = load_trained_model()
        
        # Load feature names
        feature_names = load_feature_names()
        
        # Load prediction data
        df_original = load_prediction_data_from_rds(limit=5000)
        
        # Prepare data for prediction
        df_prediction = prepare_data_for_prediction(df_original, feature_names)
        
        # Make predictions
        predictions, probabilities = make_predictions(model, df_prediction)
        
        # Create results
        results_df = create_prediction_results(df_original, predictions, probabilities)
        
        # Save results
        save_prediction_results(results_df)
        
        # Analyze predictions
        analyze_predictions(results_df)
        
        print("\n🎉 Prediction pipeline completed successfully!")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in prediction pipeline: {e}")
        raise

def predict_single_contact(contact_data, model_path="models/email_engagement_predictor_rds_v1.0.joblib"):
    """Make prediction for a single contact."""
    print("🎯 Making prediction for single contact...")
    
    try:
        # Load model
        model = load_trained_model(model_path)
        feature_names = load_feature_names()
        
        # Convert to dataframe
        df = pd.DataFrame([contact_data])
        
        # Prepare data
        df_prediction = prepare_data_for_prediction(df, feature_names)
        
        # Make prediction
        prediction = model.predict(df_prediction)[0]
        probability = model.predict_proba(df_prediction)[0]
        
        result = {
            'predicted_engagement_level': int(prediction),
            'probabilities': probability.tolist(),
            'confidence': float(np.max(probability))
        }
        
        print(f"✅ Prediction made: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error making single prediction: {e}")
        raise

if __name__ == "__main__":
    # Run the prediction pipeline
    results = main()
