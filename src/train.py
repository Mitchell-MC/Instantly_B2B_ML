"""
Enhanced Training Pipeline for Email Opening Prediction
Production-ready training with advanced preprocessing and feature engineering.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from feature_engineering import (
    enhanced_text_preprocessing, advanced_timestamp_features, 
    create_interaction_features, create_jsonb_features, handle_outliers,
    create_xgboost_optimized_features, encode_categorical_features, 
    prepare_features_for_model
)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def load_data(config):
    """Load and prepare data."""
    print("📊 Loading data...")
    
    data_path = Path(config['data']['input_file'])
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data with datetime handling
    df = pd.read_csv(data_path, on_bad_lines='warn', low_memory=False)
    
    # Standardize timestamp columns
    timestamp_cols = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 
                     'enriched_at', 'inserted_at', 'last_contacted_from']
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"✅ Data loaded. Shape: {df.shape}")
    return df

def create_target_variable(df, config):
    """Create target variable based on configuration."""
    print("🎯 Creating target variable...")
    
    target_type = config.get('data', {}).get('target_type', 'binary')
    target_variable = config['data']['target_variable']
    
    if target_type == 'binary':
        # Binary target: opened (0/1)
        if 'opened' in df.columns:
            df[target_variable] = df['opened'].astype(int)
        else:
            # Create binary target from engagement metrics
            df[target_variable] = (df['email_open_count'] > 0).astype(int)
    
    elif target_type == 'multiclass':
        # Multi-class target: engagement_level (0, 1, 2)
        conditions = [
            ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
            (df['email_open_count'] > 0)                                       # Tier 1: Open
        ]
        choices = [2, 1]
        df[target_variable] = np.select(conditions, choices, default=0)
    
    # Check target distribution
    target_dist = df[target_variable].value_counts().sort_index()
    print(f"Target distribution:\n{target_dist}")
    print(f"Class proportions: {target_dist / len(df)}")
    
    return df

def apply_enhanced_feature_engineering(df):
    """Apply comprehensive feature engineering pipeline."""
    print("🔧 Applying enhanced feature engineering...")
    
    # 1. Enhanced text preprocessing
    df = enhanced_text_preprocessing(df)
    
    # 2. Advanced timestamp features
    df = advanced_timestamp_features(df)
    
    # 3. Create interaction features
    df = create_interaction_features(df)
    
    # 4. JSONB features
    df = create_jsonb_features(df)
    
    # 5. Domain-specific XGBoost features
    df = create_xgboost_optimized_features(df)
    
    # 6. Handle outliers
    df = handle_outliers(df)
    
    print(f"✅ Feature engineering complete. Shape: {df.shape}")
    return df

def create_advanced_preprocessor():
    """Create advanced preprocessing pipeline with ColumnTransformer."""
    print("🔧 Creating advanced preprocessor...")
    
    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # More robust to outliers
    ])
    
    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            max_categories=30,  # Reduce to prevent overfitting
            sparse_output=False,
            drop='if_binary'
        ))
    ])
    
    # Text transformer
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(
            max_features=300,  # Balanced number of features
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        ))
    ])
    
    return numeric_transformer, categorical_transformer, text_transformer

def get_optimal_features(X, y, config):
    """Get optimal hyperparameters or use defaults."""
    print("🔧 Loading optimal hyperparameters...")
    
    hyperparams_path = Path(config['paths']['hyperparameters'])
    if hyperparams_path.exists():
        import json
        with open(hyperparams_path, 'r') as f:
            optimal_params = json.load(f)
        print("✅ Loaded optimal hyperparameters from tuning")
        return optimal_params
    else:
        print("⚠️ No optimal hyperparameters found, using defaults")
        return config['model']['xgboost_params']

def create_ensemble_model(config, optimal_params):
    """Create ensemble model with XGBoost and Logistic Regression."""
    print("🤖 Creating ensemble model...")
    
    # Remove random_state from optimal_params if it exists to avoid duplication
    if 'random_state' in optimal_params:
        optimal_params = optimal_params.copy()
        del optimal_params['random_state']
    
    # XGBoost model with optimal parameters
    xgb_model = xgb.XGBClassifier(
        **optimal_params,
        random_state=config['training']['random_state']
    )
    
    # Logistic Regression for diversity
    lr_model = LogisticRegression(
        random_state=config['training']['random_state'],
        max_iter=1000,
        C=1.0
    )
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lr', lr_model)
        ],
        voting=config['model']['ensemble']['voting'],
        n_jobs=config['model']['ensemble']['n_jobs']
    )
    
    return ensemble

def create_preprocessing_pipeline(X, config):
    """Create comprehensive preprocessing pipeline."""
    print("🔧 Creating preprocessing pipeline...")
    
    # Get transformers
    numeric_transformer, categorical_transformer, text_transformer = create_advanced_preprocessor()
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col.endswith('_interaction') or 
                          col in ['title', 'seniority', 'organization_industry', 'country', 'city', 
                                 'enrichment_status', 'upload_method', 'api_status', 'state']]
    text_feature = 'combined_text'
    
    # Remove categorical features from numeric features
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"Feature types:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Text feature: {text_feature if text_feature in X.columns else 'None'}")
    
    # Create transformers list
    transformers = [
        ('num', numeric_transformer, numeric_features)
    ]
    
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    if text_feature in X.columns:
        transformers.append(('text', text_transformer, text_feature))
    
    # Create final preprocessor
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Add feature selection
    final_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('variance_threshold', VarianceThreshold(threshold=0.01)),
        ('feature_selection', SelectKBest(f_classif, k=config['features']['target_features']))
    ])
    
    return final_pipeline

def evaluate_model(model, X_test, y_test, config):
    """Evaluate model performance."""
    print("📊 Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None
    
    # Calculate metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # AUC score (for binary classification)
    if y_pred_proba is not None:
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nROC AUC Score: {auc_score:.4f}")
        
        # Check against targets
        target_auc = config['training']['target_auc']
        if auc_score >= target_auc:
            print(f"✅ AUC target achieved: {auc_score:.4f} >= {target_auc}")
        else:
            print(f"⚠️ AUC below target: {auc_score:.4f} < {target_auc}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(config['paths']['confusion_matrix'], dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'accuracy': (y_pred == y_test).mean(),
        'auc': auc_score if y_pred_proba is not None else None,
        'confusion_matrix': cm
    }

def save_model_artifacts(model, preprocessor, X_train, y_train, config, performance_metrics):
    """Save all model artifacts."""
    print("💾 Saving model artifacts...")
    
    # Prepare artifacts
    artifacts = {
        'model': model,
        'preprocessor': preprocessor,
        'feature_names': X_train.columns.tolist(),
        'config': config,
        'performance_metrics': performance_metrics,
        'model_version': config['model']['version'],
        'training_shape': X_train.shape
    }
    
    # Save to file
    model_path = Path(config['paths']['model_artifact'])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(artifacts, model_path)
    print(f"✅ Model artifacts saved to: {model_path}")
    
    # Save feature importance if available
    if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
        xgb_model = model.named_estimators_['xgb']
        if hasattr(xgb_model, 'feature_importances_'):
            # Get the transformed feature names from the preprocessor
            try:
                # Transform a small sample to get feature names
                X_sample = X_train.head(1)
                X_transformed = preprocessor.transform(X_sample)
                
                # Create feature names for transformed data
                transformed_feature_names = []
                for i in range(X_transformed.shape[1]):
                    transformed_feature_names.append(f"transformed_feature_{i}")
                
                feature_importance = pd.DataFrame({
                    'feature': transformed_feature_names,
                    'importance': xgb_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                top_features = feature_importance.head(20)
                plt.barh(range(len(top_features)), top_features['importance'])
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Feature Importance')
                plt.title('Top 20 Feature Importances')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(config['paths']['feature_importance'], dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✅ Feature importance plot saved to: {config['paths']['feature_importance']}")
                
            except Exception as e:
                print(f"⚠️ Could not create feature importance plot: {e}")
                print("This is normal when using complex preprocessing pipelines")

def main():
    """Main training function."""
    print("🚀 Starting Enhanced Training Pipeline")
    print("="*50)
    
    try:
        # 1. Load configuration
        config = load_config()
        
        # 2. Load data
        df = load_data(config)
        
        # 3. Create target variable
        df = create_target_variable(df, config)
        
        # 4. Apply enhanced feature engineering
        df = apply_enhanced_feature_engineering(df)
        
        # 5. Prepare features for model
        X, y, selected_features = prepare_features_for_model(
            df, 
            target_variable=config['data']['target_variable'],
            cols_to_drop=config['features']['cols_to_drop']
        )
        
        # 6. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['data']['test_size'],
            random_state=config['data']['random_state'],
            stratify=y
        )
        
        # 7. Create preprocessing pipeline
        preprocessor = create_preprocessing_pipeline(X_train, config)
        
        # 8. Get optimal hyperparameters
        optimal_params = get_optimal_features(X_train, y_train, config)
        
        # 9. Create ensemble model
        model = create_ensemble_model(config, optimal_params)
        
        # 10. Create final pipeline
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # 11. Train model
        print("🏋️ Training model...")
        final_pipeline.fit(X_train, y_train)
        
        # 12. Transform test data for evaluation
        X_test_transformed = preprocessor.transform(X_test)
        
        # 13. Evaluate model
        performance_metrics = evaluate_model(model, X_test_transformed, y_test, config)
        
        # 14. Save model artifacts
        save_model_artifacts(model, preprocessor, X_train, y_train, config, performance_metrics)
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"Model saved to: {config['paths']['model_artifact']}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 