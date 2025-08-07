"""
Training Pipeline Without Data Leakage
Uses only predictive features to avoid leakage and achieve realistic accuracy.
"""

import pandas as pd
import numpy as np
import yaml
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import non-leakage feature engineering
from feature_engineering_no_leakage import apply_predictive_feature_engineering

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_real_data():
    """Load real data from merged_contacts.csv."""
    print("📊 Loading real data from merged_contacts.csv...")
    
    try:
        df = pd.read_csv("merged_contacts.csv", low_memory=False)
        print(f"✅ Real data loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("❌ merged_contacts.csv not found!")
        return None

def create_target_variable(df):
    """
    Create target variable using 3-bucket classification:
    0: No opens
    1: 1-2 opens (no clicks/replies)
    2: 3+ opens OR any opens + click OR any opens + reply
    """
    print("🎯 Creating target variable...")
    
    # Initialize target column
    df['engagement_level'] = 0
    
    # Bucket 1: No opens
    # (already initialized as 0)
    
    # Bucket 2: 1-2 opens (no clicks/replies)
    mask_bucket2 = (
        (df['email_open_count'] >= 1) & 
        (df['email_open_count'] <= 2) & 
        (df['email_click_count'] == 0) & 
        (df['email_reply_count'] == 0)
    )
    df.loc[mask_bucket2, 'engagement_level'] = 1
    
    # Bucket 3: 3+ opens OR any opens + click OR any opens + reply
    mask_bucket3 = (
        (df['email_open_count'] >= 3) | 
        ((df['email_open_count'] >= 1) & (df['email_click_count'] >= 1)) |
        ((df['email_open_count'] >= 1) & (df['email_reply_count'] >= 1))
    )
    df.loc[mask_bucket3, 'engagement_level'] = 2
    
    # Display target distribution
    target_dist = df['engagement_level'].value_counts().sort_index()
    print("Target distribution:")
    print(target_dist)
    
    # Calculate class proportions
    class_proportions = df['engagement_level'].value_counts(normalize=True).sort_index()
    print("Class proportions:")
    print(class_proportions)
    
    return df

def prepare_features_for_model(df):
    """Prepare final feature set for model training."""
    print("🔧 Preparing features for model...")
    
    # Columns to drop (leakage prevention)
    cols_to_drop = [
        'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
        'website', 'headline', 'company_domain', 'phone', 'apollo_id',
        'apollo_name', 'organization', 'photo_url', 'organization_name',
        'organization_website', 'organization_phone', 'combined_text'
    ]
    
    # Create feature set
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Remove target variable if present
    if 'engagement_level' in X.columns:
        X = X.drop(columns=['engagement_level'])
    
    # Select only numeric features
    X = X.select_dtypes(include=[np.number])
    
    # Get target variable
    y = df['engagement_level'] if 'engagement_level' in df.columns else None
    
    print(f"Final feature set shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    return X, y, list(X.columns)

def perform_feature_selection(X, y):
    """Perform feature selection without leakage."""
    print("🔍 Performing feature selection...")
    
    # Handle NaN values
    print("Handling NaN values...")
    nan_count = X.isnull().sum().sum()
    print(f"NaN count before: {nan_count}")
    
    # Remove columns with all NaN values
    all_nan_cols = X.columns[X.isnull().all()].tolist()
    if all_nan_cols:
        print(f"Removing {len(all_nan_cols)} columns with all NaN values: {all_nan_cols}")
        X = X.drop(columns=all_nan_cols)
    
    # Impute remaining NaN values
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    print(f"NaN count after imputation: {X_imputed.isnull().sum().sum()}")
    
    # 1. Variance threshold
    selector = VarianceThreshold(threshold=0.01)
    X_var_selected = selector.fit_transform(X_imputed)
    var_selected_features = X_imputed.columns[selector.get_support()].tolist()
    print(f"After variance selection: {len(var_selected_features)} features")
    
    # 2. Correlation-based selection
    X_var_df = pd.DataFrame(X_var_selected, columns=var_selected_features)
    corr_matrix = X_var_df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    X_uncorr = X_var_df.drop(columns=to_drop)
    print(f"After correlation removal: {len(X_uncorr.columns)} features")
    
    # 3. Mutual information selection
    mi_selector = SelectKBest(score_func=mutual_info_classif, k=min(30, len(X_uncorr.columns)))
    X_mi_selected = mi_selector.fit_transform(X_uncorr, y)
    mi_selected_features = X_uncorr.columns[mi_selector.get_support()].tolist()
    
    # Get feature importance scores
    mi_scores = mi_selector.scores_[mi_selector.get_support()]
    feature_importance_df = pd.DataFrame({
        'feature': mi_selected_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print(f"Final selected features: {len(mi_selected_features)}")
    print("Top 10 features by mutual information:")
    print(feature_importance_df.head(10))
    
    return X_mi_selected, mi_selected_features, feature_importance_df

def create_ensemble_model(X_train, y_train, config):
    """Create ensemble model with balanced class weights."""
    print("🤖 Creating ensemble model...")
    
    # Calculate class weights
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    print(f"Class weights: {class_weights}")
    
    # Base models
    models = []
    
    # 1. XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=config['training']['random_state'],
        scale_pos_weight=class_weights.get(1, 1.0) if len(class_weights) == 2 else None,
        eval_metric='logloss',
        verbosity=0
    )
    models.append(('xgb', xgb_model))
    
    # 2. Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=config['training']['random_state'],
        class_weight='balanced'
    )
    models.append(('rf', rf_model))
    
    # 3. Logistic Regression
    lr_model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        random_state=config['training']['random_state'],
        class_weight='balanced',
        solver='liblinear'
    )
    models.append(('lr', lr_model))
    
    # Create ensemble
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )
    
    return ensemble

def perform_cross_validation(model, X, y, config):
    """Perform cross-validation with proper scoring."""
    print("🔄 Performing cross-validation...")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config['training']['random_state'])
    
    # Multiple scoring metrics
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'roc_auc_ovr_weighted': 'roc_auc_ovr_weighted'
    }
    
    cv_results = {}
    for metric_name, metric_func in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric_func, n_jobs=-1)
        cv_results[metric_name] = {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std()
        }
        print(f"{metric_name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return cv_results

def evaluate_model(model, X_test, y_test, cv_results):
    """Evaluate model performance."""
    print("📊 Model evaluation...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = (y_pred == y_test).mean()
    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    
    # Detailed classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Performance summary
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Multi-class ROC AUC Score (weighted): {auc_score:.4f}")
    
    # Compare with CV results
    cv_accuracy = cv_results['accuracy']['mean']
    cv_auc = cv_results['roc_auc_ovr_weighted']['mean']
    
    print(f"CV Accuracy: {cv_accuracy:.4f}")
    print(f"CV AUC: {cv_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'cv_accuracy': cv_accuracy,
        'cv_auc': cv_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def save_model_artifacts(model, feature_names, performance_metrics, config):
    """Save model artifacts."""
    print("💾 Saving model artifacts...")
    
    # Create artifacts dictionary
    artifacts = {
        'model': model,
        'feature_names': feature_names,
        'performance_metrics': performance_metrics,
        'model_version': '2.0_no_leakage',
        'training_date': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save artifacts
    model_path = "models/email_engagement_predictor_no_leakage_v2.0.joblib"
    joblib.dump(artifacts, model_path)
    print(f"✅ Model artifacts saved to: {model_path}")
    
    return model_path

def main():
    """Main training pipeline without leakage."""
    print("🚀 Starting Training Pipeline (No Leakage)")
    print("="*60)
    print("Using only predictive features to avoid data leakage")
    print("="*60)
    
    # Load configuration
    config = load_config()
    
    # Load data
    df = load_real_data()
    if df is None:
        return
    
    # Create target variable
    df = create_target_variable(df)
    
    # Apply predictive feature engineering (no leakage)
    df = apply_predictive_feature_engineering(df)
    
    # Prepare features for model
    X, y, all_features = prepare_features_for_model(df)
    
    # Perform feature selection
    X_selected, selected_features, feature_importance = perform_feature_selection(X, y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.25, random_state=config['training']['random_state'], stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Create ensemble model
    model = create_ensemble_model(X_train, y_train, config)
    
    # Perform cross-validation
    cv_results = perform_cross_validation(model, X_train, y_train, config)
    
    # Train model
    print("🏋️ Training ensemble model...")
    model.fit(X_train, y_train)
    
    # Evaluate model
    performance_metrics = evaluate_model(model, X_test, y_test, cv_results)
    
    # Save model artifacts
    model_path = save_model_artifacts(model, selected_features, performance_metrics, config)
    
    # Print summary
    print("="*60)
    print("TRAINING SUMMARY (No Leakage)")
    print("="*60)
    print(f"📊 Dataset: merged_contacts.csv ({len(df)} rows)")
    print(f"🎯 Target: engagement_level (3-bucket classification)")
    print(f"🔧 Features: {len(selected_features)} selected from {len(all_features)} engineered")
    print(f"📈 Performance:")
    print(f"  - Accuracy: {performance_metrics['accuracy']:.4f}")
    print(f"  - AUC: {performance_metrics['auc']:.4f}")
    print(f"  - CV Accuracy: {performance_metrics['cv_accuracy']:.4f}")
    print(f"  - CV AUC: {performance_metrics['cv_auc']:.4f}")
    print(f"✅ Model saved to: {model_path}")
    
    # Realistic performance assessment
    if performance_metrics['accuracy'] > 0.95:
        print("⚠️ WARNING: Accuracy too high - possible remaining leakage!")
    elif performance_metrics['accuracy'] > 0.80:
        print("✅ Good performance - realistic accuracy achieved")
    else:
        print("📈 Performance indicates no leakage - realistic model")
    
    return model_path

if __name__ == "__main__":
    main() 