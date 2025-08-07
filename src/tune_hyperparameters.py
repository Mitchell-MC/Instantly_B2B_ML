"""
Hyperparameter Tuning Pipeline for Email Opening Prediction Model
Separate pipeline for finding optimal hyperparameters (run periodically, not daily).
"""

import sys
import os
import yaml
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))
from feature_engineering import create_xgboost_optimized_features, encode_categorical_features, prepare_features_for_model

def load_config(config_path="config/main_config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def perform_enhanced_xgboost_grid_search(X_train, y_train, config):
    """
    3-PHASE XGBOOST HYPERPARAMETER OPTIMIZATION
    Much more comprehensive than basic parameter tuning
    """
    print("\n🎯 Enhanced XGBoost Grid Search (3-Phase Optimization)")
    print("="*80)
    
    cv = StratifiedKFold(n_splits=config['training']['cv_folds'], shuffle=True, random_state=42)
    
    # Phase 1: Broad search with XGBoost-specific parameters
    print("\n📊 Phase 1: Broad XGBoost Search")
    print("-" * 50)
    
    broad_params = {
        'n_estimators': [300, 400, 500],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.08, 0.1, 0.12],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 0.95],
        'gamma': [0, 0.1, 0.2],
        'min_child_weight': [1, 3, 5]
    }
    
    xgb_broad = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2  # Handle class imbalance
    )
    
    grid_broad = GridSearchCV(
        estimator=xgb_broad,
        param_grid=broad_params,
        scoring=config['training']['scoring'],
        cv=cv,
        n_jobs=-1,
        verbose=config['training']['verbose']
    )
    
    print("🔄 Fitting broad grid search...")
    grid_broad.fit(X_train, y_train)
    
    print(f"✅ Best broad score: {grid_broad.best_score_:.4f}")
    print(f"📋 Best broad params: {grid_broad.best_params_}")
    
    # Phase 2: Detailed search around best parameters
    print("\n🎯 Phase 2: Detailed Grid Search")
    print("-" * 50)
    
    best_broad = grid_broad.best_params_
    
    detailed_params = {
        'n_estimators': [max(200, best_broad['n_estimators'] - 100), 
                        best_broad['n_estimators'], 
                        best_broad['n_estimators'] + 100],
        'max_depth': [max(6, best_broad['max_depth'] - 1), 
                     best_broad['max_depth'], 
                     min(15, best_broad['max_depth'] + 1)],
        'learning_rate': [max(0.05, best_broad['learning_rate'] - 0.02), 
                         best_broad['learning_rate'], 
                         min(0.2, best_broad['learning_rate'] + 0.02)],
        'subsample': [max(0.8, best_broad['subsample'] - 0.05), 
                     best_broad['subsample'], 
                     min(1.0, best_broad['subsample'] + 0.05)],
        'colsample_bytree': [max(0.8, best_broad['colsample_bytree'] - 0.05), 
                            best_broad['colsample_bytree'], 
                            min(1.0, best_broad['colsample_bytree'] + 0.05)],
        'reg_alpha': [0, 0.1, 0.3],
        'reg_lambda': [1, 1.5, 2]
    }
    
    xgb_detailed = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=1.2,
        gamma=best_broad['gamma'],
        min_child_weight=best_broad['min_child_weight']
    )
    
    grid_detailed = GridSearchCV(
        estimator=xgb_detailed,
        param_grid=detailed_params,
        scoring=config['training']['scoring'],
        cv=cv,
        n_jobs=-1,
        verbose=config['training']['verbose']
    )
    
    print("🔄 Fitting detailed grid search...")
    grid_detailed.fit(X_train, y_train)
    
    print(f"✅ Best detailed score: {grid_detailed.best_score_:.4f}")
    print(f"📋 Best detailed params: {grid_detailed.best_params_}")
    
    # Phase 3: Final regularization optimization
    print("\n🔬 Phase 3: Regularization Fine-tuning")
    print("-" * 50)
    
    best_detailed = grid_detailed.best_params_
    
    final_params = {
        'reg_alpha': [0, 0.1, 0.3, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5, 2.0],
        'scale_pos_weight': [1.0, 1.1, 1.2, 1.3]
    }
    
    # Combine all best parameters except those being fine-tuned
    base_params = {**best_broad, **best_detailed}
    base_params = {k: v for k, v in base_params.items() if k not in final_params}
    
    xgb_final = xgb.XGBClassifier(
        **base_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    grid_final = GridSearchCV(
        estimator=xgb_final,
        param_grid=final_params,
        scoring=config['training']['scoring'],
        cv=cv,
        n_jobs=-1,
        verbose=config['training']['verbose']
    )
    
    print("🔄 Fitting final grid search...")
    grid_final.fit(X_train, y_train)
    
    print(f"✅ Best final score: {grid_final.best_score_:.4f}")
    print(f"📋 Best final params: {grid_final.best_params_}")
    
    # Combine all optimal parameters
    optimal_params = {**base_params, **grid_final.best_params_}
    
    print(f"\n🏆 OPTIMAL XGBOOST PARAMETERS:")
    print("="*50)
    for param, value in optimal_params.items():
        print(f"{param:<20}: {value}")
    print(f"\n⭐ Optimal CV Score: {grid_final.best_score_:.4f}")
    
    return optimal_params, grid_final.best_score_

def save_hyperparameters(optimal_params, cv_score, config):
    """
    Save optimal hyperparameters to JSON file.
    
    Args:
        optimal_params (dict): Optimal hyperparameters
        cv_score (float): Cross-validation score
        config (dict): Configuration dictionary
    """
    hyperparams_path = config['paths']['hyperparameters']
    
    # Ensure config directory exists
    os.makedirs(os.path.dirname(hyperparams_path), exist_ok=True)
    
    # Create hyperparameters dictionary
    hyperparams_data = {
        'xgboost_params': optimal_params,
        'cv_score': cv_score,
        'tuning_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_version': config['model']['version']
    }
    
    # Save to JSON
    with open(hyperparams_path, 'w') as f:
        json.dump(hyperparams_data, f, indent=2)
    
    print(f"✅ Optimal hyperparameters saved to {hyperparams_path}")

def main():
    """Main hyperparameter tuning pipeline."""
    print("=== HYPERPARAMETER TUNING PIPELINE: EMAIL OPENING PREDICTION ===")
    print("🎯 Finding optimal XGBoost hyperparameters")
    print("="*65)
    
    # Load configuration
    config = load_config()
    print(f"✅ Configuration loaded from config/main_config.yaml")
    
    # Load data
    data_path = Path(config['data']['input_file'])
    if not data_path.exists():
        print(f"❌ Error: '{data_path}' not found.")
        return
    
    df = pd.read_csv(data_path, low_memory=False)
    print(f"✅ Data loaded. Shape: {df.shape}")
    
    # Create target
    target_variable = config['data']['target_variable']
    df[target_variable] = (df['email_open_count'] > 0).astype(int)
    target_dist = df[target_variable].value_counts().sort_index()
    print(f"\n📊 Target distribution:")
    print(f"  Not opened: {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  Opened: {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")
    
    # Apply feature engineering
    df = create_xgboost_optimized_features(df)
    df, label_encoders = encode_categorical_features(
        df,
        max_categories=config['features']['max_categories']
    )
    
    # Prepare features for model
    X, y, selected_features = prepare_features_for_model(
        df, 
        target_variable=target_variable,
        cols_to_drop=config['features']['cols_to_drop']
    )
    
    print(f"\n✅ Feature preparation complete.")
    print(f"  📊 Total features: {X.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'], 
        random_state=config['data']['random_state'], 
        stratify=y
    )
    
    print(f"\n📊 Data split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Preprocessing
    print("\n🔄 Preprocessing data...")
    
    # Handle any remaining NaN columns before imputation
    nan_cols = X_train.columns[X_train.isna().all()].tolist()
    if nan_cols:
        print(f"⚠️  Dropping {len(nan_cols)} all-NaN columns: {nan_cols[:5]}...")
        X_train = X_train.drop(columns=nan_cols)
        X_test = X_test.drop(columns=nan_cols)
        X = X.drop(columns=nan_cols)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Convert back to DataFrame for feature selection
    X_train_df = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(config['features']['target_features'], X_train_df.shape[1]))
    X_train_selected = selector.fit_transform(X_train_df, y_train)
    X_test_selected = selector.transform(X_test_df)
    
    selected_features = [X_train_df.columns[i] for i, selected in enumerate(selector.get_support()) if selected]
    
    print(f"\n✅ Preprocessing complete:")
    print(f"  📊 Selected features: {len(selected_features)}")
    print(f"  📊 Training shape: {X_train_selected.shape}")
    print(f"  📊 Test shape: {X_test_selected.shape}")
    
    # Perform hyperparameter tuning
    optimal_params, cv_score = perform_enhanced_xgboost_grid_search(X_train_selected, y_train, config)
    
    # Save optimal hyperparameters
    save_hyperparameters(optimal_params, cv_score, config)
    
    # Test the optimal parameters on test set
    print(f"\n🧪 Testing optimal parameters on test set...")
    
    xgb_optimal = xgb.XGBClassifier(
        **optimal_params,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    xgb_optimal.fit(X_train_selected, y_train)
    y_pred_proba = xgb_optimal.predict_proba(X_test_selected)[:, 1]
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Test set performance with optimal parameters:")
    print(f"  📈 Test AUC: {test_auc:.4f}")
    print(f"  📊 CV Score: {cv_score:.4f}")
    
    # Performance comparison
    baseline_auc = 0.7800
    improvement_auc = test_auc - baseline_auc
    
    print(f"\n" + "="*70)
    print("🚀 HYPERPARAMETER TUNING SUMMARY:")
    print("="*70)
    print(f"✅ Optimal hyperparameters found and saved!")
    print(f"📊 PERFORMANCE METRICS:")
    print(f"  📈 Test AUC: {test_auc:.4f} vs {baseline_auc:.4f} baseline")
    print(f"     Improvement: {improvement_auc:+.4f} ({improvement_auc*100:+.2f}%)")
    print(f"  📊 CV Score: {cv_score:.4f}")
    print(f"  🔧 Features used: {len(selected_features)}")
    print(f"")
    print(f"💾 Next steps:")
    print(f"  1. Update config/main_config.yaml with new parameters")
    print(f"  2. Run src/train.py to train model with optimal parameters")
    print(f"  3. Run src/predict.py to make predictions")
    print("="*70)
    
    return optimal_params

if __name__ == "__main__":
    optimal_params = main() 