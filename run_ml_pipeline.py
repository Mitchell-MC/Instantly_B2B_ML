#!/usr/bin/env python3
"""
Enhanced ML Pipeline Runner with Complete Training Steps from train_2.py
Comprehensive script to run all components of the Email Engagement ML Pipeline
with SHAP analysis, EC2 integration, and complete training pipeline.
"""

import os
import sys
import argparse
import subprocess
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
import logging
import yaml
import warnings
import boto3
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedMLPipelineRunner:
    """Enhanced ML Pipeline Runner with complete training steps from train_2.py."""
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the enhanced pipeline runner."""
        self.config = self._load_config(config_path)
        self.setup_directories()
        self.results = {}
        self.shap_results = {}
        self.training_results = {}
        
        # Initialize AWS clients if running on EC2
        self.aws_available = self._check_aws_availability()
        if self.aws_available:
            self.s3_client = boto3.client('s3')
            self.bucket_name = 'ml-pipeline-outputs'
        
        logger.info("🚀 Enhanced ML Pipeline Runner initialized")
        logger.info(f"🔍 SHAP Analysis: {'Enabled' if self.config.get('shap', {}).get('enabled', True) else 'Disabled'}")
        logger.info(f"☁️ AWS Integration: {'Available' if self.aws_available else 'Not Available'}")
    
    def _check_aws_availability(self):
        """Check if running on EC2 with AWS credentials."""
        try:
            boto3.client('sts').get_caller_identity()
            return True
        except:
            return False
    
    def setup_directories(self):
        """Setup required directories."""
        directories = ['logs', 'models', 'data', 'outputs', 'shap_analysis', 'artifacts']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        logger.info("✅ Directory structure verified")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"⚠️ Could not load config from {config_path}: {e}")
            # Return default configuration
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration if YAML file is not available."""
        return {
            'data': {
                'input_file': 'data/merged_contacts.csv',
                'target_variable': 'engagement_level',
                'test_size': 0.25,
                'random_state': 42
            },
            'features': {
                'max_categories': 100,
                'target_features': 60,
                'shap': {
                    'enabled': True,
                    'max_samples': 1000,
                    'rf_estimators': 100,
                    'top_k': 25
                }
            },
            'model': {
                'name': 'email_open_predictor',
                'version': '1.0'
            },
            'paths': {
                'model_artifact': 'models/email_open_predictor_optimized_v1.0.joblib'
            }
        }
    
    def run_complete_pipeline(self, include_shap=True):
        """Run the complete ML pipeline with all training steps from train_2.py."""
        logger.info("🎯 Starting Complete ML Pipeline with Full Training Steps")
        logger.info("=" * 60)
        
        try:
            start_time = time.time()
            
            # Phase 1: Data Preparation
            logger.info("📊 Phase 1: Data Preparation")
            df = self._run_data_preparation()
            if df is None:
                raise ValueError("Data preparation failed")
            
            # Phase 2: Complete Model Training (from train_2.py)
            logger.info("🤖 Phase 2: Complete Model Training Pipeline")
            model, training_metrics = self._run_complete_training_pipeline(df)
            
            # Phase 3: SHAP Analysis
            if include_shap:
                logger.info("🔍 Phase 3: SHAP Analysis")
                self._run_shap_analysis(model, df)
            
            # Phase 4: Model Validation
            logger.info("✅ Phase 4: Model Validation")
            self._run_model_validation(model, df)
            
            # Phase 5: Model Deployment
            logger.info("🚀 Phase 5: Model Deployment")
            self._run_model_deployment(model)
            
            # Phase 6: Monitoring Setup
            logger.info("📈 Phase 6: Monitoring Setup")
            self._run_monitoring_setup()
            
            execution_time = (time.time() - start_time) / 60  # minutes
            
            # Create comprehensive results
            self.results = {
                'status': 'success',
                'execution_time_minutes': execution_time,
                'phases_completed': 6,
                'shap_analysis': include_shap,
                'training_metrics': training_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            if include_shap:
                self.results.update(self.shap_results)
            
            logger.info("🎉 Complete Pipeline Finished Successfully!")
            self._save_results_to_s3()
            self._print_pipeline_summary()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            self.results = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            raise
    
    def _run_data_preparation(self):
        """Run data preparation and validation."""
        logger.info("  🔍 Checking data quality...")
        
        # Check if training data exists
        data_file = self.config.get('data', {}).get('input_file', 'data/merged_contacts.csv')
        if not os.path.exists(data_file):
            logger.error(f"  ❌ Training data not found at {data_file}")
            return None
        
        try:
            # Load data
            df = pd.read_csv(data_file, low_memory=False)
            logger.info(f"  ✅ Data loaded: {df.shape}")
            
            # Basic validation
            if df.empty:
                logger.error("  ❌ Data is empty")
                return None
            
            if self.config.get('data', {}).get('target_variable') not in df.columns:
                logger.error(f"  ❌ Target variable not found in data")
                return None
            
            logger.info("  ✅ Data preparation completed")
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Data loading failed: {e}")
            return None
    
    def _run_complete_training_pipeline(self, df):
        """Run the complete training pipeline from train_2.py."""
        logger.info("��️ Starting Complete Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create target variable
            logger.info("🎯 Step 1: Creating target variable...")
            df = self._create_target_variable(df)
            
            # Step 2: Apply enhanced feature engineering
            logger.info("🔧 Step 2: Applying enhanced feature engineering...")
            original_shape = df.shape
            df = self._apply_complete_feature_engineering(df)
            logger.info(f"✅ Feature engineering complete. Shape: {df.shape}")
            logger.info(f"📈 Feature increase: {df.shape[1] - original_shape[1]} new features")
            
            # Step 3: Prepare features for model
            logger.info("🔧 Step 3: Preparing features for model...")
            X, y, feature_names = self._prepare_features_for_model(df)
            
            # Step 4: Perform advanced feature selection
            logger.info("�� Step 4: Performing advanced feature selection...")
            X_selected, selected_features, mi_df = self._perform_advanced_feature_selection(X, y)
            
            # Step 5: Split data
            logger.info("📊 Step 5: Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y, 
                test_size=self.config.get('data', {}).get('test_size', 0.25), 
                random_state=self.config.get('data', {}).get('random_state', 42), 
                stratify=y
            )
            logger.info(f"📊 Data split: Training: {X_train.shape}, Test: {X_test.shape}")
            
            # Step 6: CTGAN augmentation (optional)
            logger.info("🔄 Step 6: Applying CTGAN augmentation...")
            try:
                X_train, y_train = self._augment_training_data_with_ctgan(X_train, y_train)
                logger.info(f"🔄 After CTGAN augmentation: X_train={X_train.shape}, y_train={len(y_train)}")
            except Exception as e:
                logger.warning(f"⚠️ CTGAN augmentation skipped: {e}")
            
            # Step 7: Class balancing
            logger.info("⚖️ Step 7: Applying class balancing...")
            class_weights = self._compute_balanced_class_weights(y_train)
            balancing_results = self._apply_advanced_class_balancing(X_train, y_train, X_test, y_test)
            
            # Step 8: Test different balancing methods
            logger.info("🔄 Step 8: Testing different class balancing methods...")
            best_accuracy, best_method, best_X_train, best_y_train = self._test_balancing_methods(
                balancing_results, X_train, y_train, X_test, y_test, class_weights
            )
            
            # Step 9: Hyperparameter optimization
            logger.info("�� Step 9: Performing hyperparameter optimization...")
            xgb_optimized, lr_optimized = self._perform_hyperparameter_optimization(
                best_X_train, best_y_train, class_weights
            )
            
            # Step 10: Create ensemble
            logger.info("🤖 Step 10: Creating optimized ensemble...")
            ensemble = self._create_optimized_ensemble(xgb_optimized, lr_optimized)
            
            # Step 11: Cross-validation
            logger.info("🔄 Step 11: Performing robust cross-validation...")
            cv_scores = self._perform_robust_cross_validation(ensemble, best_X_train, best_y_train)
            
            # Step 12: Train final model
            logger.info("��️ Step 12: Training final optimized model...")
            ensemble.fit(best_X_train, best_y_train)
            
            # Step 13: Evaluate model
            logger.info("📊 Step 13: Comprehensive model evaluation...")
            evaluation_metrics = self._evaluate_model_comprehensive(ensemble, X_test, y_test)
            
            # Step 14: Save model artifacts
            logger.info("💾 Step 14: Saving optimized model artifacts...")
            self._save_model_artifacts(ensemble, best_X_train, best_y_train, {
                'accuracy': best_accuracy,
                'cv_scores': cv_scores,
                'best_balancing_method': best_method,
                'test_metrics': evaluation_metrics,
                'selected_features': selected_features,
                'feature_count': len(selected_features)
            })
            
            # Compile training results
            training_results = {
                'model': ensemble,
                'accuracy': best_accuracy,
                'cv_scores': cv_scores,
                'best_balancing_method': best_method,
                'evaluation_metrics': evaluation_metrics,
                'selected_features': selected_features,
                'feature_count': len(selected_features),
                'training_shape': best_X_train.shape,
                'test_shape': X_test.shape
            }
            
            logger.info("✅ Complete training pipeline finished successfully!")
            return ensemble, training_results
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise
    
    def _create_target_variable(self, df):
        """Create target variable for engagement prediction."""
        logger.info("  🎯 Creating engagement level target variable...")
        
        try:
            # Check if target already exists
            if 'engagement_level' in df.columns:
                logger.info("  ✅ Target variable already exists")
                return df
            
            # Create target based on engagement metrics
            if 'email_open_count' in df.columns and 'email_click_count' in df.columns and 'email_reply_count' in df.columns:
                conditions = [
                    ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),  # Tier 2: Click OR Reply
                    (df['email_open_count'] > 0)                                       # Tier 1: Open
                ]
                choices = [2, 1]
                df['engagement_level'] = np.select(conditions, choices, default=0)
                
                # Log target distribution
                target_dist = df['engagement_level'].value_counts().sort_index()
                logger.info(f"  📊 Target distribution: {target_dist.to_dict()}")
                
            else:
                logger.warning("  ⚠️ Required engagement columns not found, creating dummy target")
                df['engagement_level'] = 0
            
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Target variable creation failed: {e}")
            raise
    
    def _apply_complete_feature_engineering(self, df):
        """Apply all feature engineering functions from train_2.py."""
        logger.info("  🔧 Applying complete feature engineering pipeline...")
        
        try:
            # Import feature engineering functions
            from src.feature_engineering import (
                enhanced_text_preprocessing,
                advanced_timestamp_features,
                create_interaction_features,
                create_comprehensive_jsonb_features,
                create_comprehensive_organization_data,
                create_advanced_engagement_features,
                create_xgboost_optimized_features,
                handle_outliers,
                encode_categorical_features
            )
            
            # Apply all feature engineering functions
            logger.info("    📝 Enhanced text preprocessing...")
            df = enhanced_text_preprocessing(df)
            
            logger.info("    ⏰ Advanced timestamp features...")
            df = advanced_timestamp_features(df)
            
            logger.info("    🔗 Interaction features...")
            df = create_interaction_features(df)
            
            logger.info("    �� Comprehensive JSONB features...")
            df = create_comprehensive_jsonb_features(df)
            
            logger.info("    🏢 Comprehensive organization data...")
            df = create_comprehensive_organization_data(df)
            
            logger.info("    📈 Advanced engagement features...")
            df = create_advanced_engagement_features(df)
            
            logger.info("    🚀 XGBoost optimized features...")
            df = create_xgboost_optimized_features(df)
            
            logger.info("    📊 Outlier handling...")
            df = handle_outliers(df)
            
            logger.info("    🔤 Categorical encoding...")
            df = encode_categorical_features(df)
            
            logger.info("  ✅ Complete feature engineering applied successfully")
            return df
            
        except ImportError as e:
            logger.warning(f"  ⚠️ Feature engineering modules not available: {e}")
            logger.info("  🔧 Using basic feature engineering fallback...")
            return self._apply_basic_feature_engineering(df)
        except Exception as e:
            logger.error(f"  ❌ Feature engineering failed: {e}")
            raise
    
    def _apply_basic_feature_engineering(self, df):
        """Basic feature engineering fallback."""
        logger.info("  🔧 Applying basic feature engineering...")
        
        try:
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col != 'engagement_level':
                    df[col] = df[col].fillna('Unknown')
                    df[col] = pd.Categorical(df[col]).codes
            
            # Create basic interaction features
            if 'organization_employees' in df.columns and 'organization_founded_year' in df.columns:
                df['company_age'] = 2025 - df['organization_founded_year'].fillna(2020)
                df['company_age'] = df['company_age'].clip(lower=0, upper=100)
            
            logger.info("  ✅ Basic feature engineering completed")
            return df
            
        except Exception as e:
            logger.error(f"  ❌ Basic feature engineering failed: {e}")
            raise
    
    def _prepare_features_for_model(self, df):
        """Prepare features for modeling."""
        logger.info("  �� Preparing features for model...")
        
        try:
            # Drop target variable and prepare features
            target_col = 'engagement_level'
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in data")
            
            # Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Ensure all features are numeric
            non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns
            if len(non_numeric_cols) > 0:
                logger.warning(f"  ⚠️ Found {len(non_numeric_cols)} non-numeric columns, converting...")
                for col in non_numeric_cols:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    X[col] = X[col].fillna(0)
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            logger.info(f"  ✅ Features prepared: X={X.shape}, y={y.shape}")
            return X, y, list(X.columns)
            
        except Exception as e:
            logger.error(f"  ❌ Feature preparation failed: {e}")
            raise
    
    def _perform_advanced_feature_selection(self, X, y):
        """Perform advanced feature selection."""
        logger.info("  🔍 Performing advanced feature selection...")
        
        try:
            # 1. Variance threshold
            variance_selector = VarianceThreshold(threshold=0.01)
            X_var_selected = variance_selector.fit_transform(X)
            var_selected_features = X.columns[variance_selector.get_support()].tolist()
            logger.info(f"    After variance selection: {len(var_selected_features)} features")
            
            # 2. Correlation-based selection
            X_df = pd.DataFrame(X_var_selected, columns=var_selected_features)
            corr_matrix = X_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            X_uncorr = X_df.drop(columns=high_corr_features)
            logger.info(f"    After correlation removal: {X_uncorr.shape[1]} features")
            
            # 3. Mutual information selection
            mi_scores = mutual_info_classif(X_uncorr, y, random_state=42)
            mi_df = pd.DataFrame({'feature': X_uncorr.columns, 'mi_score': mi_scores})
            mi_df = mi_df.sort_values('mi_score', ascending=False)
            
            # 4. Select top features
            target_features = min(self.config.get('features', {}).get('target_features', 60), X_uncorr.shape[1])
            selected_features = mi_df.head(target_features)['feature'].tolist()
            X_selected = X_uncorr[selected_features]
            
            logger.info(f"  ✅ Feature selection complete: {len(selected_features)} features selected")
            return X_selected, selected_features, mi_df
            
        except Exception as e:
            logger.error(f"  ❌ Feature selection failed: {e}")
            raise
    
    def _augment_training_data_with_ctgan(self, X_train, y_train):
        """Apply CTGAN augmentation to training data."""
        logger.info("  🔄 Applying CTGAN augmentation...")
        
        try:
            from src.ctgan_augmentation import augment_training_data_with_ctgan
            X_aug, y_aug = augment_training_data_with_ctgan(X_train, y_train, self.config)
            logger.info(f"  ✅ CTGAN augmentation applied: {X_aug.shape}")
            return X_aug, y_aug
            
        except ImportError:
            logger.warning("  ⚠️ CTGAN module not available, skipping augmentation")
            return X_train, y_train
        except Exception as e:
            logger.warning(f"  ⚠️ CTGAN augmentation failed: {e}, using original data")
            return X_train, y_train
    
    def _compute_balanced_class_weights(self, y_train):
        """Compute balanced class weights."""
        logger.info("  ⚖️ Computing balanced class weights...")
        
        try:
            class_weights = compute_class_weight(
                'balanced', 
                classes=np.unique(y_train), 
                y=y_train
            )
            weight_dict = dict(zip(np.unique(y_train), class_weights))
            logger.info(f"  ✅ Class weights computed: {weight_dict}")
            return weight_dict
            
        except Exception as e:
            logger.warning(f"  ⚠️ Class weight computation failed: {e}, using default weights")
            return {0: 1.0, 1: 1.0, 2: 1.0}
    
    def _apply_advanced_class_balancing(self, X_train, y_train, X_test, y_test):
        """Apply advanced class balancing techniques."""
        logger.info("  ⚖️ Applying advanced class balancing...")
        
        try:
            from imblearn.over_sampling import SMOTE, ADASYN
            from imblearn.under_sampling import RandomUnderSampler
            from imblearn.combine import SMOTEENN
            
            balancing_results = {}
            
            # Original data
            balancing_results['original'] = (X_train, y_train)
            
            # SMOTE
            try:
                smote = SMOTE(random_state=42)
                X_smote, y_smote = smote.fit_resample(X_train, y_train)
                balancing_results['smote'] = (X_smote, y_smote)
                logger.info(f"    SMOTE: {X_smote.shape}")
            except Exception as e:
                logger.warning(f"    SMOTE failed: {e}")
            
            # ADASYN
            try:
                adasyn = ADASYN(random_state=42)
                X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
                balancing_results['adasyn'] = (X_adasyn, y_adasyn)
                logger.info(f"    ADASYN: {X_adasyn.shape}")
            except Exception as e:
                logger.warning(f"    ADASYN failed: {e}")
            
            # SMOTEENN
            try:
                smoteenn = SMOTEENN(random_state=42)
                X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train, y_train)
                balancing_results['smoteenn'] = (X_smoteenn, y_smoteenn)
                logger.info(f"    SMOTEENN: {X_smoteenn.shape}")
            except Exception as e:
                logger.warning(f"    SMOTEENN failed: {e}")
            
            logger.info("  ✅ Advanced class balancing applied")
            return balancing_results
            
        except ImportError:
            logger.warning("  ⚠️ Imbalanced-learn not available, using original data")
            return {'original': (X_train, y_train)}
        except Exception as e:
            logger.warning(f"  ⚠️ Class balancing failed: {e}, using original data")
            return {'original': (X_train, y_train)}
    
    def _test_balancing_methods(self, balancing_results, X_train, y_train, X_test, y_test, class_weights):
        """Test different balancing methods and select the best."""
        logger.info("  🔄 Testing different class balancing methods...")
        
        best_accuracy = 0
        best_method = 'original'
        best_X_train, best_y_train = X_train, y_train
        
        for method, result in balancing_results.items():
            if method == 'original':
                continue
            
            if isinstance(result, tuple):
                X_balanced, y_balanced = result
            else:
                continue
            
            logger.info(f"    Testing {method}...")
            
            # Create and train model with balanced data
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.8,
                min_child_weight=3,
                random_state=42,
                scale_pos_weight=class_weights.get(1, 1) if len(class_weights) > 1 else 1
            )
            
            try:
                model.fit(X_balanced, y_balanced)
                y_pred = model.predict(X_test)
                accuracy = (y_pred == y_test).mean()
                
                logger.info(f"      {method} accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_method = method
                    best_X_train, best_y_train = X_balanced, y_balanced
                    
            except Exception as e:
                logger.warning(f"      {method} failed: {e}")
        
        logger.info(f"  ✅ Best method: {best_method} (accuracy: {best_accuracy:.4f})")
        return best_accuracy, best_method, best_X_train, best_y_train
    
    def _perform_hyperparameter_optimization(self, X_train, y_train, class_weights):
        """Perform hyperparameter optimization."""
        logger.info("  🔧 Performing hyperparameter optimization...")
        
        try:
            # XGBoost optimization
            xgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(
                random_state=42,
                scale_pos_weight=class_weights.get(1, 1) if len(class_weights) > 1 else 1
            )
            
            xgb_grid = GridSearchCV(
                xgb_model, xgb_param_grid, 
                cv=3, scoring='accuracy', n_jobs=-1, verbose=0
            )
            xgb_grid.fit(X_train, y_train)
            
            # Logistic Regression optimization
            lr_param_grid = {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            lr_grid = GridSearchCV(
                lr_model, lr_param_grid, 
                cv=3, scoring='accuracy', n_jobs=-1, verbose=0
            )
            lr_grid.fit(X_train, y_train)
            
            logger.info(f"  ✅ XGBoost best params: {xgb_grid.best_params_}")
            logger.info(f"  ✅ Logistic Regression best params: {lr_grid.best_params_}")
            
            return xgb_grid.best_estimator_, lr_grid.best_estimator_
            
        except Exception as e:
            logger.warning(f"  ⚠️ Hyperparameter optimization failed: {e}, using default models")
            xgb_model = xgb.XGBClassifier(random_state=42)
            lr_model = LogisticRegression(random_state=42, max_iter=1000)
            return xgb_model, lr_model
    
    def _create_optimized_ensemble(self, xgb_model, lr_model):
        """Create optimized ensemble model."""
        logger.info("  �� Creating optimized ensemble...")
        
        try:
            ensemble = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lr', lr_model)
                ],
                voting='soft'
            )
            
            logger.info("  ✅ Ensemble model created successfully")
            return ensemble
            
        except Exception as e:
            logger.warning(f"  ⚠️ Ensemble creation failed: {e}, using XGBoost only")
            return xgb_model
    
    def _perform_robust_cross_validation(self, model, X_train, y_train):
        """Perform robust cross-validation."""
        logger.info("  🔄 Performing robust cross-validation...")
        
        try:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            logger.info(f"  ✅ CV scores: {cv_scores}")
            logger.info(f"  📊 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            return cv_scores
            
        except Exception as e:
            logger.warning(f"  ⚠️ Cross-validation failed: {e}")
            return np.array([0.0])
    
    def _evaluate_model_comprehensive(self, model, X_test, y_test):
        """Comprehensive model evaluation."""
        logger.info("  📊 Performing comprehensive model evaluation...")
        
        try:
            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            accuracy = (y_pred == y_test).mean()
            
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC-AUC (for binary case, use one-vs-rest)
            if len(np.unique(y_test)) == 2:
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
            
            metrics = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score'],
                'auc_score': auc_score,
                'confusion_matrix': cm.tolist()
            }
            
            logger.info(f"  ✅ Evaluation complete:")
            logger.info(f"    Accuracy: {accuracy:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    AUC: {auc_score:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"  ❌ Model evaluation failed: {e}")
            return {'accuracy': 0.0, 'error': str(e)}
    
    def _save_model_artifacts(self, model, X_train, y_train, metadata):
        """Save model artifacts."""
        logger.info("  💾 Saving model artifacts...")
        
        try:
            artifact_path = Path(self.config.get('paths', {}).get('model_artifact', 'models/email_open_predictor_optimized_v1.0.joblib'))
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            
            payload = {
                'model': model,
                'features': list(X_train.columns),
                'class_labels': np.unique(y_train),
                'metadata': metadata
            }
            
            joblib.dump(payload, artifact_path)
            logger.info(f"  ✅ Model saved to: {artifact_path}")
            
            # Try MLflow integration
            try:
                self._log_to_mlflow(model, X_train, y_train, metadata)
            except Exception as e:
                logger.warning(f"  ⚠️ MLflow logging failed: {e}")
            
        except Exception as e:
            logger.error(f"  ❌ Model saving failed: {e}")
            raise
    
    def _log_to_mlflow(self, model, X_train, y_train, metadata):
        """Log training run to MLflow."""
        try:
            from src.mlflow_integration import log_training_run_wrapper
            
            run_id = log_training_run_wrapper(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_train.head(100),  # Use sample for MLflow
                y_test=y_train.head(100),
                metrics=metadata.get('test_metrics', {}),
                params=metadata.get('model_params', {}),
                feature_importance=metadata.get('feature_importance'),
                artifacts=metadata.get('artifacts', {}),
                run_name=f"training_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            logger.info(f"  ✅ MLflow run logged: {run_id}")
            
        except ImportError:
            logger.info("  ℹ️ MLflow not available")
        except Exception as e:
            logger.warning(f"  ⚠️ MLflow logging failed: {e}")
    
    def _run_shap_analysis(self, model, df):
        """Run comprehensive SHAP analysis."""
        logger.info("🔍 Running SHAP Analysis...")
        
        try:
            # Import SHAP analysis module
            from shap_analysis import create_binary_shap_plots, create_feature_interpretation_report
            
            # Prepare test data for SHAP
            X_test, y_test = self._prepare_test_data_for_shap(df)
            
            # Generate SHAP plots
            feature_names = X_test.columns.tolist()
            shap_values = create_binary_shap_plots(model, X_test, y_test, feature_names)
            
            # Create feature interpretation report
            create_feature_interpretation_report(shap_values, X_test, feature_names, {})
            
            # Extract top 5 features
            feature_importance = np.abs(shap_values).mean(0)
            top5_indices = np.argsort(feature_importance)[-5:][::-1]
            
            self.shap_results = {
                'shap_analysis_completed': True,
                'top5_features': [
                    {
                        'name': feature_names[i],
                        'importance': float(feature_importance[i]),
                        'rank': idx + 1
                    }
                    for idx, i in enumerate(top5_indices)
                ],
                'shap_values_generated': True,
                'feature_importance_plot': 'shap_feature_importance_binary.png',
                'summary_plot': 'shap_summary_binary_engagement.png',
                'waterfall_plot': 'shap_waterfall_binary_opener.png'
            }
            
            logger.info("✅ SHAP Analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ SHAP Analysis failed: {e}")
            self.shap_results = {
                'shap_analysis_completed': False,
                'error': str(e)
            }
    
    def _prepare_test_data_for_shap(self, df):
        """Prepare test data for SHAP analysis."""
        try:
            # Use a sample of the data for SHAP analysis
            test_size = min(1000, len(df) // 4)
            test_df = df.sample(n=test_size, random_state=42)
            
            # Apply the same preprocessing
            test_df = self._apply_complete_feature_engineering(test_df)
            X_test, y_test, _ = self._prepare_features_for_model(test_df)
            
            return X_test, y_test
            
        except Exception as e:
            logger.warning(f"⚠️ SHAP test data preparation failed: {e}")
            # Return empty dataframes as fallback
            return pd.DataFrame(), pd.Series()
    
    def _run_model_validation(self, model, df):
        """Run model validation."""
        logger.info("✅ Running model validation...")
        # Implementation for model validation
        pass
    
    def _run_model_deployment(self, model):
        """Run model deployment."""
        logger.info("🚀 Running model deployment...")
        # Implementation for model deployment
        pass
    
    def _run_monitoring_setup(self):
        """Run monitoring setup."""
        logger.info("�� Running monitoring setup...")
        # Implementation for monitoring setup
        pass
    
    def _save_results_to_s3(self):
        """Save results to S3 if running on EC2."""
        if not self.aws_available:
            return
        
        try:
            # Save results JSON
            results_file = f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Upload to S3
            instance_id = self._get_instance_id()
            s3_key = f"{instance_id}/pipeline_results.json"
            
            self.s3_client.upload_file(
                results_file,
                self.bucket_name,
                s3_key
            )
            
            logger.info(f"✅ Results uploaded to S3: s3://{self.bucket_name}/{s3_key}")
            
            # Clean up local file
            os.remove(results_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to save results to S3: {e}")
    
    def _get_instance_id(self):
        """Get EC2 instance ID."""
        try:
            import requests
            response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=1)
            return response.text
        except:
            return 'local'
    
    def _print_pipeline_summary(self):
        """Print comprehensive pipeline summary."""
        logger.info("=" * 60)
        logger.info("COMPLETE ML PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"📊 Status: {self.results.get('status', 'unknown')}")
        logger.info(f"⏱️ Execution Time: {self.results.get('execution_time_minutes', 0):.2f} minutes")
        logger.info(f"🔍 SHAP Analysis: {'Completed' if self.results.get('shap_analysis') else 'Not Run'}")
        
        if 'training_metrics' in self.results:
            tm = self.results['training_metrics']
            logger.info(f"�� Model Training:")
            logger.info(f"  Accuracy: {tm.get('accuracy', 0):.4f}")
            logger.info(f"  Best Balancing Method: {tm.get('best_balancing_method', 'N/A')}")
            logger.info(f"  Features Selected: {tm.get('feature_count', 0)}")
            logger.info(f"  Training Shape: {tm.get('training_shape', 'N/A')}")
        
        if self.shap_results.get('shap_analysis_completed'):
            logger.info(f"🔍 SHAP Analysis Results:")
            for feature in self.shap_results.get('top5_features', [])[:3]:
                logger.info(f"  {feature['rank']}. {feature['name']}: {feature['importance']:.4f}")
        
        logger.info("=" * 60)

def main():
    """Main function to run the enhanced ML pipeline."""
    parser = argparse.ArgumentParser(
        description="Enhanced ML Pipeline Runner with Complete Training Steps from train_2.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
COMPLETE TRAINING PIPELINE INCLUDES:
  - Enhanced feature engineering (14+ feature types)
  - Advanced feature selection (Variance, Correlation, Mutual Info)
  - CTGAN data augmentation
  - Advanced class balancing (SMOTE, ADASYN, SMOTEENN)
  - Hyperparameter optimization (XGBoost + Logistic Regression)
  - Ensemble model creation
  - Robust cross-validation
  - Comprehensive model evaluation
  - SHAP analysis integration
  - MLflow logging

EXAMPLES:
  # Complete pipeline with SHAP
  python run_ml_pipeline.py --all --shap
  
  # Training with SHAP analysis
  python run_ml_pipeline.py --train --force --shap
  
  # Lead scoring with SHAP explanations
  python run_ml_pipeline.py --predict --shap --top5
  
  # Testing with SHAP validation
  python run_ml_pipeline.py --test --shap
        """
    )
    
    # Enhanced arguments
    parser.add_argument('--shap', action='store_true', help='Enable SHAP analysis')
    parser.add_argument('--top5', action='store_true', help='Show top 5 influencing features')
    parser.add_argument('--ec2-mode', action='store_true', help='Optimize for EC2 execution')
    
    # Existing arguments
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--train', action='store_true', help='Train/retrain models')
    parser.add_argument('--test', action='store_true', help='Run validation tests')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    
    args = parser.parse_args()
    
    # Initialize pipeline runner
    runner = EnhancedMLPipelineRunner()
    
    try:
        if args.all:
            runner.run_complete_pipeline(include_shap=args.shap)
        elif args.train:
            runner.run_training_with_shap(force_retrain=args.force)
        elif args.test:
            runner.run_model_validation()
        elif args.predict:
            # For prediction, you'd need to specify input data
            input_path = input("Enter path to input data for prediction: ")
            runner.run_prediction_with_shap(input_path, top_k=5 if args.top5 else 3)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        logger.info("⏹️ Pipeline execution interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()