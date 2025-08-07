"""
Model Comparison Script for B2B Email Marketing Engagement Prediction

This script compares multiple machine learning models against the XGBoost baseline,
using the same data processing pipeline and evaluation metrics to ensure fair comparison.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# ML Models to compare
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional TensorFlow import - handle gracefully if not available
TENSORFLOW_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow successfully imported - Neural Network will be included")
except ImportError as e:
    print(f"⚠️  TensorFlow not available - Neural Network will be skipped")
    print(f"    Error: {str(e)}")
    tf = None

# Import configuration from existing XGBoost script
CSV_FILE_PATH = Path("merged_contacts.csv")
TARGET_VARIABLE = "engagement_level"
TEXT_COLS_FOR_FEATURE = ['campaign_id', 'email_subjects', 'email_bodies']
TIMESTAMP_COLS = ['timestamp_created', 'timestamp_last_contact', 'retrieval_timestamp', 'enriched_at', 'inserted_at', 'last_contacted_from']
CATEGORICAL_COLS = ['title', 'seniority', 'organization_industry', 'country', 'city', 'enrichment_status', 'upload_method', 'api_status', 'state']
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

COLS_TO_DROP = [
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',
    'email_reply_count', 'email_opened_variant', 'email_opened_step', 
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click', 
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated', 
    'status_summary', 'email_clicked_variant', 'email_clicked_step',
    'personalization', 'payload', 'list_id', 'assigned_to', 'campaign', 'uploaded_by_user',
    'auto_variant_select', 'verification_status'
]

def load_data(file_path: Path) -> pd.DataFrame:
    """Load and standardize datetime columns."""
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
    print(f"Data successfully loaded. Shape: {df.shape}")

    # Standardize timestamp columns
    all_potential_timestamps = list(set(TIMESTAMP_COLS + [col for col in COLS_TO_DROP if 'timestamp' in col]))
    for col in all_potential_timestamps:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].dt.tz_localize('UTC', nonexistent='NaT')
    for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
        if df[col].dt.tz != 'UTC':
            df[col] = df[col].dt.tz_convert('UTC')
    return df

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply the same feature engineering as the XGBoost script."""
    print("Starting feature engineering...")

    # Define target variable
    if 'email_reply_count' not in df.columns or 'email_click_count' not in df.columns or 'email_open_count' not in df.columns:
        print("Error: Source columns for engagement level not found.")
        sys.exit(1)

    conditions = [
        ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),
        (df['email_open_count'] > 0)
    ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)

    # Print class distribution
    print("\nClass Distribution:")
    class_counts = df[TARGET_VARIABLE].value_counts().sort_index()
    for class_val, count in class_counts.items():
        percentage = (count / len(df)) * 100
        class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
        print(f"  {class_name} ({class_val}): {count:,} ({percentage:.1f}%)")

    # Combine text columns
    df['combined_text'] = ""
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '

    # Engineer timestamp features
    if 'timestamp_created' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp_created']):
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_hour'] = df['timestamp_created'].dt.hour
        ref_date = df['timestamp_created']
        for col in TIMESTAMP_COLS:
            if col in df.columns and col != 'timestamp_created' and pd.api.types.is_datetime64_any_dtype(df[col]):
                feature_name = f"days_between_{col.replace('timestamp_', '')}_and_created"
                df[feature_name] = (df[col] - ref_date).dt.days

    # Engineer JSONB presence features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)

    # Create interaction features
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        df['industry_seniority'] = df['organization_industry'].fillna('Unknown') + '_' + df['seniority'].fillna('Unknown')
    
    if 'country' in df.columns and 'organization_industry' in df.columns:
        df['country_industry'] = df['country'].fillna('Unknown') + '_' + df['organization_industry'].fillna('Unknown')

    # Separate features and target
    y = df[TARGET_VARIABLE]
    all_source_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS_FOR_FEATURE + all_source_cols + [TARGET_VARIABLE]))
    X = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')

    print(f"Feature engineering complete. Shape: {X.shape}")
    return X, y

def create_preprocessing_pipeline(numeric_features, categorical_features, max_text_features=1000):
    """Create preprocessing pipeline that can be reused across models."""
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())  # Important for SVM, NN, etc.
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False))
    ])
    
    # Text preprocessing
    text_transformer = TfidfVectorizer(
        max_features=max_text_features, 
        stop_words='english', 
        lowercase=True, 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    # Combine transformers
    transformers = []
    if numeric_features:
        transformers.append(('num', numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', categorical_transformer, categorical_features))
    transformers.append(('text', text_transformer, 'combined_text'))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    return preprocessor

# Define Neural Network wrapper only if TensorFlow is available
if TENSORFLOW_AVAILABLE:
    class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
        """Wrapper to make Keras models compatible with sklearn."""
        
        def __init__(self, input_dim=None, random_state=42):
            self.input_dim = input_dim
            self.random_state = random_state
            self.model = None
            self.classes_ = None
            
        def _create_model(self, input_dim):
            """Create and compile the neural network model."""
            tf.random.set_seed(self.random_state)
            
            model = Sequential([
                Dense(512, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.2),
                Dense(3, activation='softmax')  # 3 classes
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            return model
        
        def fit(self, X, y):
            """Fit the neural network."""
            self.classes_ = np.unique(y)
            
            if self.input_dim is None:
                self.input_dim = X.shape[1]
                
            self.model = self._create_model(self.input_dim)
            
            # Early stopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            )
            
            # Train the model
            self.model.fit(
                X, y,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=0
            )
            
            return self
        
        def predict(self, X):
            """Make predictions."""
            predictions = self.model.predict(X, verbose=0)
            return np.argmax(predictions, axis=1)
        
        def predict_proba(self, X):
            """Predict class probabilities."""
            return self.model.predict(X, verbose=0)
else:
    # Dummy class if TensorFlow is not available
    class KerasClassifierWrapper:
        """Placeholder class when TensorFlow is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("TensorFlow is not available - Neural Network cannot be used")

def get_model_configurations(class_weights=None):
    """Define all models to compare with their hyperparameters."""
    
    models = {
        'XGBoost': {
            'model': xgb.XGBClassifier(
                objective='multi:softmax',
                num_class=3,
                max_depth=6,
                n_estimators=300,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'feature_selection': True
        },
        
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'feature_selection': True
        },
        
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'feature_selection': True
        },
        
        'Extra Trees': {
            'model': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'feature_selection': True
        },
        
        'Support Vector Machine': {
            'model': SVC(
                C=1.0,
                kernel='linear',  # Changed from 'rbf' to 'linear' for efficiency
                class_weight='balanced',
                probability=True,
                random_state=42,
                max_iter=1000  # Add iteration limit for stability
            ),
            'feature_selection': True
        },
        
        'Logistic Regression': {
            'model': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'feature_selection': False
        },
        
        'AdaBoost': {
            'model': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            'feature_selection': True
        },
        
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                n_jobs=-1
            ),
            'feature_selection': True
        }
    }
    
    # Add Neural Network only if TensorFlow is available
    if TENSORFLOW_AVAILABLE:
        models['Neural Network'] = {
            'model': KerasClassifierWrapper(random_state=42),
            'feature_selection': True
        }
    
    return models

def create_model_pipeline(model_config, preprocessor, use_feature_selection=True, k_features=1000):
    """Create a complete pipeline for a model."""
    
    pipeline_steps = [('preprocessor', preprocessor)]
    
    if use_feature_selection and model_config['feature_selection']:
        # Use fewer features for computationally intensive models
        model_name = model_config.get('model_name', '')
        if 'SVM' in str(type(model_config['model'])) or 'KNeighbors' in str(type(model_config['model'])):
            k_features = min(500, k_features)  # Reduce features for SVM and KNN
        
        pipeline_steps.append(('feature_selection', SelectKBest(f_classif, k=k_features)))
    
    pipeline_steps.append(('classifier', model_config['model']))
    
    return Pipeline(pipeline_steps)

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model."""
    print(f"\nTraining {model_name}...")
    
    try:
        import time
        start_time = time.time()
        
        # Fit the model with progress indication
        print(f"  🔄 Fitting {model_name}...")
        pipeline.fit(X_train, y_train)
        fit_time = time.time() - start_time
        print(f"  ⏱️  Training completed in {fit_time:.1f} seconds")
        
        # Make predictions
        print(f"  🔄 Making predictions...")
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Cross-validation score (with timeout protection)
        print(f"  🔄 Running cross-validation...")
        cv_start = time.time()
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy', n_jobs=1)  # Use single job for stability
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            cv_time = time.time() - cv_start
            print(f"  ⏱️  Cross-validation completed in {cv_time:.1f} seconds")
        except Exception as cv_error:
            print(f"  ⚠️  Cross-validation failed: {str(cv_error)}")
            cv_mean, cv_std = accuracy, 0.0  # Fallback to test accuracy
        
        total_time = time.time() - start_time
        print(f"  ✅ {model_name} completed successfully in {total_time:.1f} seconds")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    CV Score: {cv_mean:.4f} (±{cv_std:.4f})")
        
        return {
            'model_name': model_name,
            'pipeline': pipeline,
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'training_time': total_time,
            'status': 'success'
        }
        
    except MemoryError:
        print(f"  ❌ {model_name} failed: Out of memory")
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': 'Out of memory - dataset too large for this model'
        }
    except Exception as e:
        print(f"  ❌ {model_name} failed: {str(e)}")
        return {
            'model_name': model_name,
            'status': 'failed',
            'error': str(e)
        }

def compare_models(X, y):
    """Compare all models and return results."""
    print("=== MODEL COMPARISON FRAMEWORK ===")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]:,} samples")
    print(f"  Test set: {X_test.shape[0]:,} samples")
    
    # Identify feature types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in CATEGORICAL_COLS + ['industry_seniority', 'country_industry'] if col in X.columns]
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"\nFeature breakdown:")
    print(f"  Numeric features: {len(numeric_features)}")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Text features: 1 (combined_text)")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    print(f"\nClass weights calculated:")
    for class_val, weight in weight_dict.items():
        class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
        print(f"  {class_name}: {weight:.3f}")
    
    # Get model configurations
    models = get_model_configurations(weight_dict)
    
    # Train and evaluate all models
    results = []
    successful_results = []
    
    print(f"\n{'='*60}")
    print("TRAINING AND EVALUATING MODELS")
    print(f"{'='*60}")
    
    for model_name, model_config in models.items():
        try:
            pipeline = create_model_pipeline(model_config, preprocessor)
            result = evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name)
            results.append(result)
            
            if result['status'] == 'success':
                successful_results.append(result)
            else:
                print(f"  ⚠️  {model_name} will be excluded from final comparison")
                
        except Exception as e:
            print(f"  ❌ Failed to create pipeline for {model_name}: {str(e)}")
            results.append({
                'model_name': model_name,
                'status': 'failed',
                'error': f'Pipeline creation failed: {str(e)}'
            })
    
    # Summary of results
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful models: {len(successful_results)}/{len(models)}")
    print(f"❌ Failed models: {len(models) - len(successful_results)}")
    
    if successful_results:
        best_model = max(successful_results, key=lambda x: x['accuracy'])
        print(f"🏆 Best performing model: {best_model['model_name']} ({best_model['accuracy']:.4f} accuracy)")
    
    return successful_results, y_test

def visualize_results(results, y_test):
    """Create comprehensive visualizations comparing all models."""
    
    if not results:
        print("No successful results to visualize.")
        return
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Accuracy': r['accuracy'],
            'F1-Macro': r['f1_macro'],
            'F1-Weighted': r['f1_weighted'],
            'Precision': r['precision_macro'],
            'Recall': r['recall_macro'],
            'CV-Mean': r['cv_mean'],
            'CV-Std': r['cv_std']
        }
        for r in results
    ])
    
    # Sort by accuracy
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*80}")
    print(results_df.round(4).to_string(index=False))
    
    # 1. Performance comparison bar plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    bars1 = ax1.bar(results_df['Model'], results_df['Accuracy'], color='skyblue', alpha=0.7)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # F1-Score comparison
    bars2 = ax2.bar(results_df['Model'], results_df['F1-Weighted'], color='lightgreen', alpha=0.7)
    ax2.set_title('F1-Score (Weighted) Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1-Score (Weighted)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Cross-validation scores with error bars
    ax3.bar(results_df['Model'], results_df['CV-Mean'], 
            yerr=results_df['CV-Std'], capsize=5, color='orange', alpha=0.7)
    ax3.set_title('Cross-Validation Accuracy (±std)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('CV Accuracy')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(axis='y', alpha=0.3)
    
    # Precision vs Recall
    ax4.scatter(results_df['Recall'], results_df['Precision'], 
               s=100, alpha=0.7, c=results_df['Accuracy'], cmap='viridis')
    for i, model in enumerate(results_df['Model']):
        ax4.annotate(model, (results_df['Recall'].iloc[i], results_df['Precision'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    ax4.set_xlabel('Recall (Macro)')
    ax4.set_ylabel('Precision (Macro)')
    ax4.set_title('Precision vs Recall (colored by Accuracy)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Detailed metrics heatmap
    plt.figure(figsize=(12, 8))
    metrics_for_heatmap = results_df.set_index('Model')[['Accuracy', 'F1-Macro', 'F1-Weighted', 'Precision', 'Recall']]
    sns.heatmap(metrics_for_heatmap.T, annot=True, cmap='RdYlBu_r', center=0.5, 
                fmt='.3f', cbar_kws={'label': 'Score'})
    plt.title('Model Performance Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig('model_comparison_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Confusion matrices for top models
    top_models = results[:min(3, len(results))]  # Top 3 by accuracy or fewer if less available
    
    if top_models:
        fig, axes = plt.subplots(1, len(top_models), figsize=(5*len(top_models), 4))
        if len(top_models) == 1:
            axes = [axes]
        
        for i, result in enumerate(top_models):
            cm = confusion_matrix(y_test, result['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['No Engagement', 'Opener', 'Clicker'],
                       yticklabels=['No Engagement', 'Opener', 'Clicker'])
            axes[i].set_title(f'{result["model_name"]}\nAccuracy: {result["accuracy"]:.4f}')
            axes[i].set_ylabel('Actual')
            axes[i].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('top_models_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("⚠️ No successful models to create confusion matrices")
    
    return results_df

def generate_summary_report(results_df):
    """Generate a comprehensive summary report."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    best_accuracy = results_df.iloc[0]
    best_f1 = results_df.loc[results_df['F1-Weighted'].idxmax()]
    most_stable = results_df.loc[results_df['CV-Std'].idxmin()]
    
    print(f"\n🏆 BEST PERFORMING MODELS:")
    print(f"   Best Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
    print(f"   Best F1-Score: {best_f1['Model']} ({best_f1['F1-Weighted']:.4f})")
    print(f"   Most Stable:   {most_stable['Model']} (CV-Std: {most_stable['CV-Std']:.4f})")
    
    print(f"\n📊 PERFORMANCE ANALYSIS:")
    accuracy_range = results_df['Accuracy'].max() - results_df['Accuracy'].min()
    print(f"   Accuracy range: {accuracy_range:.4f}")
    print(f"   Mean accuracy: {results_df['Accuracy'].mean():.4f}")
    print(f"   Std accuracy: {results_df['Accuracy'].std():.4f}")
    
    # Model categories analysis
    tree_models = results_df[results_df['Model'].isin(['XGBoost', 'Random Forest', 'Extra Trees', 'Gradient Boosting'])]
    if not tree_models.empty:
        print(f"\n🌳 TREE-BASED MODELS:")
        print(f"   Average accuracy: {tree_models['Accuracy'].mean():.4f}")
        print(f"   Best: {tree_models.loc[tree_models['Accuracy'].idxmax(), 'Model']}")
    
    linear_models = results_df[results_df['Model'].isin(['Logistic Regression', 'Support Vector Machine'])]
    if not linear_models.empty:
        print(f"\n📈 LINEAR/KERNEL MODELS:")
        print(f"   Average accuracy: {linear_models['Accuracy'].mean():.4f}")
        print(f"   Best: {linear_models.loc[linear_models['Accuracy'].idxmax(), 'Model']}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    if best_accuracy['Accuracy'] - results_df['Accuracy'].iloc[1] > 0.01:
        print(f"   • {best_accuracy['Model']} shows clear superiority for this dataset")
    else:
        print(f"   • Multiple models perform similarly - consider ensemble methods")
    
    if most_stable['CV-Std'] < 0.01:
        print(f"   • {most_stable['Model']} shows excellent stability across folds")
    
    print(f"\n📁 Generated files:")
    print(f"   • model_comparison_overview.png")
    print(f"   • model_comparison_heatmap.png") 
    print(f"   • top_models_confusion_matrices.png")

def main():
    """Main function to run the model comparison."""
    print("=== B2B EMAIL MARKETING: MODEL COMPARISON ANALYSIS ===")
    
    # Load and process data (same as XGBoost script)
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    
    # Run model comparison
    results, y_test = compare_models(X, y)
    
    if not results:
        print("No models completed successfully!")
        return
    
    # Visualize and analyze results
    results_df = visualize_results(results, y_test)
    generate_summary_report(results_df)
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON COMPLETE!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 