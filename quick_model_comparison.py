"""
Quick Model Comparison Script for B2B Email Marketing

This script provides a faster comparison of the top 5 most important models
against your XGBoost baseline. Use this for quick testing and validation.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Import key models
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Use the same configuration as the main XGBoost script
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

def load_and_engineer_features():
    """Load data and apply feature engineering (simplified from main script)."""
    print(f"Loading data from '{CSV_FILE_PATH}'...")
    
    if not CSV_FILE_PATH.is_file():
        print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE_PATH, on_bad_lines='warn', low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")

    # Create target variable
    conditions = [
        ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),
        (df['email_open_count'] > 0)
    ]
    choices = [2, 1]
    df[TARGET_VARIABLE] = np.select(conditions, choices, default=0)

    # Print class distribution
    class_counts = df[TARGET_VARIABLE].value_counts().sort_index()
    print("\nClass Distribution:")
    for class_val, count in class_counts.items():
        class_name = ['No Engagement', 'Opener', 'Clicker'][class_val]
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count:,} ({percentage:.1f}%)")

    # Basic feature engineering
    df['combined_text'] = ""
    for col in TEXT_COLS_FOR_FEATURE:
        if col in df.columns:
            df['combined_text'] += df[col].fillna('') + ' '

    # Timestamp features
    if 'timestamp_created' in df.columns:
        df['timestamp_created'] = pd.to_datetime(df['timestamp_created'], errors='coerce')
        df['created_day_of_week'] = df['timestamp_created'].dt.dayofweek
        df['created_month'] = df['timestamp_created'].dt.month
        df['created_hour'] = df['timestamp_created'].dt.hour

    # JSONB presence features
    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)

    # Interaction features
    if 'organization_industry' in df.columns and 'seniority' in df.columns:
        df['industry_seniority'] = df['organization_industry'].fillna('Unknown') + '_' + df['seniority'].fillna('Unknown')

    # Separate features and target
    y = df[TARGET_VARIABLE]
    all_source_cols = ['email_open_count', 'email_click_count', 'email_reply_count']
    all_cols_to_drop = list(set(COLS_TO_DROP + TIMESTAMP_COLS + JSONB_COLS + TEXT_COLS_FOR_FEATURE + all_source_cols + [TARGET_VARIABLE]))
    X = df.drop(columns=[col for col in all_cols_to_drop if col in df.columns], errors='ignore')

    return X, y

def create_preprocessing_pipeline(X):
    """Create a simple preprocessing pipeline."""
    
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = [col for col in CATEGORICAL_COLS + ['industry_seniority'] if col in X.columns]
    numeric_features = [col for col in numeric_features if col not in categorical_features]
    
    print(f"Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical, 1 text")
    
    # Create transformers
    transformers = []
    
    if numeric_features:
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', max_categories=50, sparse_output=False))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
    
    text_transformer = TfidfVectorizer(
        max_features=800, 
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    transformers.append(('text', text_transformer, 'combined_text'))
    
    return ColumnTransformer(transformers=transformers, remainder='drop')

def get_quick_models():
    """Define the top 5 models for quick comparison."""
    return {
        'XGBoost (Baseline)': xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,
            max_depth=6,
            n_estimators=200,  # Reduced for speed
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss'
        ),
        
        'Random Forest': RandomForestClassifier(
            n_estimators=200,  # Reduced for speed
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,  # Reduced for speed
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        ),
        
        'Logistic Regression': LogisticRegression(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        
        'Support Vector Machine': SVC(
            C=1.0,
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }

def evaluate_models_quickly(X, y):
    """Run quick model comparison."""
    print("\n" + "="*60)
    print("QUICK MODEL COMPARISON")
    print("="*60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"Training: {X_train.shape[0]:,} samples | Testing: {X_test.shape[0]:,} samples")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(X)
    
    # Get models
    models = get_quick_models()
    results = []
    
    print(f"\nTraining {len(models)} models...")
    
    for name, model in models.items():
        print(f"\n🔄 {name}...")
        
        try:
            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('feature_selection', SelectKBest(f_classif, k=1000)),
                ('classifier', model)
            ])
            
            # Fit and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
            cv_mean = cv_scores.mean()
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'CV-Score': cv_mean,
                'y_pred': y_pred
            })
            
            print(f"   ✅ Accuracy: {accuracy:.4f} | F1: {f1:.4f} | CV: {cv_mean:.4f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            continue
    
    return results, y_test

def visualize_quick_results(results, y_test):
    """Create quick visualization of results."""
    
    if not results:
        print("No results to visualize!")
        return
    
    # Create results dataframe
    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print("="*60)
    print(df[['Model', 'Accuracy', 'F1-Score', 'CV-Score']].round(4).to_string(index=False))
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Performance comparison
    x_pos = range(len(df))
    bars = ax1.bar(x_pos, df['Accuracy'], alpha=0.7, color='skyblue')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, df['Accuracy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Best model confusion matrix
    best_model = results[0]  # First result after sorting
    cm = confusion_matrix(y_test, best_model['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
               xticklabels=['No Engagement', 'Opener', 'Clicker'],
               yticklabels=['No Engagement', 'Opener', 'Clicker'])
    ax2.set_title(f'Best Model: {best_model["Model"]}\nAccuracy: {best_model["Accuracy"]:.4f}')
    ax2.set_ylabel('Actual')
    ax2.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('quick_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary insights
    best = df.iloc[0]
    worst = df.iloc[-1]
    improvement = best['Accuracy'] - worst['Accuracy']
    
    print(f"\n🏆 BEST MODEL: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
    print(f"📊 PERFORMANCE RANGE: {improvement:.4f} ({improvement*100:.1f}% improvement)")
    
    if 'XGBoost' in best['Model']:
        print("✅ XGBoost maintains its lead!")
    else:
        print(f"🔄 {best['Model']} outperforms XGBoost baseline!")
    
    print(f"\n📁 Visualization saved as: quick_model_comparison.png")

def main():
    """Main function for quick model comparison."""
    print("=== QUICK MODEL COMPARISON FOR B2B EMAIL MARKETING ===")
    print("Testing top 5 models against your XGBoost baseline...\n")
    
    # Load and prepare data
    X, y = load_and_engineer_features()
    
    # Run comparison
    results, y_test = evaluate_models_quickly(X, y)
    
    # Visualize results
    visualize_quick_results(results, y_test)
    
    print(f"\n{'='*60}")
    print("QUICK COMPARISON COMPLETE!")
    print("For comprehensive analysis, run: python model_comparison.py")
    print("="*60)

if __name__ == "__main__":
    main() 