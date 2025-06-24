"""
This script loads contact data, preprocesses it to create features, trains an
XGBoost model to predict email opens, and then uses SHAP to interpret the
model's predictions, generating several explanatory plots.
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from typing import Tuple

# --- Configuration ---
CSV_FILE_PATH = Path("enriched_contacts.csv")
TARGET_VARIABLE = "opened"

# Define column types for processing
# UPDATED: Removed all ambiguous or post-open engagement timestamps.
# These are the only timestamps considered safe pre-engagement signals.
TIMESTAMP_COLS = [
    'timestamp_created',
    'timestamp_last_contact',
    'retrieval_timestamp',
    'enriched_at',
    'inserted_at',
    'last_contacted_from'
]
# These categorical columns will be one-hot encoded for the model.
CATEGORICAL_COLS = [
    'title', 'seniority', 'organization_industry', 'country', 'city',
    'verification_status', 'enrichment_status', 'upload_method', 'api_status',
    'state'
]
# These columns (likely containing JSON) will be converted to a binary 'has_data' feature.
JSONB_COLS = ['employment_history', 'organization_data', 'account_data', 'api_response_raw']

# This list contains all columns to be removed before training the model.
COLS_TO_DROP = [
    # --- Identifiers & High-Cardinality Text (Not useful for generalization) ---
    'id', 'email', 'first_name', 'last_name', 'company_name', 'linkedin_url',
    'website', 'headline', 'company_domain', 'phone', 'apollo_id',
    'apollo_name', 'organization', 'photo_url', 'organization_name',
    'organization_website', 'organization_phone',

    # --- Leaky Features (Contain information about the outcome) ---
    'email_reply_count', 'email_click_count', 'email_opened_variant', 'email_opened_step',
    'timestamp_last_open', 'timestamp_last_reply', 'timestamp_last_click',
    # UPDATED: Added all other potential post-open engagement timestamps
    'timestamp_last_touch', 'timestamp_last_interest_change', 'timestamp_updated',


    # --- Unstructured or Internal-Use Columns ---
    'personalization', 'status_summary', 'payload', 'list_id',
    'assigned_to', 'campaign', 'uploaded_by_user',
]

def load_data(file_path: Path) -> pd.DataFrame:
    """Loads and performs initial datetime standardization on the dataset."""
    print(f"Loading data from '{file_path}'...")
    if not file_path.is_file():
        print(f"Error: The file '{file_path}' was not found.")
        sys.exit(1)

    try:
        df = pd.read_csv(file_path, on_bad_lines='warn', low_memory=False)
        print("Data successfully loaded.")
        # Parse all potential timestamp columns to avoid errors, even those we plan to drop.
        all_potential_timestamps = list(set(TIMESTAMP_COLS + [col for col in COLS_TO_DROP if 'timestamp' in col]))
        for col in all_potential_timestamps:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        # Standardize all parsed dates to the UTC timezone
        for col in df.select_dtypes(include=['datetime64[ns]']).columns:
             df[col] = df[col].dt.tz_localize('UTC', nonexistent='NaT')
        for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
             if df[col].dt.tz != 'UTC':
                df[col] = df[col].dt.tz_convert('UTC')
        return df
    except Exception as e:
        print(f"An error occurred while loading or parsing the CSV: {e}")
        sys.exit(1)

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Applies all feature engineering and preprocessing steps."""
    print("Starting feature engineering...")
    if 'email_open_count' not in df.columns:
        print(f"Error: '{TARGET_VARIABLE}' source column 'email_open_count' not found.")
        sys.exit(1)
    df[TARGET_VARIABLE] = (df['email_open_count'] > 0).astype(int)
    df = df.drop(columns=['email_open_count'])

    # Engineer 'days_since_' features from the safe list of timestamps
    current_time_utc = pd.Timestamp.now(tz='UTC')
    for col in TIMESTAMP_COLS:
        if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
            feature_name = f"days_since_{col.replace('timestamp_', '')}"
            df[feature_name] = (current_time_utc - df[col]).dt.days
    df = df.drop(columns=[col for col in TIMESTAMP_COLS if col in df.columns])

    for col in JSONB_COLS:
        if col in df.columns:
            df[f'has_{col}'] = df[col].notna().astype(int)
    df = df.drop(columns=[col for col in JSONB_COLS if col in df.columns])

    cols_to_onehot = [col for col in CATEGORICAL_COLS if col in df.columns]
    for col in cols_to_onehot:
        df[col] = df[col].fillna('Missing').astype(str)
        if df[col].nunique() > 50:
            top_n = df[col].value_counts().nlargest(49).index
            df[col] = df[col].where(df[col].isin(top_n), 'Other')
    df = pd.get_dummies(df, columns=cols_to_onehot, dtype=int)

    df = df.drop(columns=[col for col in COLS_TO_DROP if col in df.columns], errors='ignore')
    
    if TARGET_VARIABLE not in df.columns:
        print(f"Error: Target variable '{TARGET_VARIABLE}' not found after processing.")
        sys.exit(1)
    
    X = df.drop(columns=[TARGET_VARIABLE])
    y = df[TARGET_VARIABLE]

    X = X.select_dtypes(include=np.number)

    cols_to_drop_all_nan = X.columns[X.isna().all()].tolist()
    if cols_to_drop_all_nan:
        print(f"INFO: Dropping completely empty columns: {cols_to_drop_all_nan}")
        X = X.drop(columns=cols_to_drop_all_nan)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X = pd.DataFrame(X_imputed, index=X.index, columns=X.columns)

    print(f"Feature engineering complete. Final feature shape: {X.shape}")
    return X, y

def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
    """Splits data, trains an XGBoost model with early stopping, and returns it."""
    print("Splitting data and training model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        n_estimators=1000,
        random_state=42,
        early_stopping_rounds=50
    )
    print("Starting model training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    print(f"Model training complete. Best iteration: {model.best_iteration}")
    return model

def generate_shap_plots(model: xgb.XGBClassifier, X: pd.DataFrame, y: pd.Series):
    """Calculates SHAP values and generates key interpretation plots."""
    print("\nCalculating SHAP values...")
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    explainer = shap.TreeExplainer(model)
    shap_values_array = explainer.shap_values(X_train)

    print("Generating SHAP plots...")
    
    plt.figure()
    shap.summary_plot(shap_values_array, X_train, plot_type="bar", max_display=20, show=False)
    plt.title("Mean Absolute SHAP values (Feature Importance)")
    plt.tight_layout()
    plt.show()

    plt.figure()
    shap.summary_plot(shap_values_array, X_train, plot_type="dot", max_display=20, show=False)
    plt.title("SHAP Feature Importance (Beeswarm Plot)")
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function for the script."""
    print(f"--- Script started ---")
    print(f"XGBoost version: {xgb.__version__}, Pandas version: {pd.__version__}")
    df = load_data(CSV_FILE_PATH)
    X, y = engineer_features(df)
    if X.empty:
        print("Error: No features remaining after preprocessing. Exiting.")
        sys.exit(1)
    model = train_model(X, y)
    generate_shap_plots(model, X, y)
    print("\n--- Script finished successfully ---")

if __name__ == "__main__":
    main()