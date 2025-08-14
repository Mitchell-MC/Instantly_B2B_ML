"""
Fast Training Pipeline with Reduced Runtime Complexity

Key changes vs train_2.py:
- Use RandomizedSearchCV (n_iter small) with 3-fold CV
- Remove RFE and heavy SHAP; use MI + correlation pruning (on top-K only)
- Single-model (XGBoost)
- Reintroduced CTGAN augmentation (config-controlled) and SMOTE balancing
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import joblib
import yaml
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb

from src.feature_engineering import (
    enhanced_text_preprocessing,
    advanced_timestamp_features,
    create_interaction_features,
    create_comprehensive_jsonb_features,
    create_comprehensive_organization_data,
    create_advanced_engagement_features,
    create_xgboost_optimized_features,
    handle_outliers,
    encode_categorical_features,
)
from src.ctgan_augmentation import augment_training_data_with_ctgan
from imblearn.over_sampling import SMOTE


def load_config() -> dict:
    config_path = Path("config/main_config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_real_data() -> pd.DataFrame:
    data_path = Path("merged_contacts.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Real data file not found: {data_path}")
    df = pd.read_csv(data_path, on_bad_lines="warn", low_memory=False)
    # Standardize timestamp columns used in feature engineering
    for col in [
        "timestamp_created",
        "timestamp_last_contact",
        "retrieval_timestamp",
        "enriched_at",
        "inserted_at",
        "last_contacted_from",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def create_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    # 3-bucket engagement
    df = df.copy()
    df["engagement_level"] = 0
    mask_bucket2 = (
        (df.get("email_open_count", 0) >= 1)
        & (df.get("email_open_count", 0) <= 2)
        & (df.get("email_click_count", 0) == 0)
        & (df.get("email_reply_count", 0) == 0)
    )
    df.loc[mask_bucket2, "engagement_level"] = 1
    mask_bucket3 = (
        (df.get("email_open_count", 0) >= 3)
        | ((df.get("email_open_count", 0) >= 1) & (df.get("email_click_count", 0) >= 1))
        | ((df.get("email_open_count", 0) >= 1) & (df.get("email_reply_count", 0) >= 1))
    )
    df.loc[mask_bucket3, "engagement_level"] = 2
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = enhanced_text_preprocessing(df)
    df = advanced_timestamp_features(df)
    df = create_interaction_features(df)
    df = create_comprehensive_jsonb_features(df)
    df = create_comprehensive_organization_data(df)
    df = create_advanced_engagement_features(df)
    df = create_xgboost_optimized_features(df)
    df = handle_outliers(df)
    df = encode_categorical_features(df)
    return df


def prepare_features(df: pd.DataFrame, target_col: str = "engagement_level"):
    # Drop obvious ID/text columns and the target if present
    drop_cols = [
        "id",
        "email",
        "first_name",
        "last_name",
        "company_name",
        "linkedin_url",
        "website",
        "headline",
        "company_domain",
        "phone",
        "apollo_id",
        "apollo_name",
        "organization",
        "photo_url",
        "organization_name",
        "organization_website",
        "organization_phone",
        "combined_text",
        target_col,
        # remove raw engagement columns to avoid leakage
        "email_open_count",
        "email_click_count",
        "email_reply_count",
        "email_opened_variant",
        "email_clicked_variant",
        "email_replied_variant",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore").select_dtypes(include=[np.number])
    y = df[target_col].astype(int)
    return X, y


def feature_selection_fast(X: pd.DataFrame, y: pd.Series, top_k: int = 100) -> pd.DataFrame:
    # Remove all-NaN columns
    X = X.loc[:, ~X.isnull().all(axis=0)]

    # Median impute once
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    # Low-variance filter
    var_sel = VarianceThreshold(threshold=0.01)
    X_var = pd.DataFrame(var_sel.fit_transform(X_imp), columns=X_imp.columns[var_sel.get_support()], index=X_imp.index)

    # Mutual information on reduced set
    from sklearn.feature_selection import mutual_info_classif

    mi = mutual_info_classif(X_var, y, random_state=42)
    mi_df = pd.DataFrame({"feature": X_var.columns, "mi": mi}).sort_values("mi", ascending=False)
    top_features = mi_df.head(min(top_k, len(mi_df)))['feature'].tolist()
    X_top = X_var[top_features]

    # Correlation pruning only within top-K
    corr = X_top.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    X_final = X_top.drop(columns=to_drop, errors="ignore")
    return X_final


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    weight_map = {c: w for c, w in zip(classes, class_weights)}
    return y.map(weight_map).values


def build_xgb_random_search(random_state: int = 42, n_iter: int = 20) -> RandomizedSearchCV:
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=300,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=random_state,
        verbosity=0,
    )

    param_distributions = {
        "max_depth": [3, 4, 5, 6],
        "learning_rate": np.linspace(0.03, 0.3, 10),
        "subsample": np.linspace(0.7, 1.0, 7),
        "colsample_bytree": np.linspace(0.7, 1.0, 7),
        "min_child_weight": [1, 2, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="roc_auc_ovr_weighted",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    return search


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except Exception:
        auc = None

    # Save confusion matrix PNG
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix - Fast Model")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig("confusion_matrix_fast.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        pass

    # Save classification report as JSON and TXT
    try:
        with open("classification_report_fast.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        with open("classification_report_fast.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y_test, y_pred))
    except Exception:
        pass

    return {"report": report, "confusion_matrix": cm, "auc": auc}


def main():
    print("🚀 Starting Fast Training Pipeline")
    config = load_config()

    # Load and label data
    df = load_real_data()
    df = create_target_variable(df)

    # Feature engineering
    df = apply_feature_engineering(df)

    # Prepare features
    X_all, y_all = prepare_features(df, target_col="engagement_level")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.25, random_state=config["training"]["random_state"], stratify=y_all
    )

    # Fast feature selection
    top_k = int(config.get("features", {}).get("target_features", 100))
    X_train_sel = feature_selection_fast(X_train, y_train, top_k=top_k)
    X_test_sel = X_test[X_train_sel.columns]

    # Optional CTGAN augmentation (config-controlled)
    try:
        if config.get("augmentation", {}).get("ctgan", {}).get("enabled", False):
            X_train_sel, y_train = augment_training_data_with_ctgan(X_train_sel, y_train, config)
            print(f"🔄 After CTGAN augmentation: X_train={X_train_sel.shape}, y_train={len(y_train)}")
    except Exception as e:
        print(f"⚠️ CTGAN augmentation skipped due to error: {e}")

    # Apply SMOTE once to balance classes
    try:
        smote = SMOTE(random_state=config["training"]["random_state"], k_neighbors=5)
        X_train_sel, y_train = smote.fit_resample(X_train_sel, y_train)
        print(f"🔄 After SMOTE: class_counts={np.bincount(y_train)}")
    except Exception as e:
        print(f"⚠️ SMOTE failed/skipped: {e}")

    # Sample weights (still useful after SMOTE for slight reweighting)
    sample_weight = compute_sample_weights(y_train)

    # Randomized hyperparameter search (3-fold)
    search = build_xgb_random_search(
        random_state=config["training"]["random_state"], n_iter=20
    )
    search.fit(X_train_sel, y_train, sample_weight=sample_weight)

    best_params = search.best_params_
    best_n_estimators = getattr(search.best_estimator_, 'n_estimators', 300)
    print(f"Best params: {best_params}")

    # Final fit with early stopping on a validation split
    X_tr, X_val, y_tr, y_val, sw_tr, sw_val = train_test_split(
        X_train_sel,
        y_train,
        sample_weight,
        test_size=0.2,
        random_state=config["training"]["random_state"],
        stratify=y_train,
    )

    final_model = xgb.XGBClassifier(
        objective="multi:softprob",
        n_estimators=best_n_estimators,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=config["training"]["random_state"],
        verbosity=0,
        **best_params,
    )

    final_model.fit(
        X_tr,
        y_tr,
        sample_weight=sw_tr,
    )

    # Evaluate
    metrics = evaluate(final_model, X_test_sel, y_test)
    print("AUC:", metrics["auc"])  # may be None for edge cases
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    # Save
    model_path = Path(config["paths"]["model_artifact"]).with_name("email_open_predictor_fast.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "features": list(X_train_sel.columns),
        "best_params": best_params,
        "metrics": metrics,
    }, model_path)
    print(f"✅ Saved fast model to {model_path}")


if __name__ == "__main__":
    main()


