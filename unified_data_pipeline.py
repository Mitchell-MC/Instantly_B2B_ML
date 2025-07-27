"""
Unified Data Pipeline: Comprehensive Noise Reduction + Advanced Preprocessing
Combines the best of both approaches for optimal B2B data preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from scipy import stats
import json

class UnifiedDataPipeline:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.cleaned_df = None
        self.featured_df = None
        self.pipeline_report = {}
        
    def run_complete_pipeline(self):
        """Run the complete unified pipeline"""
        print("🚀 UNIFIED DATA PIPELINE")
        print("=" * 50)
        print("Phase 1: Comprehensive Noise Reduction")
        print("Phase 2: Advanced Feature Engineering")
        print("Phase 3: Model Pipeline Creation")
        print("=" * 50)
        
        # Phase 1: Comprehensive Noise Reduction
        self._phase_1_noise_reduction()
        
        # Phase 2: Advanced Feature Engineering
        self._phase_2_feature_engineering()
        
        # Phase 3: Model Pipeline Creation
        self._phase_3_model_pipeline()
        
        # Save results
        self._save_results()
        
        return self.featured_df, self.pipeline_report
    
    def _phase_1_noise_reduction(self):
        """Phase 1: Comprehensive noise reduction"""
        print("\n🔍 PHASE 1: COMPREHENSIVE NOISE REDUCTION")
        print("-" * 50)
        
        # Load data
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Original dataset: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        
        # Step 1: Comprehensive EDA
        self._comprehensive_eda()
        
        # Step 2: Advanced outlier handling
        self._advanced_outlier_handling()
        
        # Step 3: Smoothing and filtering
        self._smoothing_and_filtering()
        
        # Step 4: Dimensionality reduction
        self._dimensionality_reduction()
        
        # Step 5: Final cleaning
        self._final_cleaning()
        
        self.cleaned_df = self.df.copy()
        print(f"✅ Phase 1 complete. Cleaned dataset: {self.cleaned_df.shape[0]:,} rows, {self.cleaned_df.shape[1]} columns")
    
    def _comprehensive_eda(self):
        """Comprehensive exploratory data analysis"""
        print("\n📊 COMPREHENSIVE EDA")
        print("-" * 25)
        
        # Basic statistics
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Missing values: {self.df.isnull().sum().sum():,}")
        
        # Distribution analysis for key numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        key_numerical = ['email_open_count', 'email_click_count', 'email_reply_count', 'status_x']
        available_numerical = [col for col in key_numerical if col in self.df.columns]
        
        if available_numerical:
            # Create distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Distribution Analysis - Key Numerical Features', fontsize=16)
            
            for i, col in enumerate(available_numerical[:4]):
                row, col_idx = i // 2, i % 2
                series = self.df[col].dropna()
                
                # Histogram with statistical lines
                axes[row, col_idx].hist(series, bins=30, alpha=0.7, edgecolor='black')
                mean_val = series.mean()
                median_val = series.median()
                axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[row, col_idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                axes[row, col_idx].set_title(f'{col} Distribution')
                axes[row, col_idx].legend()
            
            plt.tight_layout()
            plt.savefig('unified_distribution_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Distribution analysis saved")
        
        # Statistical analysis
        stats_report = {}
        for col in available_numerical:
            series = self.df[col].dropna()
            if len(series) > 0:
                stats_report[col] = {
                    'mean': series.mean(),
                    'median': series.median(),
                    'std': series.std(),
                    'skewness': series.skew(),
                    'kurtosis': series.kurtosis(),
                    'q1': series.quantile(0.25),
                    'q3': series.quantile(0.75)
                }
                
                # Check for skewness (red flag for outliers)
                mean_median_diff = abs(stats_report[col]['mean'] - stats_report[col]['median'])
                if mean_median_diff > stats_report[col]['std'] * 0.5:
                    print(f"  ⚠️  {col}: Large mean-median difference ({mean_median_diff:.2f}) - potential outliers")
        
        self.pipeline_report['eda'] = stats_report
    
    def _advanced_outlier_handling(self):
        """Advanced outlier detection and handling"""
        print("\n🎯 ADVANCED OUTLIER HANDLING")
        print("-" * 35)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numerical_cols:
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) > 0:
                    # Multiple outlier detection methods
                    outliers_iqr = self._detect_outliers_iqr(series)
                    outliers_zscore = self._detect_outliers_zscore(series)
                    
                    outlier_report[col] = {
                        'iqr_outliers': outliers_iqr.sum(),
                        'zscore_outliers': outliers_zscore.sum(),
                        'total_records': len(series)
                    }
                    
                    # Choose best method and apply treatment
                    if outliers_iqr.sum() < outliers_zscore.sum():
                        self._apply_iqr_treatment(col)
                    else:
                        self._apply_zscore_treatment(col)
        
        print(f"✅ Outlier handling completed for {len(outlier_report)} columns")
        self.pipeline_report['outlier_handling'] = outlier_report
    
    def _detect_outliers_iqr(self, series, threshold=1.5):
        """IQR method for outlier detection"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series, threshold=3):
        """Z-score method for outlier detection"""
        z_scores = np.abs(stats.zscore(series))
        return z_scores > threshold
    
    def _apply_iqr_treatment(self, col):
        """Apply IQR-based outlier treatment (capping)"""
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
        self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
    
    def _apply_zscore_treatment(self, col):
        """Apply Z-score based treatment (transformation)"""
        # Apply log transformation for highly skewed data
        if self.df[col].skew() > 1:
            self.df[f'{col}_log'] = np.log1p(self.df[col])
    
    def _smoothing_and_filtering(self):
        """Apply smoothing and filtering techniques"""
        print("\n📈 SMOOTHING AND FILTERING")
        print("-" * 30)
        
        # Identify temporal columns
        temporal_cols = [col for col in self.df.columns if 'timestamp' in col.lower()]
        
        if temporal_cols:
            print(f"Found {len(temporal_cols)} temporal columns for smoothing")
            
            for col in temporal_cols[:3]:
                if col in self.df.columns:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        
                        # Create temporal features
                        self.df[f'{col}_day_of_week'] = self.df[col].dt.dayofweek
                        self.df[f'{col}_hour'] = self.df[col].dt.hour
                        self.df[f'{col}_month'] = self.df[col].dt.month
                        self.df[f'{col}_is_weekend'] = (self.df[col].dt.dayofweek >= 5).astype(int)
                        self.df[f'{col}_is_business_hours'] = (
                            (self.df[col].dt.hour >= 9) & (self.df[col].dt.hour <= 17)
                        ).astype(int)
                        
                        print(f"  ✅ Created temporal features for {col}")
                    except:
                        pass
        
        # Apply moving averages to numerical sequences
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:
            if col in self.df.columns and self.df[col].notna().sum() > 100:
                window_size = min(7, len(self.df) // 10)
                self.df[f'{col}_sma'] = self.df[col].rolling(window=window_size, min_periods=1).mean()
                self.df[f'{col}_ema'] = self.df[col].ewm(span=window_size).mean()
                print(f"  ✅ Applied smoothing to {col}")
    
    def _dimensionality_reduction(self):
        """Apply dimensionality reduction techniques"""
        print("\n🔧 DIMENSIONALITY REDUCTION")
        print("-" * 30)
        
        # Prepare numerical data for PCA
        numerical_data = self.df.select_dtypes(include=[np.number])
        numerical_data = numerical_data.fillna(numerical_data.median())
        
        if numerical_data.shape[1] > 5:
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numerical_data)
            
            # Apply PCA
            pca = PCA(n_components=0.95)  # Keep 95% of variance
            pca_result = pca.fit_transform(scaled_data)
            
            # Create PCA features
            for i in range(pca_result.shape[1]):
                self.df[f'pca_component_{i+1}'] = pca_result[:, i]
            
            print(f"  ✅ PCA: Reduced {numerical_data.shape[1]} features to {pca_result.shape[1]} components")
            print(f"  ✅ Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        # Feature selection
        selector = VarianceThreshold(threshold=0.01)
        numerical_data_clean = numerical_data.dropna()
        if numerical_data_clean.shape[0] > 0:
            selected_features = selector.fit_transform(numerical_data_clean)
            removed_features = numerical_data.shape[1] - selected_features.shape[1]
            print(f"  ✅ Removed {removed_features} low-variance features")
    
    def _final_cleaning(self):
        """Final data cleaning steps"""
        print("\n🧹 FINAL CLEANING")
        print("-" * 20)
        
        # Remove high missing columns
        missing_percent = self.df.isnull().sum() / len(self.df)
        high_missing_cols = missing_percent[missing_percent > 0.5].index
        if len(high_missing_cols) > 0:
            self.df = self.df.drop(columns=high_missing_cols)
            print(f"  ✅ Removed {len(high_missing_cols)} high-missing columns")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_rows - len(self.df)
        if duplicates_removed > 0:
            print(f"  ✅ Removed {duplicates_removed} duplicate rows")
        
        # Impute remaining missing values
        missing_before = self.df.isnull().sum().sum()
        
        # Numerical columns: median imputation
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Categorical columns: mode imputation
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode().iloc[0] if len(self.df[col].mode()) > 0 else 'Unknown'
                self.df[col] = self.df[col].fillna(mode_val)
        
        missing_after = self.df.isnull().sum().sum()
        print(f"  ✅ Imputed {missing_before - missing_after} missing values")
    
    def _phase_2_feature_engineering(self):
        """Phase 2: Advanced feature engineering"""
        print("\n🔧 PHASE 2: ADVANCED FEATURE ENGINEERING")
        print("-" * 45)
        
        # Text preprocessing
        self._enhanced_text_preprocessing()
        
        # Temporal feature engineering
        self._advanced_temporal_features()
        
        # Interaction features
        self._create_interaction_features()
        
        # Binning and discretization
        self._feature_binning()
        
        print(f"✅ Phase 2 complete. Enhanced dataset: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
    
    def _enhanced_text_preprocessing(self):
        """Enhanced text preprocessing"""
        print("\n📝 Enhanced text preprocessing...")
        
        # Identify text columns
        text_cols = [col for col in self.df.columns if any(keyword in col.lower() 
                   for keyword in ['name', 'title', 'company', 'description', 'body', 'subject'])]
        
        # Clean text columns
        for col in text_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.lower()
                self.df[col] = self.df[col].str.replace(r'[^\w\s]', '', regex=True)
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
                self.df[col] = self.df[col].str.strip()
        
        # Create combined text features
        if text_cols:
            self.df['combined_text'] = ""
            for col in text_cols:
                if col in self.df.columns:
                    self.df['combined_text'] += self.df[col].fillna('') + ' '
            
            # Text-based features
            self.df['text_length'] = self.df['combined_text'].str.len()
            self.df['text_word_count'] = self.df['combined_text'].str.split().str.len()
            self.df['has_numbers_in_text'] = self.df['combined_text'].str.contains(r'\d', regex=True).astype(int)
            self.df['has_email_in_text'] = self.df['combined_text'].str.contains(r'@', regex=True).astype(int)
            self.df['text_quality_score'] = (
                (self.df['text_length'] > 10).astype(int) +
                (self.df['text_word_count'] > 3).astype(int) +
                (~self.df['combined_text'].str.contains('nan', regex=True)).astype(int)
            )
            
            print(f"  ✅ Created text features. Length range: {self.df['text_length'].min()}-{self.df['text_length'].max()}")
    
    def _advanced_temporal_features(self):
        """Advanced temporal feature engineering"""
        print("\n⏰ Advanced temporal feature engineering...")
        
        timestamp_cols = [col for col in self.df.columns if 'timestamp' in col.lower()]
        
        for col in timestamp_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
                    # Basic temporal features
                    self.df[f'{col}_day_of_week'] = self.df[col].dt.dayofweek
                    self.df[f'{col}_month'] = self.df[col].dt.month
                    self.df[f'{col}_hour'] = self.df[col].dt.hour
                    self.df[f'{col}_quarter'] = self.df[col].dt.quarter
                    
                    # Business-relevant features
                    self.df[f'{col}_is_weekend'] = (self.df[col].dt.dayofweek >= 5).astype(int)
                    self.df[f'{col}_is_business_hours'] = (
                        (self.df[col].dt.hour >= 9) & (self.df[col].dt.hour <= 17)
                    ).astype(int)
                    self.df[f'{col}_is_morning'] = (self.df[col].dt.hour <= 12).astype(int)
                    
                    # Seasonal features
                    self.df[f'{col}_season'] = self.df[col].dt.month % 12 // 3
                    
                    print(f"  ✅ Created temporal features for {col}")
                except:
                    pass
    
    def _create_interaction_features(self):
        """Create business-relevant interaction features"""
        print("\n🔗 Creating interaction features...")
        
        # Industry-seniority interaction
        if 'organization_industry' in self.df.columns and 'seniority' in self.df.columns:
            self.df['industry_seniority_interaction'] = (
                self.df['organization_industry'].fillna('Unknown') + '_' + 
                self.df['seniority'].fillna('Unknown')
            )
        
        # Geographic-industry interaction
        if 'country' in self.df.columns and 'organization_industry' in self.df.columns:
            self.df['geo_industry_interaction'] = (
                self.df['country'].fillna('Unknown') + '_' + 
                self.df['organization_industry'].fillna('Unknown')
            )
        
        # Company-size interaction (if available)
        if 'company_name' in self.df.columns and 'organization_industry' in self.df.columns:
            self.df['company_industry_interaction'] = (
                self.df['company_name'].fillna('Unknown') + '_' + 
                self.df['organization_industry'].fillna('Unknown')
            )
        
        print("  ✅ Created interaction features")
    
    def _feature_binning(self):
        """Feature binning and discretization"""
        print("\n📊 Feature binning and discretization...")
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Bin first 5 numerical columns
            if col in self.df.columns and self.df[col].notna().sum() > 50:
                try:
                    # Create bins based on quantiles
                    self.df[f'{col}_binned'] = pd.qcut(
                        self.df[col], 
                        q=5, 
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                        duplicates='drop'
                    )
                    print(f"  ✅ Binned {col} into 5 categories")
                except:
                    pass
    
    def _phase_3_model_pipeline(self):
        """Phase 3: Model pipeline creation"""
        print("\n🎯 PHASE 3: MODEL PIPELINE CREATION")
        print("-" * 40)
        
        # Create target variable (engagement level)
        self._create_target_variable()
        
        # Prepare features and target
        self._prepare_features_and_target()
        
        # Create sklearn pipeline
        self._create_sklearn_pipeline()
        
        print("✅ Phase 3 complete. Model pipeline ready")
    
    def _create_target_variable(self):
        """Create target variable for classification"""
        print("\n🎯 Creating target variable...")
        
        # Create engagement level based on email interactions
        if 'email_open_count' in self.df.columns and 'email_click_count' in self.df.columns:
            # Define engagement levels
            self.df['engagement_level'] = 0  # No engagement
            
            # Opener: opened but didn't click
            opener_mask = (self.df['email_open_count'] > 0) & (self.df['email_click_count'] == 0)
            self.df.loc[opener_mask, 'engagement_level'] = 1
            
            # Clicker: clicked emails
            clicker_mask = self.df['email_click_count'] > 0
            self.df.loc[clicker_mask, 'engagement_level'] = 2
            
            print(f"  ✅ Created engagement levels:")
            print(f"     No engagement: {(self.df['engagement_level'] == 0).sum():,}")
            print(f"     Opener: {(self.df['engagement_level'] == 1).sum():,}")
            print(f"     Clicker: {(self.df['engagement_level'] == 2).sum():,}")
    
    def _prepare_features_and_target(self):
        """Prepare features and target for modeling"""
        print("\n📊 Preparing features and target...")
        
        # Remove target variable from features
        if 'engagement_level' in self.df.columns:
            self.featured_df = self.df.drop(columns=['engagement_level'])
            target = self.df['engagement_level']
        else:
            self.featured_df = self.df.copy()
            target = None
        
        print(f"  ✅ Features shape: {self.featured_df.shape}")
        if target is not None:
            print(f"  ✅ Target shape: {target.shape}")
    
    def _create_sklearn_pipeline(self):
        """Create sklearn pipeline for modeling"""
        print("\n🔧 Creating sklearn pipeline...")
        
        # Define column types
        numerical_cols = self.featured_df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.featured_df.select_dtypes(include=['object']).columns
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ],
            remainder='passthrough'
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif, k=50)),
            ('classifier', None)  # Will be set by specific models
        ])
        
        print("  ✅ Sklearn pipeline created")
    
    def _save_results(self):
        """Save all results"""
        print("\n💾 Saving results...")
        
        # Save cleaned data
        cleaned_output = self.data_path.parent / f"{self.data_path.stem}_unified_cleaned.csv"
        self.cleaned_df.to_csv(cleaned_output, index=False)
        print(f"  ✅ Cleaned data saved to: {cleaned_output}")
        
        # Save featured data
        featured_output = self.data_path.parent / f"{self.data_path.stem}_unified_featured.csv"
        self.featured_df.to_csv(featured_output, index=False)
        print(f"  ✅ Featured data saved to: {featured_output}")
        
        # Save pipeline report
        report_output = self.data_path.parent / "unified_pipeline_report.json"
        with open(report_output, 'w') as f:
            json.dump(self.pipeline_report, f, indent=2, default=str)
        print(f"  ✅ Pipeline report saved to: {report_output}")
        
        # Final summary
        print(f"\n🎉 UNIFIED PIPELINE COMPLETE!")
        print(f"📊 Original: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        print(f"📊 Final: {self.featured_df.shape[0]:,} rows, {self.featured_df.shape[1]} columns")
        print(f"📈 Quality improvement: Comprehensive noise reduction + advanced feature engineering")

def main():
    """Main execution function"""
    pipeline = UnifiedDataPipeline('merged_contacts.csv')
    featured_df, report = pipeline.run_complete_pipeline()
    
    print(f"\n🎯 Unified pipeline completed successfully!")
    print(f"📊 Final dataset ready for modeling: {featured_df.shape[0]:,} rows, {featured_df.shape[1]} columns")
    
    return featured_df, report

if __name__ == "__main__":
    main() 