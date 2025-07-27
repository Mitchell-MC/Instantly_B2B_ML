"""
Comprehensive Noise Reduction for B2B Email Marketing Dataset
Covers ALL aspects: EDA, Outliers, Smoothing, Feature Engineering, PCA
Senior Data Scientist Approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Advanced libraries for comprehensive analysis
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy import stats
from scipy.signal import savgol_filter

class ComprehensiveNoiseReducer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.df = None
        self.cleaned_df = None
        self.analysis_report = {}
        
    def step_1_diagnose_data(self):
        """Step 1: Comprehensive EDA and Data Diagnosis"""
        print("🔍 STEP 1: COMPREHENSIVE DATA DIAGNOSIS")
        print("=" * 60)
        
        # Load data with proper error handling
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            if self.df is None or self.df.empty:
                raise ValueError("Dataset is empty or failed to load")
            print(f"Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"Trying to find alternative data files...")
            # Try to find alternative data files
            import glob
            csv_files = glob.glob("*.csv")
            if csv_files:
                print(f"Found CSV files: {csv_files}")
                # Try the first available CSV file
                try:
                    self.df = pd.read_csv(csv_files[0])
                    print(f"Loaded alternative file: {csv_files[0]}")
                except Exception as e2:
                    print(f"❌ Failed to load alternative file: {e2}")
                    raise ValueError("No valid data file found")
            else:
                raise ValueError("No CSV files found in directory")
        
        # Basic statistics
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"Missing values: {self.df.isnull().sum().sum():,}")
        
        # Comprehensive EDA
        self._exploratory_data_analysis()
        self._statistical_analysis()
        self._source_analysis()
        
        return self.df
    
    def _exploratory_data_analysis(self):
        """Comprehensive EDA with visualizations"""
        print("\n📊 EXPLORATORY DATA ANALYSIS")
        print("-" * 40)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for analysis")
            return
        
        # 1. Distribution Analysis
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"Numerical columns: {len(numerical_cols)}")
        
        # Create distribution plots for key numerical columns
        key_numerical = ['email_open_count', 'email_click_count', 'email_reply_count', 'status_x']
        available_numerical = [col for col in key_numerical if col in self.df.columns]
        
        if available_numerical:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Distribution Analysis of Key Numerical Features', fontsize=16)
            
            for i, col in enumerate(available_numerical[:4]):
                row, col_idx = i // 2, i % 2
                
                # Histogram
                axes[row, col_idx].hist(self.df[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[row, col_idx].set_title(f'{col} Distribution')
                axes[row, col_idx].set_xlabel(col)
                axes[row, col_idx].set_ylabel('Frequency')
                
                # Add mean and median lines
                mean_val = self.df[col].mean()
                median_val = self.df[col].median()
                axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                axes[row, col_idx].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                axes[row, col_idx].legend()
            
            plt.tight_layout()
            plt.savefig('distribution_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Distribution plots saved to 'distribution_analysis.png'")
        
        # 2. Box plots for outlier detection
        if available_numerical:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Box Plots for Outlier Detection', fontsize=16)
            
            for i, col in enumerate(available_numerical[:4]):
                row, col_idx = i // 2, i % 2
                axes[row, col_idx].boxplot(self.df[col].dropna())
                axes[row, col_idx].set_title(f'{col} Box Plot')
                axes[row, col_idx].set_ylabel(col)
            
            plt.tight_layout()
            plt.savefig('outlier_detection_boxplots.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Box plots saved to 'outlier_detection_boxplots.png'")
        
        # 3. Correlation analysis
        if len(available_numerical) > 1:
            correlation_matrix = self.df[available_numerical].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✅ Correlation matrix saved to 'correlation_matrix.png'")
        
        # 4. Missing value patterns
        missing_percent = (self.df.isnull().sum() / len(self.df) * 100).sort_values(ascending=False)
        high_missing = missing_percent[missing_percent > 50]
        moderate_missing = missing_percent[(missing_percent > 10) & (missing_percent <= 50)]
        
        print(f"\nMissing value analysis:")
        print(f"  High missing (>50%): {len(high_missing)} columns")
        print(f"  Moderate missing (10-50%): {len(moderate_missing)} columns")
        print(f"  Average missing rate: {missing_percent.mean():.2f}%")
        
        self.analysis_report['eda'] = {
            'numerical_columns': len(numerical_cols),
            'high_missing_columns': len(high_missing),
            'moderate_missing_columns': len(moderate_missing),
            'avg_missing_rate': missing_percent.mean()
        }
    
    def _statistical_analysis(self):
        """Comprehensive statistical analysis"""
        print("\n📈 STATISTICAL ANALYSIS")
        print("-" * 30)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for statistical analysis")
            return
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        stats_report = {}
        
        for col in numerical_cols[:10]:  # Analyze first 10 numerical columns
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) > 0:
                    stats_report[col] = {
                        'mean': series.mean(),
                        'median': series.median(),
                        'std': series.std(),
                        'skewness': series.skew(),
                        'kurtosis': series.kurtosis(),
                        'q1': series.quantile(0.25),
                        'q3': series.quantile(0.75),
                        'iqr': series.quantile(0.75) - series.quantile(0.25)
                    }
                    
                    # Check for skewness (red flag for outliers)
                    mean_median_diff = abs(stats_report[col]['mean'] - stats_report[col]['median'])
                    if mean_median_diff > stats_report[col]['std'] * 0.5:
                        print(f"  ⚠️  {col}: Large mean-median difference ({mean_median_diff:.2f}) - potential outliers")
        
        self.analysis_report['statistical_analysis'] = stats_report
    
    def _source_analysis(self):
        """Analyze data source characteristics"""
        print("\n🔍 DATA SOURCE ANALYSIS")
        print("-" * 30)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for source analysis")
            return
        
        # Identify data source patterns
        source_patterns = {
            'web_scraping': ['timestamp', 'url', 'scraped', 'crawled'],
            'human_input': ['name', 'email', 'phone', 'company'],
            'sensor_data': ['measurement', 'reading', 'value', 'sensor'],
            'api_data': ['api', 'response', 'payload', 'json']
        }
        
        detected_sources = {}
        for source_type, keywords in source_patterns.items():
            matching_cols = []
            for col in self.df.columns:
                if any(keyword in col.lower() for keyword in keywords):
                    matching_cols.append(col)
            if matching_cols:
                detected_sources[source_type] = matching_cols
        
        print("Detected data sources:")
        for source, cols in detected_sources.items():
            print(f"  {source}: {len(cols)} columns")
        
        self.analysis_report['source_analysis'] = detected_sources
    
    def step_2_handle_outliers(self):
        """Step 2: Comprehensive outlier handling"""
        print("\n🎯 STEP 2: COMPREHENSIVE OUTLIER HANDLING")
        print("=" * 50)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for outlier handling")
            return self.df
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        
        for col in numerical_cols:
            if col in self.df.columns:
                series = self.df[col].dropna()
                if len(series) > 0:
                    # Multiple outlier detection methods
                    outliers_iqr = self._detect_outliers_iqr(series)
                    outliers_zscore = self._detect_outliers_zscore(series)
                    outliers_isolation = self._detect_outliers_isolation_forest(series)
                    
                    outlier_report[col] = {
                        'iqr_outliers': outliers_iqr.sum(),
                        'zscore_outliers': outliers_zscore.sum(),
                        'isolation_outliers': outliers_isolation.sum(),
                        'total_records': len(series)
                    }
                    
                    # Choose best method and apply treatment
                    best_method = self._choose_outlier_method(outlier_report[col])
                    self._apply_outlier_treatment(col, best_method)
        
        print(f"✅ Outlier analysis completed for {len(outlier_report)} numerical columns")
        self.analysis_report['outlier_handling'] = outlier_report
        
        return self.df
    
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
        try:
            # Convert to numpy array and handle potential issues
            series_array = np.array(series, dtype=float)
            # Manual z-score calculation to avoid type issues
            mean_val = np.mean(series_array)
            std_val = np.std(series_array)
            if std_val > 0:
                z_scores = np.abs((series_array - mean_val) / std_val)
                return z_scores > threshold
            else:
                return pd.Series([False] * len(series))
        except Exception as e:
            print(f"Warning: Z-score calculation failed: {e}")
            return pd.Series([False] * len(series))
    
    def _detect_outliers_isolation_forest(self, series):
        """Isolation Forest for outlier detection"""
        try:
            from sklearn.ensemble import IsolationForest
            # Use 'auto' for contamination parameter to avoid type issues
            iso_forest = IsolationForest(contamination='auto', random_state=42)
            # Convert to numpy array for compatibility
            series_array = np.array(series).reshape(-1, 1)
            outliers = iso_forest.fit_predict(series_array)
            return outliers == -1
        except Exception as e:
            print(f"Warning: Isolation Forest failed for series: {e}")
            return pd.Series([False] * len(series))
    
    def _choose_outlier_method(self, outlier_counts):
        """Choose the most appropriate outlier detection method"""
        methods = ['iqr', 'zscore', 'isolation']
        counts = [outlier_counts['iqr_outliers'], outlier_counts['zscore_outliers'], outlier_counts['isolation_outliers']]
        return methods[np.argmin(counts)]  # Choose method with fewest outliers
    
    def _apply_outlier_treatment(self, col, method):
        """Apply outlier treatment based on chosen method"""
        if self.df is None:
            return
            
        if method == 'iqr':
            # Capping (Winsorizing)
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df[col] = np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
            self.df[col] = np.where(self.df[col] > upper_bound, upper_bound, self.df[col])
        
        elif method == 'zscore':
            # Transformation approach
            try:
                series_array = np.array(self.df[col].dropna(), dtype=float)
                # Manual z-score calculation to avoid type issues
                mean_val = np.mean(series_array)
                std_val = np.std(series_array)
                if std_val > 0:
                    z_scores = np.abs((series_array - mean_val) / std_val)
                    if z_scores.max() > 3:
                        # Apply log transformation for highly skewed data
                        if self.df[col].skew() > 1:
                            self.df[f'{col}_log'] = np.log1p(self.df[col])
            except Exception as e:
                print(f"Warning: Z-score transformation failed for {col}: {e}")
    
    def step_3_smoothing_and_filtering(self):
        """Step 3: Apply smoothing and filtering techniques"""
        print("\n📈 STEP 3: SMOOTHING AND FILTERING")
        print("=" * 40)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for smoothing")
            return self.df
        
        # Identify time-series or sequential data
        temporal_cols = [col for col in self.df.columns if 'timestamp' in col.lower() or 'date' in col.lower()]
        
        if temporal_cols:
            print(f"Found {len(temporal_cols)} temporal columns for smoothing")
            
            for col in temporal_cols[:3]:  # Process first 3 temporal columns
                if col in self.df.columns:
                    # Convert to datetime if possible
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        
                        # Create time-based features
                        self.df[f'{col}_day_of_week'] = self.df[col].dt.dayofweek
                        self.df[f'{col}_hour'] = self.df[col].dt.hour
                        self.df[f'{col}_month'] = self.df[col].dt.month
                        
                        print(f"  ✅ Created temporal features for {col}")
                    except:
                        pass
        
        # Apply moving average to numerical sequences
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Process first 5 numerical columns
            if col in self.df.columns and self.df[col].notna().sum() > 100:
                # Simple Moving Average
                window_size = min(7, len(self.df) // 10)  # Adaptive window size
                self.df[f'{col}_sma'] = self.df[col].rolling(window=window_size, min_periods=1).mean()
                
                # Exponential Moving Average
                self.df[f'{col}_ema'] = self.df[col].ewm(span=window_size).mean()
                
                print(f"  ✅ Applied smoothing to {col}")
        
        self.analysis_report['smoothing'] = {
            'temporal_features_created': len(temporal_cols),
            'smoothing_applied': len(numerical_cols[:5])
        }
        
        return self.df
    
    def step_4_feature_engineering_and_dimensionality_reduction(self):
        """Step 4: Advanced feature engineering and dimensionality reduction"""
        print("\n🔧 STEP 4: FEATURE ENGINEERING AND DIMENSIONALITY REDUCTION")
        print("=" * 60)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for feature engineering")
            return self.df
        
        # 1. Binning (Discretization)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:5]:  # Bin first 5 numerical columns
            if col in self.df.columns and self.df[col].notna().sum() > 50:
                try:
                    # Create bins based on quantiles with safer approach
                    series = self.df[col].dropna()
                    if len(series) > 0:
                        # Use pd.cut with proper error handling
                        self.df[f'{col}_binned'] = pd.cut(
                            self.df[col], 
                            bins=5, 
                            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                            include_lowest=True
                        )
                        print(f"  ✅ Binned {col} into 5 categories")
                except Exception as e:
                    print(f"  ⚠️  Failed to bin {col}: {e}")
                    # Skip binning for this column if it fails
        
        # 2. PCA for dimensionality reduction
        # Prepare data for PCA
        numerical_data = self.df.select_dtypes(include=[np.number])
        
        # Handle missing values more robustly
        numerical_data_clean = numerical_data.copy()
        
        # Fill missing values with median for each column
        for col in numerical_data_clean.columns:
            missing_count = numerical_data_clean[col].isnull().sum()
            if missing_count > 0:
                try:
                    median_val = numerical_data_clean[col].median()
                    numerical_data_clean[col] = numerical_data_clean[col].fillna(median_val)
                except:
                    # If median calculation fails, use 0
                    numerical_data_clean[col] = numerical_data_clean[col].fillna(0)
        
        # Remove any remaining NaN values
        numerical_data_clean = numerical_data_clean.dropna()
        
        if numerical_data_clean.shape[1] > 5 and numerical_data_clean.shape[0] > 0:  # Only apply PCA if we have enough features and data
            try:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(numerical_data_clean)
                
                # Apply PCA
                pca = PCA(n_components=0.95)  # Keep 95% of variance
                pca_result = pca.fit_transform(scaled_data)
                
                # Create PCA features (only for rows that were used in PCA)
                pca_df = pd.DataFrame(pca_result, index=numerical_data_clean.index)
                for i in range(pca_result.shape[1]):
                    self.df[f'pca_component_{i+1}'] = np.nan  # Initialize with NaN
                    self.df.loc[pca_df.index, f'pca_component_{i+1}'] = pca_df.iloc[:, i]
                
                print(f"  ✅ PCA: Reduced {numerical_data_clean.shape[1]} features to {pca_result.shape[1]} components")
                print(f"  ✅ Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
            except Exception as e:
                print(f"  ⚠️  PCA failed: {e}")
        else:
            print(f"  ⚠️  Skipping PCA: insufficient data ({numerical_data_clean.shape[0]} rows, {numerical_data_clean.shape[1]} columns)")
        
        # 3. Feature selection
        # Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        numerical_data_clean = numerical_data.dropna()
        if numerical_data_clean.shape[0] > 0:
            selected_features = selector.fit_transform(numerical_data_clean)
            removed_features = numerical_data.shape[1] - selected_features.shape[1]
            print(f"  ✅ Removed {removed_features} low-variance features")
        
        # 4. Interaction features
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) >= 2:
            # Create interaction features for first two categorical columns
            col1, col2 = categorical_cols[0], categorical_cols[1]
            if col1 in self.df.columns and col2 in self.df.columns:
                self.df[f'{col1}_{col2}_interaction'] = (
                    self.df[col1].astype(str) + '_' + self.df[col2].astype(str)
                )
                print(f"  ✅ Created interaction feature: {col1}_{col2}_interaction")
        
        # 5. Create engineered features for important campaign/email tracking columns
        important_campaign_cols = [
            'email_opened_step', 'email_opened_variant', 'email_clicked_step', 'email_clicked_variant',
            'email_replied_step', 'email_replied_variant', 'email_gap'
        ]
        
        for col in important_campaign_cols:
            if col in self.df.columns:
                # Create binary flags for non-null values
                self.df[f'{col}_has_value'] = self.df[col].notna().astype(int)
                
                # For categorical columns, create dummy variables
                if self.df[col].dtype == 'object':
                    # Create a simplified categorical feature
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) <= 10:  # Only if reasonable number of categories
                        self.df[f'{col}_category'] = self.df[col].fillna('Unknown')
                
                print(f"  ✅ Created engineered features for {col}")
        
        self.analysis_report['feature_engineering'] = {
            'binned_features': len([col for col in self.df.columns if '_binned' in col]),
            'pca_components': len([col for col in self.df.columns if 'pca_component' in col]),
            'interaction_features': len([col for col in self.df.columns if '_interaction' in col])
        }
        
        return self.df
    
    def step_5_final_validation_and_cleaning(self):
        """Step 5: Final validation and comprehensive cleaning"""
        print("\n✅ STEP 5: FINAL VALIDATION AND CLEANING")
        print("=" * 50)
        
        # Check if dataframe is loaded
        if self.df is None or self.df.empty:
            print("❌ No data available for final cleaning")
            return self.df
        
        # Define important columns that should be preserved
        important_columns = [
            # Campaign tracking
            'campaign_schedule_start_date', 'campaign_schedule_end_date', 'campaign_schedule_schedules',
            'auto_variant_select', 'auto_variant_select_trigger', 'insert_unsubscribe_header',
            'prioritize_new_leads', 'list_id',
            
            # Email tracking
            'email_opened_step', 'email_opened_variant', 'email_clicked_step', 'email_clicked_variant',
            'email_replied_step', 'email_replied_variant', 'email_gap',
            'timestamp_last_open', 'timestamp_last_click', 'timestamp_last_reply',
            'timestamp_last_interest_change',
            
            # Contact info
            'phone', 'website', 'personalization',
            
            # Status fields
            'enrichment_status', 'verification_status', 'status_summary', 'lt_interest_status',
            'last_contacted_from', 'uploaded_by_user', 'stop_for_company'
        ]
        
        # Remove high missing columns, but preserve important ones
        missing_percent = self.df.isnull().sum() / len(self.df)
        high_missing_cols = missing_percent[missing_percent > 0.5].index
        
        # Filter out important columns from removal list
        cols_to_remove = [col for col in high_missing_cols if col not in important_columns]
        cols_preserved = [col for col in high_missing_cols if col in important_columns]
        
        if len(cols_to_remove) > 0:
            self.df = self.df.drop(columns=cols_to_remove)
            print(f"  ✅ Removed {len(cols_to_remove)} high-missing columns")
        
        if len(cols_preserved) > 0:
            print(f"  ⚠️  Preserved {len(cols_preserved)} important columns despite high missing values:")
            for col in cols_preserved:
                missing_pct = missing_percent[col]
                print(f"    - {col}: {missing_pct:.1f}% missing")
        
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
        
        # Final quality assessment
        quality_score = self._calculate_quality_score()
        print(f"  ✅ Final data quality score: {quality_score:.1f}%")
        
        self.analysis_report['final_cleaning'] = {
            'final_shape': self.df.shape,
            'quality_score': quality_score,
            'duplicates_removed': duplicates_removed,
            'missing_imputed': missing_before - missing_after
        }
        
        return self.df
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score"""
        if self.df is None or self.df.empty:
            return 0.0
        
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        quality_score = ((total_cells - missing_cells) / total_cells) * 100
        return quality_score
    
    def run_complete_pipeline(self):
        """Run the complete comprehensive noise reduction pipeline"""
        print("🚀 COMPREHENSIVE NOISE REDUCTION PIPELINE")
        print("=" * 60)
        print("Covers ALL aspects: EDA, Outliers, Smoothing, Feature Engineering, PCA")
        print("=" * 60)
        
        # Execute all steps
        self.step_1_diagnose_data()
        self.step_2_handle_outliers()
        self.step_3_smoothing_and_filtering()
        self.step_4_feature_engineering_and_dimensionality_reduction()
        self.step_5_final_validation_and_cleaning()
        
        # Save results
        if self.df is not None:
            self.cleaned_df = self.df.copy()
            output_path = self.data_path.parent / f"{self.data_path.stem}_comprehensive_cleaned_v2.csv"
            try:
                self.cleaned_df.to_csv(output_path, index=False)
            except PermissionError:
                print(f"⚠️  Could not save to {output_path} (file may be open)")
                # Try alternative filename
                alt_path = self.data_path.parent / f"{self.data_path.stem}_cleaned_v2.csv"
                self.cleaned_df.to_csv(alt_path, index=False)
                output_path = alt_path
            
            # Save analysis report
            import json
            report_path = self.data_path.parent / "comprehensive_analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(self.analysis_report, f, indent=2, default=str)
            
            print(f"\n💾 Comprehensive cleaned data saved to: {output_path}")
            print(f"📋 Analysis report saved to: {report_path}")
        else:
            print("❌ No data to save - pipeline failed")
            self.cleaned_df = pd.DataFrame()
        
        return self.cleaned_df, self.analysis_report

def main():
    """Main execution function"""
    reducer = ComprehensiveNoiseReducer('merged_contacts.csv')
    cleaned_df, report = reducer.run_complete_pipeline()
    
    print(f"\n🎉 Comprehensive noise reduction completed!")
    print(f"📊 Final dataset: {cleaned_df.shape[0]:,} rows, {cleaned_df.shape[1]} columns")
    if 'final_cleaning' in report:
        print(f"📈 Quality score: {report['final_cleaning']['quality_score']:.1f}%")
    
    return cleaned_df, report

if __name__ == "__main__":
    main() 