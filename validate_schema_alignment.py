#!/usr/bin/env python3
"""
Schema Validation Script for Bronze to Silver Pipeline
Validates that the pipeline configuration matches the actual database schema
"""

import sys
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
import yaml
import logging

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from bronze_to_silver_pipeline import BronzeToSilverPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaValidator:
    """Validates schema alignment between database and pipeline"""
    
    def __init__(self, config_path: str = "config/silver_layer_config.yaml"):
        self.pipeline = BronzeToSilverPipeline(config_path)
        self.engine = self.pipeline.engine
    
    def get_bronze_schema(self) -> dict:
        """Get actual bronze layer table schema"""
        try:
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'leads' 
            AND table_name = 'instantly_enriched_contacts'
            ORDER BY ordinal_position;
            """
            
            df = pd.read_sql(query, self.engine)
            return {
                'columns': df['column_name'].tolist(),
                'data_types': dict(zip(df['column_name'], df['data_type'])),
                'nullable': dict(zip(df['column_name'], df['is_nullable']))
            }
        except Exception as e:
            logger.error(f"Failed to get bronze schema: {e}")
            return {}
    
    def validate_column_mapping(self) -> dict:
        """Validate that pipeline column mappings match actual schema"""
        bronze_schema = self.get_bronze_schema()
        actual_columns = set(bronze_schema.get('columns', []))
        
        validation_results = {
            'missing_columns': [],
            'extra_columns': [],
            'data_type_mismatches': [],
            'coverage_stats': {}
        }
        
        # Check text columns
        expected_text_cols = set(self.pipeline.text_columns)
        missing_text = expected_text_cols - actual_columns
        if missing_text:
            validation_results['missing_columns'].extend([f"Text: {col}" for col in missing_text])
        
        # Check timestamp columns  
        expected_timestamp_cols = set(self.pipeline.timestamp_columns)
        missing_timestamps = expected_timestamp_cols - actual_columns
        if missing_timestamps:
            validation_results['missing_columns'].extend([f"Timestamp: {col}" for col in missing_timestamps])
        
        # Check categorical columns
        expected_categorical_cols = set(self.pipeline.categorical_columns)
        missing_categorical = expected_categorical_cols - actual_columns
        if missing_categorical:
            validation_results['missing_columns'].extend([f"Categorical: {col}" for col in missing_categorical])
        
        # Check Apollo columns
        expected_apollo_cols = set(self.pipeline.apollo_columns)
        missing_apollo = expected_apollo_cols - actual_columns
        if missing_apollo:
            validation_results['missing_columns'].extend([f"Apollo: {col}" for col in missing_apollo])
        
        # Calculate coverage statistics
        all_expected = (expected_text_cols | expected_timestamp_cols | 
                       expected_categorical_cols | expected_apollo_cols)
        
        validation_results['coverage_stats'] = {
            'total_expected_columns': len(all_expected),
            'total_actual_columns': len(actual_columns),
            'matched_columns': len(all_expected & actual_columns),
            'coverage_percentage': len(all_expected & actual_columns) / len(all_expected) * 100 if all_expected else 0
        }
        
        return validation_results
    
    def test_sample_data_extraction(self) -> dict:
        """Test extracting a small sample to validate data types"""
        try:
            query = "SELECT * FROM leads.instantly_enriched_contacts LIMIT 5"
            df = pd.read_sql(query, self.engine)
            
            return {
                'sample_extraction_success': True,
                'sample_row_count': len(df),
                'sample_columns': df.columns.tolist(),
                'sample_dtypes': df.dtypes.to_dict()
            }
        except Exception as e:
            logger.error(f"Sample data extraction failed: {e}")
            return {
                'sample_extraction_success': False,
                'error': str(e)
            }
    
    def validate_target_variable_creation(self) -> dict:
        """Test target variable creation logic"""
        try:
            # Test with a small sample using actual column names
            query = """
            SELECT I_email_open_count, I_email_click_count, I_email_reply_count
            FROM leads.instantly_enriched_contacts 
            WHERE I_email_open_count IS NOT NULL 
               OR I_email_click_count IS NOT NULL 
               OR I_email_reply_count IS NOT NULL
            LIMIT 10
            """
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return {
                    'target_validation_success': False,
                    'error': 'No engagement data found for target variable creation'
                }
            
            # Test target variable creation logic using actual column names
            engagement_cols = ['I_email_open_count', 'I_email_click_count', 'I_email_reply_count']
            for col in engagement_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Create engagement level (mapping to standard names first)
            import numpy as np
            df['email_open_count'] = df['I_email_open_count']
            df['email_click_count'] = df['I_email_click_count']
            df['email_reply_count'] = df['I_email_reply_count']
            
            conditions = [
                ((df['email_click_count'] > 0) | (df['email_reply_count'] > 0)),
                (df['email_open_count'] > 0)
            ]
            choices = [2, 1]
            df['engagement_level'] = np.select(conditions, choices, default=0)
            
            target_dist = df['engagement_level'].value_counts().sort_index()
            
            return {
                'target_validation_success': True,
                'sample_target_distribution': dict(target_dist),
                'engagement_columns_found': [col for col in engagement_cols if col in df.columns]
            }
            
        except Exception as e:
            logger.error(f"Target variable validation failed: {e}")
            return {
                'target_validation_success': False,
                'error': str(e)
            }
    
    def check_silver_layer_compatibility(self) -> dict:
        """Check if silver layer schema can accommodate pipeline output"""
        try:
            query = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_schema = 'ml_lead_scoring' 
            AND table_name = 'silver_ml_features'
            ORDER BY ordinal_position;
            """
            
            df = pd.read_sql(query, self.engine)
            
            if df.empty:
                return {
                    'silver_schema_exists': False,
                    'error': 'Silver layer table not found. Run sql/create_silver_layer_tables.sql first.'
                }
            
            silver_columns = set(df['column_name'].tolist())
            
            # Check if key pipeline output columns exist in silver schema
            pipeline_output_columns = {
                'engagement_level', 'has_apollo_enrichment', 'apollo_api_success',
                'company_size_category', 'company_size_log', 'apollo_data_completeness_pct',
                'is_high_value_title', 'is_high_value_seniority', 'is_tech_department'
            }
            
            missing_in_silver = pipeline_output_columns - silver_columns
            
            return {
                'silver_schema_exists': True,
                'total_silver_columns': len(silver_columns),
                'pipeline_output_columns': len(pipeline_output_columns),
                'missing_in_silver': list(missing_in_silver),
                'silver_compatibility': len(missing_in_silver) == 0
            }
            
        except Exception as e:
            logger.error(f"Silver layer compatibility check failed: {e}")
            return {
                'silver_schema_exists': False,
                'error': str(e)
            }
    
    def run_comprehensive_validation(self) -> dict:
        """Run all validation checks"""
        logger.info("Starting comprehensive schema validation...")
        
        results = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'bronze_schema_validation': {},
            'column_mapping_validation': {},
            'sample_data_validation': {},
            'target_variable_validation': {},
            'silver_layer_validation': {},
            'overall_status': 'UNKNOWN'
        }
        
        try:
            # 1. Validate bronze schema
            logger.info("Validating bronze schema...")
            bronze_schema = self.get_bronze_schema()
            results['bronze_schema_validation'] = {
                'schema_accessible': bool(bronze_schema),
                'total_columns': len(bronze_schema.get('columns', [])),
                'columns': bronze_schema.get('columns', [])[:10]  # First 10 columns
            }
            
            # 2. Validate column mappings
            logger.info("Validating column mappings...")
            results['column_mapping_validation'] = self.validate_column_mapping()
            
            # 3. Test sample data extraction
            logger.info("Testing sample data extraction...")
            results['sample_data_validation'] = self.test_sample_data_extraction()
            
            # 4. Validate target variable creation
            logger.info("Validating target variable creation...")
            results['target_variable_validation'] = self.validate_target_variable_creation()
            
            # 5. Check silver layer compatibility
            logger.info("Checking silver layer compatibility...")
            results['silver_layer_validation'] = self.check_silver_layer_compatibility()
            
            # Determine overall status
            critical_failures = [
                not results['bronze_schema_validation'].get('schema_accessible', False),
                not results['sample_data_validation'].get('sample_extraction_success', False),
                not results['target_variable_validation'].get('target_validation_success', False),
                not results['silver_layer_validation'].get('silver_schema_exists', False)
            ]
            
            if any(critical_failures):
                results['overall_status'] = 'FAILED'
            elif results['column_mapping_validation'].get('coverage_stats', {}).get('coverage_percentage', 0) < 80:
                results['overall_status'] = 'WARNING'
            else:
                results['overall_status'] = 'PASSED'
            
            logger.info(f"Validation completed with status: {results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}")
            results['overall_status'] = 'ERROR'
            results['error'] = str(e)
        
        return results
    
    def print_validation_report(self, results: dict):
        """Print a human-readable validation report"""
        print("\n" + "="*80)
        print("🔍 BRONZE TO SILVER PIPELINE - SCHEMA VALIDATION REPORT")
        print("="*80)
        
        status = results.get('overall_status', 'UNKNOWN')
        status_emoji = {
            'PASSED': '✅',
            'WARNING': '⚠️',
            'FAILED': '❌',
            'ERROR': '💥',
            'UNKNOWN': '❓'
        }
        
        print(f"\n📊 Overall Status: {status_emoji.get(status, '❓')} {status}")
        print(f"🕐 Validation Time: {results.get('validation_timestamp', 'N/A')}")
        
        # Bronze Schema Validation
        print(f"\n📋 Bronze Schema Validation:")
        bronze_val = results.get('bronze_schema_validation', {})
        if bronze_val.get('schema_accessible'):
            print(f"   ✅ Schema accessible with {bronze_val.get('total_columns', 0)} columns")
        else:
            print(f"   ❌ Schema not accessible")
        
        # Column Mapping Validation
        print(f"\n🗂️  Column Mapping Validation:")
        col_val = results.get('column_mapping_validation', {})
        coverage_stats = col_val.get('coverage_stats', {})
        coverage_pct = coverage_stats.get('coverage_percentage', 0)
        
        print(f"   📈 Coverage: {coverage_pct:.1f}% ({coverage_stats.get('matched_columns', 0)}/{coverage_stats.get('total_expected_columns', 0)} columns)")
        
        missing_cols = col_val.get('missing_columns', [])
        if missing_cols:
            print(f"   ⚠️  Missing columns ({len(missing_cols)}):")
            for col in missing_cols[:5]:  # Show first 5
                print(f"      - {col}")
            if len(missing_cols) > 5:
                print(f"      ... and {len(missing_cols) - 5} more")
        else:
            print(f"   ✅ All expected columns found")
        
        # Sample Data Validation
        print(f"\n🧪 Sample Data Validation:")
        sample_val = results.get('sample_data_validation', {})
        if sample_val.get('sample_extraction_success'):
            print(f"   ✅ Successfully extracted {sample_val.get('sample_row_count', 0)} sample rows")
        else:
            print(f"   ❌ Sample extraction failed: {sample_val.get('error', 'Unknown error')}")
        
        # Target Variable Validation
        print(f"\n🎯 Target Variable Validation:")
        target_val = results.get('target_variable_validation', {})
        if target_val.get('target_validation_success'):
            dist = target_val.get('sample_target_distribution', {})
            print(f"   ✅ Target variable creation successful")
            print(f"   📊 Sample distribution: {dist}")
        else:
            print(f"   ❌ Target variable validation failed: {target_val.get('error', 'Unknown error')}")
        
        # Silver Layer Validation
        print(f"\n🥈 Silver Layer Validation:")
        silver_val = results.get('silver_layer_validation', {})
        if silver_val.get('silver_schema_exists'):
            if silver_val.get('silver_compatibility', False):
                print(f"   ✅ Silver layer schema compatible")
            else:
                missing = silver_val.get('missing_in_silver', [])
                print(f"   ⚠️  Schema exists but missing {len(missing)} columns: {missing}")
        else:
            print(f"   ❌ Silver layer schema not found: {silver_val.get('error', 'Unknown error')}")
        
        print("\n" + "="*80)
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if status == 'FAILED':
            print("   1. Fix critical issues before running pipeline")
            print("   2. Ensure database connectivity and table access")
            print("   3. Run sql/create_silver_layer_tables.sql if silver layer missing")
        elif status == 'WARNING':
            print("   1. Review missing columns - some features may not be available")
            print("   2. Consider updating pipeline configuration")
            print("   3. Test with small dataset first")
        elif status == 'PASSED':
            print("   1. Pipeline ready for production deployment")
            print("   2. Consider running incremental test first")
            print("   3. Monitor data quality after deployment")
        
        print("="*80)

def main():
    """Main validation function"""
    try:
        validator = SchemaValidator()
        results = validator.run_comprehensive_validation()
        validator.print_validation_report(results)
        
        # Exit with appropriate code
        status = results.get('overall_status', 'ERROR')
        if status == 'PASSED':
            sys.exit(0)
        elif status == 'WARNING':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Validation script failed: {e}")
        sys.exit(3)

if __name__ == "__main__":
    main()
