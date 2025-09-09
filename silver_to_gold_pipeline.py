#!/usr/bin/env python3
"""
Silver to Gold Transformation Pipeline
Filters mature leads (30+ days from last contact) for ML model training
"""

import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine, text
import yaml
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SilverToGoldPipeline:
    """Transform silver layer data to gold layer with lead maturity filtering"""
    
    def __init__(self, config_path: str = "config/silver_layer_config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self._load_config(config_path)
        self.engine = self._create_db_connection()
        self.maturity_days = 30  # Lead maturity threshold
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info("Configuration loaded successfully")
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            raise
    
    def _create_db_connection(self):
        """Create database connection"""
        try:
            db_config = self.config['database']
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            engine = create_engine(connection_string)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            logging.info("Database connection established with leads schema")
            return engine
        except Exception as e:
            logging.error(f"Database connection failed: {e}")
            raise
    
    def extract_silver_data(self, incremental: bool = True, lookback_days: int = 7) -> pd.DataFrame:
        """Extract data from silver layer"""
        try:
            # First check if gold table exists
            table_check = text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'leads' AND table_name = 'gold_ml_features'
                )
            """)
            
            with self.engine.connect() as conn:
                gold_table_exists = conn.execute(table_check).scalar()
            
            if incremental and gold_table_exists:
                # Incremental load: recent silver data + potentially mature records (avoiding duplicates)
                query = text(f"""
                    SELECT * FROM leads.silver_ml_features 
                    WHERE processed_timestamp >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'
                    OR (timestamp_last_contact IS NOT NULL 
                        AND timestamp_last_contact <= CURRENT_TIMESTAMP - INTERVAL '{self.maturity_days} days'
                        AND id::text NOT IN (SELECT id FROM leads.gold_ml_features))
                    ORDER BY processed_timestamp DESC
                """)
            elif incremental and not gold_table_exists:
                # Incremental load: recent silver data + potentially mature records (no duplicates check needed)
                query = text(f"""
                    SELECT * FROM leads.silver_ml_features 
                    WHERE processed_timestamp >= CURRENT_TIMESTAMP - INTERVAL '{lookback_days} days'
                    OR (timestamp_last_contact IS NOT NULL 
                        AND timestamp_last_contact <= CURRENT_TIMESTAMP - INTERVAL '{self.maturity_days} days')
                    ORDER BY processed_timestamp DESC
                """)
            else:
                # Full load: all silver data
                query = text("SELECT * FROM leads.silver_ml_features ORDER BY processed_timestamp DESC")
            
            df = pd.read_sql(query, self.engine)
            logging.info(f"📤 Extracted {len(df)} records from silver layer")
            return df
            
        except Exception as e:
            logging.error(f"Failed to extract silver data: {e}")
            raise
    
    def apply_maturity_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply lead maturity filter (30+ days from last contact)"""
        try:
            logging.info("🔍 Applying lead maturity filter...")
            
            if df.empty:
                return df
            
            # Convert timestamp_last_contact to datetime
            df['timestamp_last_contact'] = pd.to_datetime(df['timestamp_last_contact'], errors='coerce', utc=True)
            
            # Calculate cutoff date (30 days ago)
            cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=self.maturity_days)
            
            # Count records before filtering
            initial_count = len(df)
            
            # Filter conditions
            mature_records = (
                df['timestamp_last_contact'].notna() &  # Has last contact timestamp
                (df['timestamp_last_contact'] <= cutoff_date)  # At least 30 days old
            )
            
            # Apply filter
            gold_df = df[mature_records].copy()
            
            # Add gold layer metadata
            gold_df['gold_processed_timestamp'] = pd.Timestamp.now(tz='UTC')
            gold_df['lead_maturity_days'] = (pd.Timestamp.now(tz='UTC') - gold_df['timestamp_last_contact']).dt.days
            gold_df['is_mature_lead'] = 1
            
            # Log filtering results
            filtered_count = len(gold_df)
            logging.info(f"📊 Maturity filter results:")
            logging.info(f"   Initial records: {initial_count:,}")
            logging.info(f"   Mature leads (30+ days): {filtered_count:,}")
            logging.info(f"   Filtering ratio: {filtered_count/initial_count*100:.1f}%")
            
            if filtered_count > 0:
                logging.info(f"   Maturity range: {gold_df['lead_maturity_days'].min()}-{gold_df['lead_maturity_days'].max()} days")
            
            return gold_df
            
        except Exception as e:
            logging.error(f"Maturity filtering failed: {e}")
            raise
    
    def apply_data_quality_checks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply additional data quality checks for gold layer"""
        try:
            logging.info("🔍 Applying gold layer data quality checks...")
            
            if df.empty:
                return df
            
            initial_count = len(df)
            
            # Quality filters for ML training data
            quality_filters = (
                df['engagement_level'].notna() &  # Must have target variable
                df['id'].notna() &  # Must have identifier
                df['email'].notna() &  # Must have email
                (df['data_quality_score'].fillna(0) >= 0.3)  # Minimum data quality
            )
            
            quality_df = df[quality_filters].copy()
            
            # Log quality filtering
            quality_count = len(quality_df)
            logging.info(f"📊 Data quality filter results:")
            logging.info(f"   Before quality checks: {initial_count:,}")
            logging.info(f"   After quality checks: {quality_count:,}")
            logging.info(f"   Quality retention: {quality_count/initial_count*100:.1f}%")
            
            return quality_df
            
        except Exception as e:
            logging.error(f"Data quality checks failed: {e}")
            raise
    
    def check_existing_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Check for existing records in gold layer and filter out duplicates"""
        if df.empty:
            return df
            
        logging.info(f"🔍 Checking for existing records in gold layer...")
        
        try:
            # Get list of IDs that already exist in gold layer
            existing_ids_query = text("""
                SELECT DISTINCT id 
                FROM leads.gold_ml_features 
                WHERE id = ANY(:id_list)
            """)
            
            # Convert DataFrame IDs to list
            candidate_ids = df['id'].tolist()
            
            with self.engine.connect() as conn:
                result = conn.execute(existing_ids_query, {'id_list': candidate_ids})
                existing_ids = [row[0] for row in result.fetchall()]
            
            # Filter out records that already exist
            new_records = df[~df['id'].isin(existing_ids)].copy()
            
            logging.info(f"📊 Found {len(existing_ids)} existing records, {len(new_records)} new records to process")
            
            return new_records
            
        except Exception as e:
            logging.warning(f"Could not check existing records: {e}. Processing all records.")
            return df
    
    def write_to_gold_layer(self, df: pd.DataFrame, table_name: str = 'gold_ml_features'):
        """Write processed data to gold layer"""
        try:
            if df.empty:
                logging.info("No new records to write to gold layer")
                return
            
            logging.info(f"💾 Writing {len(df)} records to gold layer...")
            
            # Check if table exists, create if not
            table_check = text(f"""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables 
                    WHERE table_schema = 'leads' AND table_name = '{table_name}'
                )
            """)
            
            with self.engine.connect() as conn:
                table_exists = conn.execute(table_check).scalar()
                
                if not table_exists:
                    logging.info(f"Creating corrected gold layer table: {table_name}")
                    # Use the corrected gold layer table schema (matches actual silver columns)
                    with open('sql/create_gold_layer_table_corrected.sql', 'r') as f:
                        create_table_sql = f.read()
                    conn.execute(text(create_table_sql))
                    conn.commit()
                    logging.info(f"Corrected gold table created: {table_name}")
            
            # Write to PostgreSQL gold layer
            df.to_sql(
                table_name,
                self.engine,
                schema='leads',
                if_exists='append',
                index=False,
                method='multi',
                chunksize=500
            )
            
            logging.info(f"✅ Successfully wrote {len(df)} records to {table_name}")
            
        except Exception as e:
            logging.error(f"Failed to write to gold layer: {e}")
            raise
    
    def run_pipeline(self, incremental: bool = True, lookback_days: int = 7):
        """Execute the complete silver to gold ETL pipeline"""
        try:
            logging.info("🚀 Starting Silver to Gold ETL Pipeline...")
            start_time = datetime.now()
            
            # Step 1: Extract silver data
            df = self.extract_silver_data(incremental=incremental, lookback_days=lookback_days)
            
            if df.empty:
                logging.info("No data to process. Pipeline completed.")
                return None
            
            # Step 2: Apply maturity filter (30+ days from last contact)
            gold_df = self.apply_maturity_filter(df)
            
            if gold_df.empty:
                logging.info("No mature leads found. Pipeline completed.")
                return None
            
            # Step 3: Apply data quality checks
            gold_df = self.apply_data_quality_checks(gold_df)
            
            if gold_df.empty:
                logging.info("No records passed quality checks. Pipeline completed.")
                return None
            
            # Step 4: Check for existing records (idempotent)
            new_gold_df = self.check_existing_records(gold_df)
            
            if new_gold_df.empty:
                logging.info("All mature records already exist in gold layer - pipeline is idempotent ✅")
                return gold_df
            
            # Step 5: Write to gold layer
            self.write_to_gold_layer(new_gold_df)
            
            # Pipeline completion
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logging.info(f"✅ Pipeline completed successfully in {duration:.2f} seconds")
            
            # Report final statistics
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT COUNT(*) FROM leads.gold_ml_features"))
                total_count = result.fetchone()[0]
                logging.info(f"📊 Total records in gold layer: {total_count:,}")
                
                # Engagement distribution in gold layer
                eng_query = text("""
                    SELECT engagement_level, COUNT(*) as count
                    FROM leads.gold_ml_features 
                    GROUP BY engagement_level 
                    ORDER BY engagement_level
                """)
                eng_result = conn.execute(eng_query).fetchall()
                
                if eng_result:
                    logging.info("🎯 Gold layer engagement distribution:")
                    level_names = {0: 'Low', 1: 'Medium', 2: 'High'}
                    for level, count in eng_result:
                        level_name = level_names.get(level, 'Unknown')
                        pct = (count / total_count) * 100 if total_count > 0 else 0
                        logging.info(f"   Level {level} ({level_name}): {count:,} ({pct:.1f}%)")
            
            return new_gold_df
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    try:
        print("🚀 RUNNING SILVER-TO-GOLD TRANSFORMATION PIPELINE")
        print("=" * 60)
        
        # Initialize pipeline
        pipeline = SilverToGoldPipeline()
        
        # Run pipeline
        print("📋 Filtering mature leads (30+ days from last contact)...")
        result = pipeline.run_pipeline(incremental=True, lookback_days=30)
        
        if result is not None:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Processed mature leads for ML training")
        else:
            print("ℹ️  No new mature leads to process")
        
    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
