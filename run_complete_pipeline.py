#!/usr/bin/env python3
"""
Complete MLOps Pipeline: Bronze → Silver → Gold
Runs the full data transformation pipeline for ML model training
"""

import logging
from datetime import datetime
from run_main_pipeline_idempotent import MainIdempotentPipeline
from silver_to_gold_pipeline import SilverToGoldPipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_complete_pipeline(incremental: bool = True, batch_size: int = None):
    """Run the complete bronze→silver→gold transformation pipeline"""
    
    print("🚀 RUNNING COMPLETE MLOps PIPELINE")
    print("=" * 60)
    print("📊 Data Flow: Bronze → Silver → Gold")
    print("🎯 Purpose: Prepare mature leads for ML model training")
    print()
    
    start_time = datetime.now()
    
    try:
        # Step 1: Bronze to Silver Transformation
        print("STEP 1: BRONZE → SILVER TRANSFORMATION")
        print("-" * 40)
        
        bronze_to_silver = MainIdempotentPipeline()
        silver_result = bronze_to_silver.run_idempotent_pipeline(
            incremental=incremental, 
            batch_size=batch_size
        )
        
        if silver_result is not None:
            print(f"✅ Silver layer updated with {len(silver_result)} records")
            print(f"🔍 Features engineered: {len(silver_result.columns)} total columns")
        else:
            print("ℹ️  No new data processed in silver layer")
        
        print()
        
        # Step 2: Silver to Gold Transformation
        print("STEP 2: SILVER → GOLD TRANSFORMATION")
        print("-" * 40)
        
        silver_to_gold = SilverToGoldPipeline()
        gold_result = silver_to_gold.run_pipeline(
            incremental=incremental, 
            lookback_days=60  # Look back further for mature leads
        )
        
        if gold_result is not None:
            print(f"✅ Gold layer updated with {len(gold_result)} mature leads")
            
            # Show maturity distribution
            if 'lead_maturity_days' in gold_result.columns:
                maturity_stats = gold_result['lead_maturity_days'].describe()
                print(f"📊 Lead maturity stats:")
                print(f"   Min: {maturity_stats['min']:.0f} days")
                print(f"   Avg: {maturity_stats['mean']:.0f} days") 
                print(f"   Max: {maturity_stats['max']:.0f} days")
        else:
            print("ℹ️  No new mature leads processed in gold layer")
        
        # Pipeline completion summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print()
        print("=" * 60)
        print("🎉 COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        print(f"⏰ Total duration: {duration:.2f} seconds")
        print()
        print("📊 PIPELINE SUMMARY:")
        print("   ✅ Bronze → Silver: Enhanced feature engineering with JSONB extraction")
        print("   ✅ Silver → Gold: Mature lead filtering (30+ days from last contact)")
        print("   🎯 Gold layer ready for ML model training")
        print()
        print("🔧 NEXT STEPS:")
        print("   • Use leads.v_gold_training_features view for model training")
        print("   • Gold layer contains only mature, high-quality leads")
        print("   • All JSONB intelligence features extracted and ready")
        
        return True
        
    except Exception as e:
        logging.error(f"Complete pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        return False

def run_full_refresh():
    """Run a complete full refresh of all layers"""
    print("🔄 RUNNING FULL REFRESH PIPELINE")
    print("⚠️  This processes ALL available data")
    print()
    
    return run_complete_pipeline(incremental=False, batch_size=None)

def run_incremental():
    """Run incremental pipeline for recent data"""
    print("⚡ RUNNING INCREMENTAL PIPELINE")
    print("📋 Processing recent data only")
    print()
    
    return run_complete_pipeline(incremental=True, batch_size=None)

def main():
    """Main execution with options"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--full":
            success = run_full_refresh()
        elif sys.argv[1] == "--incremental":
            success = run_incremental()
        else:
            print("Usage: python run_complete_pipeline.py [--full|--incremental]")
            print("  --full        : Process all data (full refresh)")
            print("  --incremental : Process recent data only") 
            return
    else:
        # Default: incremental
        success = run_incremental()
    
    if success:
        print("\n🎯 Pipeline completed successfully!")
        print("💡 Gold layer data is ready for XGBoost model training")
    else:
        print("\n❌ Pipeline failed - check logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
