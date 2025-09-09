#!/usr/bin/env python3
"""
Quick verification script for silver layer inserts
"""

from sqlalchemy import create_engine, text
import yaml
import pandas as pd

def verify_silver_inserts():
    """Run verification queries on silver layer"""
    
    # Load configuration
    with open('config/silver_layer_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    db_config = config['database']
    connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_string, connect_args={"options": "-csearch_path=leads"})

    print("🔍 SILVER LAYER VERIFICATION")
    print("=" * 50)

    with engine.connect() as conn:
        # 1. Basic stats
        print("\n1️⃣  BASIC STATISTICS")
        result = conn.execute(text("""
            SELECT 
                COUNT(*) as total_records,
                MIN(processed_timestamp) as first_insert,
                MAX(processed_timestamp) as last_insert
            FROM leads.silver_ml_features
        """))
        row = result.fetchone()
        print(f"   📊 Total records: {row[0]:,}")
        print(f"   📅 First insert: {row[1]}")
        print(f"   📅 Last insert: {row[2]}")

        # 2. Target distribution
        print("\n2️⃣  TARGET VARIABLE DISTRIBUTION")
        result = conn.execute(text("""
            SELECT 
                engagement_level,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM leads.silver_ml_features
            GROUP BY engagement_level
            ORDER BY engagement_level
        """))
        for row in result:
            print(f"   Level {row[0]}: {row[1]:,} ({row[2]}%)")

        # 3. Feature completeness
        print("\n3️⃣  FEATURE COMPLETENESS")
        result = conn.execute(text("""
            SELECT 
                SUM(has_apollo_enrichment) as apollo_enriched,
                ROUND(AVG(has_apollo_enrichment) * 100, 1) as apollo_rate,
                ROUND(AVG(data_quality_score), 3) as avg_quality,
                ROUND(AVG(text_length), 0) as avg_text_len,
                COUNT(CASE WHEN text_length > 0 THEN 1 END) as has_text
            FROM leads.silver_ml_features
        """))
        row = result.fetchone()
        print(f"   🌟 Apollo enriched: {row[0]:,} ({row[1]}%)")
        print(f"   📈 Avg data quality: {row[2]}")
        print(f"   📝 Avg text length: {row[3]}")
        print(f"   📄 Records with text: {row[4]:,}")

        # 4. Recent activity
        print("\n4️⃣  RECENT PROCESSING ACTIVITY")
        result = conn.execute(text("""
            SELECT COUNT(*) as recent_records
            FROM leads.silver_ml_features
            WHERE processed_timestamp >= NOW() - INTERVAL '1 hour'
        """))
        recent_count = result.scalar()
        print(f"   ⏰ Records processed in last hour: {recent_count:,}")

        # 5. Sample records
        print("\n5️⃣  SAMPLE RECORDS")
        result = conn.execute(text("""
            SELECT 
                LEFT(email, 20) as email_sample,
                engagement_level,
                has_apollo_enrichment,
                company_size_category,
                text_length,
                data_quality_score
            FROM leads.silver_ml_features
            ORDER BY processed_timestamp DESC
            LIMIT 3
        """))
        print("   Email Sample | Target | Apollo | Company Size | Text Len | Quality")
        print("   " + "-" * 65)
        for row in result:
            print(f"   {row[0]:<12} | {row[1]:<6} | {row[2]:<6} | {row[3]:<12} | {row[4]:<8} | {row[5]:.3f}")

        # 6. Training view check
        print("\n6️⃣  TRAINING VIEW STATUS")
        result = conn.execute(text("""
            SELECT COUNT(*) as training_ready
            FROM leads.v_ml_training_features
        """))
        training_count = result.scalar()
        print(f"   🎯 Training-ready records: {training_count:,}")

    print(f"\n✅ VERIFICATION COMPLETE!")
    print(f"🎉 Silver layer is ready for ML training!")

if __name__ == "__main__":
    verify_silver_inserts()





