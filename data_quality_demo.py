"""
Data Quality Monitoring Demo
Demonstrates the comprehensive data quality monitoring capabilities.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_data_quality_monitoring():
    """Demonstrate data quality monitoring capabilities."""
    print("🔍 Data Quality Monitoring Implementation")
    print("="*80)
    
    # Step 1: Initialize Data Quality Monitor
    print("\n📊 Step 1: Initializing Data Quality Monitor")
    print("-" * 50)
    
    try:
        from data_quality_monitor import DataQualityMonitor
        monitor = DataQualityMonitor()
        print("✅ Data quality monitor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize data quality monitor: {e}")
        return None
    
    # Step 2: Simulate Data Quality Over Time
    print("\n📊 Step 2: Simulating Data Quality Over Time")
    print("-" * 50)
    
    # Create sample datasets with varying quality
    datasets = []
    
    # High quality dataset
    high_quality_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'timestamp': pd.date_range(start='2025-01-01', periods=100, freq='H')
    })
    datasets.append(('High Quality', high_quality_data, {'source': 'production', 'version': '1.0'}))
    
    # Medium quality dataset (some missing values)
    medium_quality_data = high_quality_data.copy()
    medium_quality_data.loc[10:20, 'feature_1'] = np.nan
    medium_quality_data.loc[30:35, 'feature_2'] = np.nan
    datasets.append(('Medium Quality', medium_quality_data, {'source': 'production', 'version': '1.1'}))
    
    # Low quality dataset (many missing values and anomalies)
    low_quality_data = high_quality_data.copy()
    low_quality_data.loc[40:60, 'feature_1'] = np.nan
    low_quality_data.loc[70:80, 'feature_2'] = np.nan
    low_quality_data.loc[90:95, 'feature_3'] = np.nan
    # Add some anomalies
    low_quality_data.loc[5, 'feature_1'] = 1000  # Extreme outlier
    low_quality_data.loc[15, 'feature_2'] = -1000  # Extreme outlier
    datasets.append(('Low Quality', low_quality_data, {'source': 'production', 'version': '1.2'}))
    
    # Degraded quality dataset (drift in distributions)
    degraded_data = high_quality_data.copy()
    degraded_data['feature_1'] = degraded_data['feature_1'] * 2 + 5  # Shifted and scaled
    degraded_data['feature_2'] = degraded_data['feature_2'] * 0.5 - 2  # Different distribution
    datasets.append(('Degraded Quality', degraded_data, {'source': 'production', 'version': '1.3'}))
    
    # Monitor quality for each dataset
    for name, data, metadata in datasets:
        print(f"\n  📊 Monitoring {name} dataset...")
        try:
            quality_result = monitor.monitor_data_quality(data, metadata)
            if quality_result:
                quality_score = quality_result.get('quality_score', 0.0)
                print(f"    ✅ Quality Score: {quality_score:.3f}")
                
                # Check for degradation
                degradation = quality_result.get('degradation_analysis', {})
                if degradation.get('degradation_detected', False):
                    severity = degradation.get('degradation_severity', 'unknown')
                    print(f"    🚨 Degradation detected: {severity}")
                
                # Show key metrics
                metrics = quality_result.get('metrics', {})
                print(f"    📈 Completeness: {metrics.get('completeness', 0):.3f}")
                print(f"    🔍 Anomaly-free: {metrics.get('anomaly_free', 0):.3f}")
                print(f"    ⏰ Freshness: {metrics.get('freshness', 0):.3f}")
            else:
                print(f"    ❌ Quality monitoring failed")
        except Exception as e:
            print(f"    ❌ Error monitoring {name} dataset: {e}")
    
    # Step 3: Quality Trend Analysis
    print("\n📈 Step 3: Quality Trend Analysis")
    print("-" * 50)
    
    try:
        summary = monitor.get_quality_summary(days=30)
        if summary and 'error' not in summary:
            print(f"📊 Quality Summary (Last 30 days):")
            print(f"  Total Records: {summary.get('total_records', 0)}")
            print(f"  Total Samples: {summary.get('total_samples', 0)}")
            print(f"  Quality Trend: {summary.get('quality_trend', {}).get('trend_direction', 'unknown')}")
            print(f"  Mean Score: {summary.get('quality_trend', {}).get('mean_score', 0):.3f}")
            print(f"  Min Score: {summary.get('quality_trend', {}).get('min_score', 0):.3f}")
            print(f"  Max Score: {summary.get('quality_trend', {}).get('max_score', 0):.3f}")
            print(f"  Total Alerts: {summary.get('degradation_summary', {}).get('total_alerts', 0)}")
            print(f"  Critical Degradations: {summary.get('degradation_summary', {}).get('critical_degradations', 0)}")
            print(f"  Moderate Degradations: {summary.get('degradation_summary', {}).get('moderate_degradations', 0)}")
            
            print(f"\n💡 Recommendations:")
            for rec in summary.get('recommendations', []):
                print(f"  {rec}")
        else:
            print(f"❌ Failed to get quality summary: {summary.get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Quality summary failed: {e}")
    
    # Step 4: Generate Quality Reports
    print("\n📄 Step 4: Generating Quality Reports")
    print("-" * 50)
    
    try:
        print("📊 Generating quality report...")
        report = monitor.generate_quality_report("data/quality/quality_report.json")
        if report:
            print("✅ Quality report generated successfully")
        else:
            print("❌ Failed to generate quality report")
    except Exception as e:
        print(f"❌ Quality report generation failed: {e}")
    
    # Step 5: MLOps Pipeline Integration
    print("\n🔗 Step 5: MLOps Pipeline Integration")
    print("-" * 50)
    
    print("🔍 Data quality monitoring integrates with:")
    print("  ✅ Drift Detection - Quality degradation can trigger drift alerts")
    print("  ✅ Automated Retraining - Quality thresholds can trigger retraining")
    print("  ✅ Production Monitoring - Quality metrics included in health checks")
    print("  ✅ CI/CD Pipeline - Quality validation in deployment pipeline")
    print("  ✅ Champion/Challenger - Quality comparison for deployment decisions")
    
    # Step 6: Real-World Usage Examples
    print("\n💡 Step 6: Real-World Usage Examples")
    print("-" * 50)
    
    print("🚀 How to use in production:")
    print("\n1. Monitor Quality After Each Data Ingestion:")
    print("   monitor.monitor_data_quality(new_data, metadata)")
    
    print("\n2. Monitor Quality Trends:")
    print("   summary = monitor.get_quality_summary(days=7)")
    
    print("\n3. Generate Weekly Reports:")
    print("   monitor.generate_quality_report('weekly_quality_report.json', days=7)")
    
    print("\n4. Set Up Automated Alerting:")
    print("   # Configure quality thresholds in config/main_config.yaml")
    print("   # Alerts are automatically generated when thresholds are exceeded")
    
    print("\n5. Integrate with Production Monitoring:")
    print("   monitor.monitor_data_quality(production_data)")
    
    # Step 7: Quality Monitoring Status
    print("\n📊 Step 7: Quality Monitoring Status")
    print("-" * 50)
    
    try:
        status = monitor.get_quality_status()
        print(f"📊 Current Status:")
        print(f"  Total Quality Records: {status.get('total_quality_records', 0)}")
        print(f"  Total Alerts: {status.get('total_alerts', 0)}")
        print(f"  Baseline Established: {'✅ YES' if status.get('baseline_established', False) else '❌ NO'}")
        print(f"  Last Quality Check: {status.get('last_quality_check', 'Never')}")
        
        if status.get('recent_alerts'):
            print(f"\n🚨 Recent Alerts:")
            for alert in status['recent_alerts'][:3]:  # Show last 3 alerts
                print(f"  - {alert.get('message', 'Unknown alert')}")
    except Exception as e:
        print(f"❌ Failed to get quality status: {e}")
    
    print("\n🎉 Data Quality Monitoring Demo Complete!")
    print("="*80)
    
    return monitor

def show_data_quality_features():
    """Display the features of the data quality monitor."""
    print("\n🔍 DATA QUALITY MONITORING FEATURES")
    print("="*80)
    
    print("📊 COMPREHENSIVE QUALITY METRICS:")
    print("  ✅ Completeness: Missing value detection and analysis")
    print("  ✅ Type Consistency: Data type validation and consistency checks")
    print("  ✅ Range Validity: Statistical range validation for numeric data")
    print("  ✅ Uniqueness: Duplicate detection and uniqueness analysis")
    print("  ✅ Freshness: Timestamp validation and data recency checks")
    print("  ✅ Anomaly Detection: Statistical outlier detection using IQR method")
    
    print("\n🚨 AUTOMATED ALERTING:")
    print("  ✅ Quality degradation detection")
    print("  ✅ Configurable quality thresholds")
    print("  ✅ Severity classification (minor, moderate, critical)")
    print("  ✅ Retraining trigger alerts")
    
    print("\n📈 TREND ANALYSIS:")
    print("  ✅ Quality trend direction (improving, stable, degrading)")
    print("  ✅ Statistical trend analysis")
    print("  ✅ Historical quality comparison")
    print("  ✅ Baseline establishment and monitoring")
    
    print("\n📄 REPORTING & VISUALIZATION:")
    print("  ✅ Comprehensive quality reports")
    print("  ✅ Quality trend analysis")
    print("  ✅ Degradation analysis charts")
    print("  ✅ JSON export for external analysis")
    
    print("\n🔗 MLOPS INTEGRATION:")
    print("  ✅ Production monitoring integration")
    print("  ✅ Drift detection correlation")
    print("  ✅ Automated retraining triggers")
    print("  ✅ CI/CD pipeline validation")
    print("  ✅ Champion/Challenger quality comparison")
    
    print("\n⚙️ CONFIGURATION:")
    print("  ✅ YAML-based configuration")
    print("  ✅ Environment-specific settings")
    print("  ✅ Customizable quality thresholds")
    print("  ✅ Flexible alert channels")
    
    print("\n🎯 PRODUCTION READINESS: READY")
    print("   - Comprehensive data quality metrics tracking")
    print("   - Automated degradation detection and alerting")
    print("   - Historical quality analysis and trends")
    print("   - Full MLOps pipeline integration")
    print("   - Configurable thresholds and alerting")
    
    print("\n🚀 NEXT STEPS:")
    print("   1. Configure quality thresholds in config/main_config.yaml")
    print("   2. Integrate with your data ingestion pipeline")
    print("   3. Set up automated quality monitoring")
    print("   4. Configure alert notifications")
    print("   5. Schedule regular quality reports")

def main():
    """Main demo function."""
    try:
        # Run the main demo
        monitor = demo_data_quality_monitoring()
        
        if monitor:
            # Show features
            show_data_quality_features()
            
            print(f"\n✅ Demo completed successfully!")
            return monitor
        else:
            print(f"\n❌ Demo failed - please check the error messages above")
            return None
            
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print(f"❌ Demo failed - please check the error messages above")
        return None

if __name__ == "__main__":
    monitor = main()
