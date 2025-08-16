"""
Business Intelligence Dashboard
Comprehensive dashboard for business stakeholders to monitor ML pipeline performance and insights.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
import warnings
import joblib
import glob
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Configure Streamlit page
st.set_page_config(
    page_title="Email Engagement ML Pipeline Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BusinessDashboard:
    """Business Intelligence Dashboard for ML Pipeline."""
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the business dashboard."""
        self.config = self._load_config(config_path)
        
        # Load data
        self._load_data()
        
        st.title("📊 Email Engagement ML Pipeline - Business Dashboard")
        st.markdown("---")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            st.error(f"Failed to load config: {e}")
            return {}
    
    def _load_data(self):
        """Load data for dashboard."""
        try:
            # Load performance data
            self.performance_data = self._load_json_data("data/performance/performance_history.json")
            
            # Load quality data
            self.quality_data = self._load_json_data("data/quality/quality_history.json")
            
            # Load monitoring data
            self.monitoring_data = self._load_json_data("data/monitoring_history.json")
            
            # Load drift data
            self.drift_data = self._load_json_data("data/drift_analysis_report.json")
            
            # Load actual ML models and their performance
            self._load_ml_models()
            
            # Load feature analysis data
            self._load_feature_analysis()
            
        except Exception as e:
            st.error(f"Failed to load dashboard data: {e}")
    
    def _load_ml_models(self):
        """Load actual ML models and their performance data."""
        try:
            # Find all model files
            model_files = glob.glob("models/*.joblib")
            self.models = {}
            
            for model_file in model_files:
                model_name = os.path.basename(model_file).replace('.joblib', '')
                try:
                    # Load model metadata (not the full model to avoid memory issues)
                    self.models[model_name] = {
                        'path': model_file,
                        'size_mb': os.path.getsize(model_file) / (1024 * 1024),
                        'modified': datetime.fromtimestamp(os.path.getmtime(model_file))
                    }
                except Exception as e:
                    st.warning(f"Could not load model {model_name}: {e}")
            
            # Load actual performance reports
            self.classification_reports = {}
            report_files = [
                "classification_report_fast.json",
                "classification_report_fast.txt",
                "comprehensive_analysis_report.json"
            ]
            
            for report_file in report_files:
                if os.path.exists(report_file):
                    try:
                        if report_file.endswith('.json'):
                            with open(report_file, 'r') as f:
                                self.classification_reports[report_file] = json.load(f)
                        else:
                            with open(report_file, 'r') as f:
                                self.classification_reports[report_file] = f.read()
                    except Exception as e:
                        st.warning(f"Could not load report {report_file}: {e}")
                        
        except Exception as e:
            st.error(f"Failed to load ML models: {e}")
    
    def _load_feature_analysis(self):
        """Load feature analysis and importance data."""
        try:
            # Look for feature importance visualizations
            feature_files = glob.glob("feature_analytics/*.png")
            self.feature_analytics = {
                'visualizations': feature_files,
                'reports': []
            }
            
            # Load feature analysis reports
            report_files = glob.glob("feature_analytics/*.md")
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        self.feature_analytics['reports'].append({
                            'name': os.path.basename(report_file),
                            'content': f.read()[:500] + "..."  # First 500 chars
                        })
                except Exception:
                    pass
                    
        except Exception as e:
            st.error(f"Failed to load feature analysis: {e}")
    
    def _load_json_data(self, file_path):
        """Load JSON data from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
            return []
        except Exception:
            return []
    
    def render_dashboard(self):
        """Render the complete business dashboard."""
        
        # Sidebar for navigation
        st.sidebar.title("🎯 Dashboard Sections")
        section = st.sidebar.selectbox(
            "Select Section",
            ["Overview", "ML Models", "Performance Metrics", "Feature Analysis", "Data Quality", "Business Insights", "Alerts & Monitoring"]
        )
        
        # Render selected section
        if section == "Overview":
            self._render_overview()
        elif section == "ML Models":
            self._render_ml_models()
        elif section == "Performance Metrics":
            self._render_performance_metrics()
        elif section == "Feature Analysis":
            self._render_feature_analysis()
        elif section == "Data Quality":
            self._render_data_quality()
        elif section == "Business Insights":
            self._render_business_insights()
        elif section == "Alerts & Monitoring":
            self._render_alerts_monitoring()
    
    def _render_overview(self):
        """Render overview section."""
        st.header("📈 Pipeline Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Count actual models
            model_count = len(self.models) if hasattr(self, 'models') else 0
            st.metric(
                label="Active ML Models",
                value=model_count,
                delta=""
            )
        
        with col2:
            # Count performance reports
            report_count = len(self.classification_reports) if hasattr(self, 'classification_reports') else 0
            st.metric(
                label="Performance Reports",
                value=report_count,
                delta=""
            )
        
        with col3:
            # Count feature analytics
            feature_count = len(self.feature_analytics.get('visualizations', [])) if hasattr(self, 'feature_analytics') else 0
            st.metric(
                label="Feature Analytics",
                value=feature_count,
                delta=""
            )
        
        with col4:
            # Count monitoring data
            monitoring_count = len(self.monitoring_data) if hasattr(self, 'monitoring_data') else 0
            st.metric(
                label="Monitoring Records",
                value=monitoring_count,
                delta=""
            )
        
        # Project structure overview
        st.subheader("🏗️ Project Structure")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Core Components**")
            components = [
                "📊 Feature Engineering Pipeline",
                "🤖 ML Models (XGBoost, Random Forest)",
                "📈 Performance Tracking",
                "🔍 Data Quality Monitoring",
                "🌊 Drift Detection",
                "📱 FastAPI Service",
                "🐳 Docker Containerization"
            ]
            for component in components:
                st.write(component)
        
        with col2:
            st.write("**Data Sources**")
            data_sources = [
                "📁 Apollo Lead Data",
                "📊 Campaign Performance",
                "🔗 Link Tracking Analytics",
                "📧 Email Engagement Metrics",
                "🏢 Company Information",
                "🌍 Geographic Data"
            ]
            for source in data_sources:
                st.write(source)
        
        # Recent activity based on actual files
        st.subheader("🔄 Recent Activity")
        
        if hasattr(self, 'models') and self.models:
            st.write("**Latest Model Updates**")
            model_updates = []
            for model_name, model_info in self.models.items():
                model_updates.append({
                    "Model": model_name,
                    "Last Modified": model_info['modified'].strftime("%Y-%m-%d %H:%M"),
                    "Size (MB)": f"{model_info['size_mb']:.1f}"
                })
            
            if model_updates:
                # Sort by modification date
                model_updates.sort(key=lambda x: x['Last Modified'], reverse=True)
                st.dataframe(pd.DataFrame(model_updates[:5]))
        
        # System health
        st.subheader("🔍 System Health")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File System**")
            health_data = {
                "Models Directory": "🟢 Available" if os.path.exists("models") else "🔴 Missing",
                "Data Directory": "🟢 Available" if os.path.exists("data") else "🔴 Missing",
                "Config Files": "🟢 Available" if os.path.exists("config") else "🔴 Missing",
                "Feature Analytics": "🟢 Available" if os.path.exists("feature_analytics") else "🔴 Missing"
            }
            for service, status in health_data.items():
                st.write(f"{service}: {status}")
        
        with col2:
            st.write("**Data Availability**")
            data_health = {
                "Performance Data": "🟢 Available" if self.performance_data else "🟡 Limited",
                "Quality Data": "🟢 Available" if self.quality_data else "🟡 Limited",
                "Monitoring Data": "🟢 Available" if self.monitoring_data else "🟡 Limited",
                "Drift Analysis": "🟢 Available" if self.drift_data else "🟡 Limited"
            }
            for data_type, status in data_health.items():
                st.write(f"{data_type}: {status}")
    
    def _render_ml_models(self):
        """Render ML models section."""
        st.header("🤖 ML Models Overview")
        
        if not hasattr(self, 'models') or not self.models:
            st.warning("No ML models found. Please check the models/ directory.")
            return
        
        # Model summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", len(self.models))
        
        with col2:
            total_size = sum(model['size_mb'] for model in self.models.values())
            st.metric("Total Size", f"{total_size:.1f} MB")
        
        with col3:
            latest_model = max(self.models.values(), key=lambda x: x['modified'])
            st.metric("Latest Update", latest_model['modified'].strftime("%m/%d"))
        
        # Model details table
        st.subheader("📋 Model Details")
        
        model_data = []
        for model_name, model_info in self.models.items():
            model_data.append({
                "Model Name": model_name,
                "File Size (MB)": f"{model_info['size_mb']:.1f}",
                "Last Modified": model_info['modified'].strftime("%Y-%m-%d %H:%M:%S"),
                "Status": "✅ Active"
            })
        
        if model_data:
            st.dataframe(pd.DataFrame(model_data), use_container_width=True)
        
        # Model performance reports
        if hasattr(self, 'classification_reports') and self.classification_reports:
            st.subheader("📊 Performance Reports")
            
            for report_name, report_content in self.classification_reports.items():
                with st.expander(f"📄 {report_name}"):
                    if isinstance(report_content, dict):
                        st.json(report_content)
                    else:
                        st.text(report_content)
    
    def _render_performance_metrics(self):
        """Render performance metrics section."""
        st.header("🎯 Performance Metrics")
        
        # Performance summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Performance")
            
            # Try to load actual performance metrics from reports
            if hasattr(self, 'classification_reports') and self.classification_reports:
                # Look for JSON reports with metrics
                for report_name, report_content in self.classification_reports.items():
                    if isinstance(report_content, dict) and 'metrics' in report_content:
                        metrics = report_content['metrics']
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                st.metric(label=metric, value=f"{value:.3f}")
                        break
                else:
                    # Fallback to sample metrics
                    st.info("Using sample metrics - check classification reports for actual data")
                    metrics = {
                        "Accuracy": 0.873,
                        "Precision": 0.856,
                        "Recall": 0.891,
                        "F1-Score": 0.873
                    }
                    for metric, value in metrics.items():
                        st.metric(label=metric, value=f"{value:.3f}")
            else:
                st.warning("No performance reports found")
        
        with col2:
            st.subheader("📊 Model Comparison")
            
            # Look for actual confusion matrix images
            confusion_files = glob.glob("*.png")
            confusion_files = [f for f in confusion_files if 'confusion' in f.lower()]
            
            if confusion_files:
                st.write("**Available Confusion Matrices:**")
                for file in confusion_files:
                    st.write(f"📊 {file}")
                
                # Display the first confusion matrix
                try:
                    st.image(confusion_files[0], caption="Confusion Matrix", use_column_width=True)
                except Exception as e:
                    st.error(f"Could not display image: {e}")
            else:
                st.info("No confusion matrix images found")
        
        # Performance over time
        st.subheader("⏰ Performance Over Time")
        
        if self.performance_data:
            # Use actual performance data
            try:
                perf_df = pd.DataFrame(self.performance_data)
                if 'timestamp' in perf_df.columns and 'accuracy' in perf_df.columns:
                    fig = px.line(perf_df, x='timestamp', y='accuracy',
                                 title="Model Accuracy Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Performance data format not recognized")
            except Exception as e:
                st.error(f"Error plotting performance data: {e}")
        else:
            # Generate sample data
            st.info("No performance history data found - using sample data")
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            performance_data = []
            
            for date in dates:
                base_accuracy = 0.87
                variation = np.random.normal(0, 0.02)
                performance_data.append({
                    'Date': date,
                    'Accuracy': max(0.7, min(0.95, base_accuracy + variation))
                })
            
            perf_df = pd.DataFrame(performance_data)
            fig = px.line(perf_df, x='Date', y='Accuracy',
                         title="Sample Performance Trend (No Real Data Available)")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_feature_analysis(self):
        """Render feature analysis section."""
        st.header("🔍 Feature Analysis")
        
        if not hasattr(self, 'feature_analytics') or not self.feature_analytics:
            st.warning("No feature analysis data found")
            return
        
        # Feature importance visualizations
        if self.feature_analytics.get('visualizations'):
            st.subheader("📊 Feature Importance Visualizations")
            
            # Group visualizations by type
            feature_importance_files = [f for f in self.feature_analytics['visualizations'] 
                                      if 'feature_importance' in f.lower()]
            shap_files = [f for f in self.feature_analytics['visualizations'] 
                         if 'shap' in f.lower()]
            other_files = [f for f in self.feature_analytics['visualizations'] 
                          if f not in feature_importance_files and f not in shap_files]
            
            # Display feature importance charts
            if feature_importance_files:
                st.write("**🎯 Feature Importance Charts**")
                cols = st.columns(min(3, len(feature_importance_files)))
                
                for i, file_path in enumerate(feature_importance_files[:3]):
                    with cols[i]:
                        try:
                            st.image(file_path, caption=os.path.basename(file_path), use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display {file_path}: {e}")
            
            # Display SHAP analysis
            if shap_files:
                st.write("**🧠 SHAP Analysis**")
                cols = st.columns(min(3, len(shap_files)))
                
                for i, file_path in enumerate(shap_files[:3]):
                    with cols[i]:
                        try:
                            st.image(file_path, caption=os.path.basename(file_path), use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display {file_path}: {e}")
            
            # Display other analytics
            if other_files:
                st.write("**📈 Other Analytics**")
                cols = st.columns(min(3, len(other_files)))
                
                for i, file_path in enumerate(other_files[:3]):
                    with cols[i]:
                        try:
                            st.image(file_path, caption=os.path.basename(file_path), use_column_width=True)
                        except Exception as e:
                            st.error(f"Could not display {file_path}: {e}")
        
        # Feature analysis reports
        if self.feature_analytics.get('reports'):
            st.subheader("📄 Feature Analysis Reports")
            
            for report in self.feature_analytics['reports']:
                with st.expander(f"📋 {report['name']}"):
                    st.text(report['content'])
        
        # Business insights from features
        st.subheader("💡 Business Insights from Features")
        
        # Look for actual feature analysis results
        feature_files = glob.glob("feature_analytics/*.py")
        if feature_files:
            st.write("**Available Feature Analysis Scripts:**")
            for file in feature_files:
                st.write(f"🐍 {file}")
            
            # Try to find and display actual feature importance data
            try:
                # Look for CSV files with feature importance
                csv_files = glob.glob("feature_analytics/*.csv")
                if csv_files:
                    st.write("**📊 Feature Importance Data:**")
                    for csv_file in csv_files:
                        try:
                            df = pd.read_csv(csv_file)
                            st.write(f"**{os.path.basename(csv_file)}**")
                            st.dataframe(df.head(10), use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not read {csv_file}: {e}")
            except Exception as e:
                st.error(f"Error loading feature data: {e}")
        else:
            st.info("No feature analysis scripts found")
    
    def _render_data_quality(self):
        """Render data quality section."""
        st.header("🔍 Data Quality Metrics")
        
        # Quality scores
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if self.quality_data:
                try:
                    latest_quality = self.quality_data[-1] if isinstance(self.quality_data, list) else self.quality_data
                    if isinstance(latest_quality, dict):
                        st.metric("Completeness", f"{latest_quality.get('completeness', 'N/A')}%")
                        st.metric("Accuracy", f"{latest_quality.get('accuracy', 'N/A')}%")
                except Exception:
                    st.metric("Completeness", "N/A")
                    st.metric("Accuracy", "N/A")
            else:
                st.metric("Completeness", "N/A")
                st.metric("Accuracy", "N/A")
        
        with col2:
            if self.quality_data:
                try:
                    latest_quality = self.quality_data[-1] if isinstance(self.quality_data, list) else self.quality_data
                    if isinstance(latest_quality, dict):
                        st.metric("Consistency", f"{latest_quality.get('consistency', 'N/A')}%")
                        st.metric("Validity", f"{latest_quality.get('validity', 'N/A')}%")
                except Exception:
                    st.metric("Consistency", "N/A")
                    st.metric("Validity", "N/A")
            else:
                st.metric("Consistency", "N/A")
                st.metric("Validity", "N/A")
        
        with col3:
            if self.quality_data:
                try:
                    latest_quality = self.quality_data[-1] if isinstance(self.quality_data, list) else self.quality_data
                    if isinstance(latest_quality, dict):
                        st.metric("Uniqueness", f"{latest_quality.get('uniqueness', 'N/A')}%")
                        st.metric("Timeliness", f"{latest_quality.get('timeliness', 'N/A')}%")
                except Exception:
                    st.metric("Uniqueness", "N/A")
                    st.metric("Timeliness", "N/A")
            else:
                st.metric("Uniqueness", "N/A")
                st.metric("Timeliness", "N/A")
        
        # Quality trends
        st.subheader("📊 Data Quality Trends")
        
        if self.quality_data and len(self.quality_data) > 1:
            try:
                quality_df = pd.DataFrame(self.quality_data)
                if 'timestamp' in quality_df.columns:
                    # Plot quality trends over time
                    quality_metrics = ['completeness', 'accuracy', 'consistency', 'validity']
                    available_metrics = [col for col in quality_metrics if col in quality_df.columns]
                    
                    if available_metrics:
                        fig = px.line(quality_df, x='timestamp', y=available_metrics,
                                     title="Data Quality Metrics Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No quality metrics found in data")
                else:
                    st.info("Quality data format not recognized")
            except Exception as e:
                st.error(f"Error plotting quality data: {e}")
        else:
            st.info("No quality trend data available")
        
        # Data quality issues
        st.subheader("⚠️ Recent Quality Issues")
        
        if self.quality_data:
            try:
                latest_quality = self.quality_data[-1] if isinstance(self.quality_data, list) else self.quality_data
                if isinstance(latest_quality, dict) and 'issues' in latest_quality:
                    issues = latest_quality['issues']
                    if issues:
                        st.dataframe(pd.DataFrame(issues), use_container_width=True)
                    else:
                        st.success("✅ No quality issues reported")
                else:
                    st.info("No quality issues data available")
            except Exception as e:
                st.error(f"Error loading quality issues: {e}")
        else:
            st.info("No quality data available")
    
    def _render_business_insights(self):
        """Render business insights section."""
        st.header("💡 Business Insights")
        
        # Key business metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Try to get actual metrics from data
            if self.performance_data:
                try:
                    latest_perf = self.performance_data[-1] if isinstance(self.performance_data, list) else self.performance_data
                    if isinstance(latest_perf, dict):
                        total_predictions = latest_perf.get('total_predictions', 'N/A')
                        accuracy = latest_perf.get('accuracy', 'N/A')
                    else:
                        total_predictions = 'N/A'
                        accuracy = 'N/A'
                except Exception:
                    total_predictions = 'N/A'
                    accuracy = 'N/A'
            else:
                total_predictions = 'N/A'
                accuracy = 'N/A'
            
            st.metric("Total Predictions", total_predictions)
            st.metric("Model Accuracy", f"{accuracy}%" if accuracy != 'N/A' else 'N/A')
        
        with col2:
            # Model count and size
            model_count = len(self.models) if hasattr(self, 'models') else 0
            total_size = sum(model['size_mb'] for model in self.models.values()) if hasattr(self, 'models') else 0
            
            st.metric("Active Models", model_count)
            st.metric("Total Model Size", f"{total_size:.1f} MB" if total_size > 0 else 'N/A')
        
        with col3:
            # Feature analytics count
            feature_count = len(self.feature_analytics.get('visualizations', [])) if hasattr(self, 'feature_analytics') else 0
            report_count = len(self.feature_analytics.get('reports', [])) if hasattr(self, 'feature_analytics') else 0
            
            st.metric("Feature Charts", feature_count)
            st.metric("Analysis Reports", report_count)
        
        with col4:
            # Data availability
            data_sources = sum([
                1 if self.performance_data else 0,
                1 if self.quality_data else 0,
                1 if self.monitoring_data else 0,
                1 if self.drift_data else 0
            ])
            
            st.metric("Data Sources", f"{data_sources}/4")
            st.metric("Dashboard Status", "🟢 Active")
        
        # Business impact analysis
        st.subheader("📈 Business Impact Analysis")
        
        # Show actual project structure and capabilities
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🚀 Project Capabilities**")
            capabilities = [
                "✅ Email Engagement Prediction",
                "✅ Feature Engineering Pipeline",
                "✅ Model Performance Tracking",
                "✅ Data Quality Monitoring",
                "✅ Drift Detection",
                "✅ FastAPI Service",
                "✅ Docker Containerization",
                "✅ Business Dashboard"
            ]
            for capability in capabilities:
                st.write(capability)
        
        with col2:
            st.write("**📊 Available Analytics**")
            analytics = []
            if hasattr(self, 'models') and self.models:
                analytics.append(f"🤖 {len(self.models)} ML Models")
            if hasattr(self, 'feature_analytics') and self.feature_analytics.get('visualizations'):
                analytics.append(f"📈 {len(self.feature_analytics['visualizations'])} Feature Charts")
            if hasattr(self, 'classification_reports') and self.classification_reports:
                analytics.append(f"📋 {len(self.classification_reports)} Performance Reports")
            if self.performance_data:
                analytics.append("📊 Performance History")
            if self.quality_data:
                analytics.append("🔍 Quality Metrics")
            
            if analytics:
                for analytic in analytics:
                    st.write(analytic)
            else:
                st.write("No analytics data available")
        
        # Feature importance for business
        st.subheader("🎯 Business-Critical Features")
        
        # Look for actual feature importance data
        feature_importance_files = glob.glob("feature_analytics/*feature_importance*.png")
        if feature_importance_files:
            st.write("**Available Feature Importance Visualizations:**")
            cols = st.columns(min(3, len(feature_importance_files)))
            
            for i, file_path in enumerate(feature_importance_files[:3]):
                with cols[i]:
                    try:
                        st.image(file_path, caption=os.path.basename(file_path), use_column_width=True)
                    except Exception as e:
                        st.error(f"Could not display {file_path}: {e}")
        else:
            st.info("No feature importance visualizations found")
        
        # Recommendations based on actual data
        st.subheader("💡 Business Recommendations")
        
        recommendations = []
        
        if hasattr(self, 'models') and len(self.models) > 1:
            recommendations.append("🔄 Multiple models available - consider A/B testing for optimal performance")
        
        if hasattr(self, 'feature_analytics') and self.feature_analytics.get('visualizations'):
            recommendations.append("📊 Feature analysis available - use insights for targeted marketing")
        
        if self.performance_data:
            recommendations.append("📈 Performance tracking active - monitor model degradation")
        
        if self.quality_data:
            recommendations.append("🔍 Data quality monitoring - ensure reliable predictions")
        
        if not recommendations:
            recommendations = [
                "📚 Review available models and performance reports",
                "🔍 Analyze feature importance for business insights",
                "📊 Set up regular performance monitoring",
                "🚀 Deploy models to production using FastAPI service"
            ]
        
        for rec in recommendations:
            st.write(rec)
    
    def _render_alerts_monitoring(self):
        """Render alerts and monitoring section."""
        st.header("🚨 Alerts & Monitoring")
        
        # Current alerts
        st.subheader("⚠️ Active Alerts")
        
        # Check for actual monitoring data
        if self.monitoring_data:
            try:
                latest_monitoring = self.monitoring_data[-1] if isinstance(self.monitoring_data, list) else self.monitoring_data
                if isinstance(latest_monitoring, dict) and 'alerts' in latest_monitoring:
                    alerts = latest_monitoring['alerts']
                    if alerts:
                        st.dataframe(pd.DataFrame(alerts), use_container_width=True)
                    else:
                        st.success("✅ No active alerts")
                else:
                    st.info("No alerts data in monitoring")
            except Exception as e:
                st.error(f"Error loading alerts: {e}")
        else:
            st.info("No monitoring data available")
        
        # System monitoring
        st.subheader("🔍 System Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Service Status**")
            services = {
                "API Service": "🟢 Available" if os.path.exists("src/api_service.py") else "🔴 Missing",
                "Dashboard": "🟢 Running",
                "Models Directory": "🟢 Available" if os.path.exists("models") else "🔴 Missing",
                "Data Directory": "🟢 Available" if os.path.exists("data") else "🔴 Missing",
                "Feature Analytics": "🟢 Available" if os.path.exists("feature_analytics") else "🔴 Missing"
            }
            
            for service, status in services.items():
                st.write(f"{service}: {status}")
        
        with col2:
            st.write("**Data Availability**")
            data_status = {
                "Performance Data": "🟢 Available" if self.performance_data else "🟡 Limited",
                "Quality Data": "🟢 Available" if self.quality_data else "🟡 Limited",
                "Monitoring Data": "🟢 Available" if self.monitoring_data else "🟡 Limited",
                "Drift Analysis": "🟢 Available" if self.drift_data else "🟡 Limited"
            }
            
            for data_type, status in data_status.items():
                st.write(f"{data_type}: {status}")
        
        # Monitoring timeline
        st.subheader("📊 Monitoring Timeline")
        
        if self.monitoring_data and len(self.monitoring_data) > 1:
            try:
                monitor_df = pd.DataFrame(self.monitoring_data)
                if 'timestamp' in monitor_df.columns:
                    # Plot monitoring data over time
                    available_metrics = [col for col in monitor_df.columns if col != 'timestamp']
                    
                    if available_metrics:
                        fig = make_subplots(
                            rows=min(2, len(available_metrics)), cols=1,
                            subplot_titles=available_metrics[:2],
                            vertical_spacing=0.1
                        )
                        
                        for i, metric in enumerate(available_metrics[:2]):
                            fig.add_trace(
                                go.Scatter(x=monitor_df['timestamp'], y=monitor_df[metric], name=metric),
                                row=i+1, col=1
                            )
                        
                        fig.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No monitoring metrics found")
                else:
                    st.info("Monitoring data format not recognized")
            except Exception as e:
                st.error(f"Error plotting monitoring data: {e}")
        else:
            st.info("No monitoring timeline data available")

def main():
    """Main function to run the business dashboard."""
    try:
        dashboard = BusinessDashboard()
        dashboard.render_dashboard()
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()
