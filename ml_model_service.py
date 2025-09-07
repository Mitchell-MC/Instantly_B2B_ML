#!/usr/bin/env python3
"""
ML Model Service for Lead Scoring
Handles model training, prediction, and lifecycle management
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import os
import pickle
import joblib
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# API Framework
from flask import Flask, request, jsonify
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLModelService:
    def __init__(self, db_config: Dict[str, str], model_dir: str = "./models"):
        """Initialize ML model service"""
        self.db_config = db_config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.app = Flask(__name__)
        self.current_model = None
        self.current_scaler = None
        self.current_model_version = None
        
        self.setup_routes()
        self.load_latest_model()
        
    def get_db_connection(self):
        """Create database connection"""
        return psycopg2.connect(**self.db_config)
    
    def setup_routes(self):
        """Setup Flask API routes"""
        
        @self.app.route('/api/train-model', methods=['POST'])
        def train_model_endpoint():
            try:
                data = request.get_json() or {}
                model_type = data.get('model_type', 'random_forest')
                retrain = data.get('retrain', False)
                
                # Start training in background
                threading.Thread(
                    target=self.train_model,
                    args=(model_type, retrain)
                ).start()
                
                return jsonify({
                    'status': 'training_started',
                    'message': f'Started training {model_type} model',
                    'retrain': retrain
                })
                
            except Exception as e:
                logger.error(f"Error in train model endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/predict', methods=['POST'])
        def predict_endpoint():
            try:
                data = request.get_json()
                
                # Support both lead_ids (from database) and leads_data (direct prediction)
                lead_ids = data.get('lead_ids', [])
                leads_data = data.get('leads_data', [])
                
                if lead_ids:
                    predictions = self.predict_leads(lead_ids)
                elif leads_data:
                    predictions = self.predict_leads_direct(leads_data)
                else:
                    return jsonify({'error': 'No lead IDs or lead data provided'}), 400
                
                return jsonify({
                    'status': 'success',
                    'predictions': predictions
                })
                
            except Exception as e:
                logger.error(f"Error in predict endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/model-performance', methods=['GET'])
        def model_performance_endpoint():
            try:
                performance = self.get_model_performance()
                return jsonify(performance)
                
            except Exception as e:
                logger.error(f"Error in model performance endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/feature-importance', methods=['GET'])
        def feature_importance_endpoint():
            try:
                importance = self.get_feature_importance()
                return jsonify(importance)
                
            except Exception as e:
                logger.error(f"Error in feature importance endpoint: {str(e)}")
                return jsonify({'error': str(e)}), 500
    
    def load_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load training data from gold layer"""
        try:
            conn = self.get_db_connection()
            
            query = """
            SELECT 
                feature_vector,
                target_label,
                lead_outcome,
                final_engagement_score,
                data_maturity_days
            FROM ml_lead_scoring.gold_training_data 
            WHERE training_eligible = true
              AND feature_vector IS NOT NULL
              AND target_label IS NOT NULL
              AND data_maturity_days >= 30
            ORDER BY created_timestamp DESC
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                raise ValueError("No training data available in gold layer")
            
            logger.info(f"Loaded {len(df)} training samples from gold layer")
            
            # Parse feature vectors
            features_list = []
            targets = []
            
            for _, row in df.iterrows():
                try:
                    feature_dict = json.loads(row['feature_vector'])
                    features_list.append(feature_dict)
                    targets.append(row['target_label'])
                except Exception as e:
                    logger.warning(f"Skipping invalid feature vector: {e}")
                    continue
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list)
            targets_array = np.array(targets)
            
            logger.info(f"Processed {len(features_df)} valid feature vectors")
            logger.info(f"Target distribution: {np.bincount(targets_array)}")
            
            return features_df, targets_array
            
        except Exception as e:
            logger.error(f"Error loading training data: {str(e)}")
            raise
    
    def prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML training"""
        # Handle missing values
        features_df = features_df.fillna(0)
        
        # Feature engineering
        if 'open_rate' in features_df.columns and 'click_rate' in features_df.columns:
            features_df['click_through_rate'] = features_df['click_rate'] / (features_df['open_rate'] + 1e-6)
        
        if 'reply_rate' in features_df.columns and 'engagement_score' in features_df.columns:
            features_df['engagement_reply_ratio'] = features_df['reply_rate'] / (features_df['engagement_score'] + 1e-6)
        
        # Create interaction features
        if 'company_size_encoded' in features_df.columns and 'revenue_encoded' in features_df.columns:
            features_df['company_maturity'] = features_df['company_size_encoded'] * features_df['revenue_encoded']
        
        # Log transformation for skewed features
        skewed_features = ['days_since_activity', 'lead_age_days']
        for feature in skewed_features:
            if feature in features_df.columns:
                features_df[f'{feature}_log'] = np.log1p(features_df[feature])
        
        return features_df
    
    def balance_dataset(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
        """Balance the dataset to handle class imbalance"""
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        logger.info(f"Original class distribution: {class_distribution}")
        
        # Use SMOTE for oversampling minority classes and random undersampling for majority
        sampling_strategy = 'auto'  # Balance all classes
        
        # Combine over and under sampling
        over_sampler = SMOTE(sampling_strategy='minority', random_state=42, k_neighbors=3)
        under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=42)
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('over', over_sampler),
            ('under', under_sampler)
        ])
        
        try:
            X_resampled, y_resampled = pipeline.fit_resample(X, y)
            
            # Log new distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            new_distribution = dict(zip(unique, counts))
            logger.info(f"Resampled class distribution: {new_distribution}")
            
            return pd.DataFrame(X_resampled, columns=X.columns), y_resampled
            
        except Exception as e:
            logger.warning(f"Could not balance dataset: {e}. Using original data.")
            return X, y
    
    def train_model(self, model_type: str = 'random_forest', retrain: bool = False):
        """Train ML model for lead scoring"""
        try:
            logger.info(f"Starting model training: {model_type}")
            
            # Load and prepare data
            features_df, targets = self.load_training_data()
            features_df = self.prepare_features(features_df)
            
            # Balance dataset
            features_df, targets = self.balance_dataset(features_df, targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, targets, test_size=0.2, random_state=42, stratify=targets
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Select model
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    class_weight='balanced',
                    n_jobs=-1
                )
            elif model_type == 'gradient_boosting':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                )
            elif model_type == 'logistic_regression':
                model = LogisticRegression(
                    random_state=42,
                    class_weight='balanced',
                    max_iter=1000
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Train model
            logger.info("Training model...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # ROC AUC for multiclass
            if len(np.unique(targets)) > 2:
                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            
            # Cross validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='f1_weighted'
            )
            
            performance_metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'roc_auc': float(auc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            logger.info(f"Model performance - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")
            
            # Save model and scaler
            model_version = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.model_dir / f"{model_version}.pkl"
            scaler_path = self.model_dir / f"{model_version}_scaler.pkl"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            # Get feature importance
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                importance_dict = dict(zip(features_df.columns, model.feature_importances_))
                feature_importance = {k: float(v) for k, v in importance_dict.items()}
            
            # Save model metadata to database
            self.save_model_metadata(
                model_version, model_type, str(model_path), 
                len(X_train), list(features_df.columns), 
                performance_metrics, feature_importance
            )
            
            # Set as current model
            self.current_model = model
            self.current_scaler = scaler
            self.current_model_version = model_version
            
            logger.info(f"Model training completed: {model_version}")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def save_model_metadata(self, model_name: str, model_type: str, model_path: str,
                          training_data_size: int, features_used: List[str],
                          performance_metrics: Dict, feature_importance: Dict):
        """Save model metadata to database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Deactivate previous models
            cursor.execute("""
                UPDATE ml_lead_scoring.model_versions 
                SET is_active = false 
                WHERE model_name = %s
            """, (model_type,))
            
            # Insert new model
            cursor.execute("""
                INSERT INTO ml_lead_scoring.model_versions 
                (model_name, version, model_path, training_data_size, features_used, 
                 performance_metrics, is_active)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                model_type, model_name, model_path, training_data_size,
                json.dumps(features_used), json.dumps(performance_metrics), True
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"Saved model metadata for {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving model metadata: {str(e)}")
    
    def load_latest_model(self):
        """Load the latest active model"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM ml_lead_scoring.model_versions 
                WHERE is_active = true 
                ORDER BY created_timestamp DESC 
                LIMIT 1
            """)
            
            model_record = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if model_record:
                model_path = model_record['model_path']
                version = model_record['version']
                
                if os.path.exists(model_path):
                    self.current_model = joblib.load(model_path)
                    
                    # Load scaler
                    scaler_path = model_path.replace('.pkl', '_scaler.pkl')
                    if os.path.exists(scaler_path):
                        self.current_scaler = joblib.load(scaler_path)
                    
                    self.current_model_version = version
                    logger.info(f"Loaded model: {version}")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            else:
                logger.info("No active model found")
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
    
    def predict_leads(self, lead_ids: List[str]) -> List[Dict]:
        """Predict scores for given leads"""
        if not self.current_model or not self.current_scaler:
            raise ValueError("No trained model available")
        
        try:
            conn = self.get_db_connection()
            
            # Get features for leads
            placeholders = ','.join(['%s'] * len(lead_ids))
            query = f"""
            SELECT lead_id, feature_vector 
            FROM ml_lead_scoring.silver_ml_features 
            WHERE lead_id IN ({placeholders})
              AND feature_vector IS NOT NULL
            """
            
            df = pd.read_sql(query, conn, params=lead_ids)
            conn.close()
            
            if df.empty:
                return []
            
            # Prepare features
            features_list = []
            lead_id_list = []
            
            for _, row in df.iterrows():
                try:
                    feature_dict = json.loads(row['feature_vector'])
                    features_list.append(feature_dict)
                    lead_id_list.append(row['lead_id'])
                except Exception as e:
                    logger.warning(f"Skipping invalid feature vector for lead {row['lead_id']}: {e}")
                    continue
            
            if not features_list:
                return []
            
            features_df = pd.DataFrame(features_list)
            features_df = self.prepare_features(features_df)
            
            # Scale features
            features_scaled = self.current_scaler.transform(features_df)
            
            # Make predictions
            predictions = self.current_model.predict_proba(features_scaled)
            predicted_classes = self.current_model.predict(features_scaled)
            
            # Get feature importance for predictions
            feature_importance = {}
            if hasattr(self.current_model, 'feature_importances_'):
                feature_importance = dict(zip(features_df.columns, self.current_model.feature_importances_))
            
            # Prepare results
            results = []
            for i, lead_id in enumerate(lead_id_list):
                # Get max probability as confidence
                confidence = float(np.max(predictions[i]))
                predicted_score = float(predictions[i][2]) if len(predictions[i]) > 2 else float(predictions[i][1])
                
                result = {
                    'lead_id': lead_id,
                    'predicted_score': predicted_score,
                    'predicted_class': int(predicted_classes[i]),
                    'confidence_level': confidence,
                    'class_probabilities': {
                        str(j): float(prob) for j, prob in enumerate(predictions[i])
                    },
                    'model_version': self.current_model_version
                }
                results.append(result)
            
            # Save predictions to database
            self.save_predictions(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_leads_direct(self, leads_data: List[Dict]) -> List[Dict]:
        """Predict scores for leads with direct feature data (no database lookup)"""
        if not self.current_model or not self.current_scaler:
            raise ValueError("No trained model available")
        
        try:
            # Process leads data directly
            features_list = []
            lead_identifiers = []
            
            for lead in leads_data:
                try:
                    if 'feature_vector' in lead:
                        # Lead already has feature vector
                        feature_dict = json.loads(lead['feature_vector']) if isinstance(lead['feature_vector'], str) else lead['feature_vector']
                        features_list.append(feature_dict)
                        lead_identifiers.append(lead.get('email', f"lead_{len(lead_identifiers)}"))
                    else:
                        # Extract features from enriched lead data
                        feature_dict = self.extract_features_from_lead(lead)
                        features_list.append(feature_dict)
                        lead_identifiers.append(lead.get('email', f"lead_{len(lead_identifiers)}"))
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid lead data: {e}")
                    continue
            
            if not features_list:
                return []
            
            # Convert to DataFrame and prepare features
            features_df = pd.DataFrame(features_list)
            features_df = self.prepare_features(features_df)
            
            # Scale features
            features_scaled = self.current_scaler.transform(features_df)
            
            # Make predictions
            predictions = self.current_model.predict_proba(features_scaled)
            predicted_classes = self.current_model.predict(features_scaled)
            
            # Prepare results
            results = []
            for i, lead_id in enumerate(lead_identifiers):
                confidence = float(np.max(predictions[i]))
                predicted_score = float(predictions[i][2]) if len(predictions[i]) > 2 else float(predictions[i][1])
                
                result = {
                    'lead_identifier': lead_id,
                    'predicted_score': predicted_score,
                    'predicted_class': int(predicted_classes[i]),
                    'confidence_level': confidence,
                    'class_probabilities': {
                        str(j): float(prob) for j, prob in enumerate(predictions[i])
                    },
                    'model_version': self.current_model_version,
                    'prediction_type': 'direct'
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error making direct predictions: {str(e)}")
            raise
    
    def extract_features_from_lead(self, lead: Dict) -> Dict:
        """Extract ML features from enriched lead data"""
        # This mirrors the logic from data_processing_service.py but for single leads
        return {
            'open_rate': 0,  # New lead defaults
            'click_rate': 0,
            'reply_rate': 0,
            'bounce_rate': 0,
            'days_since_activity': 0,
            'lead_age_days': 0,
            'engagement_score': 0,
            'company_size_encoded': self.encode_company_size_direct(
                lead.get('company_size'), lead.get('employee_count')
            ),
            'revenue_encoded': self.encode_revenue_direct(lead.get('company_revenue')),
            'industry_encoded': self.encode_industry_direct(lead.get('company_industry')),
            'has_tech_stack': 1 if lead.get('technologies') and len(lead.get('technologies', [])) > 0 else 0,
            'social_presence_score': self.calculate_social_presence_direct(lead)
        }
    
    def encode_company_size_direct(self, size_str: str, employee_count: int) -> int:
        """Direct encoding for company size"""
        if employee_count:
            if employee_count < 10: return 1
            elif employee_count < 50: return 2
            elif employee_count < 250: return 3
            elif employee_count < 1000: return 4
            else: return 5
        
        if size_str:
            size_lower = size_str.lower()
            if 'startup' in size_lower or '1-10' in size_lower: return 1
            elif 'small' in size_lower or '11-50' in size_lower: return 2
            elif 'medium' in size_lower or '51-250' in size_lower: return 3
            elif 'large' in size_lower or '251-1000' in size_lower: return 4
            elif 'enterprise' in size_lower or '1000+' in size_lower: return 5
        
        return 0
    
    def encode_revenue_direct(self, revenue_str: str) -> int:
        """Direct encoding for revenue"""
        if not revenue_str: return 0
        
        revenue_lower = revenue_str.lower()
        if 'million' in revenue_lower:
            import re
            numbers = re.findall(r'\d+', revenue_lower)
            if numbers:
                revenue_mil = int(numbers[0])
                if revenue_mil < 1: return 1
                elif revenue_mil < 10: return 2
                elif revenue_mil < 100: return 3
                else: return 4
        
        return 0
    
    def encode_industry_direct(self, industry_str: str) -> int:
        """Direct encoding for industry"""
        if not industry_str: return 0
        
        industry_lower = industry_str.lower()
        if any(term in industry_lower for term in ['software', 'technology', 'tech', 'saas']): return 1
        elif any(term in industry_lower for term in ['finance', 'financial', 'banking']): return 2
        elif any(term in industry_lower for term in ['healthcare', 'medical', 'health']): return 3
        elif any(term in industry_lower for term in ['manufacturing', 'industrial']): return 4
        elif any(term in industry_lower for term in ['consulting', 'services']): return 5
        elif any(term in industry_lower for term in ['retail', 'ecommerce']): return 6
        else: return 7
    
    def calculate_social_presence_direct(self, lead: Dict) -> float:
        """Calculate social presence score from lead data"""
        score = 0.0
        if lead.get('linkedin_url'): score += 0.5
        if lead.get('twitter_url'): score += 0.3
        if lead.get('technologies') and len(lead.get('technologies', [])) > 0: score += 0.2
        return min(score, 1.0)
    
    def save_predictions(self, predictions: List[Dict]):
        """Save predictions to database"""
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Get model version ID
            cursor.execute("""
                SELECT id FROM ml_lead_scoring.model_versions 
                WHERE version = %s
            """, (self.current_model_version,))
            
            model_version_id = cursor.fetchone()[0]
            
            # Insert predictions
            for pred in predictions:
                cursor.execute("""
                    INSERT INTO ml_lead_scoring.lead_predictions 
                    (lead_id, model_version_id, predicted_score, confidence_level, feature_importance)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (lead_id, model_version_id) DO UPDATE SET
                        predicted_score = EXCLUDED.predicted_score,
                        confidence_level = EXCLUDED.confidence_level,
                        prediction_timestamp = CURRENT_TIMESTAMP
                """, (
                    pred['lead_id'], model_version_id, pred['predicted_score'],
                    pred['confidence_level'], json.dumps(pred.get('class_probabilities', {}))
                ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving predictions: {str(e)}")
    
    def get_model_performance(self) -> Dict:
        """Get current model performance metrics"""
        if not self.current_model_version:
            return {'error': 'No active model'}
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute("""
                SELECT * FROM ml_lead_scoring.model_versions 
                WHERE version = %s
            """, (self.current_model_version,))
            
            model_record = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if model_record:
                return {
                    'model_version': model_record['version'],
                    'model_name': model_record['model_name'],
                    'training_data_size': model_record['training_data_size'],
                    'performance_metrics': model_record['performance_metrics'],
                    'created_timestamp': model_record['created_timestamp'].isoformat()
                }
            else:
                return {'error': 'Model not found'}
                
        except Exception as e:
            logger.error(f"Error getting model performance: {str(e)}")
            return {'error': str(e)}
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from current model"""
        if not self.current_model:
            return {'error': 'No active model'}
        
        try:
            # Load features from latest training
            features_df, _ = self.load_training_data()
            features_df = self.prepare_features(features_df)
            
            if hasattr(self.current_model, 'feature_importances_'):
                importance_dict = dict(zip(features_df.columns, self.current_model.feature_importances_))
                
                # Sort by importance
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                
                return {
                    'model_version': self.current_model_version,
                    'feature_importance': {k: float(v) for k, v in sorted_importance},
                    'top_features': [k for k, v in sorted_importance[:10]]
                }
            else:
                return {'error': 'Model does not support feature importance'}
                
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return {'error': str(e)}
    
    def run_service(self, host='0.0.0.0', port=5001):
        """Run the Flask service"""
        logger.info(f"Starting ML model service on {host}:{port}")
        self.app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == "__main__":
    # Database configuration
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'ml_lead_scoring'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password')
    }
    
    # Initialize and run service
    ml_service = MLModelService(db_config)
    
    # Auto-retrain model daily
    def auto_retrain():
        while True:
            time.sleep(24 * 3600)  # 24 hours
            try:
                logger.info("Starting automated model retraining")
                ml_service.train_model('random_forest', retrain=True)
            except Exception as e:
                logger.error(f"Auto-retrain failed: {str(e)}")
    
    threading.Thread(target=auto_retrain, daemon=True).start()
    
    # Start the Flask service
    ml_service.run_service()
