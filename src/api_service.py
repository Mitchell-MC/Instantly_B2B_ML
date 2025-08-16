"""
FastAPI Service for Model Serving
Production-ready API service with health checks, prediction endpoints, and monitoring.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import yaml
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our custom modules
try:
    from feature_engineering import create_xgboost_optimized_features, encode_categorical_features
    from advanced_drift_detection import AdvancedDriftDetector
    from model_performance_tracker import ModelPerformanceTracker
except ImportError:
    # Handle case where modules don't exist yet
    logger.warning("Some modules not available - running in basic mode")
    create_xgboost_optimized_features = None
    encode_categorical_features = None
    AdvancedDriftDetector = None
    ModelPerformanceTracker = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a basic FastAPI app for uvicorn to import
app = FastAPI(
    title="Email Engagement Prediction API",
    description="Production API for predicting email engagement in B2B marketing campaigns",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add basic health check endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Email Engagement Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "API service is running"
    }

# Pydantic models for API requests/responses
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    data: List[Dict[str, Any]] = Field(..., description="List of records to predict")
    include_probabilities: bool = Field(True, description="Include prediction probabilities")
    include_feature_importance: bool = Field(False, description="Include feature importance")
    
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    predictions: List[int] = Field(..., description="Model predictions")
    probabilities: Optional[List[List[float]]] = Field(None, description="Prediction probabilities")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")
    timestamp: str = Field(..., description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Health check timestamp")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    system_info: Dict[str, Any] = Field(..., description="System information")

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    features: List[str] = Field(..., description="Model features")
    performance: Dict[str, float] = Field(..., description="Model performance metrics")
    last_updated: str = Field(..., description="Last model update")

class APIService:
    """FastAPI service for model serving."""
    
    def __init__(self, config_path="config/main_config.yaml"):
        """Initialize the API service."""
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = None
        self.model_metadata = {}
        
        # Initialize monitoring components
        self.drift_detector = AdvancedDriftDetector(self.config.get('drift_detection', {}))
        self.performance_tracker = ModelPerformanceTracker(config_path)
        
        # Load model
        self._load_model()
        
        # Initialize FastAPI app
        self.app = self._create_app()
        
        logger.info("🚀 API Service initialized")
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_model(self):
        """Load the trained model and metadata."""
        try:
            model_path = self.config.get('paths', {}).get('model_artifact', 'models/email_open_predictor_v1.0.joblib')
            
            if not os.path.exists(model_path):
                logger.warning(f"Model not found at {model_path}, using sample model")
                model_path = "models/sample_model.joblib"
            
            artifacts = joblib.load(model_path)
            self.model = artifacts['model']
            self.feature_columns = artifacts.get('features', [])
            self.model_metadata = {
                'name': artifacts.get('name', 'email_open_predictor'),
                'version': artifacts.get('version', '1.0'),
                'features': self.feature_columns,
                'last_updated': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"✅ Model loaded: {self.model_metadata['name']} v{self.model_metadata['version']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _create_app(self):
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Email Engagement Prediction API",
            description="Production API for predicting email engagement in B2B marketing campaigns",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add routes
        self._add_routes(app)
        
        return app
    
    def _add_routes(self, app):
        """Add API routes to the FastAPI application."""
        
        @app.get("/", response_model=Dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "Email Engagement Prediction API",
                "version": "1.0.0",
                "docs": "/docs"
            }
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                model_info=self.model_metadata,
                system_info={
                    "python_version": sys.version,
                    "feature_count": len(self.feature_columns) if self.feature_columns else 0,
                    "model_loaded": self.model is not None
                }
            )
        
        @app.get("/model/info", response_model=ModelInfo)
        async def get_model_info():
            """Get model information."""
            return ModelInfo(
                model_name=self.model_metadata['name'],
                version=self.model_metadata['version'],
                features=self.feature_columns or [],
                performance={
                    "feature_count": len(self.feature_columns) if self.feature_columns else 0,
                    "model_type": type(self.model).__name__ if self.model else "Unknown"
                },
                last_updated=self.model_metadata['last_updated']
            )
        
        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
            """Make predictions on new data."""
            try:
                # Convert request data to DataFrame
                df = pd.DataFrame(request.data)
                
                if df.empty:
                    raise HTTPException(status_code=400, detail="Empty data provided")
                
                # Prepare features
                X_prepared = self._prepare_features(df)
                
                # Make predictions
                predictions = self.model.predict(X_prepared)
                probabilities = None
                feature_importance = None
                
                if request.include_probabilities and hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(X_prepared).tolist()
                
                if request.include_feature_importance and hasattr(self.model, 'feature_importances_'):
                    feature_importance = dict(zip(self.feature_columns, self.model.feature_importances_))
                
                # Add background task for monitoring
                background_tasks.add_task(
                    self._log_prediction_batch,
                    predictions, probabilities, df
                )
                
                return PredictionResponse(
                    predictions=predictions.tolist(),
                    probabilities=probabilities,
                    feature_importance=feature_importance,
                    metadata={
                        "records_processed": len(predictions),
                        "features_used": len(self.feature_columns) if self.feature_columns else 0,
                        "model_version": self.model_metadata['version']
                    },
                    timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @app.post("/predict/batch")
        async def predict_batch(file_path: str):
            """Make predictions on a CSV file."""
            try:
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                
                # Load CSV data
                df = pd.read_csv(file_path)
                
                # Prepare features
                X_prepared = self._prepare_features(df)
                
                # Make predictions
                predictions = self.model.predict(X_prepared)
                probabilities = self.model.predict_proba(X_prepared) if hasattr(self.model, 'predict_proba') else None
                
                # Save results
                results_df = df.copy()
                results_df['prediction'] = predictions
                if probabilities is not None:
                    results_df['probability'] = np.max(probabilities, axis=1)
                
                output_path = f"data/predictions_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                results_df.to_csv(output_path, index=False)
                
                return {
                    "message": "Batch prediction completed",
                    "records_processed": len(predictions),
                    "output_file": output_path,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Batch prediction error: {e}")
                raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
        
        @app.get("/monitoring/drift")
        async def check_drift(reference_data: str = None):
            """Check for data drift."""
            try:
                if not reference_data:
                    reference_data = self.config.get('data', {}).get('input_file', 'data/sample_data.csv')
                
                if not os.path.exists(reference_data):
                    raise HTTPException(status_code=404, detail=f"Reference data not found: {reference_data}")
                
                # Load reference data
                ref_df = pd.read_csv(reference_data)
                
                # For demonstration, we'll use a sample of recent predictions
                # In production, you'd compare with actual new data
                sample_data = pd.DataFrame({
                    'organization_employees': np.random.randint(1, 10000, 100),
                    'daily_limit': np.random.randint(100, 10000, 100),
                    'esp_code': np.random.randint(1, 100, 100)
                })
                
                # Detect drift
                drift_results = self.drift_detector.comprehensive_drift_analysis(
                    ref_df, sample_data, 'engagement_level'
                )
                
                return {
                    "drift_analysis": drift_results,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Drift detection error: {e}")
                raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")
        
        @app.get("/monitoring/performance")
        async def get_performance_metrics():
            """Get current performance metrics."""
            try:
                # Get performance history
                performance_data = self.performance_tracker.get_performance_summary()
                
                return {
                    "performance_metrics": performance_data,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except Exception as e:
                logger.error(f"Performance metrics error: {e}")
                raise HTTPException(status_code=500, detail=f"Performance metrics failed: {str(e)}")
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction."""
        try:
            # Apply feature engineering
            if hasattr(self, 'feature_columns') and self.feature_columns:
                # Use the same feature engineering as training
                X_prepared = create_xgboost_optimized_features(df.copy())
                
                # Ensure all required features are present
                missing_features = set(self.feature_columns) - set(X_prepared.columns)
                if missing_features:
                    for feature in missing_features:
                        X_prepared[feature] = 0  # Default value for missing features
                
                # Select only the features used by the model
                X_prepared = X_prepared[self.feature_columns]
            else:
                # Fallback to basic feature preparation
                X_prepared = df.copy()
            
            return X_prepared
            
        except Exception as e:
            logger.error(f"Feature preparation error: {e}")
            raise
    
    def _log_prediction_batch(self, predictions, probabilities, input_data):
        """Log prediction batch for monitoring."""
        try:
            # Log to performance tracker
            if hasattr(self.performance_tracker, 'track_predictions'):
                self.performance_tracker.track_predictions(
                    predictions=predictions,
                    input_data=input_data,
                    metadata={
                        'source': 'api',
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                )
            
            logger.info(f"Logged prediction batch: {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Failed to log prediction batch: {e}")
    
    def run(self, host="0.0.0.0", port=8000, reload=False):
        """Run the FastAPI service."""
        logger.info(f"🚀 Starting API service on {host}:{port}")
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

def main():
    """Main function to run the API service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Email Engagement Prediction API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", default="config/main_config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    try:
        service = APIService(args.config)
        service.run(host=args.host, port=args.port, reload=args.reload)
    except Exception as e:
        logger.error(f"Failed to start API service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
