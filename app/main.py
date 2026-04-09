from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import re
import logging
import os

from .schemas import ArticleInput, BiasOutput, BatchInput, BatchOutput, HealthResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="Bias Prediction API",
    description="Predict political bias from article title and source",
    version="1.0.0"
)

# CORS - Allow all origins (for Django integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
metadata = None
features = []

def preprocess_text(text):
    """Clean and preprocess text"""
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def score_to_category(score):
    """Convert numeric score to category"""
    if score < -0.33:
        return 'left'
    elif score > 0.33:
        return 'right'
    else:
        return 'center'

@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, metadata, features
    
    logger.info("="*50)
    logger.info("Loading Bias Prediction Model...")
    logger.info("="*50)
    
    # Try different paths
    model_paths = [
        'models/bias_model.pkl',
        '../models/bias_model.pkl',
        '/app/models/bias_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"✅ Model loaded from {path}")
                break
            except Exception as e:
                logger.error(f"Failed to load from {path}: {e}")
    
    # Load metadata
    metadata_paths = [
        'models/model_metadata.pkl',
        '../models/model_metadata.pkl',
        '/app/models/model_metadata.pkl'
    ]
    
    for path in metadata_paths:
        if os.path.exists(path):
            try:
                metadata = joblib.load(path)
                logger.info(f"  Metadata loaded from {path}")
                features = metadata.get('features', ['title', 'source'])
                logger.info(f"   Best Model: {metadata.get('model_type')}")
                logger.info(f"   Features: {features}")
                logger.info(f"   Accuracy: {metadata.get('metrics', {}).get('category_accuracy', 0):.2%}")
                break
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
    
    if model:
        logger.info(" Model ready for predictions!")
        logger.info("="*50)
    else:
        logger.error(" Failed to load model")

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Bias Prediction API",
        "version": "1.0.0",
        "description": "Predicts political bias from article title and source",
        "output_scale": "-1 (left) to +1 (right)",
        "categories": ["left", "center", "right"],
        "features": features,
        "best_model": metadata.get('model_type') if metadata else None,
        "model_accuracy": metadata.get('metrics', {}).get('category_accuracy') if metadata else None,
        "endpoints": {
            "predict": "POST /predict",
            "batch": "POST /predict/batch",
            "health": "GET /health",
            "docs": "GET /docs"
        },
        "model_loaded": model is not None
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    return HealthResponse(
        status="healthy" if model else "degraded",
        model_loaded=model is not None,
        model_type=metadata.get('model_type') if metadata else None,
        accuracy=metadata.get('metrics', {}).get('category_accuracy') if metadata else None,
        version="1.0.0"
    )

@app.post("/predict", response_model=BiasOutput)
async def predict(article: ArticleInput):
    """Predict bias from article title and source ONLY"""
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please try again later."
        )
    
    try:
        # Validate input
        if not article.title or len(article.title.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Title cannot be empty"
            )
        
        # Preprocess title
        clean_title = preprocess_text(article.title)
        
        # Prepare features - ONLY title and source
        df = pd.DataFrame({
            'clean_title': [clean_title],
            'source': [article.source if article.source else 'unknown']
        })
        
        # Predict
        bias_score = float(np.clip(model.predict(df)[0], -1.0, 1.0))
        bias_category = score_to_category(bias_score)
        
        # Calculate confidence (using prediction variance if available)
        confidence = 0.85
        if hasattr(model.named_steps['regressor'], 'estimators_'):
            try:
                predictions = []
                for tree in model.named_steps['regressor'].estimators_:
                    pred = tree.predict(df)[0]
                    predictions.append(pred)
                std = np.std(predictions)
                confidence = 1 - min(1.0, std / 0.67)
            except:
                pass
        
        logger.info(f"Prediction: title='{article.title[:50]}...', score={bias_score:.3f}, category={bias_category}")
        
        return BiasOutput(
            bias_score=round(bias_score, 3),
            bias_category=bias_category,
            confidence=round(confidence, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(batch: BatchInput):
    """Batch prediction for multiple articles"""
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if len(batch.articles) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 50 articles per batch request"
        )
    
    try:
        predictions = []
        
        for article in batch.articles:
            clean_title = preprocess_text(article.title)
            
            df = pd.DataFrame({
                'clean_title': [clean_title],
                'source': [article.source if article.source else 'unknown']
            })
            
            bias_score = float(np.clip(model.predict(df)[0], -1.0, 1.0))
            bias_category = score_to_category(bias_score)
            
            predictions.append(BiasOutput(
                bias_score=round(bias_score, 3),
                bias_category=bias_category,
                confidence=0.8
            ))
        
        logger.info(f"Batch prediction completed: {len(predictions)} articles")
        
        return BatchOutput(
            predictions=predictions,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )