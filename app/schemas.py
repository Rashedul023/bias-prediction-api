from pydantic import BaseModel, Field
from typing import Optional, List

class ArticleInput(BaseModel):
    """Input schema for bias prediction - Title and Source only"""
    title: str = Field(..., description="Article title", min_length=1, max_length=500)
    source: Optional[str] = Field(None, description="News source (e.g., CNN, Fox News)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Democrats announce new climate change legislation",
                "source": "CNN"
            }
        }

class BiasOutput(BaseModel):
    """Output schema - bias score from -1 to +1"""
    bias_score: float = Field(
        ..., 
        description="Bias score from -1 (left) to +1 (right)",
        ge=-1.0,
        le=1.0
    )
    bias_category: str = Field(
        ..., 
        description="Bias category: left, center, or right"
    )
    confidence: float = Field(
        ..., 
        description="Prediction confidence (0-1)",
        ge=0.0,
        le=1.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "bias_score": -0.752,
                "bias_category": "left",
                "confidence": 0.92
            }
        }

class BatchInput(BaseModel):
    """Batch prediction input"""
    articles: List[ArticleInput] = Field(..., max_items=50)

class BatchOutput(BaseModel):
    """Batch prediction output"""
    predictions: List[BiasOutput]
    total_processed: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: Optional[str] = None
    accuracy: Optional[float] = None
    version: str = "1.0.0"