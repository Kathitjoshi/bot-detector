"""
FastAPI Application for Bot Detection
Serves predictions via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from contextlib import asynccontextmanager
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.feature_engineering.features import FeatureEngineer

# Global variables for model and feature engineer
model_data = None
feature_engineer = FeatureEngineer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model_data

    model_path = (
        Path(__file__).parent.parent.parent
        / "models"
        / "saved_models"
        / "best_model.joblib"
    )

    if model_path.exists():
        model_data = joblib.load(model_path)
        print(f"Model loaded: {model_data['model_name']}")
    else:
        print("Warning: No model found. Please train a model first.")

    yield

    # Cleanup if needed
    pass


app = FastAPI(
    title="Bot Detection API",
    description="Machine Learning API for detecting bot accounts on social media",
    version="1.0.0",
    lifespan=lifespan,
)

# Global variables for model and feature engineer
model_data = None
feature_engineer = FeatureEngineer()


class UserData(BaseModel):
    """Single user data schema"""

    followers_count: int = Field(..., ge=0, description="Number of followers")
    following_count: int = Field(..., ge=0, description="Number of following")
    tweet_count: int = Field(..., ge=0, description="Total tweets")
    account_age_days: int = Field(..., ge=0, description="Account age in days")
    listed_count: int = Field(0, ge=0, description="Number of lists user is on")
    verified: int = Field(0, ge=0, le=1, description="Verification status (0 or 1)")
    default_profile: int = Field(
        0, ge=0, le=1, description="Using default profile (0 or 1)"
    )
    default_profile_image: int = Field(
        0, ge=0, le=1, description="Using default image (0 or 1)"
    )
    geo_enabled: int = Field(0, ge=0, le=1, description="Geo tagging enabled (0 or 1)")
    description_length: int = Field(
        0, ge=0, description="Length of profile description"
    )
    avg_tweets_per_day: float = Field(..., ge=0, description="Average tweets per day")
    avg_retweet_ratio: float = Field(
        ..., ge=0, le=1, description="Average retweet ratio"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "followers_count": 150,
                "following_count": 2000,
                "tweet_count": 5000,
                "account_age_days": 180,
                "listed_count": 2,
                "verified": 0,
                "default_profile": 1,
                "default_profile_image": 1,
                "geo_enabled": 0,
                "description_length": 20,
                "avg_tweets_per_day": 50.5,
                "avg_retweet_ratio": 0.85,
            }
        }
    )


class BatchUserData(BaseModel):
    """Batch user data schema"""

    users: List[UserData]


class PredictionResponse(BaseModel):
    """Prediction response schema"""

    is_bot: bool
    confidence: float
    suspicion_score: float
    message: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response schema"""

    predictions: List[PredictionResponse]
    total_users: int
    bot_count: int
    human_count: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Bot Detection API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": model_data is not None,
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model_data is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict_bot(user: UserData):
    """
    Predict if a single user is a bot

    Args:
        user: User data

    Returns:
        Prediction result with confidence score
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        user_dict = user.model_dump()
        df = pd.DataFrame([user_dict])

        # Engineer features
        df = feature_engineer.create_all_features(df)

        # Remove label column if it exists
        if "label" in df.columns:
            df = df.drop("label", axis=1)

        # Make prediction
        prediction = model_data["model"].predict(df)[0]
        probability = model_data["model"].predict_proba(df)[0]

        # Calculate suspicion score (from features)
        suspicion_score = float(df["suspicion_score"].iloc[0])

        is_bot = bool(prediction == 1)
        confidence = float(probability[1] if is_bot else probability[0])

        message = (
            "This account shows bot-like behavior"
            if is_bot
            else "This account appears to be human"
        )

        return PredictionResponse(
            is_bot=is_bot,
            confidence=confidence,
            suspicion_score=suspicion_score,
            message=message,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(batch: BatchUserData):
    """
    Predict bot status for multiple users

    Args:
        batch: Batch of user data

    Returns:
        Batch prediction results
    """
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(batch.users) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 users per batch")

    try:
        # Convert to DataFrame
        users_data = [user.model_dump() for user in batch.users]
        df = pd.DataFrame(users_data)

        # Engineer features
        df = feature_engineer.create_all_features(df)

        # Remove label column if it exists
        if "label" in df.columns:
            df = df.drop("label", axis=1)

        # Make predictions
        predictions = model_data["model"].predict(df)
        probabilities = model_data["model"].predict_proba(df)

        # Prepare response
        results = []
        bot_count = 0

        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            is_bot = bool(pred == 1)
            confidence = float(prob[1] if is_bot else prob[0])
            suspicion_score = float(df["suspicion_score"].iloc[i])
            message = "Bot-like behavior detected" if is_bot else "Appears human"

            if is_bot:
                bot_count += 1

            results.append(
                PredictionResponse(
                    is_bot=is_bot,
                    confidence=confidence,
                    suspicion_score=suspicion_score,
                    message=message,
                )
            )

        return BatchPredictionResponse(
            predictions=results,
            total_users=len(batch.users),
            bot_count=bot_count,
            human_count=len(batch.users) - bot_count,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info")
async def model_info():
    """Get information about the loaded model"""
    if model_data is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_name": model_data["model_name"],
        "model_type": type(model_data["model"]).__name__,
        "feature_importance_available": model_data.get("feature_importance")
        is not None,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
