"""
Integration Tests for FastAPI Application
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app
from src.data_processing.data_loader import DataProcessor
from src.feature_engineering.features import FeatureEngineer
from src.model.train import BotDetectionModel


@pytest.fixture(scope="module")
def test_client():
    """Create test client"""
    client = TestClient(app)
    return client


@pytest.fixture(scope="module", autouse=True)
def setup_test_model():
    """Setup a test model before running integration tests"""
    # Create and train a simple model for testing
    processor = DataProcessor()
    df = processor.create_sample_dataset(n_samples=500, bot_ratio=0.3)
    df = processor.clean_data()
    
    engineer = FeatureEngineer()
    df = engineer.create_all_features(df)
    
    model = BotDetectionModel(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
        df, balance_data=False
    )
    
    model.train_models(X_train, y_train, X_val, y_val)
    
    # Save test model
    model_path = Path(__file__).parent.parent / "models" / "saved_models"
    model_path.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path / "best_model.joblib"))
    
    # Force reload the model in the app
    import src.api.main as main_module
    main_module.model_data = joblib.load(str(model_path / "best_model.joblib"))
    
    yield
    
    # Cleanup (optional)


class TestAPIEndpoints:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "status" in data
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self, test_client):
        """Test model info endpoint"""
        response = test_client.get("/model_info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "model_type" in data
    
    def test_predict_endpoint_bot(self, test_client):
        """Test prediction endpoint with bot-like data"""
        bot_data = {
            "followers_count": 50,
            "following_count": 3000,
            "tweet_count": 10000,
            "account_age_days": 90,
            "listed_count": 1,
            "verified": 0,
            "default_profile": 1,
            "default_profile_image": 1,
            "geo_enabled": 0,
            "description_length": 10,
            "avg_tweets_per_day": 100.0,
            "avg_retweet_ratio": 0.9
        }
        
        response = test_client.post("/predict", json=bot_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_bot" in data
        assert "confidence" in data
        assert "suspicion_score" in data
        assert "message" in data
        
        # Bot-like data should likely be classified as bot
        assert isinstance(data["is_bot"], bool)
        assert 0 <= data["confidence"] <= 1
        assert 0 <= data["suspicion_score"] <= 1
    
    def test_predict_endpoint_human(self, test_client):
        """Test prediction endpoint with human-like data"""
        human_data = {
            "followers_count": 500,
            "following_count": 300,
            "tweet_count": 2000,
            "account_age_days": 1200,
            "listed_count": 10,
            "verified": 0,
            "default_profile": 0,
            "default_profile_image": 0,
            "geo_enabled": 1,
            "description_length": 100,
            "avg_tweets_per_day": 5.0,
            "avg_retweet_ratio": 0.3
        }
        
        response = test_client.post("/predict", json=human_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "is_bot" in data
        assert "confidence" in data
        assert isinstance(data["is_bot"], bool)
    
    def test_predict_endpoint_invalid_data(self, test_client):
        """Test prediction endpoint with invalid data"""
        invalid_data = {
            "followers_count": -100,  # Invalid negative value
            "following_count": 1000
        }
        
        response = test_client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self, test_client):
        """Test batch prediction endpoint"""
        batch_data = {
            "users": [
                {
                    "followers_count": 50,
                    "following_count": 3000,
                    "tweet_count": 10000,
                    "account_age_days": 90,
                    "listed_count": 1,
                    "verified": 0,
                    "default_profile": 1,
                    "default_profile_image": 1,
                    "geo_enabled": 0,
                    "description_length": 10,
                    "avg_tweets_per_day": 100.0,
                    "avg_retweet_ratio": 0.9
                },
                {
                    "followers_count": 500,
                    "following_count": 300,
                    "tweet_count": 2000,
                    "account_age_days": 1200,
                    "listed_count": 10,
                    "verified": 0,
                    "default_profile": 0,
                    "default_profile_image": 0,
                    "geo_enabled": 1,
                    "description_length": 100,
                    "avg_tweets_per_day": 5.0,
                    "avg_retweet_ratio": 0.3
                }
            ]
        }
        
        response = test_client.post("/batch_predict", json=batch_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert "total_users" in data
        assert "bot_count" in data
        assert "human_count" in data
        
        assert data["total_users"] == 2
        assert len(data["predictions"]) == 2
        assert data["bot_count"] + data["human_count"] == 2
    
    def test_batch_predict_too_many_users(self, test_client):
        """Test batch prediction with too many users"""
        # Create batch with > 100 users
        batch_data = {
            "users": [
                {
                    "followers_count": 100,
                    "following_count": 200,
                    "tweet_count": 1000,
                    "account_age_days": 365,
                    "listed_count": 5,
                    "verified": 0,
                    "default_profile": 0,
                    "default_profile_image": 0,
                    "geo_enabled": 1,
                    "description_length": 50,
                    "avg_tweets_per_day": 10.0,
                    "avg_retweet_ratio": 0.5
                }
            ] * 101  # 101 users
        }
        
        response = test_client.post("/batch_predict", json=batch_data)
        assert response.status_code == 400


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline"""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to prediction"""
        # Step 1: Create data
        processor = DataProcessor()
        df = processor.create_sample_dataset(n_samples=200, bot_ratio=0.3)
        df = processor.clean_data()
        
        # Step 2: Engineer features
        engineer = FeatureEngineer()
        df = engineer.create_all_features(df)
        
        # Step 3: Train model
        model = BotDetectionModel(random_state=42)
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            df, balance_data=False
        )
        
        results = model.train_models(X_train, y_train, X_val, y_val)
        
        # Step 4: Evaluate
        metrics = model.evaluate_model(X_test, y_test)
        
        # Assertions
        assert model.best_model is not None
        assert metrics['accuracy'] > 0.5  # Should be better than random
        assert len(results) == 3  # Three models trained
        
    def test_prediction_consistency(self):
        """Test that predictions are consistent"""
        # Create test data
        test_data = pd.DataFrame({
            'followers_count': [100],
            'following_count': [200],
            'tweet_count': [1000],
            'account_age_days': [365],
            'listed_count': [5],
            'verified': [0],
            'default_profile': [0],
            'default_profile_image': [0],
            'geo_enabled': [1],
            'description_length': [50],
            'avg_tweets_per_day': [10.0],
            'avg_retweet_ratio': [0.5],
            'label': [0]
        })
        
        # Engineer features
        engineer = FeatureEngineer()
        df = engineer.create_all_features(test_data)
        
        # Load model
        model = BotDetectionModel()
        model_path = Path(__file__).parent.parent / "models" / "saved_models" / "best_model.joblib"
        model.load_model(str(model_path))
        
        # Make multiple predictions
        X = df.drop('label', axis=1)
        pred1 = model.predict(X)
        pred2 = model.predict(X)
        pred3 = model.predict(X)
        
        # Predictions should be identical
        assert np.array_equal(pred1, pred2)
        assert np.array_equal(pred2, pred3)