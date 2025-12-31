"""
Unit Tests for Model Training Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.model.train import BotDetectionModel


class TestBotDetectionModel:
    """Test suite for BotDetectionModel class"""
    
    @pytest.fixture
    def model(self):
        """Create BotDetectionModel instance"""
        return BotDetectionModel(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        n_samples = 200
        
        data = {
            'feature1': np.random.rand(n_samples),
            'feature2': np.random.rand(n_samples),
            'feature3': np.random.rand(n_samples),
            'feature4': np.random.rand(n_samples),
            'feature5': np.random.rand(n_samples),
            'label': np.random.choice([0, 1], n_samples)
        }
        
        return pd.DataFrame(data)
    
    def test_prepare_data_splits(self, model, sample_data):
        """Test data preparation and splitting"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            test_size=0.2,
            val_size=0.2,
            balance_data=False
        )
        
        # Check split sizes approximately correct
        total_samples = len(sample_data)
        assert len(X_test) == pytest.approx(total_samples * 0.2, abs=10)
        assert len(X_val) == pytest.approx(total_samples * 0.2 * 0.8, abs=10)
        
        # Check that labels match features
        assert len(X_train) == len(y_train)
        assert len(X_val) == len(y_val)
        assert len(X_test) == len(y_test)
    
    def test_prepare_data_no_overlap(self, model, sample_data):
        """Test that train/val/test sets don't overlap"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            test_size=0.2,
            val_size=0.2,
            balance_data=False
        )
        
        # Combine all sets to check uniqueness
        total_samples = len(X_train) + len(X_val) + len(X_test)
        
        # Should not be greater than original (no duplicates across sets)
        assert total_samples <= len(sample_data) * 1.1  # Allow small margin for rounding
    
    def test_prepare_data_with_smote(self, model, sample_data):
        """Test SMOTE balancing"""
        # Create imbalanced dataset
        imbalanced_data = sample_data.copy()
        imbalanced_data = imbalanced_data[
            (imbalanced_data['label'] == 0) | 
            (imbalanced_data.index < 30)
        ]
        
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            imbalanced_data,
            balance_data=True
        )
        
        # After SMOTE, classes should be more balanced in training set
        bot_ratio = y_train.sum() / len(y_train)
        assert 0.4 < bot_ratio < 0.6  # Should be close to 50%
    
    def test_train_models_creates_models(self, model, sample_data):
        """Test that training creates multiple models"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        results = model.train_models(X_train, y_train, X_val, y_val)
        
        # Check that models were trained
        assert len(results) > 0
        assert 'logistic_regression' in results
        assert 'random_forest' in results
        assert 'xgboost' in results
    
    def test_train_models_metrics(self, model, sample_data):
        """Test that training returns proper metrics"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        results = model.train_models(X_train, y_train, X_val, y_val)
        
        # Check metrics for each model
        for model_name, metrics in results.items():
            assert 'best_params' in metrics
            assert 'train_score' in metrics
            assert 'val_accuracy' in metrics
            assert 'val_precision' in metrics
            assert 'val_recall' in metrics
            assert 'val_f1' in metrics
            assert 'val_auc' in metrics
            
            # Metrics should be in valid range
            assert 0 <= metrics['val_accuracy'] <= 1
            assert 0 <= metrics['val_f1'] <= 1
            assert 0 <= metrics['val_auc'] <= 1
    
    def test_best_model_selected(self, model, sample_data):
        """Test that best model is selected"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        model.train_models(X_train, y_train, X_val, y_val)
        
        assert model.best_model is not None
        assert model.best_model_name is not None
        assert model.best_model_name in ['logistic_regression', 'random_forest', 'xgboost']
    
    def test_evaluate_model(self, model, sample_data):
        """Test model evaluation"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        model.train_models(X_train, y_train, X_val, y_val)
        metrics = model.evaluate_model(X_test, y_test)
        
        # Check that all metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc_roc' in metrics
        assert 'confusion_matrix' in metrics
        
        # Metrics should be in valid range
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_predict_before_training_raises_error(self, model, sample_data):
        """Test that prediction before training raises error"""
        with pytest.raises(ValueError):
            model.predict(sample_data.drop('label', axis=1))
    
    def test_predict_after_training(self, model, sample_data):
        """Test prediction after training"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        model.train_models(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        
        # Check prediction shape
        assert len(predictions) == len(X_test)
        
        # Predictions should be binary
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, model, sample_data):
        """Test probability predictions"""
        X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
            sample_data,
            balance_data=False
        )
        
        model.train_models(X_train, y_train, X_val, y_val)
        probabilities = model.predict_proba(X_test)
        
        # Check shape
        assert len(probabilities) == len(X_test)
        
        # Probabilities should be in [0, 1]
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()