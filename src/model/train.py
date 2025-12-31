"""
Model Training Module
Trains and evaluates bot detection models with anti-overfitting measures
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import joblib
import logging
from typing import Dict, Tuple, Any
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BotDetectionModel:
    """
    Train and evaluate bot detection models with proper anti-overfitting measures
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.2,
        balance_data: bool = True,
    ) -> Tuple:
        """
        Prepare train/validation/test splits with optional SMOTE

        Args:
            df: Input DataFrame with features and labels
            test_size: Proportion of test set
            val_size: Proportion of validation set from remaining data
            balance_data: Whether to apply SMOTE for class balancing

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Preparing data splits...")

        # Separate features and labels
        X = df.drop("label", axis=1)
        y = df["label"]

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Validation set: {len(X_val)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Apply SMOTE only to training data to prevent overfitting
        if balance_data:
            logger.info("Applying SMOTE to balance training data...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE - Train set: {len(X_train)} samples")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, Any]:
        """
        Train multiple models with hyperparameter tuning

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary of trained models and their performance
        """
        logger.info("Training models with hyperparameter tuning...")

        # Define models with anti-overfitting parameters
        models_config = {
            "logistic_regression": {
                "model": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                "params": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
            },
            "random_forest": {
                "model": RandomForestClassifier(random_state=self.random_state),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2"],
                },
            },
            "xgboost": {
                "model": XGBClassifier(
                    random_state=self.random_state,
                    use_label_encoder=False,
                    eval_metric="logloss",
                ),
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0],
                    "reg_alpha": [0, 0.1],
                    "reg_lambda": [1, 1.5],
                },
            },
        }

        results = {}

        # Use Stratified K-Fold for cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        for name, config in models_config.items():
            logger.info(f"\nTraining {name}...")

            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config["model"],
                config["params"],
                cv=cv,
                scoring="f1",
                n_jobs=-1,
                verbose=1,
            )

            grid_search.fit(X_train, y_train)

            # Store best model
            best_model = grid_search.best_estimator_
            self.models[name] = best_model

            # Validate on validation set
            y_pred = best_model.predict(X_val)
            y_pred_proba = best_model.predict_proba(X_val)[:, 1]

            # Calculate metrics
            metrics = {
                "best_params": grid_search.best_params_,
                "train_score": grid_search.best_score_,
                "val_accuracy": accuracy_score(y_val, y_pred),
                "val_precision": precision_score(y_val, y_pred),
                "val_recall": recall_score(y_val, y_pred),
                "val_f1": f1_score(y_val, y_pred),
                "val_auc": roc_auc_score(y_val, y_pred_proba),
            }

            results[name] = metrics

            logger.info(f"{name} - Best params: {metrics['best_params']}")
            logger.info(f"{name} - Validation F1: {metrics['val_f1']:.4f}")
            logger.info(f"{name} - Validation AUC: {metrics['val_auc']:.4f}")

        # Select best model based on validation F1 score
        self.best_model_name = max(results, key=lambda x: results[x]["val_f1"])
        self.best_model = self.models[self.best_model_name]

        logger.info(f"\nBest model: {self.best_model_name}")

        return results

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate best model on test set

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No model trained. Call train_models() first.")

        logger.info(f"\nEvaluating {self.best_model_name} on test set...")

        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "auc_roc": roc_auc_score(y_test, y_pred_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
        }

        logger.info(f"\nTest Set Performance:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
        logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        logger.info(f"\n{metrics['classification_report']}")

        # Get feature importance for tree-based models
        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = self.best_model.feature_importances_

        return metrics

    def save_model(self, save_path: str):
        """
        Save the best model to disk

        Args:
            save_path: Path to save the model
        """
        if self.best_model is None:
            raise ValueError("No model to save. Train a model first.")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.best_model,
            "model_name": self.best_model_name,
            "feature_importance": self.feature_importance,
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load a saved model from disk

        Args:
            load_path: Path to the saved model
        """
        model_data = joblib.load(load_path)
        self.best_model = model_data["model"]
        self.best_model_name = model_data["model_name"]
        self.feature_importance = model_data.get("feature_importance")
        logger.info(f"Model loaded from {load_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model

        Args:
            X: Features to predict

        Returns:
            Predictions array
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.best_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            X: Features to predict

        Returns:
            Probability array
        """
        if self.best_model is None:
            raise ValueError("No model loaded. Train or load a model first.")

        return self.best_model.predict_proba(X)[:, 1]
