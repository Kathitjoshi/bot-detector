"""
Main Training Script
Orchestrates the complete training pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_processing.data_loader import DataProcessor
from src.feature_engineering.features import FeatureEngineer
from src.model.train import BotDetectionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main training pipeline"""
    
    logger.info("=" * 50)
    logger.info("BOT DETECTION MODEL TRAINING PIPELINE")
    logger.info("=" * 50)
    
    # Step 1: Load and prepare data
    logger.info("\n[STEP 1] Loading and preparing data...")
    processor = DataProcessor()
    
    # Create synthetic dataset (replace with real data if available)
    df = processor.create_sample_dataset(n_samples=10000, bot_ratio=0.3)
    df = processor.clean_data()
    
    # Save raw data
    processor.save_processed_data("data/processed/cleaned_data.csv")
    
    # Step 2: Feature engineering
    logger.info("\n[STEP 2] Engineering features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_all_features(df)
    
    logger.info(f"Total features created: {len(feature_engineer.get_feature_names())}")
    
    # Step 3: Prepare train/val/test splits
    logger.info("\n[STEP 3] Preparing data splits...")
    model = BotDetectionModel(random_state=42)
    X_train, X_val, X_test, y_train, y_val, y_test = model.prepare_data(
        df_features,
        test_size=0.2,
        val_size=0.2,
        balance_data=True
    )
    
    # Step 4: Train models with hyperparameter tuning
    logger.info("\n[STEP 4] Training models with hyperparameter tuning...")
    training_results = model.train_models(X_train, y_train, X_val, y_val)
    
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("=" * 50)
    for model_name, metrics in training_results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Validation F1: {metrics['val_f1']:.4f}")
        logger.info(f"  Validation AUC: {metrics['val_auc']:.4f}")
        logger.info(f"  Best params: {metrics['best_params']}")
    
    # Step 5: Evaluate on test set
    logger.info("\n[STEP 5] Evaluating best model on test set...")
    test_metrics = model.evaluate_model(X_test, y_test)
    
    # Step 6: Save model
    logger.info("\n[STEP 6] Saving model...")
    model.save_model("models/saved_models/best_model.joblib")
    
    # Final summary
    logger.info("\n" + "=" * 50)
    logger.info("FINAL TEST SET PERFORMANCE")
    logger.info("=" * 50)
    logger.info(f"Model: {model.best_model_name}")
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Recall: {test_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")
    
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 50)
    logger.info("\nNext steps:")
    logger.info("1. Run tests: pytest tests/")
    logger.info("2. Start API: python src/api/main.py")
    logger.info("3. Deploy: docker build -t bot-detector .")


if __name__ == "__main__":
    main()