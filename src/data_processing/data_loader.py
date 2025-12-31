"""
Data Processing Module
Handles loading, cleaning, and preprocessing of user data
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Process and clean social media user data for bot detection
    """

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.df = None

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load dataset from CSV file

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame
        """
        path = file_path or self.data_path

        if not path:
            raise ValueError("No data path provided")

        logger.info(f"Loading data from {path}")

        try:
            self.df = pd.read_csv(path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def create_sample_dataset(
        self, n_samples: int = 10000, bot_ratio: float = 0.3
    ) -> pd.DataFrame:
        """
        Create synthetic dataset for testing

        Args:
            n_samples: Number of samples to generate
            bot_ratio: Proportion of bot accounts

        Returns:
            Synthetic DataFrame
        """
        np.random.seed(42)

        n_bots = int(n_samples * bot_ratio)
        n_humans = n_samples - n_bots

        # Generate bot accounts (suspicious patterns)
        bots = pd.DataFrame(
            {
                "followers_count": np.random.randint(10, 500, n_bots),
                "following_count": np.random.randint(500, 5000, n_bots),
                "tweet_count": np.random.randint(1000, 50000, n_bots),
                "account_age_days": np.random.randint(10, 300, n_bots),
                "listed_count": np.random.randint(0, 5, n_bots),
                "verified": np.zeros(n_bots),
                "default_profile": np.ones(n_bots),
                "default_profile_image": np.random.choice([0, 1], n_bots, p=[0.3, 0.7]),
                "geo_enabled": np.random.choice([0, 1], n_bots, p=[0.7, 0.3]),
                "description_length": np.random.randint(0, 50, n_bots),
                "avg_tweets_per_day": np.random.uniform(20, 200, n_bots),
                "avg_retweet_ratio": np.random.uniform(0.7, 0.95, n_bots),
                "label": np.ones(n_bots),
            }
        )

        # Generate human accounts (normal patterns)
        humans = pd.DataFrame(
            {
                "followers_count": np.random.randint(100, 5000, n_humans),
                "following_count": np.random.randint(100, 2000, n_humans),
                "tweet_count": np.random.randint(50, 10000, n_humans),
                "account_age_days": np.random.randint(365, 3650, n_humans),
                "listed_count": np.random.randint(0, 100, n_humans),
                "verified": np.random.choice([0, 1], n_humans, p=[0.95, 0.05]),
                "default_profile": np.zeros(n_humans),
                "default_profile_image": np.random.choice(
                    [0, 1], n_humans, p=[0.95, 0.05]
                ),
                "geo_enabled": np.random.choice([0, 1], n_humans, p=[0.3, 0.7]),
                "description_length": np.random.randint(50, 160, n_humans),
                "avg_tweets_per_day": np.random.uniform(1, 20, n_humans),
                "avg_retweet_ratio": np.random.uniform(0.1, 0.5, n_humans),
                "label": np.zeros(n_humans),
            }
        )

        self.df = pd.concat([bots, humans], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Created synthetic dataset with {n_samples} samples")
        logger.info(f"Bot ratio: {bot_ratio:.2%}")

        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the dataset

        Returns:
            Cleaned DataFrame
        """
        if self.df is None:
            raise ValueError(
                "No data loaded. Call load_data() or create_sample_dataset() first"
            )

        logger.info("Cleaning data...")

        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_count - len(self.df)} duplicate rows")

        # Handle missing values
        self.df = self.df.fillna(0)

        # Remove invalid values
        self.df = self.df[self.df["followers_count"] >= 0]
        self.df = self.df[self.df["following_count"] >= 0]
        self.df = self.df[self.df["tweet_count"] >= 0]

        logger.info(f"Cleaned dataset contains {len(self.df)} records")

        return self.df

    def save_processed_data(self, output_path: str):
        """
        Save processed data to CSV

        Args:
            output_path: Path to save the processed data
        """
        if self.df is None:
            raise ValueError("No data to save")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
