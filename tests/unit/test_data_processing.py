"""
Unit Tests for Data Processing Module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.data_loader import DataProcessor


class TestDataProcessor:
    """Test suite for DataProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create DataProcessor instance"""
        return DataProcessor()
    
    def test_create_sample_dataset(self, processor):
        """Test synthetic dataset creation"""
        df = processor.create_sample_dataset(n_samples=1000, bot_ratio=0.3)
        
        assert len(df) == 1000
        assert 'label' in df.columns
        assert df['label'].sum() == 300  # 30% bots
        assert df['label'].nunique() == 2
        
    def test_create_sample_dataset_columns(self, processor):
        """Test that all required columns are present"""
        df = processor.create_sample_dataset(n_samples=100)
        
        required_columns = [
            'followers_count', 'following_count', 'tweet_count',
            'account_age_days', 'listed_count', 'verified',
            'default_profile', 'default_profile_image', 'geo_enabled',
            'description_length', 'avg_tweets_per_day', 'avg_retweet_ratio',
            'label'
        ]
        
        for col in required_columns:
            assert col in df.columns
    
    def test_clean_data(self, processor):
        """Test data cleaning"""
        # Create dataset with some issues
        processor.create_sample_dataset(n_samples=100)
        
        # Add a duplicate row
        processor.df = pd.concat([processor.df, processor.df.iloc[[0]]], ignore_index=True)
        initial_count = len(processor.df)
        
        # Clean data
        cleaned_df = processor.clean_data()
        
        assert len(cleaned_df) < initial_count
        assert cleaned_df.isnull().sum().sum() == 0
    
    def test_clean_data_removes_invalid_values(self, processor):
        """Test that negative values are removed"""
        processor.create_sample_dataset(n_samples=100)
        
        # Add invalid negative values
        processor.df.loc[0, 'followers_count'] = -1
        
        cleaned_df = processor.clean_data()
        
        assert (cleaned_df['followers_count'] >= 0).all()
        assert (cleaned_df['following_count'] >= 0).all()
    
    def test_data_types(self, processor):
        """Test that data types are correct"""
        df = processor.create_sample_dataset(n_samples=100)
        
        int_columns = [
            'followers_count', 'following_count', 'tweet_count',
            'account_age_days', 'listed_count', 'verified',
            'default_profile', 'default_profile_image', 'geo_enabled',
            'description_length', 'label'
        ]
        
        for col in int_columns:
            assert df[col].dtype in ['int64', 'int32', 'float64']
    
    def test_bot_ratio_accuracy(self, processor):
        """Test that bot ratio is accurate"""
        ratios_to_test = [0.1, 0.3, 0.5, 0.7]
        
        for ratio in ratios_to_test:
            df = processor.create_sample_dataset(n_samples=1000, bot_ratio=ratio)
            actual_ratio = df['label'].sum() / len(df)
            assert abs(actual_ratio - ratio) < 0.01  # Within 1%