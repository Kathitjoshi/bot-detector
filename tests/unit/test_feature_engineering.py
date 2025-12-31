"""
Unit Tests for Feature Engineering Module
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.feature_engineering.features import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class"""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        return pd.DataFrame({
            'followers_count': [100, 500, 1000],
            'following_count': [200, 300, 800],
            'tweet_count': [1000, 5000, 10000],
            'account_age_days': [365, 730, 1095],
            'listed_count': [5, 10, 20],
            'verified': [0, 1, 0],
            'default_profile': [1, 0, 0],
            'default_profile_image': [1, 0, 0],
            'geo_enabled': [0, 1, 1],
            'description_length': [20, 100, 150],
            'avg_tweets_per_day': [50, 10, 5],
            'avg_retweet_ratio': [0.8, 0.3, 0.2],
            'label': [1, 0, 0]
        })
    
    def test_create_basic_features(self, engineer, sample_data):
        """Test basic feature creation"""
        df_with_features = engineer.create_basic_features(sample_data.copy())
        
        # Check new features exist
        expected_features = [
            'follower_following_ratio',
            'following_follower_ratio',
            'tweets_per_day',
            'activity_score',
            'profile_completeness',
            'reputation_score'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
    
    def test_follower_ratio_calculation(self, engineer, sample_data):
        """Test follower ratio calculation"""
        df = engineer.create_basic_features(sample_data.copy())
        
        # Manual calculation for first row
        expected_ratio = 100 / (200 + 1)
        assert abs(df['follower_following_ratio'].iloc[0] - expected_ratio) < 0.01
    
    def test_create_behavioral_features(self, engineer, sample_data):
        """Test behavioral feature creation"""
        df_with_features = engineer.create_behavioral_features(sample_data.copy())
        
        expected_features = [
            'retweet_score',
            'high_activity_flag',
            'low_engagement_flag',
            'new_account_flag',
            'suspicion_score'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
    
    def test_suspicion_score_range(self, engineer, sample_data):
        """Test that suspicion score is in valid range [0, 1]"""
        df = engineer.create_behavioral_features(sample_data.copy())
        
        assert (df['suspicion_score'] >= 0).all()
        assert (df['suspicion_score'] <= 1).all()
    
    def test_create_crypto_features(self, engineer, sample_data):
        """Test cryptographic feature creation"""
        df_with_features = engineer.create_crypto_features(sample_data.copy())
        
        expected_features = [
            'content_diversity_score',
            'coordination_score',
            'hash_collision_rate'
        ]
        
        for feature in expected_features:
            assert feature in df_with_features.columns
    
    def test_content_fingerprint_generation(self, engineer):
        """Test SHA-256 fingerprint generation"""
        text1 = "This is a test tweet"
        text2 = "This is a test tweet"
        text3 = "This is a different tweet"
        
        hash1 = engineer.generate_content_fingerprint(text1)
        hash2 = engineer.generate_content_fingerprint(text2)
        hash3 = engineer.generate_content_fingerprint(text3)
        
        # Same content should produce same hash
        assert hash1 == hash2
        
        # Different content should produce different hash
        assert hash1 != hash3
        
        # Hash should be 64 characters (SHA-256)
        assert len(hash1) == 64
    
    def test_detect_bot_network(self, engineer):
        """Test bot network detection"""
        user_ids = ['user1', 'user2', 'user3', 'user4']
        content_hashes = [
            'hash_a',  # user1
            'hash_a',  # user2 (same as user1)
            'hash_b',  # user3
            'hash_a'   # user4 (same as user1 and user2)
        ]
        
        networks = engineer.detect_bot_network(user_ids, content_hashes)
        
        # Should detect hash_a network with 3 users
        assert 'hash_a' in networks
        assert len(networks['hash_a']) == 3
        assert set(networks['hash_a']) == {'user1', 'user2', 'user4'}
        
        # hash_b should not be in networks (only 1 user)
        assert 'hash_b' not in networks
    
    def test_create_all_features(self, engineer, sample_data):
        """Test creating all features at once"""
        df_with_features = engineer.create_all_features(sample_data.copy())
        
        # Should have more columns than original
        assert len(df_with_features.columns) > len(sample_data.columns)
        
        # Original columns should still exist
        for col in sample_data.columns:
            assert col in df_with_features.columns
    
    def test_get_feature_names(self, engineer, sample_data):
        """Test getting feature names"""
        engineer.create_all_features(sample_data.copy())
        feature_names = engineer.get_feature_names()
        
        # Should return a list
        assert isinstance(feature_names, list)
        
        # Should not include label
        assert 'label' not in feature_names
        
        # Should have multiple features
        assert len(feature_names) > 10
    
    def test_no_nan_in_features(self, engineer, sample_data):
        """Test that no NaN values are created in features"""
        df = engineer.create_all_features(sample_data.copy())
        
        # Check for NaN values (excluding label)
        feature_cols = [col for col in df.columns if col != 'label']
        assert not df[feature_cols].isnull().any().any()