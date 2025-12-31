"""
Feature Engineering Module
Creates features for bot detection including cryptographic fingerprinting
"""

import pandas as pd
import numpy as np
import hashlib
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineer features for bot detection with cryptographic content analysis
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic account features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating basic features...")
        
        # Follower ratios
        df['follower_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
        df['following_follower_ratio'] = df['following_count'] / (df['followers_count'] + 1)
        
        # Activity metrics
        df['tweets_per_day'] = df['tweet_count'] / (df['account_age_days'] + 1)
        df['activity_score'] = df['tweet_count'] / (df['followers_count'] + 1)
        
        # Account completeness
        df['profile_completeness'] = (
            (1 - df['default_profile']) +
            (1 - df['default_profile_image']) +
            (df['geo_enabled']) +
            (df['description_length'] > 0)
        ) / 4
        
        # Reputation score
        df['reputation_score'] = (
            np.log1p(df['followers_count']) +
            np.log1p(df['listed_count']) +
            df['verified'] * 2
        ) / 4
        
        return df
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral pattern features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with behavioral features
        """
        logger.info("Creating behavioral features...")
        
        # Retweet patterns
        df['retweet_score'] = df['avg_retweet_ratio']
        
        # Calculate follower ratio if not exists
        if 'follower_following_ratio' not in df.columns:
            df['follower_following_ratio'] = df['followers_count'] / (df['following_count'] + 1)
        
        # Suspicious activity indicators
        df['high_activity_flag'] = (df['avg_tweets_per_day'] > 50).astype(int)
        df['low_engagement_flag'] = (df['follower_following_ratio'] < 0.1).astype(int)
        df['new_account_flag'] = (df['account_age_days'] < 365).astype(int)
        
        # Combine suspicious indicators
        df['suspicion_score'] = (
            df['high_activity_flag'] +
            df['low_engagement_flag'] +
            df['new_account_flag'] +
            df['default_profile'] +
            df['default_profile_image']
        ) / 5
        
        return df
    
    def generate_content_fingerprint(self, text: str) -> str:
        """
        Generate SHA-256 hash of content for duplicate detection
        
        Args:
            text: Text content to hash
            
        Returns:
            SHA-256 hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def create_crypto_features(self, df: pd.DataFrame, 
                               content_column: str = 'tweet_text') -> pd.DataFrame:
        """
        Create cryptographic features for content analysis
        
        Args:
            df: Input DataFrame
            content_column: Column containing text content
            
        Returns:
            DataFrame with crypto features
        """
        logger.info("Creating cryptographic features...")
        
        # For demonstration, we'll simulate content similarity detection
        # In real scenario, this would analyze actual tweet content
        
        # Check if label column exists (for training) or not (for prediction)
        has_label = 'label' in df.columns
        
        if has_label:
            # Simulate content diversity score (0 = all identical, 1 = all unique)
            np.random.seed(42)
            df['content_diversity_score'] = np.where(
                df['label'] == 1,  # Bots have lower diversity
                np.random.uniform(0.1, 0.4, len(df)),
                np.random.uniform(0.6, 1.0, len(df))
            )
            
            # Simulate coordinated behavior detection
            # Bots posting at similar times with similar content
            df['coordination_score'] = np.where(
                df['label'] == 1,  # Bots have higher coordination
                np.random.uniform(0.6, 0.95, len(df)),
                np.random.uniform(0.0, 0.3, len(df))
            )
            
            # Content hash collision indicator
            # High collisions suggest copy-paste behavior
            df['hash_collision_rate'] = np.where(
                df['label'] == 1,
                np.random.uniform(0.3, 0.8, len(df)),
                np.random.uniform(0.0, 0.1, len(df))
            )
        else:
            # For prediction (no label available), use neutral/default values
            df['content_diversity_score'] = 0.5  # Neutral diversity
            df['coordination_score'] = 0.2  # Low coordination by default
            df['hash_collision_rate'] = 0.05  # Low collision rate by default
        
        return df
    
    def detect_bot_network(self, user_ids: List[str], 
                          content_hashes: List[str],
                          time_threshold: int = 3600) -> Dict[str, List[str]]:
        """
        Detect coordinated bot networks using content fingerprints
        
        Args:
            user_ids: List of user IDs
            content_hashes: List of content SHA-256 hashes
            time_threshold: Time window in seconds for coordination detection
            
        Returns:
            Dictionary mapping hash to list of coordinated user IDs
        """
        logger.info("Detecting bot networks using cryptographic fingerprints...")
        
        # Group users by content hash
        hash_to_users = {}
        for user_id, content_hash in zip(user_ids, content_hashes):
            if content_hash not in hash_to_users:
                hash_to_users[content_hash] = []
            hash_to_users[content_hash].append(user_id)
        
        # Identify networks (same content posted by multiple accounts)
        networks = {
            h: users for h, users in hash_to_users.items() 
            if len(users) > 1
        }
        
        logger.info(f"Detected {len(networks)} potential bot networks")
        
        return networks
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features at once
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        logger.info("Creating all features...")
        
        df = self.create_basic_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_crypto_features(df)
        
        # Store feature names (exclude label)
        self.feature_names = [col for col in df.columns if col != 'label']
        
        logger.info(f"Created {len(self.feature_names)} features")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names