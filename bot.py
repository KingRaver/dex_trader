#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import sys
import os
import time
import requests
import re
import numpy as np
from datetime import datetime, timedelta, timezone
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff
import anthropic
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.common.keys import Keys
import random
import statistics
import threading
import queue
import json
from llm_provider import LLMProvider
from database import CryptoDatabase

from utils.logger import logger
from utils.browser import browser
from config import config
from coingecko_handler import CoinGeckoHandler
from mood_config import MoodIndicators, determine_advanced_mood, Mood, MemePhraseGenerator
from meme_phrases import MEME_PHRASES
from prediction_engine import EnhancedPredictionEngine, MachineLearningModels, StatisticalModels

# Import modules for reply functionality
from timeline_scraper import TimelineScraper
from reply_handler import ReplyHandler
from content_analyzer import ContentAnalyzer

from database import CryptoDatabase

class CryptoAnalysisBot:
    """
    Enhanced crypto analysis bot with market and tech content capabilities.
    Handles market analysis, predictions, and social media engagement.
    """

    def __init__(self, database, llm_provider, config=None):
        """
        Initialize the crypto analysis bot with improved configuration and tracking.
        Sets up connections to browser, database, and API services.
        """
        self.browser = browser
        self.config = config
        self.llm_provider = LLMProvider(self.config)  
        self.past_predictions = []
        self.meme_phrases = MEME_PHRASES
        self.market_conditions = {}
        self.last_check_time = strip_timezone(datetime.now())
        self.last_market_data = {}
        self.last_reply_time = strip_timezone(datetime.now())
        self.db = database
        
        # Multi-timeframe prediction tracking
        self.timeframes = ["1h", "24h", "7d"]
        self.timeframe_predictions = {tf: {} for tf in self.timeframes}
        self.timeframe_last_post = {tf: strip_timezone(datetime.now() - timedelta(hours=3)) for tf in self.timeframes}
       
        # Timeframe posting frequency controls (in hours)
        self.timeframe_posting_frequency = {
            "1h": 1,    # Every hour
            "24h": 6,   # Every 6 hours
            "7d": 24    # Once per day
        }
       
        # Prediction accuracy tracking by timeframe
        self.prediction_accuracy = {tf: {'correct': 0, 'total': 0} for tf in self.timeframes}
       
        # Initialize prediction engine with database and LLM Provider
        self.prediction_engine = EnhancedPredictionEngine(
            database=self.config.db,
            llm_provider=self.llm_provider
        )
       
        # Create a queue for predictions to process
        self.prediction_queue = queue.Queue()
       
        # Initialize thread for async prediction generation
        self.prediction_thread = None
        self.prediction_thread_running = False
       
        # Initialize CoinGecko handler with 60s cache duration
        self.coingecko = CoinGeckoHandler(
            base_url=self.config.COINGECKO_BASE_URL,
            cache_duration=60
        )
        
        # Target chains to analyze
        self.target_chains = {
            'BTC': 'bitcoin', 
            'ETH': 'ethereum',
            'SOL': 'solana',
            'XRP': 'ripple',
            'BNB': 'binancecoin',
            'AVAX': 'avalanche-2',
            'DOT': 'polkadot',
            'UNI': 'uniswap',
            'NEAR': 'near',
            'AAVE': 'aave',
            'FIL': 'filecoin',
            'POL': 'matic-network',
            'TRUMP': 'official-trump',
            'KAITO': 'kaito'
        }

        # All tokens for reference and comparison
        self.reference_tokens = list(self.target_chains.keys())
       
        # Chain name mapping for display
        self.chain_name_mapping = self.target_chains.copy()
       
        self.CORRELATION_THRESHOLD = 0.75  
        self.VOLUME_THRESHOLD = 0.60  
        self.TIME_WINDOW = 24
       
        # Smart money thresholds
        self.SMART_MONEY_VOLUME_THRESHOLD = 1.5  # 50% above average
        self.SMART_MONEY_ZSCORE_THRESHOLD = 2.0  # 2 standard deviations
       
        # Timeframe-specific triggers and thresholds
        self.timeframe_thresholds = {
            "1h": {
                "price_change": 3.0,    # 3% price change for 1h predictions
                "volume_change": 8.0,   # 8% volume change
                "confidence": 70,       # Minimum confidence percentage
                "fomo_factor": 1.0      # FOMO enhancement factor
            },
            "24h": {
                "price_change": 5.0,    # 5% price change for 24h predictions
                "volume_change": 12.0,  # 12% volume change
                "confidence": 65,       # Slightly lower confidence for longer timeframe
                "fomo_factor": 1.2      # Higher FOMO factor
            },
            "7d": {
                "price_change": 8.0,    # 8% price change for 7d predictions
                "volume_change": 15.0,  # 15% volume change
                "confidence": 60,       # Even lower confidence for weekly predictions
                "fomo_factor": 1.5      # Highest FOMO factor
            }
        }
       
        # Initialize scheduled timeframe posts
        self.next_scheduled_posts = {
            "1h": strip_timezone(datetime.now() + timedelta(minutes=random.randint(10, 30))),
            "24h": strip_timezone(datetime.now() + timedelta(hours=random.randint(1, 3))),
            "7d": strip_timezone(datetime.now() + timedelta(hours=random.randint(4, 8)))
        }
       
        # Initialize reply functionality components
        self.timeline_scraper = TimelineScraper(self.browser, self.config, self.config.db)
        self.reply_handler = ReplyHandler(self.browser, self.config, self.llm_provider, self.coingecko, self.config.db)
        self.content_analyzer = ContentAnalyzer(self.config, self.config.db)
       
        # Reply tracking and control
        self.last_reply_check = strip_timezone(datetime.now() - timedelta(minutes=30))  # Start checking soon
        self.reply_check_interval = 20  # Check for posts to reply to every 20 minutes
        self.max_replies_per_cycle = 10  # Maximum 10 replies per cycle
        self.reply_cooldown = 15  # Minutes between reply cycles
        self.last_reply_time = strip_timezone(datetime.now() - timedelta(minutes=self.reply_cooldown))  # Allow immediate first run

        logger.logger.info("Enhanced Prediction Engine initialized with adaptive architecture") 
        logger.log_startup()
    
    def _ensure_dict_data(self, data):
        """
        Ensure data is dictionary-like and not a list or string
    
        Args:
            data: Data to check
        
        Returns:
            Dictionary version of data or empty dict if conversion not possible
        """
        if isinstance(data, dict):
            return data
        elif isinstance(data, list):
            # Try to convert list to dict using 'symbol' as key if available
            result = {}
            for item in data:
                if isinstance(item, dict) and 'symbol' in item:
                    symbol = item['symbol'].upper()
                    result[symbol] = item
            return result
        elif isinstance(data, str):
            # Try to parse as JSON
            try:
                parsed = json.loads(data)
                return self._ensure_dict_data(parsed)  # Recursive call to handle parsed result
            except:
                return {}
        else:
            return {}

    def _standardize_market_data(self, market_data):
        """
        Standardize market data to dictionary format with token symbols as keys.
        """
        # Initialize the result dictionary
        result = {}

        # If None, return empty dict
        if market_data is None:
            return {}
    
        # If already a dictionary with expected structure, return as is
        if isinstance(market_data, dict) and any(isinstance(key, str) for key in market_data.keys()):
            return market_data
    
        # Convert list to dictionary
        if isinstance(market_data, list):
            # Log for debugging
            logger.logger.debug(f"Converting list market_data to dict, length: {len(market_data)}")
        
            # Map of CoinGecko IDs to token symbols (uppercase for consistency)
            id_to_symbol_map = {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'binancecoin': 'BNB',
                'solana': 'SOL',
                'ripple': 'XRP',
                'polkadot': 'DOT',
                'avalanche-2': 'AVAX',
                'polygon-pos': 'POL',
                'near': 'NEAR',
                'filecoin': 'FIL',
                'uniswap': 'UNI',
                'aave': 'AAVE',
                'kaito': 'KAITO',
                'trump': 'TRUMP',
                'matic-network': 'MATIC'
                # Add other mappings as needed
            }
        
            # Process each item in the list
            for item in market_data:
                if not isinstance(item, dict):
                    continue
            
                # Try to find symbol based on ID first (most reliable)
                if 'id' in item:
                    item_id = item['id']
                    # Use the mapping if available
                    if item_id in id_to_symbol_map:
                        symbol = id_to_symbol_map[item_id]
                        result[symbol] = item
                        # Also add with ID as key for backward compatibility
                        result[item_id] = item
                    # Otherwise fall back to using the ID as is
                    else:
                        result[item_id] = item
            
                # If no ID, try using symbol directly
                elif 'symbol' in item:
                    symbol = item['symbol'].upper()  # Uppercase for consistency
                    result[symbol] = item

        # Log the result for debugging
        logger.logger.debug(f"Standardized market_data has {len(result)} items")
    
        return result
    
    @ensure_naive_datetimes
    def _check_for_reply_opportunities(self, market_data: Dict[str, Any]) -> bool:
        """
        Enhanced check for posts to reply to with multiple fallback mechanisms
        and detailed logging for better debugging
    
        Args:
            market_data: Current market data dictionary
        
        Returns:
            True if any replies were posted
        """
        now = strip_timezone(datetime.now())

        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 20
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
    
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 20
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
    
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Try multiple post gathering strategies with fallbacks
            success = self._try_normal_reply_strategy(market_data)
            if success:
                return True
            
            # First fallback: Try with lower threshold for reply-worthy posts
            success = self._try_lower_threshold_reply_strategy(market_data)
            if success:
                return True
            
            # Second fallback: Try replying to trending posts even if not directly crypto-related
            success = self._try_trending_posts_reply_strategy(market_data)
            if success:
                return True
        
            # Final fallback: Try replying to any post from major crypto accounts
            success = self._try_crypto_accounts_reply_strategy(market_data)
            if success:
                return True
        
            logger.logger.warning("All reply strategies failed, no suitable posts found")
            return False
        
        except Exception as e:
            logger.log_error("Check For Reply Opportunities", str(e))
            return False

    @ensure_naive_datetimes
    def _try_normal_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Standard reply strategy with normal thresholds
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get more posts to increase chances of finding suitable ones
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False
            
            logger.logger.info(f"Timeline scraping completed - found {len(posts)} posts")
        
            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")
        
            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts")
        
            if not market_posts:
                logger.logger.warning("No market-related posts found")
                return False
            
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
        
            if not unreplied_posts:
                logger.logger.warning("All market-related posts have already been replied to")
                return False
            
            # Analyze content of each post for engagement metrics
            analyzed_posts = []
            for post in unreplied_posts:
                analysis = self.content_analyzer.analyze_post(post)
                post['content_analysis'] = analysis
                analyzed_posts.append(post)
        
            # Only reply to posts worth replying to based on analysis
            reply_worthy_posts = [post for post in analyzed_posts if post['content_analysis'].get('reply_worthy', False)]
            logger.logger.info(f"Found {len(reply_worthy_posts)} reply-worthy posts")
        
            if not reply_worthy_posts:
                logger.logger.warning("No reply-worthy posts found among market-related posts")
                return False
        
            # Balance between high value and regular posts
            high_value_posts = [post for post in reply_worthy_posts if post['content_analysis'].get('high_value', False)]
            posts_to_reply = high_value_posts[:int(self.max_replies_per_cycle * 0.7)]
            remaining_slots = self.max_replies_per_cycle - len(posts_to_reply)
        
            if remaining_slots > 0:
                medium_value_posts = [p for p in reply_worthy_posts if p not in high_value_posts]
                medium_value_posts.sort(key=lambda x: x['content_analysis'].get('reply_score', 0), reverse=True)
                posts_to_reply.extend(medium_value_posts[:remaining_slots])
        
            if not posts_to_reply:
                logger.logger.warning("No posts selected for reply after prioritization")
                return False
        
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies using normal strategy")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted using normal strategy")
                return False
            
        except Exception as e:
            logger.log_error("Normal Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_lower_threshold_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy with lower thresholds for reply-worthiness
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get fresh posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            if not posts:
                logger.logger.warning("No posts found during lower threshold timeline scraping")
                return False
            
            logger.logger.info(f"Lower threshold timeline scraping completed - found {len(posts)} posts")
        
            # Find posts with ANY crypto-related content, not just market-focused
            crypto_posts = []
            for post in posts:
                text = post.get('text', '').lower()
                # Check for ANY crypto-related terms
                if any(term in text for term in ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'token', 'coin', 'defi']):
                    crypto_posts.append(post)
        
            logger.logger.info(f"Found {len(crypto_posts)} crypto-related posts with lower threshold")
        
            if not crypto_posts:
                logger.logger.warning("No crypto-related posts found with lower threshold")
                return False
            
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(crypto_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied crypto-related posts with lower threshold")
        
            if not unreplied_posts:
                return False
            
            # Add basic content analysis but don't filter by reply_worthy
            analyzed_posts = []
            for post in unreplied_posts:
                analysis = self.content_analyzer.analyze_post(post)
                # Override reply_worthy to True for all posts in this fallback
                analysis['reply_worthy'] = True
                post['content_analysis'] = analysis
                analyzed_posts.append(post)
        
            # Just take the top N posts by engagement
            analyzed_posts.sort(key=lambda x: x.get('engagement_score', 0), reverse=True)
            posts_to_reply = analyzed_posts[:self.max_replies_per_cycle]
        
            if not posts_to_reply:
                return False
        
            # Generate and post replies with lower standards
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} posts with lower threshold")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies using lower threshold strategy")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted using lower threshold strategy")
                return False
            
        except Exception as e:
            logger.log_error("Lower Threshold Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_trending_posts_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy focusing on trending posts regardless of crypto relevance
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Get trending posts - use a different endpoint if possible
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)
            if not posts:
                return False
            
            logger.logger.info(f"Trending posts scraping completed - found {len(posts)} posts")
        
            # Sort by engagement (likes, retweets, etc.) to find trending posts
            posts.sort(key=lambda x: (
                x.get('like_count', 0) + 
                x.get('retweet_count', 0) * 2 + 
                x.get('reply_count', 0) * 0.5
            ), reverse=True)
        
            # Get the top trending posts
            trending_posts = posts[:int(self.max_replies_per_cycle * 1.5)]
        
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(trending_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied trending posts")
        
            if not unreplied_posts:
                return False
            
            # Add minimal content analysis
            for post in unreplied_posts:
                post['content_analysis'] = {'reply_worthy': True, 'reply_score': 75}
        
            # Generate and post replies to trending content
            logger.logger.info(f"Starting to reply to {len(unreplied_posts[:self.max_replies_per_cycle])} trending posts")
            successful_replies = self.reply_handler.reply_to_posts(
                unreplied_posts[:self.max_replies_per_cycle], 
                market_data, 
                max_replies=self.max_replies_per_cycle
            )
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies to trending posts")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted to trending posts")
                return False
            
        except Exception as e:
            logger.log_error("Trending Posts Reply Strategy", str(e))
            return False

    @ensure_naive_datetimes
    def _try_crypto_accounts_reply_strategy(self, market_data: Dict[str, Any]) -> bool:
        """
        Reply strategy focusing on major crypto accounts regardless of post content
    
        Args:
            market_data: Market data dictionary
        
        Returns:
            True if any replies were posted
        """
        try:
            # Major crypto accounts to target
            crypto_accounts = [
                'cz_binance', 'vitalikbuterin', 'SBF_FTX', 'aantonop', 'cryptohayes', 'coinbase',
                'kraken', 'whale_alert', 'CoinDesk', 'Cointelegraph', 'binance', 'BitcoinMagazine'
            ]
        
            all_posts = []
        
            # Try to get posts from specific accounts
            for account in crypto_accounts[:3]:  # Limit to 3 accounts to avoid too many requests
                try:
                    # This would need an account-specific scraper method
                    # For now, use regular timeline as placeholder
                    posts = self.timeline_scraper.scrape_timeline(count=5)
                    if posts:
                        all_posts.extend(posts)
                except Exception as e:
                    logger.logger.debug(f"Error getting posts for account {account}: {str(e)}")
                    continue
        
            # If no account-specific posts, get timeline posts and filter
            if not all_posts:
                posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 3)
            
                # Filter for posts from crypto accounts (based on handle or name)
                for post in posts:
                    handle = post.get('author_handle', '').lower()
                    name = post.get('author_name', '').lower()
                
                    if any(account.lower() in handle or account.lower() in name for account in crypto_accounts):
                        all_posts.append(post)
                    
                    # Also include posts with many crypto terms
                    text = post.get('text', '').lower()
                    crypto_terms = ['crypto', 'bitcoin', 'btc', 'eth', 'blockchain', 'token', 'coin', 'defi', 
                                   'altcoin', 'nft', 'mining', 'wallet', 'address', 'exchange']
                    if sum(1 for term in crypto_terms if term in text) >= 3:
                        all_posts.append(post)
        
            # Remove duplicates
            unique_posts = []
            post_ids = set()
            for post in all_posts:
                post_id = post.get('post_id')
                if post_id and post_id not in post_ids:
                    post_ids.add(post_id)
                    unique_posts.append(post)
        
            logger.logger.info(f"Found {len(unique_posts)} posts from crypto accounts")
        
            # Filter out posts we've already replied to
            unreplied_posts = self.content_analyzer.filter_already_replied_posts(unique_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied posts from crypto accounts")
        
            if not unreplied_posts:
                return False
            
            # Add minimal content analysis
            for post in unreplied_posts:
                post['content_analysis'] = {'reply_worthy': True, 'reply_score': 80}
        
            # Generate and post replies to crypto accounts
            logger.logger.info(f"Starting to reply to {len(unreplied_posts[:self.max_replies_per_cycle])} crypto account posts")
            successful_replies = self.reply_handler.reply_to_posts(
                unreplied_posts[:self.max_replies_per_cycle], 
                market_data, 
                max_replies=self.max_replies_per_cycle
            )
        
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies to crypto accounts")
                self.last_reply_time = strip_timezone(datetime.now())
                return True
            else:
                logger.logger.warning("No replies were successfully posted to crypto accounts")
                return False
            
        except Exception as e:
            logger.log_error("Crypto Accounts Reply Strategy", str(e))
            return False

    def _get_historical_volume_data(self, chain: str, minutes: int = None, timeframe: str = "1h") -> List[Dict[str, Any]]:
        """
        Get historical volume data for the specified window period
        Adjusted based on timeframe for appropriate historical context
        
        Args:
            chain: Token/chain symbol
            minutes: Time window in minutes (if None, determined by timeframe)
            timeframe: Timeframe for the data (1h, 24h, 7d)
            
        Returns:
            List of historical volume data points
        """
        try:
            # Adjust window size based on timeframe if not specifically provided
            if minutes is None:
                if timeframe == "1h":
                    minutes = self.config.VOLUME_WINDOW_MINUTES  # Default (typically 60)
                elif timeframe == "24h":
                    minutes = 24 * 60  # Last 24 hours
                elif timeframe == "7d":
                    minutes = 7 * 24 * 60  # Last 7 days
                else:
                    minutes = self.config.VOLUME_WINDOW_MINUTES
               
            window_start = strip_timezone(datetime.now() - timedelta(minutes=minutes))
            query = """
                SELECT timestamp, volume
                FROM market_data
                WHERE chain = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """
           
            conn = self.config.db.conn
            cursor = conn.cursor()
            cursor.execute(query, (chain, window_start))
            results = cursor.fetchall()
           
            volume_data = [
                {
                    'timestamp': strip_timezone(datetime.fromisoformat(row[0])),
                    'volume': float(row[1])
                }
                for row in results
            ]
           
            logger.logger.debug(
                f"Retrieved {len(volume_data)} volume data points for {chain} "
                f"over last {minutes} minutes (timeframe: {timeframe})"
            )
           
            return volume_data
           
        except Exception as e:
            logger.log_error(f"Historical Volume Data - {chain} ({timeframe})", str(e))
            return []
       
    def _is_duplicate_analysis(self, new_tweet: str, last_posts: List[str], timeframe: str = "1h") -> bool:
        """
        Enhanced duplicate detection with time-based thresholds and timeframe awareness.
        Applies different checks based on how recently similar content was posted:
        - Very recent posts (< 15 min): Check for exact matches
        - Recent posts (15-30 min): Check for high similarity
        - Older posts (> 30 min): Allow similar content
        
        Args:
            new_tweet: The new tweet text to check for duplication
            last_posts: List of recently posted tweets
            timeframe: Timeframe for the post (1h, 24h, 7d)
            
        Returns:
            Boolean indicating if the tweet is a duplicate
        """
        try:
            # Log that we're using enhanced duplicate detection
            logger.logger.info(f"Using enhanced time-based duplicate detection for {timeframe} timeframe")
           
            # Define time windows for different levels of duplicate checking
            # Adjust windows based on timeframe
            if timeframe == "1h":
                VERY_RECENT_WINDOW_MINUTES = 15
                RECENT_WINDOW_MINUTES = 30
                HIGH_SIMILARITY_THRESHOLD = 0.85  # 85% similar for recent posts
            elif timeframe == "24h":
                VERY_RECENT_WINDOW_MINUTES = 120  # 2 hours
                RECENT_WINDOW_MINUTES = 240       # 4 hours
                HIGH_SIMILARITY_THRESHOLD = 0.80  # Slightly lower threshold for daily predictions
            else:  # 7d
                VERY_RECENT_WINDOW_MINUTES = 720  # 12 hours
                RECENT_WINDOW_MINUTES = 1440      # 24 hours
                HIGH_SIMILARITY_THRESHOLD = 0.75  # Even lower threshold for weekly predictions
           
            # 1. Check for exact matches in very recent database entries
            conn = self.config.db.conn
            cursor = conn.cursor()
           
            # Very recent exact duplicates check
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            very_recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Check for exact matches in very recent posts
            for post in very_recent_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected within last {VERY_RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                    return True
           
            # 2. Check for high similarity in recent posts
            cursor.execute("""
                SELECT content FROM posted_content 
                WHERE timestamp >= datetime('now', '-' || ? || ' minutes')
                AND timestamp < datetime('now', '-' || ? || ' minutes')
                AND timeframe = ?
            """, (RECENT_WINDOW_MINUTES, VERY_RECENT_WINDOW_MINUTES, timeframe))
           
            recent_posts = [row[0] for row in cursor.fetchall()]
           
            # Calculate similarity for recent posts
            new_content = new_tweet.lower()
           
            for post in recent_posts:
                post_content = post.lower()
               
                # Calculate a simple similarity score based on word overlap
                new_words = set(new_content.split())
                post_words = set(post_content.split())
               
                if new_words and post_words:
                    overlap = len(new_words.intersection(post_words))
                    similarity = overlap / max(len(new_words), len(post_words))
                   
                    # Apply high similarity threshold for recent posts
                    if similarity > HIGH_SIMILARITY_THRESHOLD:
                        logger.logger.info(f"High similarity ({similarity:.2f}) detected within last {RECENT_WINDOW_MINUTES} minutes for {timeframe}")
                        return True
           
            # 3. Also check exact duplicates in last posts from Twitter
            # This prevents double-posting in case of database issues
            for post in last_posts:
                if post.strip() == new_tweet.strip():
                    logger.logger.info(f"Exact duplicate detected in recent Twitter posts for {timeframe}")
                    return True
           
            # If we get here, it's not a duplicate according to our criteria
            logger.logger.info(f"No duplicates detected with enhanced time-based criteria for {timeframe}")
            return False
           
        except Exception as e:
            logger.log_error(f"Duplicate Check - {timeframe}", str(e))
            # If the duplicate check fails, allow the post to be safe
            logger.logger.warning("Duplicate check failed, allowing post to proceed")
            return False

    def _start_prediction_thread(self) -> None:
        """
        Start background thread for asynchronous prediction generation
        """
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_thread_running = True
            self.prediction_thread = threading.Thread(target=self._process_prediction_queue)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            logger.logger.info("Started prediction processing thread")
           
    def _process_prediction_queue(self) -> None:
        """
        Process predictions from the queue in the background
        """
        while self.prediction_thread_running:
            try:
                # Get a prediction task from the queue with timeout
                try:
                    task = self.prediction_queue.get(timeout=10)
                except queue.Empty:
                    # No tasks, just continue the loop
                    continue
                   
                # Process the prediction task
                token, timeframe, market_data = task
               
                logger.logger.debug(f"Processing queued prediction for {token} ({timeframe})")
               
                # Generate the prediction
                prediction = self.prediction_engine.generate_prediction(
                    token=token, 
                    market_data=market_data,
                    timeframe=timeframe
                )
               
                # Store in memory for quick access
                self.timeframe_predictions[timeframe][token] = prediction
               
                # Mark task as done
                self.prediction_queue.task_done()
               
                # Short sleep to prevent CPU overuse
                time.sleep(0.5)
               
            except Exception as e:
                logger.log_error("Prediction Thread Error", str(e))
                time.sleep(5)  # Sleep longer on error
               
        logger.logger.info("Prediction processing thread stopped")

    def _login_to_twitter(self) -> bool:
        """
        Log into Twitter with enhanced verification and detection of existing sessions
        
        Returns:
            Boolean indicating login success
        """
        try:
            logger.logger.info("Starting Twitter login")
            self.browser.driver.set_page_load_timeout(45)
        
            # First navigate to Twitter home page instead of login page directly
            self.browser.driver.get('https://twitter.com')
            time.sleep(5)
        
            # Check if we're already logged in
            already_logged_in = False
            login_indicators = [
                '[data-testid="SideNav_NewTweet_Button"]',
                '[data-testid="AppTabBar_Profile_Link"]',
                '[aria-label="Tweet"]',
                '.DraftEditor-root'  # Tweet composer element
            ]
        
            for indicator in login_indicators:
                try:
                    if self.browser.check_element_exists(indicator):
                        already_logged_in = True
                        logger.logger.info("Already logged into Twitter, using existing session")
                        return True
                except Exception:
                    continue
        
            if not already_logged_in:
                logger.logger.info("Not logged in, proceeding with login process")
                self.browser.driver.get('https://twitter.com/login')
                time.sleep(5)

                username_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[autocomplete='username']"))
                )
                username_field.click()
                time.sleep(1)
                username_field.send_keys(self.config.TWITTER_USERNAME)
                time.sleep(2)

                next_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
                )
                next_button.click()
                time.sleep(3)

                password_field = WebDriverWait(self.browser.driver, 20).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "input[type='password']"))
                )
                password_field.click()
                time.sleep(1)
                password_field.send_keys(self.config.TWITTER_PASSWORD)
                time.sleep(2)

                login_button = WebDriverWait(self.browser.driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//span[text()='Log in']"))
                )
                login_button.click()
                time.sleep(10)

            return self._verify_login()

        except Exception as e:
            logger.log_error("Twitter Login", str(e))
            return False

    def _verify_login(self) -> bool:
        """
        Verify Twitter login success
        
        Returns:
            Boolean indicating if login verification succeeded
        """
        try:
            verification_methods = [
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]'))
                ),
                lambda: WebDriverWait(self.browser.driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="AppTabBar_Profile_Link"]'))
                ),
                lambda: any(path in self.browser.driver.current_url 
                          for path in ['home', 'twitter.com/home'])
            ]
           
            for method in verification_methods:
                try:
                    if method():
                        return True
                except:
                    continue
           
            return False
           
        except Exception as e:
            logger.log_error("Login Verification", str(e))
            return False

    def _queue_predictions_for_all_timeframes(self, token: str, market_data: Dict[str, Any]) -> None:
        """
        Queue predictions for all timeframes for a specific token
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
        """
        for timeframe in self.timeframes:
            # Skip if we already have a recent prediction
            if (token in self.timeframe_predictions.get(timeframe, {}) and 
               safe_datetime_diff(datetime.now(), self.timeframe_predictions[timeframe].get(token, {}).get('timestamp', 
                                                                                        datetime.now() - timedelta(hours=3))) 
               < 3600):  # Less than 1 hour old
                logger.logger.debug(f"Skipping {timeframe} prediction for {token} - already have recent prediction")
                continue
               
            # Add prediction task to queue
            self.prediction_queue.put((token, timeframe, market_data))
            logger.logger.debug(f"Queued {timeframe} prediction for {token}")

    @ensure_naive_datetimes
    def _post_analysis(self, tweet_text: str, timeframe: str = "1h") -> bool:
        """
        Post analysis to Twitter with robust button handling
        Tracks post by timeframe
        Prevents posting just "neutral" content
    
        Args:
            tweet_text: Text to post
            timeframe: Timeframe for the analysis
        
        Returns:
            Boolean indicating if posting succeeded
        """
        # Check for empty or just "neutral" content and replace it
        if not tweet_text or tweet_text.strip().lower() == "neutral":
            # Find the token from current analysis context if available
            token = None
        
            # Look for token in the most recent market data analyses
            for token_symbol in self.target_chains.keys():
                if token_symbol in self.timeframe_predictions.get(timeframe, {}):
                    token = token_symbol
                    break
        
            # If we couldn't find a specific token, use a default
            if not token:
                token = next(iter(self.target_chains.keys()), "CRYPTO")
        
            # Replace with an exciting message
            tweet_text = f"#{token} {timeframe.upper()} UPDATE: To The Moon!!!! 🚀🚀🚀"
            logger.logger.info(f"Replaced neutral/empty content with exciting message for {token} ({timeframe})")
    
        max_retries = 3
        retry_count = 0
    
        while retry_count < max_retries:
            try:
                self.browser.driver.get('https://twitter.com/compose/tweet')
                time.sleep(3)
            
                text_area = WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
                )
                text_area.click()
                time.sleep(1)
            
                # Ensure tweet text only contains BMP characters
                safe_tweet_text = ''.join(char for char in tweet_text if ord(char) < 0x10000)
            
                # Simply send the tweet text directly - no handling of hashtags needed
                text_area.send_keys(safe_tweet_text)
                time.sleep(2)

                post_button = None
                button_locators = [
                    (By.CSS_SELECTOR, '[data-testid="tweetButton"]'),
                    (By.XPATH, "//div[@role='button'][contains(., 'Post')]"),
                    (By.XPATH, "//span[text()='Post']")
                ]

                for locator in button_locators:
                    try:
                        post_button = WebDriverWait(self.browser.driver, 5).until(
                            EC.element_to_be_clickable(locator)
                        )
                        if post_button:
                            break
                    except:
                        continue

                if post_button:
                    self.browser.driver.execute_script("arguments[0].scrollIntoView(true);", post_button)
                    time.sleep(1)
                    self.browser.driver.execute_script("arguments[0].click();", post_button)
                    time.sleep(5)
                
                    # Update last post time for this timeframe
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now())
                
                    # Update next scheduled post time
                    hours_to_add = self.timeframe_posting_frequency.get(timeframe, 1)
                    # Add some randomness to prevent predictable patterns
                    jitter = random.uniform(0.8, 1.2)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=hours_to_add * jitter))
                
                    logger.logger.info(f"{timeframe} tweet posted successfully")
                    logger.logger.debug(f"Next {timeframe} post scheduled for {self.next_scheduled_posts[timeframe]}")
                    return True
                else:
                    logger.logger.error(f"Could not find post button for {timeframe} tweet")
                    retry_count += 1
                    time.sleep(2)
                
            except Exception as e:
                logger.logger.error(f"{timeframe} tweet posting error, attempt {retry_count + 1}: {str(e)}")
                retry_count += 1
                wait_time = retry_count * 10
                logger.logger.warning(f"Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                continue
    
        logger.log_error(f"Tweet Creation - {timeframe}", "Maximum retries reached")
        return False
   
    @ensure_naive_datetimes
    def _get_last_posts(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get last N posts from timeline with timeframe detection
        
        Args:
            count: Number of posts to retrieve
            
        Returns:
            List of post information including detected timeframe
        """
        max_retries = 3
        retry_count = 0
    
        while retry_count < max_retries:
            try:
                self.browser.driver.get(f'https://twitter.com/{self.config.TWITTER_USERNAME}')
                time.sleep(3)
            
                # Use explicit waits to ensure elements are loaded
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetText"]'))
                )
            
                posts = self.browser.driver.find_elements(By.CSS_SELECTOR, '[data-testid="tweetText"]')
            
                # Use an explicit wait for timestamps too
                WebDriverWait(self.browser.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, 'time'))
                )
            
                timestamps = self.browser.driver.find_elements(By.CSS_SELECTOR, 'time')
            
                # Get only the first count posts
                posts = posts[:count]
                timestamps = timestamps[:count]
            
                result = []
                for i in range(min(len(posts), len(timestamps))):
                    try:
                        post_text = posts[i].text
                        timestamp_str = timestamps[i].get_attribute('datetime') if timestamps[i].get_attribute('datetime') else None
                    
                        # Detect timeframe from post content
                        detected_timeframe = "1h"  # Default
                    
                        # Look for timeframe indicators in the post
                        if "7D PREDICTION" in post_text.upper() or "7-DAY" in post_text.upper() or "WEEKLY" in post_text.upper():
                            detected_timeframe = "7d"
                        elif "24H PREDICTION" in post_text.upper() or "24-HOUR" in post_text.upper() or "DAILY" in post_text.upper():
                            detected_timeframe = "24h"
                        elif "1H PREDICTION" in post_text.upper() or "1-HOUR" in post_text.upper() or "HOURLY" in post_text.upper():
                            detected_timeframe = "1h"
                    
                        post_info = {
                            'text': post_text,
                            'timestamp': strip_timezone(datetime.fromisoformat(timestamp_str)) if timestamp_str else None,
                            'timeframe': detected_timeframe
                        }
                    
                        result.append(post_info)
                    except Exception as element_error:
                        # Skip this element if it's stale or otherwise problematic
                        logger.logger.debug(f"Element error while extracting post {i}: {str(element_error)}")
                        continue
            
                return result
            
            except Exception as e:
                retry_count += 1
                logger.logger.warning(f"Error getting last posts (attempt {retry_count}/{max_retries}): {str(e)}")
                time.sleep(2)  # Add a small delay before retry
            
        # If all retries failed, log the error and return an empty list
        logger.log_error("Get Last Posts", f"Maximum retries ({max_retries}) reached")
        return []

    def _get_last_posts_by_timeframe(self, timeframe: str = "1h", count: int = 5) -> List[str]:
        """
        Get last N posts for a specific timeframe
        
        Args:
            timeframe: Timeframe to filter for
            count: Number of posts to retrieve
            
        Returns:
            List of post text content
        """
        all_posts = self._get_last_posts(count=20)  # Get more posts to filter from
        
        # Filter posts by the requested timeframe
        filtered_posts = [post['text'] for post in all_posts if post['timeframe'] == timeframe]
        
        # Return the requested number of posts
        return filtered_posts[:count]

    @ensure_naive_datetimes
    def _schedule_timeframe_post(self, timeframe: str, delay_hours: float = None) -> None:
        """
        Schedule the next post for a specific timeframe
        
        Args:
            timeframe: Timeframe to schedule for
            delay_hours: Optional override for delay hours (otherwise uses default frequency)
        """
        if delay_hours is None:
            # Use default frequency with some randomness
            base_hours = self.timeframe_posting_frequency.get(timeframe, 1)
            delay_hours = base_hours * random.uniform(0.9, 1.1)
        
        self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        logger.logger.debug(f"Scheduled next {timeframe} post for {self.next_scheduled_posts[timeframe]}")
   
    @ensure_naive_datetimes
    def _should_post_timeframe_now(self, timeframe: str) -> bool:
        """
        Check if it's time to post for a specific timeframe
        
        Args:
            timeframe: Timeframe to check
            
        Returns:
            Boolean indicating if it's time to post
        """
        try:
            # Debug
            logger.logger.debug(f"Checking if should post for {timeframe}")
            logger.logger.debug(f"  Last post: {self.timeframe_last_post.get(timeframe)} ({type(self.timeframe_last_post.get(timeframe))})")
            logger.logger.debug(f"  Next scheduled: {self.next_scheduled_posts.get(timeframe)} ({type(self.next_scheduled_posts.get(timeframe))})")
        
            # Check if enough time has passed since last post
            min_interval = timedelta(hours=self.timeframe_posting_frequency.get(timeframe, 1) * 0.8)
            last_post_time = self._ensure_datetime(self.timeframe_last_post.get(timeframe, datetime.min))
            logger.logger.debug(f"  Last post time (after ensure): {last_post_time} ({type(last_post_time)})")
        
            time_since_last = safe_datetime_diff(datetime.now(), last_post_time) / 3600  # Hours
            if time_since_last < min_interval.total_seconds() / 3600:
                return False
            
            # Check if scheduled time has been reached
            next_scheduled = self._ensure_datetime(self.next_scheduled_posts.get(timeframe, datetime.now()))
            logger.logger.debug(f"  Next scheduled (after ensure): {next_scheduled} ({type(next_scheduled)})")
        
            return datetime.now() >= next_scheduled
        except Exception as e:
            logger.logger.error(f"Error in _should_post_timeframe_now for {timeframe}: {str(e)}")
            # Provide a safe default
            return False
   
    @ensure_naive_datetimes
    def _post_prediction_for_timeframe(self, token: str, market_data: Dict[str, Any], timeframe: str) -> bool:
        """
        Post a prediction for a specific timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for the prediction
            
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            # Check if we have a prediction
            prediction = self.timeframe_predictions.get(timeframe, {}).get(token)
        
            # If no prediction exists, generate one
            if not prediction:
                prediction = self.prediction_engine.generate_prediction(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
            
                # Store for future use
                if timeframe not in self.timeframe_predictions:
                    self.timeframe_predictions[timeframe] = {}
                self.timeframe_predictions[timeframe][token] = prediction
        
            # Format the prediction for posting
            tweet_text = self._format_prediction_tweet(token, prediction, market_data, timeframe)
        
            # Check for duplicates - make sure we're handling datetime properly
            last_posts = self._get_last_posts_by_timeframe(timeframe=timeframe)
        
            # Ensure datetime compatibility in duplicate check
            if self._is_duplicate_analysis(tweet_text, last_posts, timeframe):
                logger.logger.warning(f"Skipping duplicate {timeframe} prediction for {token}")
                return False
            
            # Post the prediction
            if self._post_analysis(tweet_text, timeframe):
                # Store in database
                sentiment = prediction.get("sentiment", "NEUTRAL")
                price_data = {token: {'price': market_data[token]['current_price'], 
                                    'volume': market_data[token]['volume']}}
            
                # Create storage data
                storage_data = {
                    'content': tweet_text,
                    'sentiment': {token: sentiment},
                    'trigger_type': f"scheduled_{timeframe}_post",
                    'price_data': price_data,
                    'meme_phrases': {token: ""},  # No meme phrases for predictions
                    'is_prediction': True,
                    'prediction_data': prediction,
                    'timeframe': timeframe
                }
            
                # Store in database
                self.config.db.store_posted_content(**storage_data)
            
                # Update last post time for this timeframe with current datetime
                # This is important - make sure we're storing a datetime object
                self.timeframe_last_post[timeframe] = strip_timezone(datetime.now())
            
                logger.logger.info(f"Successfully posted {timeframe} prediction for {token}")
                return True
            else:
                logger.logger.error(f"Failed to post {timeframe} prediction for {token}")
                return False
            
        except Exception as e:
            logger.log_error(f"Post Prediction For Timeframe - {token} ({timeframe})", str(e))
            return False
   
    @ensure_naive_datetimes
    def _post_timeframe_rotation(self, market_data: Dict[str, Any]) -> bool:
        """
        Post predictions in a rotation across timeframes with enhanced token selection
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if a post was made
        """
        # Debug timeframe scheduling data
        logger.logger.debug("TIMEFRAME ROTATION DEBUG:")
        for tf in self.timeframes:
            try:
                now = strip_timezone(datetime.now())
                last_post_time = strip_timezone(self._ensure_datetime(self.timeframe_last_post.get(tf)))
                next_scheduled_time = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf)))
            
                time_since_last = safe_datetime_diff(now, last_post_time) / 3600
                time_until_next = safe_datetime_diff(next_scheduled_time, now) / 3600
                logger.logger.debug(f"{tf}: {time_since_last:.1f}h since last post, {time_until_next:.1f}h until next")
            except Exception as e:
                logger.logger.error(f"Error calculating timeframe timing for {tf}: {str(e)}")
        
        # First check if any timeframe is due for posting
        due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]

        if not due_timeframes:
            logger.logger.debug("No timeframes due for posting")
            return False
    
        try:
            # Pick the most overdue timeframe
            now = strip_timezone(datetime.now())
        
            chosen_timeframe = None
            max_overdue_time = timedelta(0)
        
            for tf in due_timeframes:
                next_scheduled = strip_timezone(self._ensure_datetime(self.next_scheduled_posts.get(tf, datetime.min)))
                overdue_time = safe_datetime_diff(now, next_scheduled)
            
                if overdue_time > max_overdue_time.total_seconds():
                    max_overdue_time = timedelta(seconds=overdue_time)
                    chosen_timeframe = tf
                
            if not chosen_timeframe:
                logger.logger.warning("Could not find most overdue timeframe, using first available")
                chosen_timeframe = due_timeframes[0]
            
        except ValueError as ve:
            if "arg is an empty sequence" in str(ve):
                logger.logger.warning("No timeframes available for rotation, rescheduling all timeframes")
                # Reschedule all timeframes with random delays
                now = strip_timezone(datetime.now())
                for tf in self.timeframes:
                    delay_hours = self.timeframe_posting_frequency.get(tf, 1) * random.uniform(0.1, 0.3)
                    self.next_scheduled_posts[tf] = now + timedelta(hours=delay_hours)
                return False
            else:
                raise  # Re-raise if it's a different ValueError
        
        logger.logger.info(f"Selected {chosen_timeframe} for timeframe rotation posting")

        # Enhanced token selection using content analysis and reply data
        token_to_post = self._select_best_token_for_timeframe(market_data, chosen_timeframe)
    
        if not token_to_post:
            logger.logger.warning(f"No suitable token found for {chosen_timeframe} timeframe")
            # Reschedule this timeframe for later
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
            return False
    
        # Before posting, check if there's active community discussion about this token
        # This helps align our posts with current community interests
        try:
            # Get recent timeline posts to analyze community trends
            recent_posts = self.timeline_scraper.scrape_timeline(count=25)
            if recent_posts:
                # Filter for posts related to our selected token
                token_related_posts = [p for p in recent_posts if token_to_post.upper() in p.get('text', '').upper()]
        
                # If we found significant community discussion, give this token higher priority
                if len(token_related_posts) >= 3:
                    logger.logger.info(f"Found active community discussion about {token_to_post} ({len(token_related_posts)} recent posts)")
                    # Analyze sentiment to make our post more contextually relevant
                    sentiment_stats = {
                        'positive': 0,
                        'negative': 0,
                        'neutral': 0
                    }
            
                    # Simple sentiment analysis of community posts
                    for post in token_related_posts:
                        analysis = self.content_analyzer.analyze_post(post)
                        sentiment = analysis.get('features', {}).get('sentiment', {}).get('label', 'neutral')
                        if sentiment in ['bullish', 'enthusiastic', 'positive']:
                            sentiment_stats['positive'] += 1
                        elif sentiment in ['bearish', 'negative', 'skeptical']:
                            sentiment_stats['negative'] += 1
                        else:
                            sentiment_stats['neutral'] += 1
            
                    # Log community sentiment
                    dominant_sentiment = max(sentiment_stats.items(), key=lambda x: x[1])[0]
                    logger.logger.info(f"Community sentiment for {token_to_post}: {dominant_sentiment} ({sentiment_stats})")
                else:
                    logger.logger.debug(f"Limited community discussion about {token_to_post} ({len(token_related_posts)} posts)")
        except Exception as e:
            logger.logger.warning(f"Error analyzing community trends: {str(e)}")
    
        # Post the prediction
        success = self._post_prediction_for_timeframe(token_to_post, market_data, chosen_timeframe)
    
        # If post failed, reschedule for later
        if not success:
            now = strip_timezone(datetime.now())
            self._schedule_timeframe_post(chosen_timeframe, delay_hours=1)
    
        return success

    def _analyze_tech_topics(self, market_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze tech topics for educational content generation
    
        Args:
            market_data: Optional market data for context
        
        Returns:
            Dictionary with tech topic analysis
        """
        try:
            # Get configured tech topics
            tech_topics = self.config.get_tech_topics()
        
            if not tech_topics:
                logger.logger.warning("No tech topics configured or enabled")
                return {'enabled': False}
            
            # Get recent tech posts from database
            tech_posts = {}
            last_tech_post = strip_timezone(datetime.now() - timedelta(days=1))  # Default fallback
        
            if self.config.db:
                try:
                    # Query last 24 hours of content
                    recent_posts = self.config.db.get_recent_posts(hours=24)
                
                    # Filter to tech-related posts
                    for post in recent_posts:
                        if 'tech_category' in post:
                            category = post['tech_category']
                            if category not in tech_posts:
                                tech_posts[category] = []
                            tech_posts[category].append(post)
                        
                            # Update last tech post time
                            post_time = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            if post_time > last_tech_post:
                                last_tech_post = post_time
                except Exception as db_err:
                    logger.logger.warning(f"Error retrieving tech posts: {str(db_err)}")
                
            # Analyze topics for candidacy
            candidate_topics = []
        
            for topic in tech_topics:
                category = topic['category']
                posts_today = len(tech_posts.get(category, []))
            
                # Calculate last post for this category
                category_last_post = last_tech_post
                if category in tech_posts and tech_posts[category]:
                    category_timestamps = [
                        strip_timezone(datetime.fromisoformat(p['timestamp'])) 
                        for p in tech_posts[category]
                    ]
                    if category_timestamps:
                        category_last_post = max(category_timestamps)
            
                # Check if allowed to post about this category
                allowed = self.config.is_tech_post_allowed(category, category_last_post)
            
                if allowed:
                    # Prepare topic metadata
                    topic_metadata = {
                        'category': category,
                        'priority': topic['priority'],
                        'keywords': topic['keywords'][:5],  # Just first 5 for logging
                        'posts_today': posts_today,
                        'hours_since_last_post': safe_datetime_diff(datetime.now(), category_last_post) / 3600,
                        'selected_token': self._select_token_for_tech_topic(category, market_data)
                    }
                
                    # Add to candidates
                    candidate_topics.append(topic_metadata)
        
            # Order by priority and recency
            if candidate_topics:
                candidate_topics.sort(key=lambda x: (x['priority'], x['hours_since_last_post']), reverse=True)
                logger.logger.info(f"Found {len(candidate_topics)} tech topics eligible for posting")
            
                # Return analysis results
                return {
                    'enabled': True,
                    'candidate_topics': candidate_topics,
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': self.config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
            else:
                logger.logger.info("No tech topics are currently eligible for posting")
                return {
                    'enabled': True,
                    'candidate_topics': [],
                    'tech_posts_today': sum(len(posts) for posts in tech_posts.values()),
                    'max_daily_posts': self.config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6),
                    'last_tech_post': last_tech_post
                }
        
        except Exception as e:
            logger.log_error("Tech Topic Analysis", str(e))
            return {'enabled': False, 'error': str(e)}

    def _select_token_for_tech_topic(self, tech_category: str, market_data: Dict[str, Any] = None) -> str:
        """
        Select an appropriate token to pair with a tech topic
    
        Args:
            tech_category: Tech category for pairing
            market_data: Market data for context
        
        Returns:
            Selected token symbol
        """
        try:
            if not market_data:
                # Default to a popular token if no market data
                return random.choice(['BTC', 'ETH', 'SOL'])
            
            # Define affinity between tech categories and tokens
            tech_token_affinity = {
                'ai': ['ETH', 'SOL', 'DOT'],          # Smart contract platforms
                'quantum': ['BTC', 'XRP', 'AVAX'],    # Security-focused or scaling
                'blockchain_tech': ['ETH', 'SOL', 'BNB', 'NEAR'],  # Advanced platforms
                'advanced_computing': ['SOL', 'AVAX', 'DOT']  # High performance chains
            }
        
            # Get affinity tokens for this category
            affinity_tokens = tech_token_affinity.get(tech_category, self.reference_tokens)
        
            # Filter to tokens with available market data
            available_tokens = [t for t in affinity_tokens if t in market_data]
        
            if not available_tokens:
                # Fall back to reference tokens if no affinity tokens available
                available_tokens = [t for t in self.reference_tokens if t in market_data]
            
            if not available_tokens:
                # Last resort fallback
                return random.choice(['BTC', 'ETH', 'SOL'])
            
            # Select token with interesting market movement if possible
            interesting_tokens = []
            for token in available_tokens:
                price_change = abs(market_data[token].get('price_change_percentage_24h', 0))
                if price_change > 5.0:  # >5% change is interesting
                    interesting_tokens.append(token)
                
            # Use interesting tokens if available, otherwise use all available tokens
            selection_pool = interesting_tokens if interesting_tokens else available_tokens
        
            # Select a token, weighting by market cap if possible
            if len(selection_pool) > 1:
                # Extract market caps
                market_caps = {t: market_data[t].get('market_cap', 1) for t in selection_pool}
                # Create weighted probability
                total_cap = sum(market_caps.values())
                weights = [market_caps[t]/total_cap for t in selection_pool]
                # Select with weights
                return random.choices(selection_pool, weights=weights, k=1)[0]
            else:
                # Just one token available
                return selection_pool[0]
            
        except Exception as e:
            logger.log_error("Token Selection for Tech Topic", str(e))
            # Safe fallback
            return random.choice(['BTC', 'ETH', 'SOL'])

    def _generate_tech_content(self, tech_category: str, token: str, market_data: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Generate educational tech content for posting
    
        Args:
            tech_category: Tech category to focus on
            token: Token to relate to tech content
            market_data: Market data for context
        
        Returns:
            Tuple of (content_text, metadata)
        """
        try:
            logger.logger.info(f"Generating tech content for {tech_category} related to {token}")
        
            # Get token data if available
            token_data = {}
            if market_data and token in market_data:
                token_data = market_data[token]
            
            # Determine content type - integration or educational
            integration_prob = 0.7  # 70% chance of integration content
            is_integration = random.random() < integration_prob
        
            # Select appropriate audience level
            audience_levels = ['beginner', 'intermediate', 'advanced']
            audience_weights = [0.3, 0.5, 0.2]  # Bias toward intermediate
            audience_level = random.choices(audience_levels, weights=audience_weights, k=1)[0]
        
            # Select appropriate template
            template_type = 'integration_template' if is_integration else 'educational_template'
            template = self.config.get_tech_prompt_template(template_type, audience_level)
        
            # Prepare prompt variables
            prompt_vars = {
                'tech_topic': tech_category.replace('_', ' ').title(),
                'token': token,
                'audience_level': audience_level,
                'min_length': self.config.TWEET_CONSTRAINTS['MIN_LENGTH'],
                'max_length': self.config.TWEET_CONSTRAINTS['MAX_LENGTH']
            }
        
            if is_integration:
                # Integration template - focus on connections
                # Get token sentiment for mood
                mood_words = ['enthusiastic', 'analytical', 'curious', 'balanced', 'thoughtful']
                prompt_vars['mood'] = random.choice(mood_words)
            
                # Add token price data if available
                if token_data:
                    prompt_vars['token_price'] = token_data.get('current_price', 0)
                    prompt_vars['price_change'] = token_data.get('price_change_percentage_24h', 0)
                
                # Get tech status summary
                prompt_vars['tech_status'] = self._get_tech_status_summary(tech_category)
                prompt_vars['integration_level'] = random.randint(3, 8)
            
                # Use tech analysis prompt template if we have market data
                if token_data:
                    prompt = self.config.client_TECH_ANALYSIS_PROMPT.format(
                        tech_topic=prompt_vars['tech_topic'],
                        token=token,
                        price=token_data.get('current_price', 0),
                        change=token_data.get('price_change_percentage_24h', 0),
                        tech_status_summary=prompt_vars['tech_status'],
                        integration_level=prompt_vars['integration_level'],
                        audience_level=audience_level
                    )
                else:
                    # Fall back to simpler template without market data
                    prompt = template.format(**prompt_vars)
                
            else:
                # Educational template - focus on informative content
                # Generate key points for educational content
                key_points = self._generate_tech_key_points(tech_category)
                prompt_vars['key_point_1'] = key_points[0]
                prompt_vars['key_point_2'] = key_points[1]
                prompt_vars['key_point_3'] = key_points[2]
                prompt_vars['learning_objective'] = self._generate_learning_objective(tech_category)
            
                # Format prompt with variables
                prompt = template.format(**prompt_vars)
        
            # Generate content with LLM
            logger.logger.debug(f"Generating {tech_category} content with {template_type}")
            content = self.llm_provider.generate_text(prompt, max_tokens=1000)
        
            if not content:
                raise ValueError("Failed to generate tech content")
            
            # Ensure content meets length requirements
            content = self._format_tech_content(content)
        
            # Prepare metadata for storage
            metadata = {
                'tech_category': tech_category,
                'token': token,
                'is_integration': is_integration,
                'audience_level': audience_level,
                'template_type': template_type,
                'token_data': token_data,
                'timestamp': strip_timezone(datetime.now())
            }
        
            return content, metadata
        
        except Exception as e:
            logger.log_error("Tech Content Generation", str(e))
            # Return fallback content
            fallback_content = f"Did you know that advances in {tech_category.replace('_', ' ')} technology could significantly impact the future of {token} and the broader crypto ecosystem? The intersection of these fields is creating fascinating new possibilities."
            return fallback_content, {'tech_category': tech_category, 'token': token, 'error': str(e)}

    def _post_tech_content(self, content: str, metadata: Dict[str, Any]) -> bool:
        """
        Post tech content to Twitter with proper formatting
    
        Args:
            content: Content to post
            metadata: Content metadata for database storage
        
        Returns:
            Boolean indicating if posting succeeded
        """
        try:
            # Check if content is already properly formatted
            if len(content) > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                content = self._format_tech_content(content)
            
            # Format as a tweet
            tweet_text = content
        
            # Add a subtle educational hashtag if there's room
            if len(tweet_text) < self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 20:
                tech_category = metadata.get('tech_category', 'technology')
                token = metadata.get('token', '')
            
                # Determine if we should add hashtags
                if random.random() < 0.7:  # 70% chance to add hashtags
                    # Potential hashtags
                    tech_tags = {
                        'ai': ['#AI', '#ArtificialIntelligence', '#MachineLearning'],
                        'quantum': ['#QuantumComputing', '#Quantum', '#QuantumTech'],
                        'blockchain_tech': ['#Blockchain', '#Web3', '#DLT'],
                        'advanced_computing': ['#Computing', '#TechInnovation', '#FutureTech']
                    }
                
                    # Get tech hashtags
                    tech_hashtags = tech_tags.get(tech_category, ['#Technology', '#Innovation'])
                
                    # Add tech hashtag and token
                    hashtag = random.choice(tech_hashtags)
                    if len(tweet_text) + len(hashtag) + len(token) + 2 <= self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                        tweet_text = f"{tweet_text} {hashtag}"
                    
                        # Maybe add token hashtag too
                        if len(tweet_text) + len(token) + 2 <= self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
                            tweet_text = f"{tweet_text} #{token}"
        
            # Post to Twitter
            logger.logger.info(f"Posting tech content about {metadata.get('tech_category', 'tech')} and {metadata.get('token', 'crypto')}")
            if self._post_analysis(tweet_text):
                logger.logger.info("Successfully posted tech content")
            
                # Record in database
                if self.config.db:
                    try:
                        # Extract token price data if available
                        token = metadata.get('token', '')
                        token_data = metadata.get('token_data', {})
                        price_data = {
                            token: {
                                'price': token_data.get('current_price', 0),
                                'volume': token_data.get('volume', 0)
                            }
                        }
                    
                        # Store as content with tech category
                        self.config.db.store_posted_content(
                            content=tweet_text,
                            sentiment={},  # No sentiment for educational content
                            trigger_type=f"tech_{metadata.get('tech_category', 'general')}",
                            price_data=price_data,
                            meme_phrases={},  # No meme phrases for educational content
                            tech_category=metadata.get('tech_category', 'technology'),
                            tech_metadata=metadata,
                            is_educational=True
                        )
                    except Exception as db_err:
                        logger.logger.warning(f"Failed to store tech content: {str(db_err)}")
            
                return True
            else:
                logger.logger.warning("Failed to post tech content")
                return False
            
        except Exception as e:
            logger.log_error("Tech Content Posting", str(e))
            return False

    def _format_tech_content(self, content: str) -> str:
        """
        Format tech content to meet tweet constraints
    
        Args:
            content: Raw content to format
        
        Returns:
            Formatted content
        """
        # Ensure length is within constraints
        if len(content) > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']:
            # Find a good sentence break to truncate
            last_period = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('.')
            last_question = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('?')
            last_exclamation = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind('!')
        
            # Find best break point
            break_point = max(last_period, last_question, last_exclamation)
        
            if break_point > self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] * 0.7:
                # Good sentence break found
                content = content[:break_point + 1]
            else:
                # Find word boundary
                last_space = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3].rfind(' ')
                if last_space > 0:
                    content = content[:last_space] + "..."
                else:
                    # Hard truncate with ellipsis
                    content = content[:self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH'] - 3] + "..."
    
        # Ensure minimum length is met
        if len(content) < self.config.TWEET_CONSTRAINTS['MIN_LENGTH']:
            logger.logger.warning(f"Tech content too short ({len(content)} chars). Minimum: {self.config.TWEET_CONSTRAINTS['MIN_LENGTH']}")
            # We won't try to expand too-short content
    
        return content

    def _get_tech_status_summary(self, tech_category: str) -> str:
        """
        Get a current status summary for a tech category
    
        Args:
            tech_category: Tech category
        
        Returns:
            Status summary string
        """
        # Status summaries by category
        summaries = {
            'ai': [
                "Rapid advancement in multimodal capabilities",
                "Increasing deployment in enterprise settings",
                "Rising concerns about governance and safety",
                "Growing focus on specialized models",
                "Shift toward open models and distributed research",
                "Mainstream adoption accelerating"
            ],
            'quantum': [
                "Steady progress in error correction",
                "Growing number of qubits in leading systems",
                "Early commercial applications emerging",
                "Increasing focus on quantum-resistant cryptography",
                "Major investment from government and private sectors",
                "Hardware diversity expanding beyond superconducting qubits"
            ],
            'blockchain_tech': [
                "Layer 2 solutions gaining momentum",
                "ZK-rollup technology maturing rapidly",
                "Cross-chain interoperability improving",
                "RWA tokenization expanding use cases",
                "Institutional adoption of infrastructure growing",
                "Privacy-preserving technologies advancing"
            ],
            'advanced_computing': [
                "Specialized AI hardware proliferating",
                "Edge computing deployments accelerating",
                "Neuromorphic computing showing early promise",
                "Post-Moore's Law approaches diversifying",
                "High-performance computing becoming more accessible",
                "Increasing focus on energy efficiency"
            ]
        }
    
        # Get summaries for this category
        category_summaries = summaries.get(tech_category, [
            "Steady technological progress",
            "Growing market adoption",
            "Increasing integration with existing systems",
            "Emerging commercial applications",
            "Active research and development"
        ])
    
        # Return random summary
        return random.choice(category_summaries)

    def _generate_tech_key_points(self, tech_category: str) -> List[str]:
        """
        Generate key educational points for a tech category
    
        Args:
            tech_category: Tech category
        
        Returns:
            List of key points for educational content
        """
        # Define key educational points by category
        key_points = {
            'ai': [
                "How large language models process and generate human language",
                "The difference between narrow AI and artificial general intelligence",
                "How multimodal AI combines text, image, audio, and video processing",
                "The concept of prompt engineering and its importance",
                "The role of fine-tuning in customizing AI models",
                "How AI models are trained on massive datasets",
                "The emergence of specialized AI for different industries",
                "The importance of ethical considerations in AI development",
                "How AI models handle context and memory limitations",
                "The computational resources required for modern AI systems"
            ],
            'quantum': [
                "How quantum bits (qubits) differ from classical bits",
                "The concept of quantum superposition and entanglement",
                "Why quantum computing excels at certain types of problems",
                "The challenge of quantum error correction",
                "Different physical implementations of quantum computers",
                "How quantum algorithms provide computational advantages",
                "The potential impact on cryptography and security",
                "The timeline for quantum advantage in practical applications",
                "How quantum computing complements rather than replaces classical computing",
                "The difference between quantum annealing and gate-based quantum computing"
            ],
            'blockchain_tech': [
                "How zero-knowledge proofs enable privacy while maintaining verification",
                "The concept of sharding and its role in blockchain scaling",
                "The difference between optimistic and ZK rollups",
                "How Layer 2 solutions address blockchain scalability challenges",
                "The evolution of consensus mechanisms beyond proof of work",
                "How cross-chain bridges enable interoperability between blockchains",
                "The concept of state channels for off-chain transactions",
                "How smart contracts enable programmable transactions",
                "The role of oracles in connecting blockchains to external data",
                "Different approaches to blockchain governance"
            ],
            'advanced_computing': [
                "How neuromorphic computing mimics brain functions",
                "The concept of edge computing and its advantages",
                "The evolution beyond traditional Moore's Law scaling",
                "How specialized hardware accelerates specific workloads",
                "The rise of heterogeneous computing architectures",
                "How in-memory computing reduces data movement bottlenecks",
                "The potential of optical computing for specific applications",
                "How quantum-inspired algorithms work on classical hardware",
                "The importance of energy efficiency in modern computing",
                "How cloud computing is evolving with specialized hardware"
            ]
        }
    
        # Get points for this category
        category_points = key_points.get(tech_category, [
            "The fundamental principles behind this technology",
            "Current applications and use cases",
            "Future potential developments and challenges",
            "How this technology relates to blockchain and cryptocurrency",
            "The importance of this technology for digital innovation"
        ])
    
        # Select 3 random points without replacement
        selected_points = random.sample(category_points, min(3, len(category_points)))
    
        # Ensure we have 3 points
        while len(selected_points) < 3:
            selected_points.append("How this technology impacts the future of digital assets")
        
        return selected_points

    def _generate_learning_objective(self, tech_category: str) -> str:
        """
        Generate a learning objective for educational tech content
    
        Args:
            tech_category: Tech category
        
        Returns:
            Learning objective string
        """
        # Define learning objectives by category
        objectives = {
            'ai': [
                "how AI technologies are transforming the crypto landscape",
                "the core principles behind modern AI systems",
                "how AI and blockchain technologies can complement each other",
                "the key limitations and challenges of current AI approaches",
                "how AI is being used to enhance trading, security, and analytics in crypto"
            ],
            'quantum': [
                "how quantum computing affects blockchain security",
                "the fundamentals of quantum computing in accessible terms",
                "the timeline and implications of quantum advances for cryptography",
                "how the crypto industry is preparing for quantum computing",
                "the difference between quantum threats and opportunities for blockchain"
            ],
            'blockchain_tech': [
                "how advanced blockchain technologies are addressing scalability",
                "the technical foundations of modern blockchain systems",
                "the trade-offs between different blockchain scaling approaches",
                "how blockchain privacy technologies actually work",
                "the evolution of blockchain architecture beyond first-generation systems"
            ],
            'advanced_computing': [
                "how specialized computing hardware is changing crypto mining",
                "the next generation of computing technologies on the horizon",
                "how computing advances are enabling new blockchain capabilities",
                "the relationship between energy efficiency and blockchain sustainability",
                "how distributed computing and blockchain share foundational principles"
            ]
        }
    
        # Get objectives for this category
        category_objectives = objectives.get(tech_category, [
            "the fundamentals of this technology in accessible terms",
            "how this technology relates to blockchain and cryptocurrency",
            "the potential future impact of this technological development",
            "the current state and challenges of this technology",
            "how this technology might transform digital finance"
        ])
    
        # Return random objective
        return random.choice(category_objectives)

    @ensure_naive_datetimes     
    def _select_best_token_for_timeframe(self, market_data: Dict[str, Any], timeframe: str) -> Optional[str]:
        """
        Select the best token to use for a specific timeframe post
        Uses momentum scoring, prediction accuracy, and market activity
        
        Args:
            market_data: Market data dictionary
            timeframe: Timeframe to select for
            
        Returns:
            Best token symbol for the timeframe
        """
        candidates = []
        
        # Get tokens with data
        available_tokens = [t for t in self.reference_tokens if t in market_data]
        
        # Score each token
        for token in available_tokens:
            # Calculate momentum score
            momentum_score = self._calculate_momentum_score(token, market_data, timeframe)
            
            # Calculate activity score based on recent volume and price changes
            token_data = market_data.get(token, {})
            volume = token_data.get('volume', 0)
            price_change = abs(token_data.get('price_change_percentage_24h', 0))
            
            # Get volume trend
            volume_trend, _ = self._analyze_volume_trend(volume, 
                                                    self._get_historical_volume_data(token, timeframe=timeframe),
                                                    timeframe=timeframe)
            
            # Get historical prediction accuracy
            perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
            
            # Calculate accuracy score
            accuracy_score = 0
            if perf_stats:
                accuracy = perf_stats[0].get('accuracy_rate', 0)
                total_preds = perf_stats[0].get('total_predictions', 0)
                
                # Only consider accuracy if we have enough data
                if total_preds >= 5:
                    accuracy_score = accuracy * (min(total_preds, 20) / 20)  # Scale by number of predictions up to 20
            
            # Calculate recency score - prefer tokens we haven't posted about recently
            recency_score = 0
            
            # Check when this token was last posted for this timeframe
            recent_posts = self.config.db.get_recent_posts(hours=48, timeframe=timeframe)
            
            token_posts = [p for p in recent_posts if token.upper() in p.get('content', '')]
            
            if not token_posts:
                # Never posted - maximum recency score
                recency_score = 100
            else:
                # Calculate hours since last post
                last_posts_times = [strip_timezone(datetime.fromisoformat(p.get('timestamp', datetime.min.isoformat()))) for p in token_posts]
                if last_posts_times:
                    last_post_time = max(last_posts_times)
                    hours_since = safe_datetime_diff(datetime.now(), last_post_time) / 3600
                    
                    # Scale recency score based on timeframe
                    if timeframe == "1h":
                        recency_score = min(100, hours_since * 10)  # Max score after 10 hours
                    elif timeframe == "24h":
                        recency_score = min(100, hours_since * 2)   # Max score after 50 hours
                    else:  # 7d
                        recency_score = min(100, hours_since * 0.5)  # Max score after 200 hours
            
            # Combine scores with timeframe-specific weightings
            if timeframe == "1h":
                # For hourly, momentum and price action matter most
                total_score = (
                    momentum_score * 0.5 +
                    price_change * 3.0 +
                    volume_trend * 0.7 +
                    accuracy_score * 0.3 +
                    recency_score * 0.4
                )
            elif timeframe == "24h":
                # For daily, balance between momentum, accuracy and recency
                total_score = (
                    momentum_score * 0.4 +
                    price_change * 2.0 +
                    volume_trend * 0.8 +
                    accuracy_score * 0.5 +
                    recency_score * 0.6
                )
            else:  # 7d
                # For weekly, accuracy and longer-term views matter more
                total_score = (
                    momentum_score * 0.3 +
                    price_change * 1.0 +
                    volume_trend * 1.0 +
                    accuracy_score * 0.8 +
                    recency_score * 0.8
                )
            
            candidates.append((token, total_score))
        
        # Sort by total score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        logger.logger.debug(f"Token candidates for {timeframe}: {candidates[:3]}")
        
        return candidates[0][0] if candidates else None

    def _check_for_posts_to_reply(self, market_data: Dict[str, Any]) -> bool:
        """
        Check for posts to reply to and generate replies
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Boolean indicating if any replies were posted
        """
        now = strip_timezone(datetime.now())
    
        # Check if it's time to look for posts to reply to
        time_since_last_check = safe_datetime_diff(now, self.last_reply_check) / 60
        if time_since_last_check < self.reply_check_interval:
            logger.logger.debug(f"Skipping reply check, {time_since_last_check:.1f} minutes since last check (interval: {self.reply_check_interval})")
            return False
        
        # Also check cooldown period
        time_since_last_reply = safe_datetime_diff(now, self.last_reply_time) / 60
        if time_since_last_reply < self.reply_cooldown:
            logger.logger.debug(f"In reply cooldown period, {time_since_last_reply:.1f} minutes since last reply (cooldown: {self.reply_cooldown})")
            return False
        
        logger.logger.info("Starting check for posts to reply to")
        self.last_reply_check = now
    
        try:
            # Scrape timeline for posts
            posts = self.timeline_scraper.scrape_timeline(count=self.max_replies_per_cycle * 2)  # Get more to filter
            logger.logger.info(f"Timeline scraping completed - found {len(posts) if posts else 0} posts")
        
            if not posts:
                logger.logger.warning("No posts found during timeline scraping")
                return False

            # Log sample posts for debugging
            for i, post in enumerate(posts[:3]):  # Log first 3 posts
                logger.logger.info(f"Sample post {i}: {post.get('text', '')[:100]}...")

            # Find market-related posts
            logger.logger.info(f"Finding market-related posts among {len(posts)} scraped posts")
            market_posts = self.content_analyzer.find_market_related_posts(posts)
            logger.logger.info(f"Found {len(market_posts)} market-related posts, checking which ones need replies")
            
            # Filter out posts we've already replied to
            unreplied_posts = self.timeline_scraper.filter_already_replied_posts(market_posts)
            logger.logger.info(f"Found {len(unreplied_posts)} unreplied market-related posts")
            if unreplied_posts:
                for i, post in enumerate(unreplied_posts[:3]):
                    logger.logger.info(f"Sample unreplied post {i}: {post.get('text', '')[:100]}...")
            
            if not unreplied_posts:
                return False
                
            # Prioritize posts (engagement, relevance, etc.)
            prioritized_posts = self.timeline_scraper.prioritize_posts(unreplied_posts)
            
            # Limit to max replies per cycle
            posts_to_reply = prioritized_posts[:self.max_replies_per_cycle]
            
            # Generate and post replies
            logger.logger.info(f"Starting to reply to {len(posts_to_reply)} prioritized posts")
            successful_replies = self.reply_handler.reply_to_posts(posts_to_reply, market_data, max_replies=self.max_replies_per_cycle)
            
            if successful_replies > 0:
                logger.logger.info(f"Successfully posted {successful_replies} replies")
                self.last_reply_time = now                    
                return True
            else:
                logger.logger.info("No replies were successfully posted")
                return False
                
        except Exception as e:
            logger.log_error("Check For Posts To Reply", str(e))
            return False
    @ensure_naive_datetimes
    def _get_daily_tech_post_count(self) -> Dict[str, Any]:
        """
        Get tech post count for the current calendar day with proper midnight reset
        Respects existing datetime handling and normalization functions
    
        Returns:
            Dictionary containing tech post counts and limits for today
        """
        try:
            # Get the current date and time with proper timezone handling
            now = strip_timezone(datetime.now())
        
            # Calculate the start of the current day (midnight)
            today_start = strip_timezone(datetime(now.year, now.month, now.day, 0, 0, 0))
        
            # Get configured maximum daily tech posts
            max_daily_posts = self.config.TECH_CONTENT_CONFIG.get('max_daily_tech_posts', 6)
        
            tech_posts = {}
            tech_posts_today = 0
            last_tech_post = None
        
            # Query database only for posts from the current calendar day
            if self.config.db:
                try:
                    # Calculate hours since midnight for database query
                    hours_since_midnight = safe_datetime_diff(now, today_start) / 3600
                
                    # Get posts only from the current day
                    recent_posts = self.config.db.get_recent_posts(hours=hours_since_midnight)
                
                    # Filter to tech-related posts and verify they're from today
                    for post in recent_posts:
                        if 'tech_category' in post:
                            # Verify post is from today by checking the timestamp
                            post_time = strip_timezone(datetime.fromisoformat(post['timestamp']))
                            if post_time >= today_start:  # Only count posts from today
                                category = post['tech_category']
                                if category not in tech_posts:
                                    tech_posts[category] = []
                                tech_posts[category].append(post)
                            
                                # Track the most recent tech post timestamp
                                if last_tech_post is None or post_time > last_tech_post:
                                    last_tech_post = post_time
                
                    # Count today's tech posts
                    tech_posts_today = sum(len(posts) for posts in tech_posts.values())
                
                    logger.logger.debug(
                        f"Daily tech posts: {tech_posts_today}/{max_daily_posts} since midnight " 
                        f"({hours_since_midnight:.1f} hours ago)"
                    )
                
                except Exception as db_err:
                    logger.logger.warning(f"Error retrieving tech posts: {str(db_err)}")
                    tech_posts_today = 0
                    last_tech_post = today_start  # Default to start of day if error
                
            # If no posts found today, set default last post to start of day
            if last_tech_post is None:
                last_tech_post = today_start
        
            # Check if maximum posts for today has been reached
            max_reached = tech_posts_today >= max_daily_posts
            if max_reached:
                logger.logger.info(
                    f"Maximum daily tech posts reached for today: {tech_posts_today}/{max_daily_posts}"
                )
            else:
                logger.logger.debug(
                    f"Daily tech post count: {tech_posts_today}/{max_daily_posts} - " 
                    f"additional posts allowed today"
                )
            
            # Return comprehensive stats
            return {
                'tech_posts_today': tech_posts_today,
                'max_daily_posts': max_daily_posts,
                'last_tech_post': last_tech_post,
                'day_start': today_start,
                'max_reached': max_reached,
                'categories_posted': list(tech_posts.keys()),
                'posts_by_category': {k: len(v) for k, v in tech_posts.items()}
            }
        
        except Exception as e:
            logger.log_error("Daily Tech Post Count", str(e))
            # Return safe defaults
            return {
                'tech_posts_today': 0,
                'max_daily_posts': self.config.TECH_CONTENT_CONFIG.get('max_daily_posts', 6),
                'last_tech_post': strip_timezone(datetime.now() - timedelta(hours=24)),
                'day_start': strip_timezone(datetime(now.year, now.month, now.day, 0, 0, 0)),
                'max_reached': False,
                'categories_posted': [],
                'posts_by_category': {}
            }
    def get_posts_since_timestamp(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Get all posts since a specific timestamp
    
        Args:
            timestamp: ISO format timestamp string
        
        Returns:
            List of posts
        """
        conn, cursor = self._get_connection()
    
        try:
            cursor.execute("""
                SELECT * FROM posted_content
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
            """, (timestamp,))
        
            return cursor.fetchall()
        except Exception as e:
            logger.log_error("Get Posts Since Timestamp", str(e))
            return []
    
    @ensure_naive_datetimes
    def _should_post_tech_content(self, market_data: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if tech content should be posted, and select a topic
        Uses calendar day-based counters with midnight reset
        Respects daily limits with clear logging

        Args:
            market_data: Optional market data for context

        Returns:
            Tuple of (should_post, topic_data)
        """
        try:
            # Check if tech content is enabled
            if not self.config.TECH_CONTENT_CONFIG.get('enabled', False):
                logger.logger.debug("Tech content posting disabled in configuration")
                return False, {}
    
            # Get daily post metrics with calendar day reset
            daily_metrics = self._get_daily_tech_post_count()
    
            # Check if we've hit daily maximum for today - log clearly
            if daily_metrics['max_reached']:
                logger.logger.info(
                    f"Maximum daily tech posts reached ({daily_metrics['tech_posts_today']}/"
                    f"{daily_metrics['max_daily_posts']})"
                )
                return False, {}
    
            # Analyze tech topics
            tech_analysis = self._analyze_tech_topics(market_data)
    
            if not tech_analysis.get('enabled', False):
                logger.logger.debug("Tech analysis not enabled or failed")
                return False, {}
    
            # Check if we have candidate topics
            candidates = tech_analysis.get('candidate_topics', [])
            if not candidates:
                logger.logger.debug("No candidate tech topics available")
                return False, {}
    
            # Select top candidate
            selected_topic = candidates[0]
    
            # Check if enough time has passed since last tech post
            last_tech_post = daily_metrics['last_tech_post']
            hours_since_last = safe_datetime_diff(strip_timezone(datetime.now()), last_tech_post) / 3600
            post_frequency = self.config.TECH_CONTENT_CONFIG.get('post_frequency', 4)
    
            if hours_since_last < post_frequency:
                logger.logger.info(
                    f"Not enough time since last tech post ({hours_since_last:.1f}h < {post_frequency}h)"
                )
                return False, selected_topic
    
            # At this point, we should post tech content
            logger.logger.info(
                f"Will post tech content about {selected_topic['category']} related to "
                f"{selected_topic['selected_token']} (Day count: {daily_metrics['tech_posts_today']}/"
                f"{daily_metrics['max_daily_posts']})"
            )
            return True, selected_topic
    
        except Exception as e:
            logger.log_error("Tech Content Decision", str(e))
            # On error, return False to prevent posting
            return False, {}

    def _post_tech_educational_content(self, market_data: Dict[str, Any]) -> bool:
        """
        Generate and post tech educational content
    
        Args:
            market_data: Market data for context
        
        Returns:
            Boolean indicating if content was successfully posted
        """
        try:
            # Check if we should post tech content
            should_post, topic_data = self._should_post_tech_content(market_data)
        
            if not should_post:
                return False
            
            # Generate tech content
            tech_category = topic_data.get('category', 'ai')  # Default to AI if not specified
            token = topic_data.get('selected_token', 'BTC')   # Default to BTC if not specified
        
            content, metadata = self._generate_tech_content(tech_category, token, market_data)
        
            # Post the content
            return self._post_tech_content(content, metadata)
        
        except Exception as e:
            logger.log_error("Tech Educational Content", str(e))
            return False
    @ensure_naive_datetimes
    def _cleanup(self) -> None:
        """Cleanup resources and save state"""
        try:
            # Stop prediction thread if running
            if self.prediction_thread_running:
                self.prediction_thread_running = False
                if self.prediction_thread and self.prediction_thread.is_alive():
                    self.prediction_thread.join(timeout=5)
                logger.logger.info("Stopped prediction thread")
           
            # Close browser
            if self.browser:
                logger.logger.info("Closing browser...")
                try:
                    self.browser.close_browser()
                    time.sleep(1)
                except Exception as e:
                    logger.logger.warning(f"Error during browser close: {str(e)}")
           
            # Save timeframe prediction data to database for persistence
            try:
                timeframe_state = {
                    "predictions": self.timeframe_predictions,
                    "last_post": {tf: ts.isoformat() for tf, ts in self.timeframe_last_post.items()},
                    "next_scheduled": {tf: ts.isoformat() for tf, ts in self.next_scheduled_posts.items()},
                    "accuracy": self.prediction_accuracy
                }
               
                # Store using the generic JSON data storage
                self.config.db._store_json_data(
                    data_type="timeframe_state",
                    data=timeframe_state
                )
                logger.logger.info("Saved timeframe state to database")
            except Exception as e:
                logger.logger.warning(f"Failed to save timeframe state: {str(e)}")
           
            # Close database connection
            if self.config:
                self.config.cleanup()
               
            logger.log_shutdown()
        except Exception as e:
            logger.log_error("Cleanup", str(e))

    @ensure_naive_datetimes
    def _ensure_datetime(self, value) -> datetime:
        """
        Convert value to datetime if it's a string, ensuring timezone-naive datetime
        
        Args:
            value: Value to convert
            
        Returns:
            Datetime object (timezone-naive)
        """
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return strip_timezone(dt)
            except ValueError:
                logger.logger.warning(f"Could not parse datetime string: {value}")
                return strip_timezone(datetime.min)
        elif isinstance(value, datetime):
            return strip_timezone(value)
        return strip_timezone(datetime.min)

    def _get_crypto_data(self) -> Optional[Dict[str, Any]]:
        """Fetch crypto data from CoinGecko with retries"""
        try:
            params = {
                **self.config.get_coingecko_params(),
                'ids': ','.join(self.target_chains.values()), 
                'sparkline': True,
                'price_change_percentage': '1h,24h,7d'  # Ensure this parameter is included
            }
            
            data = self.coingecko.get_market_data(params)
            if not data:
                logger.logger.error("Failed to fetch market data from CoinGecko")
                return None
        
            # Detailed inspection of the raw data
            logger.logger.debug(f"Raw CoinGecko data type: {type(data)}, length: {len(data)}")
            if isinstance(data, list) and len(data) > 0:
                sample_item = data[0]
                logger.logger.debug(f"Sample item keys: {list(sample_item.keys())}")
                if 'price_change_percentage_24h' in sample_item:
                    logger.logger.debug(f"Sample price_change_percentage_24h: {sample_item['price_change_percentage_24h']}")
                else:
                    logger.logger.debug(f"price_change_percentage_24h NOT FOUND in sample item")
                # Check if other price change keys exist
                price_keys = [k for k in sample_item.keys() if 'price_change' in k or 'change' in k]
                logger.logger.debug(f"Available price change keys: {price_keys}")
                
            formatted_data = {
                coin['symbol'].upper(): {
                    'current_price': coin['current_price'],
                    'volume': coin['total_volume'],
                    'price_change_percentage_24h': coin['price_change_percentage_24h'],
                    'sparkline': coin.get('sparkline_in_7d', {}).get('price', []),
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'total_supply': coin.get('total_supply'),
                    'max_supply': coin.get('max_supply'),
                    'circulating_supply': coin.get('circulating_supply'),
                    'ath': coin.get('ath'),
                    'ath_change_percentage': coin.get('ath_change_percentage')
                } for coin in data
            }
            
            # Map to correct symbol if needed (particularly for POL which might return as MATIC)
            symbol_corrections = {'MATIC': 'POL'}
            for old_sym, new_sym in symbol_corrections.items():
                if old_sym in formatted_data and new_sym not in formatted_data:
                    formatted_data[new_sym] = formatted_data[old_sym]
                    logger.logger.debug(f"Mapped {old_sym} data to {new_sym}")
            
            # Log API usage statistics
            stats = self.coingecko.get_request_stats()
            logger.logger.debug(
                f"CoinGecko API stats - Daily requests: {stats['daily_requests']}, "
                f"Failed: {stats['failed_requests']}, Cache size: {stats['cache_size']}"
            )
            
            # Store market data in database
            for chain, chain_data in formatted_data.items():
                self.config.db.store_market_data(chain, chain_data)
            
            # Check if all data was retrieved
            missing_tokens = [token for token in self.reference_tokens if token not in formatted_data]
            if missing_tokens:
                logger.logger.warning(f"Missing data for tokens: {', '.join(missing_tokens)}")
                
                # Try fallback mechanism for missing tokens
                if 'POL' in missing_tokens and 'MATIC' in formatted_data:
                    formatted_data['POL'] = formatted_data['MATIC']
                    missing_tokens.remove('POL')
                    logger.logger.info("Applied fallback for POL using MATIC data")
                
            logger.logger.info(f"Successfully fetched crypto data for {', '.join(formatted_data.keys())}")
            return formatted_data
                
        except Exception as e:
            logger.log_error("CoinGecko API", str(e))
            return None

    @ensure_naive_datetimes
    def _load_saved_timeframe_state(self) -> None:
        """Load previously saved timeframe state from database with enhanced datetime handling"""
        try:
            # Query the latest timeframe state
            conn, cursor = self.config.db._get_connection()
        
            cursor.execute("""
                SELECT data 
                FROM generic_json_data 
                WHERE data_type = 'timeframe_state'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
        
            result = cursor.fetchone()
        
            if not result:
                logger.logger.info("No saved timeframe state found")
                return
            
            # Parse the saved state
            state_json = result[0]
            state = json.loads(state_json)
        
            # Restore timeframe predictions
            for timeframe, predictions in state.get("predictions", {}).items():
                self.timeframe_predictions[timeframe] = predictions
        
            # Restore last post times with proper datetime handling
            for timeframe, timestamp in state.get("last_post", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    self.timeframe_last_post[timeframe] = strip_timezone(dt)
                    logger.logger.debug(f"Restored last post time for {timeframe}: {self.timeframe_last_post[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, use a safe default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} last post: {str(e)}")
                    self.timeframe_last_post[timeframe] = strip_timezone(datetime.now() - timedelta(hours=3))
        
            # Restore next scheduled posts with proper datetime handling
            for timeframe, timestamp in state.get("next_scheduled", {}).items():
                try:
                    # Convert string to datetime and ensure it's timezone-naive
                    dt = datetime.fromisoformat(timestamp)
                    scheduled_time = strip_timezone(dt)
                
                    # If scheduled time is in the past, reschedule
                    now = strip_timezone(datetime.now())
                    if scheduled_time < now:
                        delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                        self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
                        logger.logger.debug(f"Rescheduled {timeframe} post for {self.next_scheduled_posts[timeframe]}")
                    else:
                        self.next_scheduled_posts[timeframe] = scheduled_time
                        logger.logger.debug(f"Restored next scheduled time for {timeframe}: {self.next_scheduled_posts[timeframe]}")
                except (ValueError, TypeError) as e:
                    # If timestamp can't be parsed, set a default
                    logger.logger.warning(f"Could not parse timestamp for {timeframe} next scheduled post: {str(e)}")
                    delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                    self.next_scheduled_posts[timeframe] = strip_timezone(datetime.now() + timedelta(hours=delay_hours))
        
            # Restore accuracy tracking
            self.prediction_accuracy = state.get("accuracy", {timeframe: {'correct': 0, 'total': 0} for timeframe in self.timeframes})
        
            # Debug log the restored state
            logger.logger.debug("Restored timeframe state:")
            for tf in self.timeframes:
                last_post = self.timeframe_last_post.get(tf)
                next_post = self.next_scheduled_posts.get(tf)
                logger.logger.debug(f"  {tf}: last={last_post}, next={next_post}")
        
            logger.logger.info("Restored timeframe state from database")
        
        except Exception as e:
            logger.log_error("Load Timeframe State", str(e))
            # Create safe defaults for all timing data
            now = strip_timezone(datetime.now())
            for timeframe in self.timeframes:
                self.timeframe_last_post[timeframe] = now - timedelta(hours=3)
                delay_hours = self.timeframe_posting_frequency.get(timeframe, 1) * random.uniform(0.1, 0.5)
                self.next_scheduled_posts[timeframe] = now + timedelta(hours=delay_hours)
        
            logger.logger.warning("Using default timeframe state due to error")
    

    def _get_historical_price_data(self, token: str, hours: int, timeframe: str = None) -> Tuple[List[float], List[float], List[float], List[float]]:
        """
        Safely get historical price data with proper error handling
        Returns (prices, volumes, highs, lows)
    
        Args:
            token: The cryptocurrency token symbol
            hours: Number of hours of historical data to retrieve
        """
        try:
            # Get historical market data from database
            historical_data = self.db.get_recent_market_data(token, hours=hours)
        
            # Initialize empty lists
            prices = []
            volumes = []
            highs = []
            lows = []
        
            # Extract price and volume history if data exists
            if historical_data:
                for entry in reversed(historical_data):  # Oldest to newest
                    prices.append(entry.get('price', 0))
                    volumes.append(entry.get('volume', 0))
                    # Use price as high/low if not available
                    highs.append(entry.get('high', entry.get('price', 0)))
                    lows.append(entry.get('low', entry.get('price', 0)))
        
            # Return data, even if empty
            return prices, volumes, highs, lows
        except Exception as e:
            logger.log_error(f"Historical Data Retrieval - {token}", str(e))
            # Return empty lists as a fallback
            return [], [], [], []
   
    @ensure_naive_datetimes
    def _get_token_timeframe_performance(self, token: str) -> Dict[str, Dict[str, Any]]:
        """
        Get prediction performance statistics for a token across all timeframes
        
        Args:
            token: Token symbol
            
        Returns:
            Dictionary of performance statistics by timeframe
        """
        try:
            result = {}
            
            # Gather performance for each timeframe
            for timeframe in self.timeframes:
                perf_stats = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
                
                if perf_stats:
                    result[timeframe] = {
                        "accuracy": perf_stats[0].get("accuracy_rate", 0),
                        "total": perf_stats[0].get("total_predictions", 0),
                        "correct": perf_stats[0].get("correct_predictions", 0),
                        "avg_deviation": perf_stats[0].get("avg_deviation", 0)
                    }
                else:
                    result[timeframe] = {
                        "accuracy": 0,
                        "total": 0,
                        "correct": 0,
                        "avg_deviation": 0
                    }
            
            # Get cross-timeframe comparison
            cross_comparison = self.config.db.get_prediction_comparison_across_timeframes(token)
            
            if cross_comparison:
                result["best_timeframe"] = cross_comparison.get("best_timeframe", {}).get("timeframe", "1h")
                result["overall"] = cross_comparison.get("overall", {})
            
            return result
            
        except Exception as e:
            logger.log_error(f"Get Token Timeframe Performance - {token}", str(e))
            return {tf: {"accuracy": 0, "total": 0, "correct": 0, "avg_deviation": 0} for tf in self.timeframes}
   
    @ensure_naive_datetimes
    def _get_all_active_predictions(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all active predictions organized by timeframe and token
        
        Returns:
            Dictionary of active predictions by timeframe and token
        """
        try:
            result = {tf: {} for tf in self.timeframes}
            
            # Get active predictions from the database
            active_predictions = self.config.db.get_active_predictions()
            
            for prediction in active_predictions:
                timeframe = prediction.get("timeframe", "1h")
                token = prediction.get("token", "")
                
                if timeframe in result and token:
                    result[timeframe][token] = prediction
            
            # Merge with in-memory predictions which might be more recent
            for timeframe, predictions in self.timeframe_predictions.items():
                for token, prediction in predictions.items():
                    result.setdefault(timeframe, {})[token] = prediction
            
            return result
            
        except Exception as e:
            logger.log_error("Get All Active Predictions", str(e))
            return {tf: {} for tf in self.timeframes}

    @ensure_naive_datetimes
    def _evaluate_expired_timeframe_predictions(self) -> Dict[str, int]:
        """
        Find and evaluate expired predictions across all timeframes
        
        Returns:
            Dictionary with count of evaluated predictions by timeframe
        """
        try:
            # Get expired unevaluated predictions
            all_expired = self.config.db.get_expired_unevaluated_predictions()
            
            if not all_expired:
                logger.logger.debug("No expired predictions to evaluate")
                return {tf: 0 for tf in self.timeframes}
                
            # Group by timeframe
            expired_by_timeframe = {tf: [] for tf in self.timeframes}
            
            for prediction in all_expired:
                timeframe = prediction.get("timeframe", "1h")
                if timeframe in expired_by_timeframe:
                    expired_by_timeframe[timeframe].append(prediction)
            
            # Get current market data for evaluation
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return {tf: 0 for tf in self.timeframes}
            
            # Track evaluated counts
            evaluated_counts = {tf: 0 for tf in self.timeframes}
            
            # Evaluate each prediction by timeframe
            for timeframe, predictions in expired_by_timeframe.items():
                for prediction in predictions:
                    token = prediction["token"]
                    prediction_id = prediction["id"]
                    
                    # Get current price for the token
                    token_data = market_data.get(token, {})
                    if not token_data:
                        logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                        continue
                        
                    current_price = token_data.get("current_price", 0)
                    if current_price == 0:
                        logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                        continue
                        
                    # Record the outcome
                    result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                    
                    if result:
                        logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                        evaluated_counts[timeframe] += 1
                    else:
                        logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
            
            # Log evaluation summaries
            for timeframe, count in evaluated_counts.items():
                if count > 0:
                    logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
            
            # Update prediction performance metrics
            self._update_prediction_performance_metrics()
            
            return evaluated_counts
            
        except Exception as e:
            logger.log_error("Evaluate Expired Timeframe Predictions", str(e))
            return {tf: 0 for tf in self.timeframes}

    @ensure_naive_datetimes
    def _update_prediction_performance_metrics(self) -> None:
        """Update in-memory prediction performance metrics from database"""
        try:
            # Get overall performance by timeframe
            for timeframe in self.timeframes:
                performance = self.config.db.get_prediction_performance(timeframe=timeframe)
                
                total_correct = sum(p.get("correct_predictions", 0) for p in performance)
                total_predictions = sum(p.get("total_predictions", 0) for p in performance)
                
                # Update in-memory tracking
                self.prediction_accuracy[timeframe] = {
                    'correct': total_correct,
                    'total': total_predictions
                }
            
            # Log overall performance
            for timeframe, stats in self.prediction_accuracy.items():
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    logger.logger.info(f"{timeframe} prediction accuracy: {accuracy:.1f}% ({stats['correct']}/{stats['total']})")
                    
        except Exception as e:
            logger.log_error("Update Prediction Performance Metrics", str(e))

    def _analyze_volume_trend(self, current_volume: float, historical_data: List[Dict[str, Any]], 
                             timeframe: str = "1h") -> Tuple[float, str]:
        """
        Analyze volume trend over the window period, adjusted for timeframe
        
        Args:
            current_volume: Current volume value
            historical_data: Historical volume data
            timeframe: Timeframe for analysis
            
        Returns:
            Tuple of (percentage_change, trend_description)
        """
        if not historical_data:
            return 0.0, "insufficient_data"
            
        try:
            # Adjust trend thresholds based on timeframe
            if timeframe == "1h":
                SIGNIFICANT_THRESHOLD = self.config.VOLUME_TREND_THRESHOLD  # Default (usually 15%)
                MODERATE_THRESHOLD = 5.0
            elif timeframe == "24h":
                SIGNIFICANT_THRESHOLD = 20.0  # Higher threshold for daily predictions
                MODERATE_THRESHOLD = 10.0
            else:  # 7d
                SIGNIFICANT_THRESHOLD = 30.0  # Even higher for weekly predictions
                MODERATE_THRESHOLD = 15.0
            
            # Calculate average volume excluding the current volume
            historical_volumes = [entry['volume'] for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Calculate percentage change
            volume_change = ((current_volume - avg_volume) / avg_volume) * 100 if avg_volume > 0 else 0
            
            # Determine trend based on timeframe-specific thresholds
            if volume_change >= SIGNIFICANT_THRESHOLD:
                trend = "significant_increase"
            elif volume_change <= -SIGNIFICANT_THRESHOLD:
                trend = "significant_decrease"
            elif volume_change >= MODERATE_THRESHOLD:
                trend = "moderate_increase"
            elif volume_change <= -MODERATE_THRESHOLD:
                trend = "moderate_decrease"
            else:
                trend = "stable"
                
            logger.logger.debug(
                f"Volume trend analysis ({timeframe}): {volume_change:.2f}% change from average. "
                f"Current: {current_volume:,.0f}, Avg: {avg_volume:,.0f}, "
                f"Trend: {trend}"
            )
            
            return volume_change, trend
            
        except Exception as e:
            logger.log_error(f"Volume Trend Analysis - {timeframe}", str(e))
            return 0.0, "error"

    @ensure_naive_datetimes
    def _generate_weekly_summary(self) -> bool:
        """
        Generate and post a weekly summary of predictions and performance across all timeframes
        
        Returns:
            Boolean indicating if summary was successfully posted
        """
        try:
            # Check if it's Sunday (weekday 6) and around midnight
            now = strip_timezone(datetime.now())
            if now.weekday() != 6 or now.hour != 0:
                return False
                
            # Get performance stats for all timeframes
            overall_stats = {}
            for timeframe in self.timeframes:
                performance_stats = self.config.db.get_prediction_performance(timeframe=timeframe)
                
                if not performance_stats:
                    continue
                    
                # Calculate overall stats for this timeframe
                total_correct = sum(p["correct_predictions"] for p in performance_stats)
                total_predictions = sum(p["total_predictions"] for p in performance_stats)
                
                if total_predictions > 0:
                    overall_accuracy = (total_correct / total_predictions) * 100
                    overall_stats[timeframe] = {
                        "accuracy": overall_accuracy,
                        "total": total_predictions,
                        "correct": total_correct
                    }
                    
                    # Get token-specific stats
                    token_stats = {}
                    for stat in performance_stats:
                        token = stat["token"]
                        if stat["total_predictions"] > 0:
                            token_stats[token] = {
                                "accuracy": stat["accuracy_rate"],
                                "total": stat["total_predictions"]
                            }
                    
                    # Sort tokens by accuracy
                    sorted_tokens = sorted(token_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
                    overall_stats[timeframe]["top_tokens"] = sorted_tokens[:3]
                    overall_stats[timeframe]["bottom_tokens"] = sorted_tokens[-3:] if len(sorted_tokens) >= 3 else []
            
            if not overall_stats:
                return False
                
            # Generate report
            report = "📊 WEEKLY PREDICTION SUMMARY 📊\n\n"
            
            # Add summary for each timeframe
            for timeframe, stats in overall_stats.items():
                if timeframe == "1h":
                    display_tf = "1 HOUR"
                elif timeframe == "24h":
                    display_tf = "24 HOUR"
                else:  # 7d
                    display_tf = "7 DAY"
                    
                report += f"== {display_tf} PREDICTIONS ==\n"
                report += f"Overall Accuracy: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})\n\n"
                
                if stats.get("top_tokens"):
                    report += "Top Performers:\n"
                    for token, token_stats in stats["top_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                if stats.get("bottom_tokens"):
                    report += "\nBottom Performers:\n"
                    for token, token_stats in stats["bottom_tokens"]:
                        report += f"#{token}: {token_stats['accuracy']:.1f}% ({token_stats['total']} predictions)\n"
                        
                report += "\n"
                
            # Ensure report isn't too long
            max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(report) > max_length:
                # Truncate report intelligently
                sections = report.split("==")
                shortened_report = sections[0]  # Keep header
                
                # Add as many sections as will fit
                for section in sections[1:]:
                    if len(shortened_report + "==" + section) <= max_length:
                        shortened_report += "==" + section
                    else:
                        break
                        
                report = shortened_report
            
            # Post the weekly summary
            return self._post_analysis(report, timeframe="summary")
            
        except Exception as e:
            logger.log_error("Weekly Summary", str(e))
            return False

    def _prioritize_tokens(self, available_tokens: List[str], market_data: Dict[str, Any]) -> List[str]:
        """
        Prioritize tokens across all timeframes based on momentum score and other factors
        
        Args:
            available_tokens: List of available token symbols
            market_data: Market data dictionary
            
        Returns:
            Prioritized list of token symbols
        """
        try:
            token_priorities = []
        
            for token in available_tokens:
                # Calculate token-specific priority scores for each timeframe
                priority_scores = {}
                for timeframe in self.timeframes:
                    # Calculate momentum score for this timeframe
                    momentum_score = self._calculate_momentum_score(token, market_data, timeframe=timeframe)
                
                    # Get latest prediction time for this token and timeframe
                    last_prediction = self.config.db.get_active_predictions(token=token, timeframe=timeframe)
                    hours_since_prediction = 24  # Default high value
                
                    if last_prediction:
                        last_time = strip_timezone(datetime.fromisoformat(last_prediction[0]["timestamp"]))
                        hours_since_prediction = safe_datetime_diff(datetime.now(), last_time) / 3600
                
                    # Scale time factor based on timeframe
                    if timeframe == "1h":
                        time_factor = 2.0  # Regular weight for 1h
                    elif timeframe == "24h":
                        time_factor = 0.5  # Lower weight for 24h
                    else:  # 7d
                        time_factor = 0.1  # Lowest weight for 7d
                        
                    # Priority score combines momentum and time since last prediction
                    priority_scores[timeframe] = momentum_score + (hours_since_prediction * time_factor)
                
                # Combined score is weighted average across all timeframes with focus on shorter timeframes
                combined_score = (
                    priority_scores.get("1h", 0) * 0.6 +
                    priority_scores.get("24h", 0) * 0.3 +
                    priority_scores.get("7d", 0) * 0.1
                )
                
                token_priorities.append((token, combined_score))
        
            # Sort by priority score (highest first)
            sorted_tokens = [t[0] for t in sorted(token_priorities, key=lambda x: x[1], reverse=True)]
        
            return sorted_tokens
        
        except Exception as e:
            logger.log_error("Token Prioritization", str(e))
            return available_tokens  # Return original list on error

    def _generate_predictions(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Generate market predictions for a specific token at a specific timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for prediction
            
        Returns:
            Prediction data dictionary
        """
        try:
            logger.logger.info(f"Generating {timeframe} predictions for {token}")
        
            # Fix: Add try/except to handle the max() arg is an empty sequence error
            try:
                # Generate prediction for the specified timeframe
                prediction = self.prediction_engine.generate_prediction(
                    token=token,
                    market_data=market_data,
                    timeframe=timeframe
                )
            except ValueError as ve:
                # Handle the empty sequence error specifically
                if "max() arg is an empty sequence" in str(ve):
                    logger.logger.warning(f"Empty sequence error for {token} ({timeframe}), using fallback prediction")
                    # Create a basic fallback prediction
                    token_data = market_data.get(token, {})
                    current_price = token_data.get('current_price', 0)
                
                    # Adjust fallback values based on timeframe
                    if timeframe == "1h":
                        change_pct = 0.5
                        confidence = 60
                        range_factor = 0.01
                    elif timeframe == "24h":
                        change_pct = 1.2
                        confidence = 55
                        range_factor = 0.025
                    else:  # 7d
                        change_pct = 2.5
                        confidence = 50
                        range_factor = 0.05
                
                    prediction = {
                        "prediction": {
                            "price": current_price * (1 + change_pct/100),
                            "confidence": confidence,
                            "lower_bound": current_price * (1 - range_factor),
                            "upper_bound": current_price * (1 + range_factor),
                            "percent_change": change_pct,
                            "timeframe": timeframe
                        },
                        "rationale": f"Technical analysis based on recent price action for {token} over the {timeframe} timeframe.",
                        "sentiment": "NEUTRAL",
                        "key_factors": ["Technical analysis", "Recent price action", "Market conditions"],
                        "timestamp": strip_timezone(datetime.now())
                    }
                else:
                    # Re-raise other ValueError exceptions
                    raise
        
            # Store prediction in database
            prediction_id = self.config.db.store_prediction(token, prediction, timeframe=timeframe)
            logger.logger.info(f"Stored {token} {timeframe} prediction with ID {prediction_id}")
        
            return prediction
        
        except Exception as e:
            logger.log_error(f"Generate Predictions - {token} ({timeframe})", str(e))
            return {}

    @ensure_naive_datetimes
    def _run_analysis_cycle(self) -> None:
        """Run analysis and posting cycle for all tokens with multi-timeframe prediction integration"""
        try:
            # First, evaluate any expired predictions
            self._evaluate_expired_predictions()
            logger.logger.debug("TIMEFRAME DEBUGGING INFO:")
            for tf in self.timeframes:
                logger.logger.debug(f"Timeframe: {tf}")
                last_post = self.timeframe_last_post.get(tf)
                next_scheduled = self.next_scheduled_posts.get(tf)
                logger.logger.debug(f"  last_post type: {type(last_post)}, value: {last_post}")
                logger.logger.debug(f"  next_scheduled type: {type(next_scheduled)}, value: {next_scheduled}")
        
            # Get market data
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data")
                return
    
            # Get available tokens
            available_tokens = [token for token in self.reference_tokens if token in market_data]
            if not available_tokens:
                logger.logger.error("No token data available")
                return
    
            # Decide what type of content to prioritize
            post_priority = self._decide_post_type(market_data)
            post_success = False  # Track if any posting was successful

            # Act based on the decision
            if post_priority == "reply":
                # Prioritize finding and replying to posts
                post_success = self._check_for_reply_opportunities(market_data)
                if post_success:
                    logger.logger.info("Successfully posted replies based on priority decision")
                    return
                # Fall through to other post types if no reply opportunities
        
            elif post_priority == "prediction":
                # Prioritize prediction posts (try timeframe rotation first)
                post_success = self._post_timeframe_rotation(market_data)
                if post_success:
                    logger.logger.info("Posted scheduled timeframe prediction based on priority decision")
                    return
                # Fall through to token-specific predictions for 1h timeframe
        
            elif post_priority == "correlation":
                # Generate and post correlation report
                report_timeframe = self.timeframes[datetime.now().hour % len(self.timeframes)]
                correlation_report = self._generate_correlation_report(market_data, timeframe=report_timeframe)
                if correlation_report and self._post_analysis(correlation_report, timeframe=report_timeframe):
                    logger.logger.info(f"Posted {report_timeframe} correlation matrix report based on priority decision")
                    post_success = True
                    return
                
            elif post_priority == "tech":
                # Prioritize posting tech educational content
                post_success = self._post_tech_educational_content(market_data)
                if post_success:
                    logger.logger.info("Posted tech educational content based on priority decision")
                    return
                # Fall through to other post types if tech posting failed
    
            # Initialize trigger_type with a default value to prevent NoneType errors
            trigger_type = "regular_interval"
        
            # If we haven't had any successful posts yet, try 1h predictions
            if not post_success:
                # Prioritize tokens instead of just shuffling
                available_tokens = self._prioritize_tokens(available_tokens, market_data)
        
                # For 1h predictions and regular updates, try each token until we find one that's suitable
                for token_to_analyze in available_tokens:
                    should_post, token_trigger_type = self._should_post_update(token_to_analyze, market_data, timeframe="1h")
            
                    if should_post:
                        # Update the main trigger_type variable
                        trigger_type = token_trigger_type
                        logger.logger.info(f"Starting {token_to_analyze} analysis cycle - Trigger: {trigger_type}")
                
                        # Generate prediction for this token with 1h timeframe
                        prediction = self._generate_predictions(token_to_analyze, market_data, timeframe="1h")
                
                        if not prediction:
                            logger.logger.error(f"Failed to generate 1h prediction for {token_to_analyze}")
                            continue
    
                        # Get both standard analysis and prediction-focused content 
                        standard_analysis, storage_data = self._analyze_market_sentiment(
                            token_to_analyze, market_data, trigger_type, timeframe="1h"
                        )
                        prediction_tweet = self._format_prediction_tweet(token_to_analyze, prediction, market_data, timeframe="1h")
                
                        # Choose which type of content to post based on trigger and past posts
                        # For prediction-specific triggers or every third post, post prediction
                        should_post_prediction = (
                            "prediction" in trigger_type or 
                            random.random() < 0.35  # 35% chance of posting prediction instead of analysis
                        )
                
                        if should_post_prediction:
                            analysis_to_post = prediction_tweet
                            # Add prediction data to storage
                            if storage_data:
                                storage_data['is_prediction'] = True
                                storage_data['prediction_data'] = prediction
                        else:
                            analysis_to_post = standard_analysis
                            if storage_data:
                                storage_data['is_prediction'] = False
                
                        if not analysis_to_post:
                            logger.logger.error(f"Failed to generate content for {token_to_analyze}")
                            continue
                    
                        # Check for duplicates
                        last_posts = self._get_last_posts_by_timeframe(timeframe="1h")
                        if not self._is_duplicate_analysis(analysis_to_post, last_posts, timeframe="1h"):
                            if self._post_analysis(analysis_to_post, timeframe="1h"):
                                # Only store in database after successful posting
                                if storage_data:
                                    self.config.db.store_posted_content(**storage_data)
                            
                                logger.logger.info(
                                    f"Successfully posted {token_to_analyze} "
                                    f"{'prediction' if should_post_prediction else 'analysis'} - "
                                    f"Trigger: {trigger_type}"
                                )
                        
                                # Store additional smart money metrics
                                if token_to_analyze in market_data:
                                    smart_money = self._analyze_smart_money_indicators(
                                        token_to_analyze, market_data[token_to_analyze], timeframe="1h"
                                    )
                                    self.config.db.store_smart_money_indicators(token_to_analyze, smart_money)
                            
                                    # Store market comparison data
                                    vs_market = self._analyze_token_vs_market(token_to_analyze, market_data, timeframe="1h")
                                    if vs_market:
                                        self.config.db.store_token_market_comparison(
                                            token_to_analyze,
                                            vs_market.get('vs_market_avg_change', 0),
                                            vs_market.get('vs_market_volume_growth', 0),
                                            vs_market.get('outperforming_market', False),
                                            vs_market.get('correlations', {})
                                        )
                            
                                post_success = True
                                # Successfully posted, so we're done with this cycle
                                return
                            else:
                                logger.logger.error(f"Failed to post {token_to_analyze} {'prediction' if should_post_prediction else 'analysis'}")
                                continue  # Try next token
                        else:
                            logger.logger.info(f"Skipping duplicate {token_to_analyze} content - trying another token")
                            continue  # Try next token
                    else:
                        logger.logger.debug(f"No significant {token_to_analyze} changes detected, trying another token")
        
            # If we've tried everything and still haven't posted anything, try tech content
            if not post_success and post_priority != "tech":  # Only if we haven't already tried tech
                if self._post_tech_educational_content(market_data):
                    logger.logger.info("Posted tech educational content as fallback")
                    post_success = True
                    return
        
            # Alternatively try correlation reports
            if not post_success:
                # Alternate between different timeframe correlation reports
                current_hour = datetime.now().hour
                report_timeframe = self.timeframes[current_hour % len(self.timeframes)]
            
                correlation_report = self._generate_correlation_report(market_data, timeframe=report_timeframe)
                if correlation_report and self._post_analysis(correlation_report, timeframe=report_timeframe):
                    logger.logger.info(f"Posted {report_timeframe} correlation matrix report")
                    post_success = True
                    return      

            # FINAL FALLBACK: If still no post, try reply opportunities as a last resort
            # This is the override that should always be checked when other methods have maxed out
            if not post_success:
                logger.logger.info("Checking for reply opportunities as ultimate fallback")
                if self._check_for_reply_opportunities(market_data):
                    logger.logger.info("Successfully posted replies as fallback")
                    post_success = True
                    return

            # If we get here, we tried all tokens but couldn't post anything
            if not post_success:
                logger.logger.warning("Tried all available tokens but couldn't post any analysis or replies")
    
        except Exception as e:
            logger.log_error("Token Analysis Cycle", str(e))

    @ensure_naive_datetimes
    def _decide_post_type(self, market_data: Dict[str, Any]) -> str:
        """
        Make a strategic decision on what type of post to prioritize: prediction, analysis, reply, tech, or correlation
    
        Args:
            market_data: Market data dictionary
            
        Returns:
            String indicating the recommended action: "prediction", "analysis", "reply", "tech", or "correlation"
        """
        try:
            now = strip_timezone(datetime.now())
    
            # Initialize decision factors
            decision_factors = {
                'prediction': 0.0,
                'analysis': 0.0,
                'reply': 0.0,
                'correlation': 0.0,
                'tech': 0.0  # Added tech as a new decision factor
            }
    
            # Factor 1: Time since last post of each type
            # Use existing database methods instead of get_last_post_time
            try:
                # Get recent posts from the database
                recent_posts = self.config.db.get_recent_posts(hours=24)
        
                # Find the most recent posts of each type
                last_analysis_time = None
                last_prediction_time = None
                last_correlation_time = None
                last_tech_time = None  # Added tech time tracking
        
                for post in recent_posts:
                    # Convert timestamp to datetime if it's a string
                    post_timestamp = post.get('timestamp')
                    if isinstance(post_timestamp, str):
                        try:
                            post_timestamp = strip_timezone(datetime.fromisoformat(post_timestamp))
                        except ValueError:
                            continue
            
                    # Check if it's a prediction post
                    if post.get('is_prediction', False):
                        if last_prediction_time is None or post_timestamp > last_prediction_time:
                            last_prediction_time = post_timestamp
                    # Check if it's a correlation post
                    elif 'CORRELATION' in post.get('content', '').upper():
                        if last_correlation_time is None or post_timestamp > last_correlation_time:
                            last_correlation_time = post_timestamp
                    # Check if it's a tech post
                    elif post.get('tech_category', False) or post.get('tech_metadata', False) or 'tech_' in post.get('trigger_type', ''):
                        if last_tech_time is None or post_timestamp > last_tech_time:
                            last_tech_time = post_timestamp
                    # Otherwise it's an analysis post
                    else:
                        if last_analysis_time is None or post_timestamp > last_analysis_time:
                            last_analysis_time = post_timestamp
            except Exception as db_err:
                logger.logger.warning(f"Error retrieving recent posts: {str(db_err)}")
                last_analysis_time = now - timedelta(hours=12)  # Default fallback
                last_prediction_time = now - timedelta(hours=12)  # Default fallback
                last_correlation_time = now - timedelta(hours=48)  # Default fallback
                last_tech_time = now - timedelta(hours=24)  # Default fallback for tech
    
            # Set default values if no posts found
            if last_analysis_time is None:
                last_analysis_time = now - timedelta(hours=24)
            if last_prediction_time is None:
                last_prediction_time = now - timedelta(hours=24)
            if last_correlation_time is None:
                last_correlation_time = now - timedelta(hours=48)
            if last_tech_time is None:
                last_tech_time = now - timedelta(hours=24)
            
            # Calculate hours since each type of post using safe_datetime_diff
            hours_since_analysis = safe_datetime_diff(now, last_analysis_time) / 3600
            hours_since_prediction = safe_datetime_diff(now, last_prediction_time) / 3600
            hours_since_correlation = safe_datetime_diff(now, last_correlation_time) / 3600
            hours_since_tech = safe_datetime_diff(now, last_tech_time) / 3600
    
            # Check time since last reply (using our sanitized datetime)
            last_reply_time = strip_timezone(self._ensure_datetime(self.last_reply_time))
            hours_since_reply = safe_datetime_diff(now, last_reply_time) / 3600
    
            # Add time factors to decision weights (more time = higher weight)
            decision_factors['prediction'] += min(5.0, hours_since_prediction * 0.5)  # Cap at 5.0
            decision_factors['analysis'] += min(5.0, hours_since_analysis * 0.5)  # Cap at 5.0
            decision_factors['reply'] += min(5.0, hours_since_reply * 0.8)  # Higher weight for replies
            decision_factors['correlation'] += min(3.0, hours_since_correlation * 0.1)  # Lower weight for correlations
            decision_factors['tech'] += min(4.0, hours_since_tech * 0.6)  # Medium weight for tech content
    
            # Factor 2: Time of day considerations - adjust to audience activity patterns
            current_hour = now.hour
    
            # Morning hours (6-10 AM): Favor analyses, predictions and tech content for day traders
            if 6 <= current_hour <= 10:
                decision_factors['prediction'] += 2.0
                decision_factors['analysis'] += 1.5
                decision_factors['tech'] += 1.5  # Good time for educational content
                decision_factors['reply'] += 0.5
        
            # Mid-day (11-15): Balanced approach, slight favor to replies
            elif 11 <= current_hour <= 15:
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 1.0
                decision_factors['tech'] += 1.2  # Still good for tech content
                decision_factors['reply'] += 1.5
        
            # Evening hours (16-22): Strong favor to replies to engage with community
            elif 16 <= current_hour <= 22:
                decision_factors['prediction'] += 0.5
                decision_factors['analysis'] += 1.0
                decision_factors['tech'] += 0.8  # Lower priority but still relevant
                decision_factors['reply'] += 2.5
        
            # Late night (23-5): Favor analyses, tech content, deprioritize replies
            else:
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 2.0
                decision_factors['tech'] += 2.0  # Great for tech content when audience is more global
                decision_factors['reply'] += 0.5
                decision_factors['correlation'] += 1.5  # Good time for correlation reports
    
            # Factor 3: Market volatility - in volatile markets, predictions and analyses are more valuable
            market_volatility = self._calculate_market_volatility(market_data)
    
            # High volatility boosts prediction and analysis priority
            if market_volatility > 3.0:  # High volatility
                decision_factors['prediction'] += 2.0
                decision_factors['analysis'] += 1.5
                decision_factors['tech'] -= 0.5  # Less focus on educational content during high volatility
            elif market_volatility > 1.5:  # Moderate volatility
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 1.0
            else:  # Low volatility, good time for educational content
                decision_factors['tech'] += 1.0
    
            # Factor 4: Community engagement level - check for active discussions
            active_discussions = self._check_for_active_discussions(market_data)
            if active_discussions:
                # If there are active discussions, favor replies
                decision_factors['reply'] += len(active_discussions) * 0.5  # More discussions = higher priority
                
                # Check if there are tech-related discussions
                tech_discussions = [d for d in active_discussions if self._is_tech_related_post(d)]
                if tech_discussions:
                    # If tech discussions are happening, boost tech priority
                    decision_factors['tech'] += len(tech_discussions) * 0.8
                    
                logger.logger.debug(f"Found {len(active_discussions)} active discussions ({len(tech_discussions)} tech-related), boosting reply priority")
    
            # Factor 5: Check scheduled timeframe posts - these get high priority
            due_timeframes = [tf for tf in self.timeframes if self._should_post_timeframe_now(tf)]
            if due_timeframes:
                decision_factors['prediction'] += 3.0  # High priority for scheduled predictions
                logger.logger.debug(f"Scheduled timeframe posts due: {due_timeframes}")
    
            # Factor 6: Day of week considerations
            weekday = now.weekday()  # 0=Monday, 6=Sunday
    
            # Weekends: More casual engagement (replies), less formal analysis
            if weekday >= 5:  # Saturday or Sunday
                decision_factors['reply'] += 1.5
                decision_factors['tech'] += 1.0  # Good for educational content on weekends
                decision_factors['correlation'] += 0.5
            # Mid-week: Focus on predictions and analysis
            elif 1 <= weekday <= 3:  # Tuesday to Thursday
                decision_factors['prediction'] += 1.0
                decision_factors['analysis'] += 0.5
                decision_factors['tech'] += 0.5  # Steady tech content through the week
    
            # Factor 7: Tech content readiness
            tech_analysis = self._analyze_tech_topics(market_data)
            if tech_analysis.get('enabled', False) and tech_analysis.get('candidate_topics', []):
                # Boost tech priority if we have ready topics
                decision_factors['tech'] += 2.0
                logger.logger.debug(f"Tech topics ready: {len(tech_analysis.get('candidate_topics', []))}")
            
            # Log decision factors for debugging
            logger.logger.debug(f"Post type decision factors: {decision_factors}")
    
            # Determine highest priority action
            highest_priority = max(decision_factors.items(), key=lambda x: x[1])
            action = highest_priority[0]
    
            # Special case: If correlation has reasonable score and it's been a while, prioritize it
            if hours_since_correlation > 48 and decision_factors['correlation'] > 2.0:
                action = 'correlation'
                logger.logger.debug(f"Overriding to correlation post ({hours_since_correlation}h since last one)")
    
            logger.logger.info(f"Decided post type: {action} (score: {highest_priority[1]:.2f})")
            return action
    
        except Exception as e:
            logger.log_error("Decide Post Type", str(e))
            # Default to analysis as a safe fallback
            return "analysis"
        
    def _calculate_market_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate overall market volatility score based on price movements
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Volatility score (0.0-5.0)
        """
        try:
            if not market_data:
                return 1.0  # Default moderate volatility
            
            # Extract price changes for major tokens
            major_tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']
            changes = []
        
            for token in major_tokens:
                if token in market_data:
                    change = abs(market_data[token].get('price_change_percentage_24h', 0))
                    changes.append(change)
        
            if not changes:
                return 1.0
            
            # Calculate average absolute price change
            avg_change = sum(changes) / len(changes)
        
            # Calculate volatility score (normalized to a 0-5 scale)
            # <1% = Very Low, 1-2% = Low, 2-3% = Moderate, 3-5% = High, >5% = Very High
            if avg_change < 1.0:
                return 0.5  # Very low volatility
            elif avg_change < 2.0:
                return 1.0  # Low volatility
            elif avg_change < 3.0:
                return 2.0  # Moderate volatility
            elif avg_change < 5.0:
                return 3.0  # High volatility
            else:
                return 5.0  # Very high volatility
    
        except Exception as e:
            logger.log_error("Calculate Market Volatility", str(e))
            return 1.0  # Default to moderate volatility on error

    def _check_for_active_discussions(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check for active token discussions that might warrant replies
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            List of posts representing active discussions
        """
        try:
            # Get recent timeline posts
            recent_posts = self.timeline_scraper.scrape_timeline(count=15)
            if not recent_posts:
                return []
            
            # Filter for posts with engagement (replies, likes)
            engaged_posts = []
            for post in recent_posts:
                # Simple engagement check
                has_engagement = (
                    post.get('reply_count', 0) > 0 or
                    post.get('like_count', 0) > 2 or
                    post.get('retweet_count', 0) > 0
                )
            
                if has_engagement:
                    # Analyze the post content
                    analysis = self.content_analyzer.analyze_post(post)
                    post['content_analysis'] = analysis
                
                    # Check if it's a market-related post with sufficient reply score
                    if analysis.get('reply_worthy', False):
                        engaged_posts.append(post)
        
            return engaged_posts
    
        except Exception as e:
            logger.log_error("Check Active Discussions", str(e))
            return []
            
    def _analyze_smart_money_indicators(self, token: str, token_data: Dict[str, Any], 
                                      timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze potential smart money movements in a token
        Adjusted for different timeframes
        
        Args:
            token: Token symbol
            token_data: Token market data
            timeframe: Timeframe for analysis
            
        Returns:
            Smart money analysis results
        """
        try:
            # Get historical data over multiple timeframes - adjusted based on prediction timeframe
            if timeframe == "1h":
                hourly_data = self._get_historical_volume_data(token, minutes=60, timeframe=timeframe)
                daily_data = self._get_historical_volume_data(token, minutes=1440, timeframe=timeframe)
                # For 1h predictions, we care about recent volume patterns
                short_term_focus = True
            elif timeframe == "24h":
                # For 24h predictions, we want more data
                hourly_data = self._get_historical_volume_data(token, minutes=240, timeframe=timeframe)  # 4 hours
                daily_data = self._get_historical_volume_data(token, minutes=7*1440, timeframe=timeframe)  # 7 days
                short_term_focus = False
            else:  # 7d
                # For weekly predictions, we need even more historical context
                hourly_data = self._get_historical_volume_data(token, minutes=24*60, timeframe=timeframe)  # 24 hours
                daily_data = self._get_historical_volume_data(token, minutes=30*1440, timeframe=timeframe)  # 30 days
                short_term_focus = False
            
            current_volume = token_data['volume']
            current_price = token_data['current_price']
            
            # Volume anomaly detection
            hourly_volumes = [entry['volume'] for entry in hourly_data]
            daily_volumes = [entry['volume'] for entry in daily_data]
            
            # Calculate baselines
            avg_hourly_volume = statistics.mean(hourly_volumes) if hourly_volumes else current_volume
            avg_daily_volume = statistics.mean(daily_volumes) if daily_volumes else current_volume
            
            # Volume Z-score (how many standard deviations from mean)
            hourly_std = statistics.stdev(hourly_volumes) if len(hourly_volumes) > 1 else 1
            volume_z_score = (current_volume - avg_hourly_volume) / hourly_std if hourly_std != 0 else 0
            
            # Price-volume divergence
            # (Price going down while volume increasing suggests accumulation)
            price_direction = 1 if token_data['price_change_percentage_24h'] > 0 else -1
            volume_direction = 1 if current_volume > avg_daily_volume else -1
            
            # Divergence detected when price and volume move in opposite directions
            divergence = (price_direction != volume_direction)
            
            # Adjust accumulation thresholds based on timeframe
            if timeframe == "1h":
                price_change_threshold = 2.0
                volume_multiplier = 1.5
            elif timeframe == "24h":
                price_change_threshold = 3.0
                volume_multiplier = 1.8
            else:  # 7d
                price_change_threshold = 5.0
                volume_multiplier = 2.0
            
            # Check for abnormal volume with minimal price movement (potential accumulation)
            stealth_accumulation = (abs(token_data['price_change_percentage_24h']) < price_change_threshold and 
                                  (current_volume > avg_daily_volume * volume_multiplier))
            
            # Calculate volume profile - percentage of volume in each hour
            volume_profile = {}
            
            # Adjust volume profiling based on timeframe
            if timeframe == "1h":
                # For 1h predictions, look at hourly volume distribution over the day
                hours_to_analyze = 24
            elif timeframe == "24h":
                # For 24h predictions, look at volume by day over the week 
                hours_to_analyze = 7 * 24
            else:  # 7d
                # For weekly, look at entire month
                hours_to_analyze = 30 * 24
            
            if hourly_data:
                for i in range(min(hours_to_analyze, 24)):  # Cap at 24 hours for profile
                    hour_window = strip_timezone(datetime.now() - timedelta(hours=i+1))
                    hour_volume = sum(entry['volume'] for entry in hourly_data 
                                    if hour_window <= entry['timestamp'] <= hour_window + timedelta(hours=1))
                    volume_profile[f"hour_{i+1}"] = hour_volume
            
            # Detect unusual trading hours (potential institutional activity)
            total_volume = sum(volume_profile.values()) if volume_profile else 0
            unusual_hours = []
            
            # Adjust unusual hour threshold based on timeframe
            unusual_hour_threshold = 15 if timeframe == "1h" else 20 if timeframe == "24h" else 25
            
            if total_volume > 0:
                for hour, vol in volume_profile.items():
                    hour_percentage = (vol / total_volume) * 100
                    if hour_percentage > unusual_hour_threshold:  # % threshold varies by timeframe
                        unusual_hours.append(hour)
            
            # Detect volume clusters (potential accumulation zones)
            volume_cluster_detected = False
            min_cluster_size = 3 if timeframe == "1h" else 2 if timeframe == "24h" else 2
            cluster_threshold = 1.3 if timeframe == "1h" else 1.5 if timeframe == "24h" else 1.8
            
            if len(hourly_volumes) >= min_cluster_size:
                for i in range(len(hourly_volumes)-min_cluster_size+1):
                    if all(vol > avg_hourly_volume * cluster_threshold for vol in hourly_volumes[i:i+min_cluster_size]):
                        volume_cluster_detected = True
                        break           
            
            # Calculate additional metrics for longer timeframes
            pattern_metrics = {}
            
            if timeframe in ["24h", "7d"]:
                # Calculate volume trends over different periods
                if len(daily_volumes) >= 7:
                    week1_avg = statistics.mean(daily_volumes[:7])
                    week2_avg = statistics.mean(daily_volumes[7:14]) if len(daily_volumes) >= 14 else week1_avg
                    week3_avg = statistics.mean(daily_volumes[14:21]) if len(daily_volumes) >= 21 else week1_avg
                    
                    pattern_metrics["volume_trend_week1_to_week2"] = ((week1_avg / week2_avg) - 1) * 100 if week2_avg > 0 else 0
                    pattern_metrics["volume_trend_week2_to_week3"] = ((week2_avg / week3_avg) - 1) * 100 if week3_avg > 0 else 0
                
                # Check for volume breakout patterns
                if len(hourly_volumes) >= 48:
                    recent_max = max(hourly_volumes[:24])
                    previous_max = max(hourly_volumes[24:48])
                    
                    pattern_metrics["volume_breakout"] = recent_max > previous_max * 1.5
                
                # Check for consistent high volume days
                if len(daily_volumes) >= 14:
                    high_volume_days = [vol > avg_daily_volume * 1.3 for vol in daily_volumes[:14]]
                    pattern_metrics["consistent_high_volume"] = sum(high_volume_days) >= 5
            
            # Results
            results = {
                'volume_z_score': volume_z_score,
                'price_volume_divergence': divergence,
                'stealth_accumulation': stealth_accumulation,
                'abnormal_volume': abs(volume_z_score) > self.SMART_MONEY_ZSCORE_THRESHOLD,
                'volume_vs_hourly_avg': (current_volume / avg_hourly_volume) - 1,
                'volume_vs_daily_avg': (current_volume / avg_daily_volume) - 1,
                'unusual_trading_hours': unusual_hours,
                'volume_cluster_detected': volume_cluster_detected,
                'timeframe': timeframe
            }
            
            # Add pattern metrics for longer timeframes
            if pattern_metrics:
                results['pattern_metrics'] = pattern_metrics
            
            # Store in database
            self.config.db.store_smart_money_indicators(token, results)
            
            return results
        except Exception as e:
            logger.log_error(f"Smart Money Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}
    
    def _analyze_volume_profile(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze volume distribution and patterns for a token
        Returns different volume metrics based on timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Volume profile analysis results
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            current_volume = token_data.get('volume', 0)
            
            # Adjust analysis window based on timeframe
            if timeframe == "1h":
                hours_to_analyze = 24
                days_to_analyze = 1
            elif timeframe == "24h":
                hours_to_analyze = 7 * 24
                days_to_analyze = 7
            else:  # 7d
                hours_to_analyze = 30 * 24
                days_to_analyze = 30
            
            # Get historical data
            historical_data = self._get_historical_volume_data(token, minutes=hours_to_analyze * 60, timeframe=timeframe)
            
            # Create volume profile by hour of day
            hourly_profile = {}
            for hour in range(24):
                hourly_profile[hour] = 0
            
            # Fill the profile
            for entry in historical_data:
                timestamp = entry.get('timestamp')
                if timestamp:
                    hour = timestamp.hour
                    hourly_profile[hour] += entry.get('volume', 0)
            
            # Calculate daily pattern
            total_volume = sum(hourly_profile.values())
            if total_volume > 0:
                hourly_percentage = {hour: (volume / total_volume) * 100 for hour, volume in hourly_profile.items()}
            else:
                hourly_percentage = {hour: 0 for hour in range(24)}
            
            # Find peak volume hours
            peak_hours = sorted(hourly_percentage.items(), key=lambda x: x[1], reverse=True)[:3]
            low_hours = sorted(hourly_percentage.items(), key=lambda x: x[1])[:3]
            
            # Check for consistent daily patterns
            historical_volumes = [entry.get('volume', 0) for entry in historical_data]
            avg_volume = statistics.mean(historical_volumes) if historical_volumes else current_volume
            
            # Create day of week profile for longer timeframes
            day_of_week_profile = {}
            if timeframe in ["24h", "7d"] and len(historical_data) >= 7 * 24:
                for day in range(7):
                    day_of_week_profile[day] = 0
                
                # Fill the profile
                for entry in historical_data:
                    timestamp = entry.get('timestamp')
                    if timestamp:
                        day = timestamp.weekday()
                        day_of_week_profile[day] += entry.get('volume', 0)
                
                # Calculate percentages
                dow_total = sum(day_of_week_profile.values())
                if dow_total > 0:
                    day_of_week_percentage = {day: (volume / dow_total) * 100 
                                           for day, volume in day_of_week_profile.items()}
                else:
                    day_of_week_percentage = {day: 0 for day in range(7)}
                
                # Find peak trading days
                peak_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1], reverse=True)[:2]
                low_days = sorted(day_of_week_percentage.items(), key=lambda x: x[1])[:2]
            else:
                day_of_week_percentage = {}
                peak_days = []
                low_days = []
            
            # Calculate volume consistency
            if len(historical_volumes) > 0:
                volume_std = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0
                volume_variability = (volume_std / avg_volume) * 100 if avg_volume > 0 else 0
                
                # Volume consistency score (0-100)
                volume_consistency = max(0, 100 - volume_variability)
            else:
                volume_consistency = 50  # Default if not enough data
            
            # Calculate volume trend over the period
            if len(historical_volumes) >= 2:
                earliest_volume = historical_volumes[0]
                latest_volume = historical_volumes[-1]
                period_change = ((latest_volume - earliest_volume) / earliest_volume) * 100 if earliest_volume > 0 else 0
            else:
                period_change = 0
            
            # Assemble results
            volume_profile_results = {
                'hourly_profile': hourly_percentage,
                'peak_hours': peak_hours,
                'low_hours': low_hours,
                'avg_volume': avg_volume,
                'current_volume': current_volume,
                'current_vs_avg': ((current_volume / avg_volume) - 1) * 100 if avg_volume > 0 else 0,
                'volume_consistency': volume_consistency,
                'period_change': period_change,
                'timeframe': timeframe
            }
            
            # Add day of week profile for longer timeframes
            if day_of_week_percentage:
                volume_profile_results['day_of_week_profile'] = day_of_week_percentage
                volume_profile_results['peak_days'] = peak_days
                volume_profile_results['low_days'] = low_days
            
            return volume_profile_results
            
        except Exception as e:
            logger.log_error(f"Volume Profile Analysis - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}

    def _detect_volume_anomalies(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, Any]:
        """
        Detect volume anomalies and unusual patterns
        Adjust detection thresholds based on timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Volume anomaly detection results
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {}
            
            # Adjust anomaly detection window and thresholds based on timeframe
            if timeframe == "1h":
                detection_window = 24  # 24 hours for hourly predictions
                z_score_threshold = 2.0
                volume_spike_threshold = 3.0
                volume_drop_threshold = 0.3
            elif timeframe == "24h":
                detection_window = 7 * 24  # 7 days for daily predictions
                z_score_threshold = 2.5
                volume_spike_threshold = 4.0
                volume_drop_threshold = 0.25
            else:  # 7d
                detection_window = 30 * 24  # 30 days for weekly predictions
                z_score_threshold = 3.0
                volume_spike_threshold = 5.0
                volume_drop_threshold = 0.2
            
            # Get historical data
            volume_data = self._get_historical_volume_data(token, minutes=detection_window * 60, timeframe=timeframe)
            
            volumes = [entry.get('volume', 0) for entry in volume_data] 
            if len(volumes) < 5:
                return {'insufficient_data': True, 'timeframe': timeframe}
            
            current_volume = token_data.get('volume', 0)
            
            # Calculate metrics
            avg_volume = statistics.mean(volumes)
            if len(volumes) > 1:
                vol_std = statistics.stdev(volumes)
                # Z-score: how many standard deviations from the mean
                volume_z_score = (current_volume - avg_volume) / vol_std if vol_std > 0 else 0
            else:
                volume_z_score = 0
            
            # Moving average calculation
            if len(volumes) >= 10:
                ma_window = 5 if timeframe == "1h" else 7 if timeframe == "24h" else 10
                moving_avgs = []
                
                for i in range(len(volumes) - ma_window + 1):
                    window = volumes[i:i+ma_window]
                    moving_avgs.append(sum(window) / len(window))
                
                # Calculate rate of change in moving average
                if len(moving_avgs) >= 2:
                    ma_change = ((moving_avgs[-1] / moving_avgs[0]) - 1) * 100
                else:
                    ma_change = 0
            else:
                ma_change = 0
            
            # Volume spike detection
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            has_volume_spike = volume_ratio > volume_spike_threshold
            
            # Volume drop detection
            has_volume_drop = volume_ratio < volume_drop_threshold
            
            # Detect sustained high/low volume
            if len(volumes) >= 5:
                recent_volumes = volumes[-5:]
                avg_recent_volume = sum(recent_volumes) / len(recent_volumes)
                sustained_high_volume = avg_recent_volume > avg_volume * 1.5
                sustained_low_volume = avg_recent_volume < avg_volume * 0.5
            else:
                sustained_high_volume = False
                sustained_low_volume = False
            
            # Detect volume patterns for longer timeframes
            pattern_detection = {}
            
            if timeframe in ["24h", "7d"] and len(volumes) >= 14:
                # Check for "volume climax" pattern (increasing volumes culminating in a spike)
                vol_changes = [volumes[i]/volumes[i-1] if volumes[i-1] > 0 else 1 for i in range(1, len(volumes))]
                
                if len(vol_changes) >= 5:
                    recent_changes = vol_changes[-5:]
                    climax_pattern = (sum(1 for change in recent_changes if change > 1.1) >= 3) and has_volume_spike
                    pattern_detection["volume_climax"] = climax_pattern
                
                # Check for "volume exhaustion" pattern (decreasing volumes after a spike)
                if len(volumes) >= 10:
                    peak_idx = volumes.index(max(volumes[-10:]))
                    if peak_idx < len(volumes) - 3:
                        post_peak = volumes[peak_idx+1:]
                        exhaustion_pattern = all(post_peak[i] < post_peak[i-1] for i in range(1, len(post_peak)))
                        pattern_detection["volume_exhaustion"] = exhaustion_pattern
            
            # Assemble results
            anomaly_results = {
                'volume_z_score': volume_z_score,
                'volume_ratio': volume_ratio,
                'has_volume_spike': has_volume_spike,
                'has_volume_drop': has_volume_drop,
                'ma_change': ma_change,
                'sustained_high_volume': sustained_high_volume,
                'sustained_low_volume': sustained_low_volume,
                'abnormal_volume': abs(volume_z_score) > z_score_threshold,
                'timeframe': timeframe
            }
            
            # Add pattern detection for longer timeframes
            if pattern_detection:
                anomaly_results['patterns'] = pattern_detection
            
            return anomaly_results
            
        except Exception as e:
            logger.log_error(f"Volume Anomaly Detection - {token} ({timeframe})", str(e))
            return {'timeframe': timeframe}   


    def _analyze_token_vs_market(self, token: str, market_data: Any, timeframe: str = "1h") -> Dict[str, Any]:
        """
        Analyze token performance relative to the overall crypto market.
        Enhanced method with extensive error handling, logging, and different timeframe support.
    
        Args:
            token: Token symbol (e.g., 'BTC', 'ETH')
            market_data: Market data (can be dict or list)
            timeframe: Timeframe for analysis ('1h', '24h', '7d')
        
        Returns:
            Dictionary containing market analysis results
        """
        logger.logger.debug(f"ENTERING _analyze_token_vs_market for {token} ({timeframe})")
    
        # Default response if anything fails
        default_response = {
            "vs_market_avg_change": 0.0,
            "vs_market_percentile": 50.0,
            "market_correlation": 0.0,
            "market_sentiment": "neutral",
            "timeframe": timeframe
        }
    
        try:
            # CRITICAL: First validate and standardize market_data at the start
            if market_data is None:
                logger.logger.error(f"market_data is None for {token} ({timeframe})")
                return default_response
            
            # Log market_data type before conversion
            if isinstance(market_data, list):
                logger.logger.warning(f"CRITICAL: market_data is a list with {len(market_data)} items")
                if len(market_data) > 0:
                    first_item = market_data[0]
                    logger.logger.debug(f"First item type: {type(first_item)}")
                    if isinstance(first_item, dict):
                        logger.logger.debug(f"First item keys: {list(first_item.keys())}")

            if isinstance(market_data, dict) and len(market_data) > 0:
                sample_token = next(iter(market_data))
                sample_data = market_data[sample_token]
                logger.logger.debug(f"Sample token data for {sample_token}. Keys: {list(sample_data.keys())}")
                # If price_change_percentage_24h exists, show its value
                if 'price_change_percentage_24h' in sample_data:
                    logger.logger.debug(f"Sample 24h change value: {sample_data['price_change_percentage_24h']}")
                else:
                    logger.logger.debug(f"price_change_percentage_24h not found in sample data")

            # Convert to standardized dictionary format
            if not isinstance(market_data, dict):
                logger.logger.info(f"Converting non-dict market_data for {token}: {type(market_data)}")
                try:
                    market_data = self._standardize_market_data(market_data)
                    logger.logger.info(f"After standardization, market_data has {len(market_data)} items")
                except Exception as std_error:
                    logger.logger.error(f"Error standardizing market data: {str(std_error)}")
                    return default_response
                
            # If standardization failed or returned empty data
            if not market_data:
                logger.logger.warning(f"Failed to standardize market data for {token}")
                return default_response
        
            # Now safely access token data with case-insensitive lookup
            token_data = market_data.get(token, {})
            if not token_data and token.upper() in market_data:
                token_data = market_data.get(token.upper(), {})
            
            # Also try lowercase if needed
            if not token_data and token.lower() in market_data:
                token_data = market_data.get(token.lower(), {})
        
            # If still no token data, return default
            if not token_data:
                logger.logger.warning(f"No data found for token {token} in market_data")
                return default_response
            
            # Verify token_data is a dictionary
            if not isinstance(token_data, dict):
                logger.logger.warning(f"Token data for {token} is not a dictionary: {type(token_data)}")
                return default_response
            
            # Get reference tokens for comparison (excluding the current token)
            reference_tokens = []
            if hasattr(self, 'reference_tokens'):
                reference_tokens = [t for t in self.reference_tokens if t != token]
            else:
                # Fallback if reference_tokens not available
                reference_tokens = [t for t in market_data.keys() if t != token]
            
            # Select appropriate tokens based on timeframe
            if timeframe == "1h":
                # For hourly, focus on major tokens
                filtered_ref_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP"]
            elif timeframe == "24h":
                # For daily, use more tokens
                filtered_ref_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "DOT", "POL"]
            else:  # 7d
                # For weekly, use all reference tokens
                filtered_ref_tokens = reference_tokens
            
            # Keep only tokens that exist in market_data
            reference_tokens = [t for t in filtered_ref_tokens if t in market_data]
        
            # Validate we have reference tokens
            if not reference_tokens:
                logger.logger.warning(f"No reference tokens found for comparison with {token}")
                return default_response
            
            # Calculate market metrics
            # ----- 1. Price Changes -----
            market_changes = []
        
            for ref_token in reference_tokens:
                ref_data = market_data.get(ref_token, {})
            
                # Skip if not a dict
                if not isinstance(ref_data, dict):
                    continue
                
                # Extract price change based on timeframe
                if timeframe == "1h":
                    change_keys = ['price_change_percentage_1h_in_currency', 'price_change_1h', 'change_1h', '1h_change']
                elif timeframe == "24h":
                    change_keys = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
                else:  # 7d
                    change_keys = ['price_change_percentage_7d_in_currency', 'price_change_7d', 'change_7d', '7d_change']
                
                # Try each key
                for change_key in change_keys:
                    if change_key in ref_data:
                        try:
                            change_value = float(ref_data[change_key])
                            market_changes.append(change_value)
                            break  # Found valid value
                        except (ValueError, TypeError):
                            continue  # Try next key
        
            # Verify we have market changes data
            if not market_changes:
                logger.logger.warning(f"No market change data available for comparison with {token}")
                return default_response
            
            # Calculate market average
            market_avg_change = statistics.mean(market_changes) if market_changes else 0.0
        
            # ----- 2. Extract Token Change -----
            token_change = 0.0
        
            # Determine which keys to try based on timeframe
            if timeframe == "1h":
                change_keys = ['price_change_percentage_1h_in_currency', 'price_change_1h', 'change_1h', '1h_change']
            elif timeframe == "24h":
                change_keys = ['price_change_percentage_24h', 'price_change_24h', 'change_24h', '24h_change']
            else:  # 7d
                change_keys = ['price_change_percentage_7d_in_currency', 'price_change_7d', 'change_7d', '7d_change']
            
            # Try each key
            for change_key in change_keys:
                if change_key in token_data:
                    try:
                        token_change = float(token_data[change_key])
                        break  # Found valid value
                    except (ValueError, TypeError):
                        continue
                    
            # ----- 3. Volume Data -----
            # Compare volume changes
            try:
                # Determine volume window based on timeframe
                if timeframe == "1h":
                    volume_window_minutes = 60  # 1 hour
                elif timeframe == "24h":
                    volume_window_minutes = 24 * 60  # 24 hours
                else:  # 7d
                    volume_window_minutes = 7 * 24 * 60  # 7 days
                
                # Get token volume
                token_volume = token_data.get('volume', 0)
                if not token_volume and 'total_volume' in token_data:
                    token_volume = token_data['total_volume']
                
                # Try to get historical volume data for token
                token_historical = []
                if hasattr(self, '_get_historical_volume_data'):
                    try:
                        token_historical = self._get_historical_volume_data(
                            token, 
                            minutes=volume_window_minutes, 
                            timeframe=timeframe
                        )
                    except Exception as vol_err:
                        logger.logger.debug(f"Error getting historical volume: {str(vol_err)}")
                    
                # Calculate token volume trend
                token_volume_change = 0.0
                if token_historical and hasattr(self, '_analyze_volume_trend'):
                    try:
                        token_volume_change, _ = self._analyze_volume_trend(
                            token_volume,
                            token_historical,
                            timeframe=timeframe
                        )
                    except Exception as trend_err:
                        logger.logger.debug(f"Error analyzing volume trend: {str(trend_err)}")
                    
                # Calculate market average volume change
                market_volume_changes = []
                for ref_token in reference_tokens:
                    try:
                        ref_data = market_data.get(ref_token, {})
                        ref_volume = ref_data.get('volume', ref_data.get('total_volume', 0))
                    
                        if hasattr(self, '_get_historical_volume_data') and hasattr(self, '_analyze_volume_trend'):
                            ref_historical = self._get_historical_volume_data(
                                ref_token, 
                                minutes=volume_window_minutes, 
                                timeframe=timeframe
                            )
                        
                            if ref_historical:
                                vol_change, _ = self._analyze_volume_trend(
                                    ref_volume,
                                    ref_historical,
                                    timeframe=timeframe
                                )
                                market_volume_changes.append(vol_change)
                    except Exception as ref_err:
                        logger.logger.debug(f"Error processing volume for {ref_token}: {str(ref_err)}")
                        continue
                    
                # Calculate average market volume change
                market_avg_volume_change = statistics.mean(market_volume_changes) if market_volume_changes else 0.0
                volume_growth_diff = token_volume_change - market_avg_volume_change
            
            except Exception as volume_err:
                logger.logger.warning(f"Error calculating volume metrics: {str(volume_err)}")
                volume_growth_diff = 0.0
            
            # ----- 4. Calculate Correlations -----
            correlations = {}
        
            try:
                # Get historical price data for correlation calculation
                # Time window depends on timeframe
                if timeframe == "1h":
                    history_hours = 24  # Last 24 hours for hourly
                elif timeframe == "24h":
                    history_hours = 7 * 24  # Last 7 days for daily
                else:  # 7d
                    history_hours = 30 * 24  # Last 30 days for weekly
                
                # Get token price history if the method exists
                token_prices = []
                if hasattr(self, '_get_historical_price_data'):
                    try:
                        token_history = self._get_historical_price_data(token, hours=history_hours, timeframe=timeframe)
                        token_prices = [entry.get('price', 0) for entry in token_history]
                    except Exception as hist_err:
                        logger.logger.debug(f"Error getting token price history: {str(hist_err)}")
                    
                # Calculate correlation with each reference token
                for ref_token in reference_tokens:
                    if ref_token in market_data:
                        ref_data = market_data.get(ref_token, {})
                    
                        # Simple direction correlation
                        token_direction = 1 if token_change > 0 else -1
                        ref_change = 0.0
                    
                        # Get ref token change with same keys as token
                        for change_key in change_keys:
                            if change_key in ref_data:
                                try:
                                    ref_change = float(ref_data[change_key])
                                    break
                                except (ValueError, TypeError):
                                    continue
                                
                        ref_direction = 1 if ref_change > 0 else -1
                        direction_match = token_direction == ref_direction
                    
                        # Advanced price correlation if we have history
                        price_correlation = 0.0
                        if token_prices and hasattr(self, '_get_historical_price_data'):
                            try:
                                ref_history = self._get_historical_price_data(ref_token, hours=history_hours, timeframe=timeframe)
                                ref_prices = [entry.get('price', 0) for entry in ref_history]
                            
                                if len(token_prices) > 5 and len(ref_prices) > 5:
                                    # Match lengths
                                    min_length = min(len(token_prices), len(ref_prices))
                                    token_prices_adj = token_prices[:min_length]
                                    ref_prices_adj = ref_prices[:min_length]
                                
                                    if len(token_prices_adj) > 1 and len(ref_prices_adj) > 1:
                                        # Calculate correlation with numpy
                                        price_correlation = np.corrcoef(token_prices_adj, ref_prices_adj)[0, 1]
                                        # Handle NaN
                                        if np.isnan(price_correlation):
                                            price_correlation = 0.0
                            except Exception as corr_err:
                                logger.logger.debug(f"Error calculating correlation: {str(corr_err)}")
                            
                        # Store correlation data
                        correlations[ref_token] = {
                            'price_correlation': price_correlation,
                            'direction_match': direction_match,
                            'token_change': token_change,
                            'ref_token_change': ref_change
                        }
                    
            except Exception as corr_err:
                logger.logger.warning(f"Error calculating correlations: {str(corr_err)}")
            
            # ----- 5. Advanced Metrics for Longer Timeframes -----
            extended_metrics = {}
        
            if timeframe in ["24h", "7d"]:
                try:
                    # Sector performance analysis
                    defi_tokens = [t for t in reference_tokens if t in ["UNI", "AAVE"]]
                    layer1_tokens = [t for t in reference_tokens if t in ["ETH", "SOL", "AVAX", "NEAR"]]
                
                    # Calculate sector averages
                    if defi_tokens:
                        defi_changes = []
                        for t in defi_tokens:
                            if t in market_data:
                                for key in change_keys:
                                    if key in market_data[t]:
                                        try:
                                            defi_changes.append(float(market_data[t][key]))
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                        
                        if defi_changes:
                            defi_avg_change = statistics.mean(defi_changes)
                            extended_metrics['defi_sector_diff'] = token_change - defi_avg_change
                        
                    if layer1_tokens:
                        layer1_changes = []
                        for t in layer1_tokens:
                            if t in market_data:
                                for key in change_keys:
                                    if key in market_data[t]:
                                        try:
                                            layer1_changes.append(float(market_data[t][key]))
                                            break
                                        except (ValueError, TypeError):
                                            continue
                                        
                        if layer1_changes:
                            layer1_avg_change = statistics.mean(layer1_changes)
                            extended_metrics['layer1_sector_diff'] = token_change - layer1_avg_change
                        
                    # Calculate BTC dominance if BTC data available
                    if 'BTC' in market_data:
                        btc_data = market_data['BTC']
                        if 'market_cap' in btc_data:
                            btc_mc = btc_data['market_cap']
                            # Sum market caps
                            total_mc = 0
                            for t, data in market_data.items():
                                if isinstance(data, dict) and 'market_cap' in data:
                                    try:
                                        total_mc += float(data['market_cap'])
                                    except (ValueError, TypeError):
                                        continue
                                    
                            if total_mc > 0:
                                btc_dominance = (btc_mc / total_mc) * 100
                                extended_metrics['btc_dominance'] = btc_dominance
                            
                    # Calculate relative volatility if method exists
                    if hasattr(self, '_calculate_relative_volatility'):
                        try:
                            token_volatility = self._calculate_relative_volatility(
                                token, 
                                reference_tokens, 
                                market_data, 
                                timeframe
                            )
                            if token_volatility is not None:
                                extended_metrics['relative_volatility'] = token_volatility
                        except Exception as vol_err:
                            logger.logger.debug(f"Error calculating volatility: {str(vol_err)}")
                        
                except Exception as ext_err:
                    logger.logger.warning(f"Error calculating extended metrics: {str(ext_err)}")
                
            # ----- 6. Calculate Final Metrics and Market Sentiment -----
            # Performance difference vs market
            vs_market_change = token_change - market_avg_change
        
            # Calculate token's market percentile (% of tokens it's outperforming)
            tokens_outperforming = sum(1 for change in market_changes if token_change > change)
            vs_market_percentile = (tokens_outperforming / len(market_changes)) * 100 if market_changes else 50.0
        
            # Determine if outperforming market
            outperforming = vs_market_change > 0
        
            # Calculate BTC correlation specifically
            btc_correlation = correlations.get('BTC', {}).get('price_correlation', 0)
        
            # Determine market sentiment based on performance
            if vs_market_change > 3.0:
                market_sentiment = "strongly outperforming"
            elif vs_market_change > 1.0:
                market_sentiment = "outperforming"
            elif vs_market_change < -3.0:
                market_sentiment = "strongly underperforming"
            elif vs_market_change < -1.0:
                market_sentiment = "underperforming"
            else:
                market_sentiment = "neutral"
            
            # ----- 7. Store results in database if method exists -----
            try:
                if hasattr(self, 'config') and hasattr(self.config, 'db') and hasattr(self.config.db, 'store_token_market_comparison'):
                    self.config.db.store_token_market_comparison(
                        token,
                        vs_market_change,
                        volume_growth_diff,
                        outperforming,
                        correlations
                    )
            except Exception as db_err:
                logger.logger.warning(f"Error storing comparison data: {str(db_err)}")
            
            # ----- 8. Prepare final result dictionary -----
            result = {
                'vs_market_avg_change': vs_market_change,
                'vs_market_percentile': vs_market_percentile,
                'vs_market_volume_growth': volume_growth_diff,
                'market_correlation': btc_correlation,  # Use BTC correlation as primary metric
                'market_sentiment': market_sentiment,
                'correlations': correlations,
                'outperforming_market': outperforming,
                'btc_correlation': btc_correlation,
                'timeframe': timeframe
            }
        
            # Add extended metrics if available
            if extended_metrics:
                result['extended_metrics'] = extended_metrics
            
            logger.logger.debug(f"Successfully analyzed {token} vs market ({timeframe})")
            return result
        
        except Exception as e:
            error_message = str(e)
            logger.log_error(f"Token vs Market Analysis - {token} ({timeframe})", error_message)
        
            # Return default values to prevent crashes
            return {
                "vs_market_avg_change": 0.0,
                "vs_market_percentile": 50.0,
                "market_correlation": 0.0,
                "market_sentiment": "neutral",
                "timeframe": timeframe
            }
        
    def _calculate_relative_volatility(self, token: str, reference_tokens: List[str], 
                                     market_data: Dict[str, Any], timeframe: str) -> Optional[float]:
        """
        Calculate token's volatility relative to market average
        Returns a ratio where >1 means more volatile than market, <1 means less volatile
        
        Args:
            token: Token symbol
            reference_tokens: List of reference token symbols
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
        
        Returns:
            Relative volatility ratio or None if insufficient data
        """
        try:
            # Get historical data with appropriate window for the timeframe
            if timeframe == "1h":
                hours = 24
            elif timeframe == "24h":
                hours = 7 * 24
            else:  # 7d
                hours = 30 * 24
            
            # Get token history
            token_history = self._get_historical_price_data(token, hours=hours, timeframe=timeframe)
            if len(token_history) < 5:
                return None
            
            token_prices = [entry.get('price', 0) for entry in token_history]
            
            # Calculate token volatility (standard deviation of percent changes)
            token_changes = []
            for i in range(1, len(token_prices)):
                if token_prices[i-1] > 0:
                    pct_change = ((token_prices[i] / token_prices[i-1]) - 1) * 100
                    token_changes.append(pct_change)
                    
            if not token_changes:
                return None
                
            token_volatility = statistics.stdev(token_changes) if len(token_changes) > 1 else 0
            
            # Calculate market average volatility
            market_volatilities = []
            
            for ref_token in reference_tokens:
                if ref_token in market_data:
                    ref_history = self._get_historical_price_data(ref_token, hours=hours, timeframe=timeframe)
                    if len(ref_history) < 5:
                        continue
                        
                    ref_prices = [entry.get('price', 0) for entry in ref_history]
                    
                    ref_changes = []
                    for i in range(1, len(ref_prices)):
                        if ref_prices[i-1] > 0:
                            pct_change = ((ref_prices[i] / ref_prices[i-1]) - 1) * 100
                            ref_changes.append(pct_change)
                            
                    if len(ref_changes) > 1:
                        ref_volatility = statistics.stdev(ref_changes)
                        market_volatilities.append(ref_volatility)
            
            # Calculate relative volatility
            if market_volatilities:
                market_avg_volatility = statistics.mean(market_volatilities)
                if market_avg_volatility > 0:
                    return token_volatility / market_avg_volatility
            
            return None
            
        except Exception as e:
            logger.log_error(f"Calculate Relative Volatility - {token} ({timeframe})", str(e))
            return None

    def _calculate_correlations(self, token: str, market_data: Dict[str, Any], 
                              timeframe: str = "1h") -> Dict[str, float]:
        """
        Calculate token correlations with the market
        Adjust correlation window based on timeframe
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Dictionary of correlation metrics
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return {'timeframe': timeframe}
                
            # Filter out the token itself from reference tokens to avoid self-comparison
            reference_tokens = [t for t in self.reference_tokens if t != token]
            
            # Select appropriate reference tokens based on timeframe and relevance
            if timeframe == "1h":
                # For hourly, just use major tokens
                reference_tokens = ["BTC", "ETH", "SOL"]
            elif timeframe == "24h":
                # For daily, use more tokens
                reference_tokens = ["BTC", "ETH", "SOL", "BNB", "XRP"]
            # For weekly, use all tokens (default)
            
            correlations = {}
            
            # Calculate correlation with each reference token
            for ref_token in reference_tokens:
                if ref_token not in market_data:
                    continue
                    
                ref_data = market_data[ref_token]
                
                # Time window for correlation calculation based on timeframe
                if timeframe == "1h":
                    # Use 24h change for hourly predictions (short-term)
                    price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                elif timeframe == "24h":
                    # For daily, check if we have 7d change data available
                    if 'price_change_percentage_7d' in token_data and 'price_change_percentage_7d' in ref_data:
                        price_correlation_metric = abs(token_data['price_change_percentage_7d'] - ref_data['price_change_percentage_7d'])
                    else:
                        # Fall back to 24h change if 7d not available
                        price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                else:  # 7d
                    # For weekly, use historical correlation if available
                    # Get historical data with longer window
                    token_history = self._get_historical_price_data(token, hours=30*24, timeframe=timeframe)
                    ref_history = self._get_historical_price_data(ref_token, hours=30*24, timeframe=timeframe)
                    
                    if len(token_history) >= 14 and len(ref_history) >= 14:
                        # Calculate 14-day rolling correlation
                        token_prices = [entry.get('price', 0) for entry in token_history[:14]]
                        ref_prices = [entry.get('price', 0) for entry in ref_history[:14]]
                        
                        if len(token_prices) == len(ref_prices) and len(token_prices) > 2:
                            try:
                                # Calculate correlation coefficient
                                historical_corr = np.corrcoef(token_prices, ref_prices)[0, 1]
                                price_correlation_metric = abs(1 - historical_corr)
                            except:
                                # Fall back to 24h change if correlation fails
                                price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                        else:
                            price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                    else:
                        price_correlation_metric = abs(token_data['price_change_percentage_24h'] - ref_data['price_change_percentage_24h'])
                
                # Calculate price correlation (convert difference to correlation coefficient)
                # Smaller difference = higher correlation
                max_diff = 15 if timeframe == "1h" else 25 if timeframe == "24h" else 40
                price_correlation = 1 - min(1, price_correlation_metric / max_diff)
                
                # Volume correlation (simplified)
                volume_correlation = abs(
                    (token_data['volume'] - ref_data['volume']) / 
                    max(token_data['volume'], ref_data['volume'])
                )
                volume_correlation = 1 - volume_correlation  # Convert to correlation coefficient
                
                correlations[f'price_correlation_{ref_token}'] = price_correlation
                correlations[f'volume_correlation_{ref_token}'] = volume_correlation
            
            # Calculate average correlations
            price_correlations = [v for k, v in correlations.items() if 'price_correlation_' in k]
            volume_correlations = [v for k, v in correlations.items() if 'volume_correlation_' in k]
            
            correlations['avg_price_correlation'] = statistics.mean(price_correlations) if price_correlations else 0
            correlations['avg_volume_correlation'] = statistics.mean(volume_correlations) if volume_correlations else 0
            
            # Add BTC dominance correlation for longer timeframes
            if timeframe in ["24h", "7d"] and 'BTC' in market_data:
                btc_mc = market_data['BTC'].get('market_cap', 0)
                total_mc = sum([data.get('market_cap', 0) for data in market_data.values()])
                
                if total_mc > 0:
                    btc_dominance = (btc_mc / total_mc) * 100
                    btc_change = market_data['BTC'].get('price_change_percentage_24h', 0)
                    token_change = token_data.get('price_change_percentage_24h', 0)
                    
                    # Simple heuristic: if token moves opposite to BTC and dominance is high,
                    # it might be experiencing a rotation from/to BTC
                    btc_rotation_indicator = (btc_change * token_change < 0) and (btc_dominance > 50)
                    
                    correlations['btc_dominance'] = btc_dominance
                    correlations['btc_rotation_indicator'] = btc_rotation_indicator
            
            # Store correlation data for any token using the generic method
            self.config.db.store_token_correlations(token, correlations)
            
            logger.logger.debug(
                f"{token} correlations calculated ({timeframe}) - Avg Price: {correlations['avg_price_correlation']:.2f}, "
                f"Avg Volume: {correlations['avg_volume_correlation']:.2f}"
            )
            
            return correlations
            
        except Exception as e:
            logger.log_error(f"Correlation Calculation - {token} ({timeframe})", str(e))
            return {
                'avg_price_correlation': 0.0,
                'avg_volume_correlation': 0.0,
                'timeframe': timeframe
            }

    def _generate_correlation_report(self, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Generate a report of correlations between top tokens
        Customized based on timeframe with duplicate detection
    
        Args:
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
        
        Returns:
            Formatted correlation report as string
        """
        try:
            # Fix 1: Add check for market_data being None
            if not market_data:
                return f"Failed to generate {timeframe} correlation report: No market data available"
            
            # Select tokens to include based on timeframe
            if timeframe == "1h":
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP']  # Focus on major tokens for hourly
            elif timeframe == "24h":
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'XRP']  # More tokens for daily
            else:  # 7d
                tokens = ['BTC', 'ETH', 'SOL', 'BNB', 'AVAX', 'DOT', 'XRP']  # Most tokens for weekly
    
            # Create correlation matrix and include a report ID for tracking
            correlation_matrix = {}
            report_id = f"corr_matrix_{timeframe}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
            for token1 in tokens:
                correlation_matrix[token1] = {}
                # Fix 2: Properly nest the token2 loop inside the token1 loop
                for token2 in tokens:
                    if token1 == token2:
                        correlation_matrix[token1][token2] = 1.0
                        continue
                
                    if token1 not in market_data or token2 not in market_data:
                        correlation_matrix[token1][token2] = 0.0
                        continue
                
                    # Adjust correlation calculation based on timeframe
                    if timeframe == "1h":
                        # For hourly, use 24h price change
                        price_change1 = market_data[token1]['price_change_percentage_24h']
                        price_change2 = market_data[token2]['price_change_percentage_24h']
                    
                        # Calculate simple correlation
                        price_direction1 = 1 if price_change1 > 0 else -1
                        price_direction2 = 1 if price_change2 > 0 else -1
                    
                        # Basic correlation (-1.0 to 1.0)
                        correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                    else:
                        # For longer timeframes, try to use more sophisticated correlation
                        # Get historical data
                        token1_history = self._get_historical_price_data(token1, timeframe=timeframe)
                        token2_history = self._get_historical_price_data(token2, timeframe=timeframe)
                    
                        if len(token1_history) >= 5 and len(token2_history) >= 5:
                            # Extract prices for correlation calculation
                            prices1 = [entry.get('price', 0) for entry in token1_history]
                            prices2 = [entry.get('price', 0) for entry in token2_history]
                        
                            try:
                                # Calculate Pearson correlation
                                correlation = np.corrcoef(prices1, prices2)[0, 1]
                                if np.isnan(correlation):
                                    # Fall back to simple method if NaN
                                    price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                                    price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                                    correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                            except:
                                # Fall back to simple method if calculation fails
                                price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                                price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                                correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                        else:
                            # Not enough historical data, use simple method
                            price_direction1 = 1 if market_data[token1]['price_change_percentage_24h'] > 0 else -1
                            price_direction2 = 1 if market_data[token2]['price_change_percentage_24h'] > 0 else -1
                            correlation = 1.0 if price_direction1 == price_direction2 else -1.0
                        
                    correlation_matrix[token1][token2] = correlation
        
            # Check if this matrix is similar to recent posts to prevent duplication
            if self._is_matrix_duplicate(correlation_matrix, timeframe):
                logger.logger.warning(f"Detected duplicate {timeframe} correlation matrix, skipping")
                return None  # Return None to signal duplicate
    
            # Format the report text
            if timeframe == "1h":
                report = "1H CORRELATION MATRIX:\n\n"
            elif timeframe == "24h":
                report = "24H CORRELATION MATRIX:\n\n"
            else:
                report = "7D CORRELATION MATRIX:\n\n"
    
            # Create ASCII art heatmap
            for token1 in tokens:
                report += f"{token1} "
                for token2 in tokens:
                    corr = correlation_matrix[token1][token2]
                    if token1 == token2:
                        report += "✦ "  # Self correlation
                    elif corr > 0.5:
                        report += "➕ "  # Strong positive
                    elif corr > 0:
                        report += "📈 "  # Positive
                    elif corr < -0.5:
                        report += "➖ "  # Strong negative
                    else:
                        report += "📉 "  # Negative
                report += "\n"
        
            report += "\nKey: ✦=Same ➕=Strong+ 📈=Weak+ ➖=Strong- 📉=Weak-"
        
            # Add timeframe-specific insights
            if timeframe == "24h" or timeframe == "7d":
                # For longer timeframes, add sector analysis
                defi_tokens = [t for t in tokens if t in ["UNI", "AAVE"]]
                layer1_tokens = [t for t in tokens if t in ["ETH", "SOL", "AVAX", "NEAR"]]
            
                # Check if we have enough tokens from each sector
                if len(defi_tokens) >= 2 and len(layer1_tokens) >= 2:
                    # Calculate average intra-sector correlation
                    defi_corrs = []
                    for i in range(len(defi_tokens)):
                        for j in range(i+1, len(defi_tokens)):
                            t1, t2 = defi_tokens[i], defi_tokens[j]
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                defi_corrs.append(correlation_matrix[t1][t2])
                
                    layer1_corrs = []
                    for i in range(len(layer1_tokens)):
                        for j in range(i+1, len(layer1_tokens)):
                            t1, t2 = layer1_tokens[i], layer1_tokens[j]
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                layer1_corrs.append(correlation_matrix[t1][t2])
                
                    # Calculate cross-sector correlation
                    cross_corrs = []
                    for t1 in defi_tokens:
                        for t2 in layer1_tokens:
                            if t1 in correlation_matrix and t2 in correlation_matrix[t1]:
                                cross_corrs.append(correlation_matrix[t1][t2])
                
                    # Add to report if we have correlation data
                    if defi_corrs and layer1_corrs and cross_corrs:
                        avg_defi_corr = sum(defi_corrs) / len(defi_corrs)
                        avg_layer1_corr = sum(layer1_corrs) / len(layer1_corrs)
                        avg_cross_corr = sum(cross_corrs) / len(cross_corrs)
                    
                        report += f"\n\nSector Analysis:"
                        report += f"\nDeFi internal correlation: {avg_defi_corr:.2f}"
                        report += f"\nLayer1 internal correlation: {avg_layer1_corr:.2f}"
                        report += f"\nCross-sector correlation: {avg_cross_corr:.2f}"
                    
                        # Interpret sector rotation
                        if avg_cross_corr < min(avg_defi_corr, avg_layer1_corr) - 0.3:
                            report += "\nPossible sector rotation detected!"
        
            # Store report details in the database for tracking
            self._save_correlation_report(report_id, correlation_matrix, timeframe, report)
        
            return report
    
        except Exception as e:
            logger.log_error(f"Correlation Report - {timeframe}", str(e))
            return f"Failed to generate {timeframe} correlation report."

    def _is_matrix_duplicate(self, matrix: Dict[str, Dict[str, float]], timeframe: str) -> bool:
        """
        Stricter check for duplicate correlation matrices with direct content examination
    
        Args:
            matrix: Correlation matrix to check
            timeframe: Timeframe for analysis
        
        Returns:
            True if duplicate detected, False otherwise
        """
        try:
            # First, check posted_content table directly with a strong timeframe filter
            conn, cursor = self.config.db._get_connection()
        
            # Define timeframe prefix explicitly
            timeframe_prefix = ""
            if timeframe == "1h":
                timeframe_prefix = "1H CORRELATION MATRIX"
            elif timeframe == "24h":
                timeframe_prefix = "24H CORRELATION MATRIX"
            else:  # 7d
                timeframe_prefix = "7D CORRELATION MATRIX"
            
            # Check for any recent posts with this exact prefix - stricter window for hourly matrices
            window_hours = 3 if timeframe == "1h" else 12 if timeframe == "24h" else 48
        
            # Direct check for ANY recent matrix of this timeframe
            cursor.execute("""
                SELECT content, timestamp FROM posted_content
                WHERE content LIKE ?
                AND timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
                LIMIT 1
            """, (f"{timeframe_prefix}%", window_hours))
        
            recent_post = cursor.fetchone()
        
            if recent_post:
                post_time = datetime.fromisoformat(recent_post['timestamp']) if isinstance(recent_post['timestamp'], str) else recent_post['timestamp']
                now = datetime.now()
                hours_since_post = (now - post_time).total_seconds() / 3600
            
                logger.logger.warning(f"Found recent {timeframe} matrix posted {hours_since_post:.1f} hours ago")
                return True
            
            logger.logger.info(f"No recent {timeframe} matrix found in post history, safe to post")
            return False
            
        except Exception as e:
            logger.log_error(f"Matrix Duplication Check - {timeframe}", str(e))
            # On error, be cautious and assume it might be a duplicate
            logger.logger.warning(f"Error in duplicate check, assuming duplicate to be safe: {str(e)}")
            return True

    def _generate_correlation_report(self, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Generate a report of correlations between top tokens with improved duplicate prevention
    
        Args:
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
        
        Returns:
            Formatted correlation report as string or None if would be duplicate
        """
        try:
            # Check for duplicates FIRST before generating content
            if self._is_matrix_duplicate(None, timeframe):
                logger.logger.warning(f"Pre-emptive duplicate check: recent {timeframe} matrix already posted")
                return None
        
            # Rest of your correlation report code...
            # [Code omitted for brevity]
        
        except Exception as e:
            logger.log_error(f"Correlation Report - {timeframe}", str(e))
            return None

    def _save_correlation_report(self, report_id: str, matrix: Dict[str, Dict[str, float]], 
                                timeframe: str, report_text: str) -> None:
        """
        Save correlation report data for tracking and duplicate prevention
    
        Args:
            report_id: Unique ID for the report
            matrix: Correlation matrix data
            timeframe: Timeframe used for analysis
            report_text: Formatted report text
        """
        try:
            # Create a hash of the matrix for comparison
            matrix_str = json.dumps(matrix, sort_keys=True)
            import hashlib
            matrix_hash = hashlib.md5(matrix_str.encode()).hexdigest()
        
            # Prepare data for storage
            report_data = {
                'id': report_id,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'matrix': matrix,
                'hash': matrix_hash,
                'text': report_text
            }
        
            # Store in database (using generic_json_data table)
            self.config.db._store_json_data(
                data_type='correlation_report',
                data=report_data
            )
        
            logger.logger.debug(f"Saved {timeframe} correlation report with ID: {report_id}")
        
        except Exception as e:
            logger.log_error(f"Save Correlation Report - {report_id}", str(e))
        
    def _calculate_momentum_score(self, token: str, market_data: Dict[str, Any], timeframe: str = "1h") -> float:
        """
        Calculate a momentum score (0-100) for a token based on various metrics
        Adjusted for different timeframes
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            timeframe: Timeframe for analysis
            
        Returns:
            Momentum score (0-100)
        """
        try:
            token_data = market_data.get(token, {})
            if not token_data:
                return 50.0  # Neutral score
            
            # Get basic metrics
            price_change = token_data.get('price_change_percentage_24h', 0)
            volume = token_data.get('volume', 0)
        
            # Get historical volume for volume change - adjust window based on timeframe
            if timeframe == "1h":
                window_minutes = 60  # Last hour for hourly predictions
            elif timeframe == "24h":
                window_minutes = 24 * 60  # Last day for daily predictions
            else:  # 7d
                window_minutes = 7 * 24 * 60  # Last week for weekly predictions
                
            historical_volume = self._get_historical_volume_data(token, minutes=window_minutes, timeframe=timeframe)
            volume_change, _ = self._analyze_volume_trend(volume, historical_volume, timeframe=timeframe)
        
            # Get smart money indicators
            smart_money = self._analyze_smart_money_indicators(token, token_data, timeframe=timeframe)
        
            # Get market comparison
            vs_market = self._analyze_token_vs_market(token, market_data, timeframe=timeframe)
        
            # Calculate score components (0-20 points each)
            # Adjust price score scaling based on timeframe
            if timeframe == "1h":
                price_range = 5.0  # ±5% for hourly
            elif timeframe == "24h":
                price_range = 10.0  # ±10% for daily
            else:  # 7d
                price_range = 20.0  # ±20% for weekly
                
            price_score = min(20, max(0, (price_change + price_range) * (20 / (2 * price_range))))
        
            # Adjust volume score scaling based on timeframe
            if timeframe == "1h":
                volume_range = 10.0  # ±10% for hourly
            elif timeframe == "24h":
                volume_range = 20.0  # ±20% for daily
            else:  # 7d
                volume_range = 40.0  # ±40% for weekly
                
            volume_score = min(20, max(0, (volume_change + volume_range) * (20 / (2 * volume_range))))
        
            # Smart money score - additional indicators for longer timeframes
            smart_money_score = 0
            if smart_money.get('abnormal_volume', False):
                smart_money_score += 5
            if smart_money.get('stealth_accumulation', False):
                smart_money_score += 5
            if smart_money.get('volume_cluster_detected', False):
                smart_money_score += 5
            if smart_money.get('volume_z_score', 0) > 1.0:
                smart_money_score += 5
                
            # Add pattern metrics for longer timeframes
            if timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                pattern_metrics = smart_money['pattern_metrics']
                if pattern_metrics.get('volume_breakout', False):
                    smart_money_score += 5
                if pattern_metrics.get('consistent_high_volume', False):
                    smart_money_score += 5
                    
            smart_money_score = min(20, smart_money_score)
        
            # Market comparison score
            market_score = 0
            if vs_market.get('outperforming_market', False):
                market_score += 10
            market_score += min(10, max(0, (vs_market.get('vs_market_avg_change', 0) + 5)))
            market_score = min(20, market_score)
        
            # Trend consistency score - higher standards for longer timeframes
            if timeframe == "1h":
                trend_score = 20 if all([price_score > 10, volume_score > 10, smart_money_score > 5, market_score > 10]) else 0
            elif timeframe == "24h":
                trend_score = 20 if all([price_score > 12, volume_score > 12, smart_money_score > 8, market_score > 12]) else 0
            else:  # 7d
                trend_score = 20 if all([price_score > 15, volume_score > 15, smart_money_score > 10, market_score > 15]) else 0
        
            # Calculate total score (0-100)
            # Adjust component weights based on timeframe
            if timeframe == "1h":
                # For hourly, recent price action and smart money more important
                total_score = (
                    price_score * 0.25 +
                    volume_score * 0.2 +
                    smart_money_score * 0.25 +
                    market_score * 0.15 +
                    trend_score * 0.15
                ) * 1.0
            elif timeframe == "24h":
                # For daily, balance factors with more weight to market comparison
                total_score = (
                    price_score * 0.2 +
                    volume_score * 0.2 +
                    smart_money_score * 0.2 +
                    market_score * 0.25 +
                    trend_score * 0.15
                ) * 1.0
            else:  # 7d
                # For weekly, market factors and trend consistency more important
                total_score = (
                    price_score * 0.15 +
                    volume_score * 0.15 +
                    smart_money_score * 0.2 +
                    market_score * 0.3 +
                    trend_score * 0.2
                ) * 1.0
        
            return total_score
        
        except Exception as e:
            logger.log_error(f"Momentum Score - {token} ({timeframe})", str(e))
            return 50.0  # Neutral score on error

    def _format_prediction_tweet(self, token: str, prediction: Dict[str, Any], market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Format a prediction into a tweet with FOMO-inducing content
        Supports multiple timeframes (1h, 24h, 7d)
        
        Args:
            token: Token symbol
            prediction: Prediction data dictionary
            market_data: Market data dictionary
            timeframe: Timeframe for the prediction
            
        Returns:
            Formatted prediction tweet
        """
        try:
            # Get prediction details
            pred_data = prediction.get("prediction", {})
            sentiment = prediction.get("sentiment", "NEUTRAL")
            rationale = prediction.get("rationale", "")
        
            # Format prediction values
            price = pred_data.get("price", 0)
            confidence = pred_data.get("confidence", 70)
            lower_bound = pred_data.get("lower_bound", 0)
            upper_bound = pred_data.get("upper_bound", 0)
            percent_change = pred_data.get("percent_change", 0)
        
            # Get current price
            token_data = market_data.get(token, {})
            current_price = token_data.get("current_price", 0)
        
            # Format timeframe for display
            if timeframe == "1h":
                display_timeframe = "1HR"
            elif timeframe == "24h":
                display_timeframe = "24HR"
            else:  # 7d
                display_timeframe = "7DAY"
            
            # Format the tweet
            tweet = f"#{token} {display_timeframe} PREDICTION:\n\n"
        
            # Sentiment-based formatting
            if sentiment == "BULLISH":
                tweet += "BULLISH ALERT\n"
            elif sentiment == "BEARISH":
                tweet += "BEARISH WARNING\n"
            else:
                tweet += "MARKET ANALYSIS\n"
            
            # Add prediction with confidence
            tweet += f"Target: ${price:.4f} ({percent_change:+.2f}%)\n"
            tweet += f"Range: ${lower_bound:.4f} - ${upper_bound:.4f}\n"
            tweet += f"Confidence: {confidence}%\n\n"
        
            # Add rationale - adjust length based on timeframe
            if timeframe == "7d":
                # For weekly predictions, add more detail to rationale
                tweet += f"{rationale}\n\n"
            else:
                # For shorter timeframes, keep it brief
                if len(rationale) > 100:
                    # Truncate at a sensible point
                    last_period = rationale[:100].rfind('. ')
                    if last_period > 50:
                        rationale = rationale[:last_period+1]
                    else:
                        rationale = rationale[:100] + "..."
                tweet += f"{rationale}\n\n"
        
            # Add accuracy tracking if available
            performance = self.config.db.get_prediction_performance(token=token, timeframe=timeframe)
            if performance and performance[0]["total_predictions"] > 0:
                accuracy = performance[0]["accuracy_rate"]
                tweet += f"Accuracy: {accuracy:.1f}% on {performance[0]['total_predictions']} predictions"
            
            # Ensure tweet is within the hard stop length
            max_length = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
            if len(tweet) > max_length:
                # Smart truncate to preserve essential info
                last_paragraph = tweet.rfind("\n\n")
                if last_paragraph > max_length * 0.7:
                    # Truncate at the last paragraph break
                    tweet = tweet[:last_paragraph].strip()
                else:
                    # Simply truncate with ellipsis
                    tweet = tweet[:max_length-3] + "..."
            
            return tweet
        
        except Exception as e:
            logger.log_error(f"Format Prediction Tweet - {token} ({timeframe})", str(e))
            return f"#{token} {timeframe.upper()} PREDICTION: ${price:.4f} ({percent_change:+.2f}%) - {sentiment}"

    @ensure_naive_datetimes
    def _track_prediction(self, token: str, prediction: Dict[str, Any], relevant_tokens: List[str], timeframe: str = "1h") -> None:
        """
        Track predictions for future callbacks and analysis
        Supports multiple timeframes (1h, 24h, 7d)
        
        Args:
            token: Token symbol
            prediction: Prediction data dictionary
            relevant_tokens: List of relevant token symbols
            timeframe: Timeframe for the prediction
        """
        MAX_PREDICTIONS = 20  
    
        # Get current prices of relevant tokens from prediction
        current_prices = {chain: prediction.get(f'{chain.upper()}_price', 0) for chain in relevant_tokens if f'{chain.upper()}_price' in prediction}
    
        # Add the prediction to the tracking list with timeframe info
        self.past_predictions.append({
            'timestamp': strip_timezone(datetime.now()),
            'token': token,
            'prediction': prediction['analysis'],
            'prices': current_prices,
            'sentiment': prediction['sentiment'],
            'timeframe': timeframe,
            'outcome': None
        })
    
        # Keep only predictions from the last 24 hours, up to MAX_PREDICTIONS
        self.past_predictions = [p for p in self.past_predictions 
                                 if safe_datetime_diff(datetime.now(), p['timestamp']) < 86400]
    
        # Trim to max predictions if needed
        if len(self.past_predictions) > MAX_PREDICTIONS:
            self.past_predictions = self.past_predictions[-MAX_PREDICTIONS:]
        
        logger.logger.debug(f"Tracked {timeframe} prediction for {token}")

    @ensure_naive_datetimes
    def _validate_past_prediction(self, prediction: Dict[str, Any], current_prices: Dict[str, float]) -> str:
        """
        Check if a past prediction was accurate
        
        Args:
            prediction: Prediction data dictionary
            current_prices: Dictionary of current prices
            
        Returns:
            Evaluation outcome: 'right', 'wrong', or 'undetermined'
        """
        sentiment_map = {
            'bullish': 1,
            'bearish': -1,
            'neutral': 0,
            'volatile': 0,
            'recovering': 0.5
        }
    
        # Apply different thresholds based on the timeframe
        timeframe = prediction.get('timeframe', '1h')
        if timeframe == '1h':
            threshold = 2.0  # 2% for 1-hour predictions
        elif timeframe == '24h':
            threshold = 4.0  # 4% for 24-hour predictions
        else:  # 7d
            threshold = 7.0  # 7% for 7-day predictions
    
        wrong_tokens = []
        for token, old_price in prediction['prices'].items():
            if token in current_prices and old_price > 0:
                price_change = ((current_prices[token] - old_price) / old_price) * 100
            
                # Get sentiment for this token
                token_sentiment_key = token.upper() if token.upper() in prediction['sentiment'] else token
                token_sentiment_value = prediction['sentiment'].get(token_sentiment_key)
            
                # Handle nested dictionary structure
                if isinstance(token_sentiment_value, dict) and 'mood' in token_sentiment_value:
                    token_sentiment = sentiment_map.get(token_sentiment_value['mood'], 0)
                else:
                    token_sentiment = sentiment_map.get(token_sentiment_value, 0)
            
                # A prediction is wrong if:
                # 1. Bullish but price dropped more than threshold%
                # 2. Bearish but price rose more than threshold%
                if (token_sentiment > 0 and price_change < -threshold) or (token_sentiment < 0 and price_change > threshold):
                    wrong_tokens.append(token)
    
        return 'wrong' if wrong_tokens else 'right'
    
    @ensure_naive_datetimes
    def _get_spicy_callback(self, token: str, current_prices: Dict[str, float], timeframe: str = "1h") -> Optional[str]:
        """
        Generate witty callbacks to past terrible predictions
        Supports multiple timeframes
        
        Args:
            token: Token symbol
            current_prices: Dictionary of current prices
            timeframe: Timeframe for the callback
            
        Returns:
            Callback text or None if no suitable callback found
        """
        # Look for the most recent prediction for this token and timeframe
        recent_predictions = [p for p in self.past_predictions 
                             if safe_datetime_diff(datetime.now(), p['timestamp']) < 24*3600
                             and p['token'] == token
                             and p.get('timeframe', '1h') == timeframe]
    
        if not recent_predictions:
            return None
        
        # Evaluate any unvalidated predictions
        for pred in recent_predictions:
            if pred['outcome'] is None:
                pred['outcome'] = self._validate_past_prediction(pred, current_prices)
            
        # Find any wrong predictions
        wrong_predictions = [p for p in recent_predictions if p['outcome'] == 'wrong']
        if wrong_predictions:
            worst_pred = wrong_predictions[-1]
            time_ago = int(safe_datetime_diff(datetime.now(), worst_pred['timestamp']) / 3600)
        
            # If time_ago is 0, set it to 1 to avoid awkward phrasing
            if time_ago == 0:
                time_ago = 1
        
            # Format timeframe for display
            time_unit = "hr" if timeframe in ["1h", "24h"] else "day"
            time_display = f"{time_ago}{time_unit}"
        
            # Token-specific callbacks
            callbacks = [
                f"(Unlike my galaxy-brain take {time_display} ago about {worst_pred['prediction'].split('.')[0]}... this time I'm sure!)",
                f"(Looks like my {time_display} old prediction about {token} aged like milk. But trust me bro!)",
                f"(That awkward moment when your {time_display} old {token} analysis was completely wrong... but this one's different!)",
                f"(My {token} trading bot would be down bad after that {time_display} old take. Good thing I'm just an analyst!)",
                f"(Excuse the {time_display} old miss on {token}. Even the best crypto analysts are wrong sometimes... just not usually THIS wrong!)"
            ]
        
            # Select a callback deterministically but with variation
            callback_seed = f"{datetime.now().date()}_{token}_{timeframe}"
            callback_index = hash(callback_seed) % len(callbacks)
        
            return callbacks[callback_index]
        
        return None

    def _format_tweet_analysis(self, token: str, analysis: str, market_data: Dict[str, Any], timeframe: str = "1h") -> str:
        """
        Format analysis for Twitter with no hashtags to maximize content
        Supports multiple timeframes (1h, 24h, 7d)
        
        Args:
            token: Token symbol
            analysis: Analysis text
            market_data: Market data dictionary
            timeframe: Timeframe for the analysis
            
        Returns:
            Formatted analysis tweet
        """
        # Check if we need to add timeframe prefix
        if timeframe != "1h" and not any(prefix in analysis.upper() for prefix in [f"{timeframe.upper()} ", f"{timeframe}-"]):
            # Add timeframe prefix if not already present
            if timeframe == "24h":
                prefix = "24H ANALYSIS: "
            else:  # 7d
                prefix = "7DAY OUTLOOK: "
            
            # Only add prefix if not already present in some form
            analysis = prefix + analysis
    
        # Simply use the analysis text with no hashtags
        tweet = analysis
    
        # Sanitize text to remove non-BMP characters that ChromeDriver can't handle
        tweet = ''.join(char for char in tweet if ord(char) < 0x10000)
    
        # Check for minimum length
        min_length = self.config.TWEET_CONSTRAINTS['MIN_LENGTH']
        if len(tweet) < min_length:
            logger.logger.warning(f"{timeframe} analysis too short ({len(tweet)} chars). Minimum: {min_length}")
            # Not much we can do here since Claude should have generated the right length
            # We'll log but not try to fix, as Claude should be instructed correctly
    
        # Check for maximum length
        max_length = self.config.TWEET_CONSTRAINTS['MAX_LENGTH']
        if len(tweet) > max_length:
            logger.logger.warning(f"{timeframe} analysis too long ({len(tweet)} chars). Maximum: {max_length}")
    
        # Check for hard stop length
        hard_stop = self.config.TWEET_CONSTRAINTS['HARD_STOP_LENGTH']
        if len(tweet) > hard_stop:
            # Smart truncation - find the last sentence boundary before the limit
            # First try to end on a period, question mark, or exclamation
            last_period = tweet[:hard_stop-3].rfind('. ')
            last_question = tweet[:hard_stop-3].rfind('? ')
            last_exclamation = tweet[:hard_stop-3].rfind('! ')
        
            # Find the last sentence-ending punctuation
            last_sentence_end = max(last_period, last_question, last_exclamation)
        
            if last_sentence_end > hard_stop * 0.7:  # If we can find a good sentence break in the latter 30% of the text
                # Truncate at the end of a sentence and add no ellipsis
                tweet = tweet[:last_sentence_end+1]  # Include the punctuation
            else:
                # Fallback: find the last word boundary
                last_space = tweet[:hard_stop-3].rfind(' ')
                if last_space > 0:
                    tweet = tweet[:last_space] + "..."
                else:
                    # Last resort: hard truncation
                    tweet = tweet[:hard_stop-3] + "..."
            
            logger.logger.warning(f"Trimmed {timeframe} analysis to {len(tweet)} chars using smart truncation")
    
        return tweet

    @ensure_naive_datetimes
    def _analyze_market_sentiment(self, token, market_data, trigger_type=None, timeframe="1h"):
        """
        Analyze market sentiment for a token with enhanced error handling
        and compatibility with _get_vs_market_analysis

        Args:
            token: Token symbol
            market_data: Market data dictionary/list
            trigger_type: The type of trigger that initiated this analysis (optional)
            timeframe: Timeframe for analysis ("1h", "24h", or "7d")
    
        Returns:
            Tuple of (sentiment, market_context)
        """
        import traceback
        from datetime import datetime, timedelta

        # Initialize defaults
        sentiment = "neutral"
        market_context = ""

        try:
            # Log trigger_type for context if available
            if trigger_type:
                logger.logger.debug(f"Market sentiment analysis for {token} triggered by: {trigger_type}")
            
            # Get vs market analysis with defensive error handling
            try:
                # Initialize vs_market with default values
                vs_market = {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral"
                }
        
                # Ensure market_data is standardized if possible
                standardized_data = None
                if hasattr(self, '_standardize_market_data'):
                    try:
                        standardized_data = self._standardize_market_data(market_data)
                        logger.logger.debug(f"Standardized market data type: {type(standardized_data)}")
                    except Exception as std_error:
                        logger.logger.error(f"Error in standardizing market data for {token}: {str(std_error)}")
                        standardized_data = None
        
                # Only proceed with _get_vs_market_analysis if we have dictionary data
                if standardized_data and isinstance(standardized_data, dict):
                    # Use standardized data
                    if hasattr(self, '_get_vs_market_analysis'):
                        vs_market = self._get_vs_market_analysis(token, standardized_data, timeframe)
                    elif hasattr(self, '_analyze_token_vs_market'):
                        # Fallback to newer method if available
                        vs_market_result = self._analyze_token_vs_market(token, standardized_data, timeframe)
                        # Extract relevant keys for compatibility
                        vs_market = {
                            "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                            "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                            "market_correlation": vs_market_result.get("market_correlation", 0.0),
                            "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                        }
                elif isinstance(market_data, dict):
                    # If standardization failed but original data is a dict, use it
                    if hasattr(self, '_get_vs_market_analysis'):
                        vs_market = self._get_vs_market_analysis(token, market_data, timeframe)
                    elif hasattr(self, '_analyze_token_vs_market'):
                        # Fallback to newer method if available
                        vs_market_result = self._analyze_token_vs_market(token, market_data, timeframe)
                        # Extract relevant keys for compatibility
                        vs_market = {
                            "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                            "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                            "market_correlation": vs_market_result.get("market_correlation", 0.0),
                            "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                        }
                else:
                    # Handle the case where market_data is a list and standardization failed
                    logger.logger.warning(f"Cannot analyze market for {token}: market_data is {type(market_data)} and standardization failed")
            
                    # Create an emergency backup dictionary if needed
                    if isinstance(market_data, list):
                        temp_dict = {}
                        for item in market_data:
                            if isinstance(item, dict) and 'symbol' in item:
                                symbol = item['symbol'].upper()
                                temp_dict[symbol] = item
                
                        # If we have a reasonable dictionary now, try to use it
                        if temp_dict:
                            if hasattr(self, '_get_vs_market_analysis'):
                                logger.logger.debug(f"Using emergency backup dictionary for {token} with {len(temp_dict)} items")
                                try:
                                    vs_market = self._get_vs_market_analysis(token, temp_dict, timeframe)
                                except Exception as emergency_error:
                                    logger.logger.error(f"Error in emergency market analysis for {token}: {str(emergency_error)}")
                            elif hasattr(self, '_analyze_token_vs_market'):
                                logger.logger.debug(f"Using emergency backup dictionary with newer method for {token}")
                                try:
                                    vs_market_result = self._analyze_token_vs_market(token, temp_dict, timeframe)
                                    # Extract relevant keys for compatibility
                                    vs_market = {
                                        "vs_market_avg_change": vs_market_result.get("vs_market_avg_change", 0.0),
                                        "vs_market_percentile": vs_market_result.get("vs_market_percentile", 50.0),
                                        "market_correlation": vs_market_result.get("market_correlation", 0.0),
                                        "market_sentiment": vs_market_result.get("market_sentiment", "neutral")
                                    }
                                except Exception as emergency_error:
                                    logger.logger.error(f"Error in emergency market analysis with newer method for {token}: {str(emergency_error)}")
                                
                # Validate result is a dictionary with required keys
                if not isinstance(vs_market, dict):
                    logger.logger.warning(f"VS market analysis returned non-dict for {token}: {type(vs_market)}")
                    vs_market = {
                        "vs_market_avg_change": 0.0,
                        "vs_market_percentile": 50.0,
                        "market_correlation": 0.0,
                        "market_sentiment": "neutral"
                    }
                elif 'vs_market_avg_change' not in vs_market:
                    logger.logger.warning(f"Missing 'vs_market_avg_change' for {token}")
                    vs_market['vs_market_avg_change'] = 0.0
        
                if 'vs_market_sentiment' not in vs_market and 'market_sentiment' in vs_market:
                    # Copy market_sentiment to vs_market_sentiment for compatibility
                    vs_market['vs_market_sentiment'] = vs_market['market_sentiment']
                elif 'vs_market_sentiment' not in vs_market:
                    vs_market['vs_market_sentiment'] = "neutral"
            
            except Exception as vs_error:
                logger.logger.error(f"Error in market analysis for {token}: {str(vs_error)}")
                logger.logger.debug(traceback.format_exc())
                vs_market = {
                    "vs_market_avg_change": 0.0,
                    "vs_market_percentile": 50.0,
                    "market_correlation": 0.0,
                    "market_sentiment": "neutral"
                }
    
            # Get overall market trend
            try:
                market_conditions = {}
                if hasattr(self, 'market_conditions'):
                    market_conditions = self.market_conditions
    
                market_trend = market_conditions.get('market_trend', 'neutral')
            except Exception as trend_error:
                logger.logger.error(f"Error getting market trend for {token}: {str(trend_error)}")
                market_trend = 'neutral'

            # Handle any datetime-related operations that might be needed
            try:
                # Get current time and appropriate time window based on timeframe
                current_time = datetime.now()
    
                if timeframe == "1h":
                    time_window = timedelta(hours=1)
                    time_desc = "hourly"
                elif timeframe == "24h":
                    time_window = timedelta(hours=24)
                    time_desc = "daily"
                else:  # 7d
                    time_window = timedelta(days=7)
                    time_desc = "weekly"
        
                # Calculate prediction target time (when this prediction will be evaluated)
                target_time = current_time + time_window
    
                # Format times appropriately for displaying in context
                target_time_str = target_time.strftime("%H:%M") if timeframe == "1h" else target_time.strftime("%b %d")
    
                # Get historical volatility if available
                volatility = 0.0
                token_data = None
        
                # Safely access token data based on the data structure
                if isinstance(market_data, dict) and token in market_data:
                    token_data = market_data[token]
                elif isinstance(standardized_data, dict) and token in standardized_data:
                    token_data = standardized_data[token]
                elif isinstance(market_data, list):
                    # Try to find token in the list
                    for item in market_data:
                        if isinstance(item, dict) and item.get('symbol', '').upper() == token:
                            token_data = item
                            break
        
                # Extract volatility if token_data is available
                if isinstance(token_data, dict) and 'volatility' in token_data:
                    volatility = token_data['volatility']
    
            except Exception as time_error:
                logger.logger.error(f"Error processing time data for {token}: {str(time_error)}")
                time_desc = timeframe
                target_time_str = "upcoming " + timeframe
                volatility = 0.0

            # Now safely analyze sentiment based on available data
            try:
                # Determine sentiment based on market performance
                vs_sentiment = vs_market.get('vs_market_sentiment', 'neutral')
                vs_change = vs_market.get('vs_market_avg_change', 0.0)
    
                if vs_sentiment in ['strongly outperforming', 'outperforming']:
                    sentiment = "bullish"
                    market_context = f"\n{token} outperforming market average by {abs(vs_change):.1f}%"
                elif vs_sentiment in ['strongly underperforming', 'underperforming']:
                    sentiment = "bearish"
                    market_context = f"\n{token} underperforming market average by {abs(vs_change):.1f}%"
                else:
                    # Neutral market performance
                    sentiment = "neutral"
                    market_context = f"\n{token} performing close to market average"
        
                # Add market trend context
                if market_trend in ['strongly bullish', 'bullish']:
                    market_context += f"\nOverall market trend: bullish"
                    # In bullish market, amplify token's performance
                    if sentiment == "bullish":
                        sentiment = "strongly bullish"
                elif market_trend in ['strongly bearish', 'bearish']:
                    market_context += f"\nOverall market trend: bearish"
                    # In bearish market, amplify token's performance
                    if sentiment == "bearish":
                        sentiment = "strongly bearish"
                else:
                    market_context += f"\nOverall market trend: neutral"
        
                # Add time-based context
                market_context += f"\nAnalysis for {time_desc} timeframe (until {target_time_str})"
    
                # Add volatility context if available
                if volatility > 0:
                    market_context += f"\nCurrent volatility: {volatility:.1f}%"
        
                    # Adjust sentiment based on volatility
                    if volatility > 10 and sentiment in ["bullish", "bearish"]:
                        sentiment = f"volatile {sentiment}"
                    
                # Incorporate trigger_type into the analysis if available
                if trigger_type:
                    # Adjust sentiment based on trigger type
                    if 'price_change' in trigger_type:
                        market_context += f"\nTriggered by significant price movement"
                    elif 'volume_change' in trigger_type or 'volume_trend' in trigger_type:
                        market_context += f"\nTriggered by notable volume activity"
                    elif 'smart_money' in trigger_type:
                        market_context += f"\nTriggered by smart money indicators"
                        # Emphasize sentiment for smart money triggers
                        if sentiment in ["bullish", "bearish"]:
                            sentiment = f"smart money {sentiment}"
                    elif 'prediction' in trigger_type:
                        market_context += f"\nBased on predictive model analysis"

            except Exception as analysis_error:
                logger.logger.error(f"Error in sentiment analysis for {token}: {str(analysis_error)}")
                logger.logger.debug(traceback.format_exc())
                sentiment = "neutral"
                market_context = f"\n{token} market sentiment analysis unavailable"
    
            # Ensure we handle any required timezone conversions
            try:
                # If there's a utility function for timezone handling, use it
                if hasattr(self, 'strip_timezone') and callable(self.strip_timezone):
                    # Apply timezone handling to any datetime objects that need it
                    pass
        
                # If there's a need to ensure datetimes are timezone-naive
                if hasattr(self, 'ensure_naive_datetimes') and callable(self.ensure_naive_datetimes):
                    # Apply to any datetime objects that need it
                    pass
            except Exception as tz_error:
                logger.logger.error(f"Error handling timezone data for {token}: {str(tz_error)}")
    
            # Prepare storage_data for database
            storage_data = {
                'content': None,  # Will be filled in by the caller
                'sentiment': {token: sentiment},
                'trigger_type': trigger_type if trigger_type else "regular_interval",
                'timeframe': timeframe,
                'market_context': market_context.strip(),
                'vs_market_change': vs_market.get('vs_market_avg_change', 0),
                'market_sentiment': vs_market.get('market_sentiment', 'neutral'),
                'timestamp': strip_timezone(datetime.now()) if hasattr(self, 'strip_timezone') else datetime.now()
            }
    
            return sentiment, storage_data
    
        except Exception as e:
            # Catch-all exception handler
            logger.logger.error(f"Error in _analyze_market_sentiment for {token} ({timeframe}): {str(e)}")
            logger.logger.debug(traceback.format_exc())
            logger.logger.error(f"{timeframe} analysis error details: {str(e)}")
        
            # Return minimal valid data to prevent downstream errors
            default_storage_data = {
                'content': None,
                'sentiment': {token: "neutral"},
                'trigger_type': "error_recovery",
                'timeframe': timeframe,
                'market_context': "",
                'timestamp': strip_timezone(datetime.now()) if hasattr(self, 'strip_timezone') else datetime.now()
            }
            return "neutral", default_storage_data

    @ensure_naive_datetimes
    def _should_post_update(self, token: str, new_data: Dict[str, Any], timeframe: str = "1h") -> Tuple[bool, str]:
        """
        Determine if we should post an update based on market changes for a specific timeframe
        
        Args:
            token: Token symbol
            new_data: Latest market data dictionary
            timeframe: Timeframe for the analysis
            
        Returns:
            Tuple of (should_post, trigger_reason)
        """
        if not self.last_market_data:
            self.last_market_data = new_data
            return True, f"initial_post_{timeframe}"

        trigger_reason = None

        # Check token for significant changes
        if token in new_data and token in self.last_market_data:
            # Get timeframe-specific thresholds
            thresholds = self.timeframe_thresholds.get(timeframe, self.timeframe_thresholds["1h"])
        
            # Calculate immediate price change since last check
            price_change = abs(
                (new_data[token]['current_price'] - self.last_market_data[token]['current_price']) /
                self.last_market_data[token]['current_price'] * 100
            )
        
            # Calculate immediate volume change since last check
            immediate_volume_change = abs(
                (new_data[token]['volume'] - self.last_market_data[token]['volume']) /
                self.last_market_data[token]['volume'] * 100
            )

            logger.logger.debug(
                f"{token} immediate changes ({timeframe}) - "
                f"Price: {price_change:.2f}%, Volume: {immediate_volume_change:.2f}%"
            )

            # Check immediate price change against timeframe threshold
            price_threshold = thresholds["price_change"]
            if price_change >= price_threshold:
                trigger_reason = f"price_change_{token.lower()}_{timeframe}"
                logger.logger.info(
                    f"Significant price change detected for {token} ({timeframe}): "
                    f"{price_change:.2f}% (threshold: {price_threshold}%)"
                )
            # Check immediate volume change against timeframe threshold
            else:
                volume_threshold = thresholds["volume_change"]
                if immediate_volume_change >= volume_threshold:
                    trigger_reason = f"volume_change_{token.lower()}_{timeframe}"
                    logger.logger.info(
                        f"Significant immediate volume change detected for {token} ({timeframe}): "
                        f"{immediate_volume_change:.2f}% (threshold: {volume_threshold}%)"
                )
                # Check rolling window volume trend
                else:
                    historical_volume = self._get_historical_volume_data(token, timeframe=timeframe)
                    if historical_volume:
                        volume_change_pct, trend = self._analyze_volume_trend(
                            new_data[token]['volume'],
                            historical_volume,
                            timeframe=timeframe
                        )
                
                    # Log the volume trend
                    logger.logger.debug(
                        f"{token} rolling window volume trend ({timeframe}): {volume_change_pct:.2f}% ({trend})"
                    )
                
                    # Check if trend is significant enough to trigger
                    if trend in ["significant_increase", "significant_decrease"]:
                        trigger_reason = f"volume_trend_{token.lower()}_{trend}_{timeframe}"
                        logger.logger.info(
                            f"Significant volume trend detected for {token} ({timeframe}): "
                            f"{volume_change_pct:.2f}% - {trend}"
                        )
        
            # Check for smart money indicators
            if not trigger_reason:
                smart_money = self._analyze_smart_money_indicators(token, new_data[token], timeframe=timeframe)
                if smart_money.get('abnormal_volume') or smart_money.get('stealth_accumulation'):
                    trigger_reason = f"smart_money_{token.lower()}_{timeframe}"
                    logger.logger.info(f"Smart money movement detected for {token} ({timeframe})")
                
                # Check for pattern metrics in longer timeframes
                elif timeframe in ["24h", "7d"] and 'pattern_metrics' in smart_money:
                    pattern_metrics = smart_money['pattern_metrics']
                    if pattern_metrics.get('volume_breakout', False) or pattern_metrics.get('consistent_high_volume', False):
                        trigger_reason = f"pattern_metrics_{token.lower()}_{timeframe}"
                        logger.logger.info(f"Advanced pattern metrics detected for {token} ({timeframe})")
        
            # Check for significant outperformance vs market
            if not trigger_reason:
                vs_market = self._analyze_token_vs_market(token, new_data, timeframe=timeframe)
                outperformance_threshold = 3.0 if timeframe == "1h" else 5.0 if timeframe == "24h" else 8.0
            
                if vs_market.get('outperforming_market') and abs(vs_market.get('vs_market_avg_change', 0)) > outperformance_threshold:
                    trigger_reason = f"{token.lower()}_outperforming_market_{timeframe}"
                    logger.logger.info(f"{token} significantly outperforming market ({timeframe})")
                
                # Check if we need to post prediction update
                # Trigger prediction post based on time since last prediction
                if not trigger_reason:
                    # Check when the last prediction was posted
                    last_prediction = self.config.db.get_active_predictions(token=token, timeframe=timeframe)
                    if not last_prediction:
                        # No recent predictions for this timeframe, should post one
                        trigger_reason = f"prediction_needed_{token.lower()}_{timeframe}"
                        logger.logger.info(f"No recent {timeframe} prediction for {token}, triggering prediction post")

        # Check if regular interval has passed (only for 1h timeframe)
        if not trigger_reason and timeframe == "1h":
            time_since_last = safe_datetime_diff(datetime.now(), self.last_check_time)
            if time_since_last >= self.config.BASE_INTERVAL:
                trigger_reason = f"regular_interval_{timeframe}"
                logger.logger.debug(f"Regular interval check triggered for {timeframe}")

        should_post = trigger_reason is not None
        if should_post:
            self.last_market_data = new_data
            logger.logger.info(f"Update triggered by: {trigger_reason}")
        else:
            logger.logger.debug(f"No {timeframe} triggers activated for {token}, skipping update")

        return should_post, trigger_reason

    @ensure_naive_datetimes
    def _evaluate_expired_predictions(self) -> None:
        """
        Find and evaluate expired predictions across all timeframes
        """
        try:
            # Get expired unevaluated predictions for all timeframes
            expired_predictions = self.config.db.get_expired_unevaluated_predictions()
        
            if not expired_predictions:
                logger.logger.debug("No expired predictions to evaluate")
                return
            
            # Group by timeframe
            expired_by_timeframe = {tf: [] for tf in self.timeframes}
        
            for prediction in expired_predictions:
                timeframe = prediction.get("timeframe", "1h")
                if timeframe in expired_by_timeframe:
                    expired_by_timeframe[timeframe].append(prediction)
        
            # Log count of expired predictions by timeframe
            for timeframe, preds in expired_by_timeframe.items():
                if preds:
                    logger.logger.info(f"Found {len(preds)} expired {timeframe} predictions to evaluate")
            
            # Get current market data for evaluation
            market_data = self._get_crypto_data()
            if not market_data:
                logger.logger.error("Failed to fetch market data for prediction evaluation")
                return
            
            # Track evaluated counts
            evaluated_counts = {tf: 0 for tf in self.timeframes}
            
            # Evaluate each prediction by timeframe
            for timeframe, predictions in expired_by_timeframe.items():
                for prediction in predictions:
                    token = prediction["token"]
                    prediction_id = prediction["id"]
                    
                    # Get current price for the token
                    token_data = market_data.get(token, {})
                    if not token_data:
                        logger.logger.warning(f"No current price data for {token}, skipping evaluation")
                        continue
                        
                    current_price = token_data.get("current_price", 0)
                    if current_price == 0:
                        logger.logger.warning(f"Zero price for {token}, skipping evaluation")
                        continue
                        
                    # Record the outcome
                    result = self.config.db.record_prediction_outcome(prediction_id, current_price)
                    
                    if result:
                        logger.logger.debug(f"Evaluated {timeframe} prediction {prediction_id} for {token}")
                        evaluated_counts[timeframe] += 1
                    else:
                        logger.logger.error(f"Failed to evaluate {timeframe} prediction {prediction_id} for {token}")
            
            # Log evaluation summaries
            for timeframe, count in evaluated_counts.items():
                if count > 0:
                    logger.logger.info(f"Evaluated {count} expired {timeframe} predictions")
            
            # Update prediction performance metrics
            self._update_prediction_performance_metrics()
            
        except Exception as e:
            logger.log_error("Evaluate Expired Predictions", str(e))

    @ensure_naive_datetimes
    def start(self) -> None:
        """
        Main bot execution loop with multi-timeframe support and reply functionality
        """
        try:
            retry_count = 0
            max_setup_retries = 3
            
            # Start the prediction thread early
            self._start_prediction_thread()
            
            # Load saved timeframe state
            self._load_saved_timeframe_state()
            
            # Initialize the browser and login
            while retry_count < max_setup_retries:
                if not self.browser.initialize_driver():
                    retry_count += 1
                    logger.logger.warning(f"Browser initialization attempt {retry_count} failed, retrying...")
                    time.sleep(10)
                    continue
                    
                if not self._login_to_twitter():
                    retry_count += 1
                    logger.logger.warning(f"Twitter login attempt {retry_count} failed, retrying...")
                    time.sleep(15)
                    continue
                    
                break
            
            if retry_count >= max_setup_retries:
                raise Exception("Failed to initialize bot after maximum retries")

            logger.logger.info("Bot initialized successfully")
            
            # Log the timeframes that will be used
            logger.logger.info(f"Bot configured with timeframes: {', '.join(self.timeframes)}")
            logger.logger.info(f"Timeframe posting frequencies: {self.timeframe_posting_frequency}")
            logger.logger.info(f"Reply checking interval: {self.reply_check_interval} minutes")

            # Pre-queue predictions for all tokens and timeframes
            market_data = self._get_crypto_data()
            if market_data:
                available_tokens = [token for token in self.reference_tokens if token in market_data]
                
                # Only queue predictions for the most important tokens to avoid overloading
                top_tokens = self._prioritize_tokens(available_tokens, market_data)[:5]
                
                logger.logger.info(f"Pre-queueing predictions for top tokens: {', '.join(top_tokens)}")
                for token in top_tokens:
                    self._queue_predictions_for_all_timeframes(token, market_data)

            while True:
                try:
                    self._run_analysis_cycle()
                    
                    # Calculate sleep time until next regular check
                    time_since_last = safe_datetime_diff(datetime.now(), self.last_check_time)
                    sleep_time = max(0, self.config.BASE_INTERVAL - time_since_last)
                    
                    # Check if we should post a weekly summary
                    if self._generate_weekly_summary():
                        logger.logger.info("Posted weekly performance summary")   

                    logger.logger.debug(f"Sleeping for {sleep_time:.1f}s until next check")
                    time.sleep(sleep_time)
                    
                    self.last_check_time = strip_timezone(datetime.now())
                    
                except Exception as e:
                    logger.log_error("Analysis Cycle", str(e), exc_info=True)
                    time.sleep(60)  # Shorter sleep on error
                    continue

        except KeyboardInterrupt:
            logger.logger.info("Bot stopped by user")
        except Exception as e:
            logger.log_error("Bot Execution", str(e))
        finally:
            self._cleanup()

if __name__ == "__main__":
    try:
        # Import necessary components
        from config import config  # This already has the database initialized
        from llm_provider import LLMProvider
        
        # Create LLM provider
        llm_provider = LLMProvider(config)
        
        # Create the bot using the database from config
        bot = CryptoAnalysisBot(
            database=config.db,  # Use the database that's already initialized in config
            llm_provider=llm_provider,
            config=config
        )
        
        # Start the bot
        bot.start()
    except Exception as e:
        from utils.logger import logger
        logger.log_error("Bot Startup", str(e))
