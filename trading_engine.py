#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Any, Union, Tuple
import os
import time
from datetime import datetime, timedelta
import logging
import json
import traceback
import ccxt
import numpy as np
from web3 import Web3
from dotenv import load_dotenv

from utils.logger import logger
from datetime_utils import strip_timezone, ensure_naive_datetimes, safe_datetime_diff
from config import config
from database import CryptoDatabase

class TradingExecutionEngine:
    """
    Trading execution engine that integrates with the CryptoAnalysisBot.
    
    Provides functionality to execute trades based on predictions across different
    exchanges (CeFi) and protocols (DeFi).
    """
    
    def __init__(self, database, config=None):
        """
        Initialize the trading execution engine with configuration and connections.
        
        Args:
            database: Database instance for storing trade data
            config: Configuration object
        """
        self.db = database
        self.config = config
        self.load_env_variables()
        
        # Initialize exchange clients
        self.exchanges = {}
        self.initialize_exchanges()
        
        # Initialize Web3 connections for DeFi trading
        self.web3_connections = {}
        self.initialize_web3_connections()
        
        # Trade tracking
        self.active_trades = {}
        self.pending_orders = {}
        self.load_active_trades()
        
        # Risk management parameters
        self.position_size_percentage = float(os.getenv('POSITION_SIZE_PERCENTAGE', '1.0'))  # Default 1% of account
        self.max_trades_per_token = int(os.getenv('MAX_TRADES_PER_TOKEN', '1'))
        self.max_concurrent_trades = int(os.getenv('MAX_CONCURRENT_TRADES', '3'))
        self.stop_loss_percentage = float(os.getenv('STOP_LOSS_PERCENTAGE', '2.0'))
        self.take_profit_percentage = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '4.0'))
        
        # Default timeframe to use for trading (1h, 24h, 7d)
        self.default_timeframe = os.getenv('DEFAULT_TRADING_TIMEFRAME', '1h')
        
        # Minimum confidence for auto-trading
        self.min_confidence_threshold = float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '75.0'))
        
        # Trading mode: 'paper', 'live', or 'simulation'
        self.trading_mode = os.getenv('TRADING_MODE', 'paper').lower()
        
        logger.logger.info(f"Trading execution engine initialized in {self.trading_mode} mode")
        logger.logger.info(f"Position size: {self.position_size_percentage}%, Stop-loss: {self.stop_loss_percentage}%, Take-profit: {self.take_profit_percentage}%")
    
    def load_env_variables(self):
        """Load trading-specific environment variables"""
        # CeFi API keys
        self.binance_api_key = os.getenv('BINANCE_API_KEY', '')
        self.binance_api_secret = os.getenv('BINANCE_API_SECRET', '')
        self.bybit_api_key = os.getenv('BYBIT_API_KEY', '')
        self.bybit_api_secret = os.getenv('BYBIT_API_SECRET', '')
        
        # DeFi wallet details
        self.eth_private_key = os.getenv('ETH_PRIVATE_KEY', '')
        self.eth_wallet_address = os.getenv('ETH_WALLET_ADDRESS', '')
        
        # RPC endpoints
        self.ethereum_rpc = os.getenv('ETHEREUM_RPC', 'https://mainnet.infura.io/v3/your-infura-id')
        self.arbitrum_rpc = os.getenv('ARBITRUM_RPC', 'https://arb1.arbitrum.io/rpc')
        self.base_rpc = os.getenv('BASE_RPC', 'https://mainnet.base.org')
        
        # Contract addresses for DeFi protocols
        self.gmx_router_address = os.getenv('GMX_ROUTER_ADDRESS', '')
        self.dydx_address = os.getenv('DYDX_ADDRESS', '')
    
    def initialize_exchanges(self):
        """Initialize connections to supported centralized exchanges"""
        try:
            # Only initialize if API keys are provided
            if self.binance_api_key and self.binance_api_secret:
                self.exchanges['binance'] = ccxt.binance({
                    'apiKey': self.binance_api_key,
                    'secret': self.binance_api_secret,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'future'}
                })
                logger.logger.info("Binance Futures exchange initialized")
            
            if self.bybit_api_key and self.bybit_api_secret:
                self.exchanges['bybit'] = ccxt.bybit({
                    'apiKey': self.bybit_api_key,
                    'secret': self.bybit_api_secret,
                    'enableRateLimit': True
                })
                logger.logger.info("Bybit exchange initialized")
            
            # Paper trading exchange simulator
            if self.trading_mode == 'paper':
                self.exchanges['paper'] = self.PaperTradingExchange()
                logger.logger.info("Paper trading exchange initialized")
            
        except Exception as e:
            logger.log_error("Exchange Initialization", str(e))
    
    def initialize_web3_connections(self):
        """Initialize Web3 connections for various chains"""
        try:
            if self.ethereum_rpc:
                self.web3_connections['ethereum'] = Web3(Web3.HTTPProvider(self.ethereum_rpc))
                eth_connected = self.web3_connections['ethereum'].is_connected()
                logger.logger.info(f"Ethereum Web3 connection: {'Connected' if eth_connected else 'Failed'}")
            
            if self.arbitrum_rpc:
                self.web3_connections['arbitrum'] = Web3(Web3.HTTPProvider(self.arbitrum_rpc))
                arb_connected = self.web3_connections['arbitrum'].is_connected()
                logger.logger.info(f"Arbitrum Web3 connection: {'Connected' if arb_connected else 'Failed'}")
            
            if self.base_rpc:
                self.web3_connections['base'] = Web3(Web3.HTTPProvider(self.base_rpc))
                base_connected = self.web3_connections['base'].is_connected()
                logger.logger.info(f"Base Web3 connection: {'Connected' if base_connected else 'Failed'}")
                
        except Exception as e:
            logger.log_error("Web3 Initialization", str(e))
    
    def load_active_trades(self):
        """Load active trades from the database"""
        try:
            trades = self.db.get_active_trades()
            for trade in trades:
                trade_id = trade.get('trade_id')
                if trade_id:
                    self.active_trades[trade_id] = trade
            
            logger.logger.info(f"Loaded {len(self.active_trades)} active trades from database")
        except Exception as e:
            logger.log_error("Loading Active Trades", str(e))
    
    def process_prediction(self, token: str, prediction: Dict[str, Any], market_data: Dict[str, Any], timeframe: str = None) -> bool:
        """
        Process a prediction and decide whether to execute a trade
        
        Args:
            token: Token symbol (e.g., 'BTC', 'ETH')
            prediction: Prediction data dictionary
            market_data: Market data dictionary
            timeframe: Timeframe for the prediction
            
        Returns:
            Boolean indicating if a trade was executed
        """
        if timeframe is None:
            timeframe = self.default_timeframe
        
        try:
            logger.logger.info(f"Processing {timeframe} prediction for {token}")
            
            # Extract prediction details
            pred_data = prediction.get("prediction", {})
            sentiment = prediction.get("sentiment", "NEUTRAL")
            confidence = pred_data.get("confidence", 0)
            percent_change = pred_data.get("percent_change", 0)
            
            # Determine trade direction
            if sentiment == "BULLISH":
                direction = "buy"
            elif sentiment == "BEARISH":
                direction = "sell"
            else:
                logger.logger.info(f"Neutral sentiment for {token}, no trade signal")
                return False
            
            # Check confidence threshold
            if confidence < self.min_confidence_threshold:
                logger.logger.info(f"Confidence too low for {token} ({confidence}% < {self.min_confidence_threshold}%), skipping trade")
                return False
            
            # Check if we have too many active trades for this token
            token_trades = [t for t in self.active_trades.values() if t.get('token') == token]
            if len(token_trades) >= self.max_trades_per_token:
                logger.logger.info(f"Already have maximum trades ({self.max_trades_per_token}) for {token}, skipping")
                return False
            
            # Check if we have too many concurrent trades overall
            if len(self.active_trades) >= self.max_concurrent_trades:
                logger.logger.info(f"Maximum concurrent trades ({self.max_concurrent_trades}) reached, skipping")
                return False
            
            # Get current price
            current_price = self._get_current_price(token, market_data)
            if current_price <= 0:
                logger.logger.warning(f"Invalid price for {token}: {current_price}")
                return False
            
            # Calculate position size based on account balance
            position_size = self._calculate_position_size(token, current_price)
            
            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_exit_levels(
                direction, current_price, self.stop_loss_percentage, self.take_profit_percentage
            )
            
            # Determine leverage based on timeframe and volatility
            leverage = self._determine_leverage(token, timeframe, market_data)
            
            # Execute the trade
            trade_id = self._execute_trade(
                token=token,
                direction=direction,
                amount=position_size,
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                prediction_id=prediction.get("id"),
                timeframe=timeframe,
                confidence=confidence
            )
            
            if trade_id:
                logger.logger.info(f"Trade executed for {token}: {direction.upper()} {position_size} at ${current_price:.2f}")
                return True
            else:
                logger.logger.warning(f"Failed to execute trade for {token}")
                return False
            
        except Exception as e:
            logger.log_error(f"Process Prediction - {token} ({timeframe})", str(e))
            return False
    
    def _get_current_price(self, token: str, market_data: Dict[str, Any]) -> float:
        """
        Get current price for a token from market data
        
        Args:
            token: Token symbol
            market_data: Market data dictionary
            
        Returns:
            Current price as float
        """
        try:
            # Try to get from market data first
            if token in market_data:
                return market_data[token].get('current_price', 0)
            
            # If not in market data, try to get from exchange
            if self.exchanges:
                for exchange_id, exchange in self.exchanges.items():
                    try:
                        # Skip paper trading exchange
                        if exchange_id == 'paper':
                            continue
                        
                        # Format the symbol for the exchange
                        symbol = f"{token}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        if ticker and 'last' in ticker and ticker['last'] > 0:
                            return ticker['last']
                    except:
                        continue
            
            # Last resort: try a direct API call if we have market data for any token
            if market_data and list(market_data.values())[0].get('current_price', 0) > 0:
                # Just use the relative price from market data (not accurate but better than 0)
                logger.logger.warning(f"Using estimated price for {token} based on relative market data")
                return 100.0  # Default placeholder value
            
            # If all else fails
            return 0
            
        except Exception as e:
            logger.log_error(f"Get Current Price - {token}", str(e))
            return 0
    
    def _calculate_position_size(self, token: str, price: float) -> float:
        """
        Calculate position size based on account balance and risk parameters
        
        Args:
            token: Token symbol
            price: Current token price
            
        Returns:
            Position size in token units
        """
        try:
            # Get account balance from primary exchange
            balance = self._get_account_balance()
            
            # Calculate position value in USD
            position_value = balance * (self.position_size_percentage / 100)
            
            # Convert to token units
            if price > 0:
                position_size = position_value / price
            else:
                position_size = 0
                
            # Apply minimum position size
            min_position_value = 10.0  # $10 minimum
            if position_value < min_position_value:
                position_size = min_position_value / price if price > 0 else 0
                
            # Apply maximum position size (only if non-zero position calculated)
            max_position_value = 1000.0  # $1000 maximum
            if position_value > max_position_value and position_size > 0:
                position_size = max_position_value / price
                
            # Round to appropriate precision
            if token in ['BTC']:
                position_size = round(position_size, 5)  # 0.00001 BTC precision
            elif token in ['ETH']:
                position_size = round(position_size, 4)  # 0.0001 ETH precision
            else:
                position_size = round(position_size, 2)  # 0.01 precision for other tokens
                
            return position_size
            
        except Exception as e:
            logger.log_error(f"Calculate Position Size - {token}", str(e))
            # Return a safe default value
            return 0.01  # Small default position
    
    def _get_account_balance(self) -> float:
        """
        Get account balance from primary exchange
        
        Returns:
            Account balance in USD
        """
        try:
            # If in paper trading mode, return simulated balance
            if self.trading_mode == 'paper':
                paper_balance = float(os.getenv('PAPER_TRADING_BALANCE', '10000.0'))
                return paper_balance
            
            # Try each exchange until we get a valid balance
            for exchange_id, exchange in self.exchanges.items():
                try:
                    # Skip paper trading exchange for live balance
                    if exchange_id == 'paper':
                        continue
                    
                    balance = exchange.fetch_balance()
                    if 'USDT' in balance['total']:
                        return balance['total']['USDT']
                    elif 'USD' in balance['total']:
                        return balance['total']['USD']
                    elif 'BUSD' in balance['total']:
                        return balance['total']['BUSD']
                except:
                    continue
            
            # If no balance found, use default
            logger.logger.warning("Could not fetch account balance, using default")
            return 10000.0  # Default balance
            
        except Exception as e:
            logger.log_error("Get Account Balance", str(e))
            return 10000.0  # Default balance on error
    
    def _calculate_exit_levels(self, direction: str, price: float, 
                               stop_loss_pct: float, take_profit_pct: float) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels
        
        Args:
            direction: Trade direction ('buy' or 'sell')
            price: Entry price
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if direction == 'buy':
            stop_loss = price * (1 - stop_loss_pct / 100)
            take_profit = price * (1 + take_profit_pct / 100)
        else:  # sell
            stop_loss = price * (1 + stop_loss_pct / 100)
            take_profit = price * (1 - take_profit_pct / 100)
            
        return round(stop_loss, 4), round(take_profit, 4)
    
    def _determine_leverage(self, token: str, timeframe: str, market_data: Dict[str, Any]) -> int:
        """
        Determine appropriate leverage based on token, timeframe, and volatility
        
        Args:
            token: Token symbol
            timeframe: Prediction timeframe
            market_data: Market data with volatility info
            
        Returns:
            Leverage as integer
        """
        # Base leverage by timeframe
        if timeframe == '1h':
            base_leverage = 5  # Higher leverage for short timeframe
        elif timeframe == '24h':
            base_leverage = 3  # Moderate leverage for daily timeframe
        else:  # 7d
            base_leverage = 2  # Low leverage for weekly timeframe
        
        # Adjust for volatility if available
        volatility = 0
        if token in market_data and 'volatility' in market_data[token]:
            volatility = market_data[token]['volatility']
        elif token in market_data and 'price_change_percentage_24h' in market_data[token]:
            # Use 24h price change as proxy for volatility
            volatility = abs(market_data[token]['price_change_percentage_24h'])
        
        # Reduce leverage for high volatility
        if volatility > 10:
            leverage_reduction = min(base_leverage - 1, (volatility - 10) // 5 + 1)
            base_leverage = max(1, base_leverage - leverage_reduction)
        
        # Adjust by token
        if token == 'BTC':
            # Bitcoin tends to be less volatile, so can use slightly higher leverage
            base_leverage += 1
        elif token in ['SOL', 'AVAX', 'DOT']:
            # More volatile altcoins, use lower leverage
            base_leverage = max(1, base_leverage - 1)
        
        # Ensure leverage is within reasonable bounds
        return min(10, max(1, base_leverage))
    
    def _execute_trade(self, token: str, direction: str, amount: float, price: float,
                       stop_loss: float, take_profit: float, leverage: int,
                       prediction_id: str, timeframe: str, confidence: float) -> Optional[str]:
        """
        Execute a trade on the configured exchange
        
        Args:
            token: Token symbol
            direction: Trade direction ('buy' or 'sell')
            amount: Position size in token units
            price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Leverage to use
            prediction_id: Associated prediction ID
            timeframe: Prediction timeframe
            confidence: Prediction confidence
            
        Returns:
            Trade ID if successful, None otherwise
        """
        if amount <= 0 or price <= 0:
            logger.logger.warning(f"Invalid trade parameters: amount={amount}, price={price}")
            return None
        
        try:
            trade_exchange = self._select_exchange(token)
            if not trade_exchange:
                logger.logger.warning(f"No suitable exchange found for {token}")
                return None
            
            # Generate a unique trade ID
            trade_id = f"{token}_{direction}_{int(time.time())}"
            
            # Format parameters for the exchange
            symbol = f"{token}/USDT" if trade_exchange != "defi" else token
            
            # Execute the trade based on trading mode
            order_details = None
            
            if self.trading_mode == 'paper':
                # Simulate the trade for paper trading
                order_details = self._execute_paper_trade(
                    symbol, direction, amount, price, leverage, stop_loss, take_profit
                )
                
            elif self.trading_mode == 'simulation':
                # Just log the trade, don't actually execute
                logger.logger.info(
                    f"SIMULATION: Would execute {direction} {amount} {token} at ${price:.2f} "
                    f"with {leverage}x leverage (SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})"
                )
                # Create simulated order details
                order_details = {
                    'id': f"sim_{trade_id}",
                    'status': 'open',
                    'symbol': symbol,
                    'type': 'market',
                    'side': direction,
                    'amount': amount,
                    'price': price,
                    'leverage': leverage,
                    'timestamp': int(time.time() * 1000)
                }
                
            elif trade_exchange == "defi":
                # Execute on DeFi protocol
                order_details = self._execute_defi_trade(
                    token, direction, amount, price, leverage
                )
                
            else:
                # Execute on CeFi exchange
                exchange = self.exchanges.get(trade_exchange)
                
                # Set leverage first
                if hasattr(exchange, 'set_leverage'):
                    try:
                        exchange.set_leverage(leverage, symbol)
                    except Exception as lev_e:
                        logger.logger.warning(f"Could not set leverage for {symbol}: {str(lev_e)}")
                
                # Create market order
                order = exchange.create_market_order(
                    symbol=symbol,
                    side=direction,
                    amount=amount
                )
                
                # Add stop loss and take profit orders if supported
                if order and 'id' in order:
                    try:
                        sl_params = {
                            'symbol': symbol,
                            'side': 'sell' if direction == 'buy' else 'buy',
                            'type': 'stop',
                            'amount': amount,
                            'price': stop_loss,
                            'params': {'stopPrice': stop_loss, 'reduceOnly': True}
                        }
                        sl_order = exchange.create_order(**sl_params)
                        
                        tp_params = {
                            'symbol': symbol,
                            'side': 'sell' if direction == 'buy' else 'buy',
                            'type': 'limit',
                            'amount': amount,
                            'price': take_profit,
                            'params': {'reduceOnly': True}
                        }
                        tp_order = exchange.create_order(**tp_params)
                        
                        # Combine orders for tracking
                        order_details = {
                            **order,
                            'stop_loss_order': sl_order.get('id') if sl_order else None,
                            'take_profit_order': tp_order.get('id') if tp_order else None
                        }
                    except Exception as exit_e:
                        logger.logger.warning(f"Could not set exit orders: {str(exit_e)}")
                        order_details = order
            
            # Record the trade in the database
            if order_details:
                trade_data = {
                    'trade_id': trade_id,
                    'token': token,
                    'exchange': trade_exchange,
                    'direction': direction,
                    'amount': amount,
                    'entry_price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'leverage': leverage,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'open',
                    'prediction_id': prediction_id,
                    'timeframe': timeframe,
                    'confidence': confidence,
                    'order_details': json.dumps(order_details)
                }
                
                # Store in database
                self.db.store_trade(trade_data)
                
                # Add to active trades
                self.active_trades[trade_id] = trade_data
                
                return trade_id
            else:
                logger.logger.warning(f"No order details for {token} trade")
                return None
            
        except Exception as e:
            logger.log_error(f"Execute Trade - {token}", str(e))
            return None
    
    def _select_exchange(self, token: str) -> Optional[str]:
        """
        Select the best exchange for trading a specific token
        
        Args:
            token: Token symbol
            
        Returns:
            Exchange ID or None if no suitable exchange found
        """
        # If paper trading, always use paper exchange
        if self.trading_mode == 'paper' or self.trading_mode == 'simulation':
            return 'paper'
        
        # Check each exchange for the token's availability
        for exchange_id, exchange in self.exchanges.items():
            try:
                # Skip paper trading exchange for live trading
                if exchange_id == 'paper':
                    continue
                
                # Format the symbol for the exchange
                symbol = f"{token}/USDT"
                
                # Check if the symbol is available
                markets = exchange.load_markets()
                if symbol in markets:
                    return exchange_id
            except:
                continue
        
        # If no CeFi exchange found, check if DeFi trading is possible
        if token in ['ETH', 'BTC', 'LINK', 'UNI'] and self.eth_private_key:
            return "defi"
        
        # If no exchange found, default to paper trading
        logger.logger.warning(f"No suitable exchange found for {token}, falling back to paper trading")
        return 'paper'
    
    def _execute_paper_trade(self, symbol: str, direction: str, amount: float, price: float,
                            leverage: int, stop_loss: float, take_profit: float) -> Dict[str, Any]:
        """
        Execute a simulated paper trade
        
        Args:
            symbol: Trading symbol
            direction: Trade direction ('buy' or 'sell')
            amount: Position size
            price: Entry price
            leverage: Leverage
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            Simulated order details
        """
        try:
            # Get the paper trading exchange
            paper_exchange = self.exchanges.get('paper')
            if not paper_exchange:
                # Create one if it doesn't exist
                paper_exchange = self.PaperTradingExchange()
                self.exchanges['paper'] = paper_exchange
            
            # Execute the paper trade
            order = paper_exchange.create_market_order(
                symbol=symbol,
                side=direction,
                amount=amount,
                price=price,
                leverage=leverage
            )
            
            # Add stop loss and take profit to the order
            order['stop_loss'] = stop_loss
            order['take_profit'] = take_profit
            
            return order
            
        except Exception as e:
            logger.log_error("Paper Trade Execution", str(e))
            # Return minimal valid order
            return {
                'id': f"paper_{int(time.time())}",
                'status': 'open',
                'symbol': symbol,
                'type': 'market',
                'side': direction,
                'amount': amount,
                'price': price,
                'leverage': leverage,
                'timestamp': int(time.time() * 1000),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
    
    def _execute_defi_trade(self, token: str, direction: str, amount: float, 
                           price: float, leverage: int) -> Dict[str, Any]:
        """
        Execute a trade on DeFi (placeholder for implementation)
        
        Args:
            token: Token symbol
            direction: Trade direction
            amount: Position size
            price: Entry price
            leverage: Leverage
            
        Returns:
            Order details dictionary
        """
        # This is a placeholder for actual DeFi trading implementation
        logger.logger.warning("DeFi trading not fully implemented, simulating trade")
        
        # Return simulated order
        return {
            'id': f"defi_{int(time.time())}",
            'status': 'open',
            'token': token,
            'type': 'defi_perp',
            'side': direction,
            'amount': amount,
            'price': price,
            'leverage': leverage,
            'timestamp': int(time.time() * 1000),
            'protocol': 'simulation'
        }
    
    def update_active_trades(self, market_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Update status of all active trades and handle exits if conditions are met
        
        Args:
            market_data: Current market data
            
        Returns:
            Dictionary with lists of trade IDs for different status changes
        """
        result = {
            'closed': [],
            'stop_loss': [],
            'take_profit': [],
            'error': []
        }
        
        # Skip if no active trades
        if not self.active_trades:
            return result
        
        # For each active trade, check if exit conditions are met
        trades_to_remove = []
        
        for trade_id, trade in self.active_trades.items():
            try:
                token = trade.get('token')
                direction = trade.get('direction')
                entry_price = trade.get('entry_price', 0)
                stop_loss = trade.get('stop_loss', 0)
                take_profit = trade.get('take_profit', 0)
                
                # Skip if missing essential data
                if not token or not direction or entry_price <= 0:
                    continue
                
                # Get current price
                current_price = self._get_current_price(token, market_data)
                
                # Skip if unable to get current price
                if current_price <= 0:
                    continue
                
                # Check for stop loss hit
                stop_loss_hit = False
                if stop_loss > 0:
                    if direction == 'buy' and current_price <= stop_loss:
                        stop_loss_hit = True
                    elif direction == 'sell' and current_price >= stop_loss:
                        stop_loss_hit = True
                
                # Check for take profit hit
                take_profit_hit = False
                if take_profit > 0:
                    if direction == 'buy' and current_price >= take_profit:
                        take_profit_hit = True
                    elif direction == 'sell' and current_price <= take_profit:
                        take_profit_hit = True
                
                # Handle exit if condition met
                if stop_loss_hit or take_profit_hit:
                    # Determine exit reason
                    exit_reason = 'stop_loss' if stop_loss_hit else 'take_profit'
                    
                    # Calculate PnL
                    if direction == 'buy':
                        pnl_percentage = ((current_price - entry_price) / entry_price) * 100
                    else:  # sell
                        pnl_percentage = ((entry_price - current_price) / entry_price) * 100
                    
                    # Apply leverage to PnL
                    pnl_percentage *= trade.get('leverage', 1)
                    
                    # Close the trade
                    close_success = self._close_trade(
                        trade_id=trade_id,
                        exit_price=current_price,
                        exit_reason=exit_reason,
                        pnl_percentage=pnl_percentage
                    )
                    
                    if close_success:
                        trades_to_remove.append(trade_id)
                        result[exit_reason].append(trade_id)
                        logger.logger.info(
                            f"Closed trade {trade_id} ({token} {direction}) due to {exit_reason}: "
                            f"Entry: ${entry_price:.4f}, Exit: ${current_price:.4f}, "
                            f"PnL: {pnl_percentage:.2f}% with {trade.get('leverage', 1)}x leverage"
                        )
                
            except Exception as e:
                logger.log_error(f"Update Trade - {trade_id}", str(e))
                result['error'].append(trade_id)
        
        # Remove closed trades from active trades
        for trade_id in trades_to_remove:
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]
                
        return result
    
    @ensure_naive_datetimes
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str, 
                    pnl_percentage: float) -> bool:
        """
        Close a trade and update the database record
        
        Args:
            trade_id: Unique trade identifier
            exit_price: Closing price
            exit_reason: Reason for closing (stop_loss, take_profit, manual)
            pnl_percentage: Profit/loss percentage
            
        Returns:
            True if successful, False otherwise
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                logger.logger.warning(f"Trade {trade_id} not found in active trades")
                return False
            
            # Get trade parameters
            token = trade.get('token')
            exchange_id = trade.get('exchange')
            amount = trade.get('amount', 0)
            direction = trade.get('direction')
            
            # Prepare close parameters
            close_side = 'sell' if direction == 'buy' else 'buy'
            symbol = f"{token}/USDT"
            
            # Execute the close order on the exchange if in live mode
            order_result = None
            
            if self.trading_mode == 'live' and exchange_id != 'paper' and exchange_id != 'defi':
                exchange = self.exchanges.get(exchange_id)
                if exchange:
                    try:
                        order_result = exchange.create_market_order(
                            symbol=symbol,
                            side=close_side,
                            amount=amount
                        )
                    except Exception as order_e:
                        logger.log_error(f"Close Trade Order - {trade_id}", str(order_e))
            
            # For paper trading or if live order failed, create simulated result
            if not order_result:
                order_result = {
                    'id': f"close_{trade_id}",
                    'status': 'closed',
                    'symbol': symbol,
                    'side': close_side,
                    'amount': amount,
                    'price': exit_price,
                    'timestamp': int(time.time() * 1000)
                }
            
            # Update trade record in database
            close_time = strip_timezone(datetime.now())
            
            update_data = {
                'status': 'closed',
                'exit_price': exit_price,
                'exit_time': close_time.isoformat(),
                'exit_reason': exit_reason,
                'pnl_percentage': pnl_percentage,
                'close_order_details': json.dumps(order_result)
            }
            
            # Update database
            update_success = self.db.update_trade(trade_id, update_data)
            
            return update_success
            
        except Exception as e:
            logger.log_error(f"Close Trade - {trade_id}", str(e))
            return False
    
    @ensure_naive_datetimes
    def get_trade_performance(self, token: str = None, timeframe: str = None, 
                             days: int = 30) -> Dict[str, Any]:
        """
        Calculate performance statistics for trades
        
        Args:
            token: Optional token to filter by
            timeframe: Optional timeframe to filter by
            days: Number of days to include
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Calculate date range with proper timezone handling
            end_date = strip_timezone(datetime.now())
            start_date = strip_timezone(end_date - timedelta(days=days))
            
            # Get closed trades from database
            trades = self.db.get_closed_trades(
                token=token,
                timeframe=timeframe,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat()
            )
            
            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'total_pnl': 0,
                    'best_trade': None,
                    'worst_trade': None,
                    'token': token,
                    'timeframe': timeframe,
                    'days': days
                }
            
            # Calculate performance metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.get('pnl_percentage', 0) > 0]
            losing_trades = [t for t in trades if t.get('pnl_percentage', 0) <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            
            avg_profit = (sum(t.get('pnl_percentage', 0) for t in winning_trades) / win_count) if win_count > 0 else 0
            avg_loss = (sum(t.get('pnl_percentage', 0) for t in losing_trades) / loss_count) if loss_count > 0 else 0
            
            total_pnl = sum(t.get('pnl_percentage', 0) for t in trades)
            
            best_trade = max(trades, key=lambda x: x.get('pnl_percentage', 0)) if trades else None
            worst_trade = min(trades, key=lambda x: x.get('pnl_percentage', 0)) if trades else None
            
            # Group by token
            performance_by_token = {}
            if not token:
                for t in trades:
                    token_symbol = t.get('token')
                    if token_symbol:
                        if token_symbol not in performance_by_token:
                            performance_by_token[token_symbol] = {
                                'trades': 0,
                                'wins': 0,
                                'losses': 0,
                                'total_pnl': 0
                            }
                        
                        performance_by_token[token_symbol]['trades'] += 1
                        if t.get('pnl_percentage', 0) > 0:
                            performance_by_token[token_symbol]['wins'] += 1
                        else:
                            performance_by_token[token_symbol]['losses'] += 1
                        
                        performance_by_token[token_symbol]['total_pnl'] += t.get('pnl_percentage', 0)
            
            # Group by timeframe
            performance_by_timeframe = {}
            if not timeframe:
                for t in trades:
                    tf = t.get('timeframe')
                    if tf:
                        if tf not in performance_by_timeframe:
                            performance_by_timeframe[tf] = {
                                'trades': 0,
                                'wins': 0,
                                'losses': 0,
                                'total_pnl': 0
                            }
                        
                        performance_by_timeframe[tf]['trades'] += 1
                        if t.get('pnl_percentage', 0) > 0:
                            performance_by_timeframe[tf]['wins'] += 1
                        else:
                            performance_by_timeframe[tf]['losses'] += 1
                        
                        performance_by_timeframe[tf]['total_pnl'] += t.get('pnl_percentage', 0)
            
            # Result
            return {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'total_pnl': total_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'by_token': performance_by_token,
                'by_timeframe': performance_by_timeframe,
                'token': token,
                'timeframe': timeframe,
                'days': days,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat()
            }
            
        except Exception as e:
            logger.log_error("Trade Performance Analysis", str(e))
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'total_pnl': 0,
                'best_trade': None,
                'worst_trade': None,
                'error': str(e),
                'token': token,
                'timeframe': timeframe,
                'days': days
            }
    
    @ensure_naive_datetimes
    def calculate_trading_effectiveness(self, token: str = None, timeframe: str = None) -> Dict[str, Any]:
        """
        Calculate effectiveness of trading strategy by comparing trade results with predictions
        
        Args:
            token: Optional token to filter by
            timeframe: Optional timeframe to filter by
            
        Returns:
            Dictionary of effectiveness metrics
        """
        try:
            # Get closed trades
            trades = self.db.get_closed_trades(token=token, timeframe=timeframe)
            
            if not trades:
                return {
                    'total_trades': 0,
                    'average_effectiveness': 0,
                    'strategy_roi': 0,
                    'hold_roi': 0,
                    'token': token,
                    'timeframe': timeframe
                }
            
            total_effectiveness = 0
            total_trades = len(trades)
            
            # Calculate effectiveness for each trade
            trade_effectiveness = []
            
            for trade in trades:
                trade_id = trade.get('trade_id')
                prediction_id = trade.get('prediction_id')
                
                if not prediction_id:
                    continue
                
                # Get the associated prediction
                prediction = self.db.get_prediction_by_id(prediction_id)
                
                if not prediction:
                    continue
                
                # Calculate trade direction effectiveness
                pred_data = prediction.get('prediction', {})
                pred_direction = 'buy' if pred_data.get('percent_change', 0) > 0 else 'sell'
                trade_direction = trade.get('direction')
                
                direction_match = pred_direction == trade_direction
                
                # Calculate price target effectiveness
                pred_price = pred_data.get('price', 0)
                exit_price = trade.get('exit_price', 0)
                entry_price = trade.get('entry_price', 0)
                
                if pred_price <= 0 or exit_price <= 0 or entry_price <= 0:
                    continue
                
                # How close was the exit price to the prediction target?
                price_accuracy = 1 - min(1, abs(exit_price - pred_price) / pred_price)
                
                # Was the trade profitable?
                trade_profitable = trade.get('pnl_percentage', 0) > 0
                
                # Calculate overall effectiveness (0-100)
                effectiveness = (
                    (direction_match * 40) +  # 40% weight on direction match
                    (price_accuracy * 30) +   # 30% weight on price accuracy
                    (trade_profitable * 30)   # 30% weight on profitability
                )
                
                trade_effectiveness.append({
                    'trade_id': trade_id,
                    'effectiveness': effectiveness,
                    'direction_match': direction_match,
                    'price_accuracy': price_accuracy,
                    'profitable': trade_profitable,
                    'pnl': trade.get('pnl_percentage', 0)
                })
                
                total_effectiveness += effectiveness
            
            # Calculate average effectiveness
            average_effectiveness = total_effectiveness / len(trade_effectiveness) if trade_effectiveness else 0
            
            # Calculate strategy ROI vs buy and hold
            strategy_roi = sum(t.get('pnl_percentage', 0) for t in trades)
            
            # Calculate hold ROI (simplified - assumes equal weighting of tokens)
            hold_roi = 0
            if token:
                # For a single token, get price change over the period
                earliest_trade = min(trades, key=lambda x: x.get('timestamp', ''))
                latest_trade = max(trades, key=lambda x: x.get('exit_time', ''))
                
                start_time = strip_timezone(datetime.fromisoformat(earliest_trade.get('timestamp')))
                end_time = strip_timezone(datetime.fromisoformat(latest_trade.get('exit_time')))
                
                # Get price change for the token
                start_price = self.db.get_token_price_at_time(token, start_time.isoformat())
                end_price = self.db.get_token_price_at_time(token, end_time.isoformat())
                
                if start_price and end_price and start_price > 0:
                    hold_roi = ((end_price - start_price) / start_price) * 100
            
            # Result
            return {
                'total_trades': total_trades,
                'evaluated_trades': len(trade_effectiveness),
                'average_effectiveness': average_effectiveness,
                'strategy_roi': strategy_roi,
                'hold_roi': hold_roi,
                'trade_details': trade_effectiveness,
                'token': token,
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.log_error("Trading Effectiveness Analysis", str(e))
            return {
                'total_trades': 0,
                'average_effectiveness': 0,
                'strategy_roi': 0,
                'hold_roi': 0,
                'error': str(e),
                'token': token,
                'timeframe': timeframe
            }
    
    def cancel_trade(self, trade_id: str) -> bool:
        """
        Cancel an active trade
        
        Args:
            trade_id: Trade ID to cancel
            
        Returns:
            True if successful, False otherwise
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                logger.logger.warning(f"Trade {trade_id} not found in active trades")
                return False
            
            # Mark as manually closed with current price
            token = trade.get('token')
            current_price = self._get_current_price(token, {})
            
            close_success = self._close_trade(
                trade_id=trade_id,
                exit_price=current_price,
                exit_reason='manual',
                pnl_percentage=0  # Calculate this properly if needed
            )
            
            if close_success:
                # Remove from active trades
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
                
                logger.logger.info(f"Manually closed trade {trade_id}")
                return True
            else:
                logger.logger.warning(f"Failed to manually close trade {trade_id}")
                return False
            
        except Exception as e:
            logger.log_error(f"Cancel Trade - {trade_id}", str(e))
            return False

    def modify_trade_parameters(self, trade_id: str, stop_loss: float = None, 
                               take_profit: float = None) -> bool:
        """
        Modify parameters of an active trade
        
        Args:
            trade_id: Trade ID to modify
            stop_loss: New stop loss price (optional)
            take_profit: New take profit price (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            trade = self.active_trades.get(trade_id)
            if not trade:
                logger.logger.warning(f"Trade {trade_id} not found in active trades")
                return False
            
            # Update parameters
            changes = {}
            
            if stop_loss is not None:
                changes['stop_loss'] = stop_loss
                # Update in active trades
                trade['stop_loss'] = stop_loss
            
            if take_profit is not None:
                changes['take_profit'] = take_profit
                # Update in active trades
                trade['take_profit'] = take_profit
            
            # No changes requested
            if not changes:
                return True
            
            # Update in database
            update_success = self.db.update_trade(trade_id, changes)
            
            if update_success:
                logger.logger.info(f"Modified trade {trade_id} parameters: {changes}")
                return True
            else:
                logger.logger.warning(f"Failed to modify trade {trade_id} parameters")
                return False
            
        except Exception as e:
            logger.log_error(f"Modify Trade - {trade_id}", str(e))
            return False
    
    def set_risk_parameters(self, position_size_pct: float = None, stop_loss_pct: float = None,
                          take_profit_pct: float = None, max_trades: int = None) -> bool:
        """
        Update risk management parameters
        
        Args:
            position_size_pct: Position size as percentage of account
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_trades: Maximum concurrent trades
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if position_size_pct is not None:
                self.position_size_percentage = max(0.1, min(10.0, position_size_pct))
            
            if stop_loss_pct is not None:
                self.stop_loss_percentage = max(0.5, min(10.0, stop_loss_pct))
            
            if take_profit_pct is not None:
                self.take_profit_percentage = max(0.5, min(20.0, take_profit_pct))
            
            if max_trades is not None:
                self.max_concurrent_trades = max(1, min(10, max_trades))
            
            logger.logger.info(
                f"Updated risk parameters: Position={self.position_size_percentage}%, "
                f"SL={self.stop_loss_percentage}%, TP={self.take_profit_percentage}%, "
                f"MaxTrades={self.max_concurrent_trades}"
            )
            
            return True
            
        except Exception as e:
            logger.log_error("Set Risk Parameters", str(e))
            return False
        
    class PaperTradingExchange:
        """
        Paper trading exchange simulator
        """
        
        def __init__(self):
            """Initialize paper trading exchange"""
            self.orders = {}
            self.balance = float(os.getenv('PAPER_TRADING_BALANCE', '10000.0'))
            self.positions = {}
            self.order_id_counter = 1000
        
        def create_market_order(self, symbol: str, side: str, amount: float, 
                              price: float = None, leverage: int = 1) -> Dict[str, Any]:
            """
            Create a simulated market order
            
            Args:
                symbol: Trading symbol
                side: Order side ('buy' or 'sell')
                amount: Order amount
                price: Optional price (if None, uses current market price)
                leverage: Leverage to use
                
            Returns:
                Order details
            """
            # Generate order ID
            order_id = f"paper_{self.order_id_counter}"
            self.order_id_counter += 1
            
            # Use provided price or simulate market price
            if price is None:
                # In a real implementation, we'd get the current market price
                price = 100.0  # Placeholder price
            
            # Calculate order value
            order_value = amount * price
            
            # Update positions
            position_key = f"{symbol}_{leverage}x"
            
            if side == 'buy':
                # Add to position
                if position_key in self.positions:
                    # Average in
                    current_amount = self.positions[position_key]['amount']
                    current_price = self.positions[position_key]['price']
                    
                    # Calculate new average price
                    total_value = (current_amount * current_price) + order_value
                    total_amount = current_amount + amount
                    avg_price = total_value / total_amount if total_amount > 0 else price
                    
                    self.positions[position_key] = {
                        'symbol': symbol,
                        'side': 'long',
                        'amount': total_amount,
                        'price': avg_price,
                        'leverage': leverage,
                        'value': total_value,
                        'timestamp': int(time.time() * 1000)
                    }
                else:
                    # New position
                    self.positions[position_key] = {
                        'symbol': symbol,
                        'side': 'long',
                        'amount': amount,
                        'price': price,
                        'leverage': leverage,
                        'value': order_value,
                        'timestamp': int(time.time() * 1000)
                    }
            else:  # sell
                # Reduce or reverse position
                if position_key in self.positions:
                    current_amount = self.positions[position_key]['amount']
                    current_price = self.positions[position_key]['price']
                    
                    if current_amount >= amount:
                        # Reduce position
                        new_amount = current_amount - amount
                        if new_amount > 0:
                            # Update position
                            self.positions[position_key]['amount'] = new_amount
                            self.positions[position_key]['value'] = new_amount * current_price
                        else:
                            # Close position
                            del self.positions[position_key]
                    else:
                        # Reverse position (short)
                        new_amount = amount - current_amount
                        self.positions[position_key] = {
                            'symbol': symbol,
                            'side': 'short',
                            'amount': new_amount,
                            'price': price,
                            'leverage': leverage,
                            'value': new_amount * price,
                            'timestamp': int(time.time() * 1000)
                        }
                else:
                    # New short position
                    self.positions[position_key] = {
                        'symbol': symbol,
                        'side': 'short',
                        'amount': amount,
                        'price': price,
                        'leverage': leverage,
                        'value': order_value,
                        'timestamp': int(time.time() * 1000)
                    }
            
            # Record the order
            order = {
                'id': order_id,
                'status': 'closed',
                'symbol': symbol,
                'type': 'market',
                'side': side,
                'amount': amount,
                'price': price,
                'leverage': leverage,
                'cost': order_value,
                'timestamp': int(time.time() * 1000)
            }
            
            self.orders[order_id] = order
            
            return order
        
        def fetch_balance(self) -> Dict[str, Any]:
            """
            Get account balance
            
            Returns:
                Balance information
            """
            return {
                'total': {'USDT': self.balance},
                'used': {'USDT': sum(p['value'] for p in self.positions.values())},
                'free': {'USDT': self.balance - sum(p['value'] for p in self.positions.values())}
            }
        
        def fetch_positions(self) -> List[Dict[str, Any]]:
            """
            Get open positions
            
            Returns:
                List of position details
            """
            return list(self.positions.values())
        
        def fetch_order(self, order_id: str) -> Dict[str, Any]:
            """
            Get order details
            
            Args:
                order_id: Order ID
                
            Returns:
                Order details
            """
            return self.orders.get(order_id, {})
        
        def set_leverage(self, leverage: int, symbol: str) -> Dict[str, Any]:
            """
            Set leverage for a symbol (no-op for paper trading)
            
            Args:
                leverage: Leverage value
                symbol: Trading symbol
                
            Returns:
                Success response
            """
            return {'leverage': leverage, 'symbol': symbol, 'success': True}
