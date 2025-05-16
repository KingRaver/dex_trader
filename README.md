# 🚀 Dex Trader: Advanced Cryptocurrency Trading System

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Status](https://img.shields.io/badge/status-active-green.svg)

Tokenetics is a comprehensive, AI-powered cryptocurrency trading system that combines market analysis, sentiment detection, and multi-timeframe prediction with automated trade execution across both centralized and decentralized exchanges.

## ✨ Features

### 🧠 Intelligent Analysis & Prediction
- **Multi-timeframe Analysis**: Separate prediction engines for 1h, 24h, and 7d horizons
- **Enhanced Prediction Engine**: Combines technical analysis, market sentiment, and volume profiling
- **Smart Money Detection**: Identifies potential institutional movements and accumulation patterns
- **Market Correlation Analysis**: Automatically compares token performance against broader market trends
- **Volume Anomaly Detection**: Finds unusual trading patterns that often precede significant price moves

### 📊 Trading Execution
- **Multi-Exchange Support**: Integrated with major CeFi platforms (Binance, Bybit) and DeFi protocols
- **Risk Management**: Configurable position sizing, stop-loss, take-profit and leverage controls
- **Paper Trading**: Test strategies without risking real capital
- **Performance Tracking**: Comprehensive trade history with detailed analytics

### 🌐 Social & Community Analysis
- **Timeline Scraping**: Monitor market-related discussions to inform trading decisions
- **Sentiment Analysis**: Detect market mood shifts for timing entries and exits
- **Reply Capabilities**: Engage with relevant content on trading platforms
- **Educational Content**: Generate technical explainers for broader market context

### 🔧 System Design
- **Modular Architecture**: Easily extendable for new analysis methods and exchange integrations
- **Database Integration**: Store predictions, trades, and performance metrics
- **Async Processing**: Background processing for prediction generation
- **Extensive Logging**: Detailed operation logs for monitoring and debugging

## 🏗️ Architecture

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│  CryptoAnalysisBot  │───▶│  PredictionEngine   │───▶│TimeframePredictions │
│                     │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                          │
         ▼                           ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│                     │    │                     │    │                     │
│ContentAnalyzer/Scraper  │    │  MarketDataHandler  │    │TradingExecutionEngine│
│                     │    │                     │    │                     │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                          │
         └───────────────────────────┼──────────────────────────┘
                                     ▼
                           ┌─────────────────────┐
                           │                     │
                           │     Database        │
                           │                     │
                           └─────────────────────┘
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required Python packages (`requirements.txt`):
  - ccxt
  - numpy
  - pandas
  - requests
  - web3
  - anthropic (for AI integration)
  - selenium (for timeline scraping)
  - PyMySQL/SQLite (database support)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dex_trader.git
cd dex_trader
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Copy the example env file
cp .env.example .env

# Edit with your exchange API keys and configuration
nano .env
```

4. Initialize the database:
```bash
python setup_database.py
```

### Configuration

The system is highly configurable through environment variables or a config file:

```python
# config.py example
TWITTER_USERNAME = "your_twitter_username"
TWITTER_PASSWORD = "your_twitter_password"

# Exchange API keys
BINANCE_API_KEY = "your_binance_key"
BINANCE_API_SECRET = "your_binance_secret"
BYBIT_API_KEY = "your_bybit_key"
BYBIT_API_SECRET = "your_bybit_secret"

# DeFi wallet config
ETH_PRIVATE_KEY = "your_eth_private_key"
ETH_WALLET_ADDRESS = "your_eth_address"

# Risk management settings
POSITION_SIZE_PERCENTAGE = 1.0  # 1% of account
STOP_LOSS_PERCENTAGE = 2.0
TAKE_PROFIT_PERCENTAGE = 4.0
MAX_TRADES_PER_TOKEN = 1
MAX_CONCURRENT_TRADES = 3
TRADING_MODE = "paper"  # Options: paper, live, simulation
```

### Running the System

Start the CryptoTrader:

```bash
python main.py
```

For a production environment, it's recommended to run as a service:

```bash
# Example systemd service
sudo nano /etc/systemd/system/dextrader.service

# Add the following
[Unit]
Description=Dex Trader Service
After=network.target

[Service]
User=yourusername
WorkingDirectory=/path/to/dex_trader
ExecStart=/usr/bin/python /path/to/dex_trader/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable and start the service
sudo systemctl enable dextrader
sudo systemctl start dextrader
```

## 📊 Performance Monitoring

The system stores detailed performance metrics that can be accessed via the database:

```python
# Example: Get trading performance
performance = trader.get_trade_performance(days=30)
print(f"Win rate: {performance['win_rate']}%")
print(f"Average profit: {performance['avg_profit']}%")
print(f"Total PnL: {performance['total_pnl']}%")

# Example: Get prediction accuracy by timeframe
accuracy = trader.get_prediction_performance(timeframe="1h")
```

## 🧩 Extending the System

### Adding New Exchanges

1. Add exchange configuration in `config.py`
2. Implement exchange-specific methods in `TradingExecutionEngine`
3. Register the exchange in the `initialize_exchanges` method

### Adding New Analysis Methods

1. Create a new analysis method in the appropriate class
2. Integrate it with the prediction engine
3. Add thresholds and configuration parameters

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

Project Link: [https://github.com/kingraver/dex_trader](https://github.com/kingraver/dex_trader)

---

**Disclaimer**: Trading cryptocurrencies involves significant risk. This software is for educational purposes only. Always do your own research before trading. Past performance is not indicative of future results.
