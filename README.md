# DEX Trading System with Reinforcement Learning

## System Overview
This project implements an automated trading system for the Ergo blockchain DEX using reinforcement learning. The system consists of two main components:
1. Data Collection & Processing Pipeline
2. Reinforcement Learning Trading Agent

## Key Components

### 1. Data Collection System (`dex_collector.py`)
- Collects and processes DEX swap transactions
- Tracks price movements and trading patterns
- Maintains trader profiles and analytics
- Uses SQLite for data storage

Required files from base project:
- `clients.py` (Explorer client)
- `models.py` (Data models)
- `services.py` (Helper services)

Database Schema:
```sql
- swaps: Transaction data
- address_profiles: Trader behavior data
- price_history: Token price tracking
```

### 2. RL Environment (`dex_environment.py`)
- Implements OpenAI Gym interface
- Manages state representation
- Handles action execution
- Calculates rewards

### 3. Training Pipeline (`train.py`)
- Manages offline training process
- Handles data preprocessing
- Implements training loop
- Saves and evaluates models

### 4. Live Trading System (`live_trader.py`)
- Implements paper trading
- Manages live trading execution
- Handles risk management
- Monitors performance

## Implementation Requirements

### Configuration (`config.yaml`)
```yaml
dex:
  address: "DEX_ADDRESS"
  tokens: ["TOKEN_IDs"]
  
training:
  episode_length: 24  # hours
  lookback_window: 100
  batch_size: 64
  
agent:
  model_type: "DQN"
  learning_rate: 0.001
  discount_factor: 0.99
  
trading:
  max_position_size: 100
  stop_loss: 0.02
  take_profit: 0.05
```

### Project Structure
```
project/
├── data/
│   └── dex_data.db
├── models/
│   └── saved_models/
├── src/
│   ├── collectors/
│   │   └── dex_collector.py
│   ├── environment/
│   │   └── dex_environment.py
│   ├── agents/
│   │   └── dqn_agent.py
│   ├── trading/
│   │   └── live_trader.py
│   └── utils/
│       ├── data_preprocessing.py
│       └── risk_management.py
├── config.yaml
└── requirements.txt
```

### Required Dependencies
```
torch
gym
numpy
pandas
sqlite3
aiohttp
pyyaml
```

## Key Features

### Data Collection
- Real-time swap tracking
- Price calculation from swap ratios
- Trading pattern analysis
- Address profiling

### RL Environment
- State: Price, volume, liquidity metrics
- Actions: Buy, sell, hold
- Rewards: PnL with risk adjustment
- Episode: 24-hour trading periods

### Training Process
1. Historical data collection
2. Episode generation
3. Model training
4. Performance evaluation
5. Paper trading validation

### Live Trading
1. Real-time data processing
2. Model inference
3. Trade execution/simulation
4. Performance monitoring

## Implementation Notes

### State Representation
- Price data (multiple timeframes)
- Volume indicators
- Liquidity metrics
- Trading patterns
- Market sentiment indicators

### Action Space
- Discrete actions (buy/sell/hold)
- Position sizing
- Entry/exit timing

### Reward Function
```python
reward = pnl - risk_penalty + alpha * sharp_ratio
```

### Risk Management
- Position size limits
- Stop-loss mechanisms
- Portfolio diversification
- Drawdown controls

## Usage Examples

1. Data Collection:
```python
collector = DEXDataCollector(explorer_client)
await collector.collect_dex_data(dex_address)
```

2. Training:
```python
env = DEXTradingEnv(historical_data)
agent = DQNAgent(env)
trainer = ModelTrainer(agent, env)
await trainer.train(episodes=1000)
```

3. Live Trading:
```python
trader = LiveTrader(model_path="models/best_model.pth")
await trader.start_paper_trading()
```

## Extension Points

1. Advanced Features
- Multi-token trading
- Portfolio optimization
- Advanced order types
- Market making strategies

2. Improvements
- Enhanced state representation
- More sophisticated reward functions
- Advanced risk management
- Additional trading strategies

This system can be extended with additional features such as:
- Market making capabilities
- Multi-token portfolio management
- Advanced order types
- Social trading features
