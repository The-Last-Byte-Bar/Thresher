# DEX Trading System with Reinforcement Learning

## System Overview
This project implements an automated trading system for the Ergo blockchain DEX using reinforcement learning. The system consists of several main components:
1. Data Collection & Analysis Pipeline
2. Trading Environment
3. Reinforcement Learning Training Pipeline
4. Dashboard & Monitoring System

## Key Components

### 1. Data Collection System
#### Node Client (`client.py`)
- Manages blockchain node communication
- Fetches transaction data within specified block ranges
- Handles API authentication and rate limiting
- Provides robust error handling and retry logic

#### DEX Analyzer (`dex_analyzer.py`)
- Processes DEX transactions
- Identifies and analyzes swap events
- Tracks token information and metrics
- Handles token registry and metadata

#### DEX Monitor (`dex_monitor.py`)
- Coordinates data collection and analysis
- Manages historical data retrieval
- Updates metrics in real-time
- Exports analytics and metrics

### 2. Metrics & Analytics
#### Token Registry
- Maintains token metadata
- Tracks token volumes and trades
- Identifies LP tokens
- Manages token decimals and scaling

#### Trading Metrics (`metrics.py`)
- Calculates comprehensive trading metrics:
  - Price movements and volatility
  - Volume analysis
  - Liquidity metrics
  - Trading patterns
  - Risk metrics (RSI, Bollinger Bands)
- Exports metrics to CSV for analysis

### 3. Trading Environment (`env.py`)
- Implements OpenAI Gymnasium interface
- Manages portfolio state and actions
- Calculates sophisticated rewards including:
  - Portfolio returns
  - Risk-adjusted metrics
  - Trading costs
  - Volatility penalties
- Handles position sizing and risk limits

### 4. Training Pipeline (`train_agent.py`)
- Implements PPO-based training
- Handles data preprocessing
- Manages model training and evaluation
- Integrates with Weights & Biases for monitoring
- Implements callbacks for tracking metrics

### 5. Dashboard (`dex_dash.py`)
- Interactive trading dashboard using Dash
- Real-time price and volume charts
- Trading metrics visualization
- Portfolio performance monitoring
- Technical indicators display

## Implementation Details

### Data Models
```python
@dataclass
class SwapEvent:
    tx_id: str
    timestamp: int
    height: int
    input_token: str
    output_token: str
    input_amount: float
    output_amount: float
    price: float
    fee_erg: float
    # Additional fields for analysis
    price_impact: float
    erg_liquidity: float
    token_liquidity: float

@dataclass
class TokenMetrics:
    token_name: str
    current_price: float
    price_ma_5: float
    price_ma_20: float
    volume_ma_5: float
    price_volatility: float
    # Advanced metrics
    rsi_14: float
    bb_position: Dict[str, float]
    market_impact: float
```

### Environment Design
The trading environment (`DEXTradingEnv`) implements:
- State space: 23 dimensions including price, volume, and portfolio metrics
- Action space: HOLD, BUY, SELL, REBALANCE
- Reward function: Combines multiple components:
  ```python
  reward = (
      portfolio_return * 0.4 +    # Base performance
      window_reward * 0.2 +       # Medium-term performance
      sharpe_component * 0.2 +    # Risk-adjusted performance
      drawdown_penalty * 0.1 +    # Drawdown penalty
      volatility_penalty * 0.1 +  # Volatility penalty
      action_penalty              # Trading cost penalty
  )
  ```

### Project Structure
```
project/
├── data/
│   ├── metrics/
│   └── swaps/
├── client.py
├── dex_analyzer.py
├── dex_monitor.py
├── dex_dash.py
├── env.py
├── metrics.py
├── swap_storage.py
├── train_agent.py
└── export_metrics.py
```

### Required Dependencies
```
torch
stable-baselines3
gymnasium
dash
plotly
numpy
pandas
aiohttp
wandb
```

## Key Features

### Data Collection & Analysis
- Real-time swap event tracking
- Token registry management
- Advanced metrics calculation
- Data export and storage

### Trading Environment
- Comprehensive state representation
- Risk-aware reward function
- Position management
- Trading fee simulation

### Training Process
1. Data preprocessing
2. Environment setup and validation
3. PPO model training
4. Performance evaluation
5. Model export and metrics logging

### Dashboard & Monitoring
- Interactive price charts
- Volume analysis
- Technical indicators
- Portfolio metrics
- Trading signals

## Usage Examples

### 1. Data Collection
```python
monitor = await DEXMonitor.create(NODE_URL, DEX_ADDRESS)
transactions = await monitor.fetch_time_range(days_back=30)
swaps = await monitor.analyze_transactions(transactions)
analytics = monitor.save_and_analyze_swaps(swaps)
```

### 2. Training
```python
# Prepare environment
env = DEXTradingEnv(data=training_data)
env = DummyVecEnv([lambda: env])
env = VecNormalize(env)

# Create and train model
model = PPO("MlpPolicy", env)
model.learn(total_timesteps=1_000_000)
```

### 3. Dashboard
```python
dashboard = EnhancedDEXDashboard()
dashboard.load_data()
dashboard.run_server()
```

## Extension Points

### 1. Advanced Features
- Multi-token portfolio optimization
- Advanced order types
- Market making strategies
- Social trading features

### 2. Potential Improvements
- Enhanced state representation
- More sophisticated reward functions
- Advanced risk management
- Additional technical indicators

### 3. Infrastructure
- Distributed data collection
- Real-time model updates
- Advanced monitoring systems
- Automated retraining pipeline