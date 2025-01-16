import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Deque
from enum import Enum
from collections import deque

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2
    REBALANCE = 3

@dataclass
class PortfolioState:
    erg_balance: float
    token_balance: float
    token_value_in_erg: float
    total_value_in_erg: float
    last_action: Action
    current_price: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.erg_balance,
            self.token_balance,
            self.token_value_in_erg,
            self.total_value_in_erg,
            self.last_action.value,
            self.current_price
        ])

class DEXTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10.0,
                 transaction_fee: float = 0.002,
                 reward_window: int = 5,
                 trade_size: float = 1.0,
                 max_position: float = 50.0,
                 risk_free_rate: float = 0.02,  # Annual risk-free rate
                 volatility_penalty_factor: float = 0.5,
                 render_mode: str = None):
        super().__init__()
        
        # Data setup
        self.data = data
        self.current_step = 0
        self.reward_window = reward_window
        self.trade_size = trade_size
        self.max_position = max_position
        self.risk_free_rate = risk_free_rate / 365  # Convert to daily
        self.volatility_penalty_factor = volatility_penalty_factor
        self.render_mode = render_mode
        
        # Trading params
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        
        # Risk metrics tracking
        self.returns_history = deque(maxlen=100)  # For Sharpe ratio
        self.peak_value = initial_balance  # For max drawdown
        self.max_drawdown = 0.0
        
        # Portfolio tracking
        self.portfolio = PortfolioState(
            erg_balance=initial_balance,
            token_balance=0.0,
            token_value_in_erg=0.0,
            total_value_in_erg=initial_balance,
            last_action=Action.HOLD,
            current_price=0.0
        )
        
        # Portfolio history for reward calculation
        self.portfolio_history = []
        
        # Action space: HOLD, BUY, SELL, REBALANCE
        self.action_space = spaces.Discrete(len(Action))
        
        # Observation space: combination of market and portfolio features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(23,),  # Added max_drawdown and sharpe_ratio to observation
            dtype=np.float32
        )

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio using return history"""
        if len(self.returns_history) < 2:
            return 0.0
            
        returns = np.array(self.returns_history)
        excess_returns = returns - self.risk_free_rate
        
        if np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.sqrt(365) * (np.mean(excess_returns) / np.std(excess_returns))
        return float(sharpe)

    def _update_max_drawdown(self) -> float:
        """Update and return the maximum drawdown"""
        current_value = self.portfolio.total_value_in_erg
        self.peak_value = max(self.peak_value, current_value)
        
        drawdown = (self.peak_value - current_value) / self.peak_value
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        return self.max_drawdown

    def _calculate_volatility_penalty(self) -> float:
        """Calculate penalty for holding large positions during high volatility"""
        current_data = self.data.iloc[self.current_step]
        
        # Get current volatility and normalize it
        current_volatility = current_data.volatility
        token_position_ratio = self.portfolio.token_value_in_erg / self.portfolio.total_value_in_erg
        
        # Penalize based on position size and volatility
        penalty = (
            current_volatility * 
            token_position_ratio * 
            self.volatility_penalty_factor
        )
        
        return min(penalty, 1.0)  # Cap penalty at 1.0

    def _get_observation(self) -> np.ndarray:
        """Combine market data and portfolio state into observation"""
        current_data = self.data.iloc[self.current_step]
        
        # Market features
        market_features = np.array([
            current_data.price,
            current_data.price_ma_5,
            current_data.price_ma_20,
            current_data.volume_ma_5,
            current_data.volatility,
            current_data.price_momentum_5,
            current_data.price_momentum_20,
            current_data.volume_momentum_5,
            current_data.bb_position_5,
            current_data.bb_position_20,
            current_data.erg_liquidity,
            current_data.token_liquidity,
            current_data.volume_erg,
            current_data.returns,
            current_data.bb_std_20,
            self.max_drawdown,
            self._calculate_sharpe_ratio()
        ], dtype=np.float32)
        
        # Combine with portfolio state
        return np.concatenate([
            market_features,
            self.portfolio.to_array()
        ]).astype(np.float32)

    def _calculate_reward(self) -> float:
        """Calculate reward based on portfolio performance and risk metrics"""
        # Calculate base portfolio return
        portfolio_return = (
            self.portfolio.total_value_in_erg - self.initial_balance
        ) / self.initial_balance
        
        # Update returns history for Sharpe ratio
        if len(self.portfolio_history) > 1:
            daily_return = (
                self.portfolio.total_value_in_erg - 
                self.portfolio_history[-1].total_value_in_erg
            ) / self.portfolio_history[-1].total_value_in_erg
            self.returns_history.append(daily_return)
        
        # Calculate Sharpe ratio component
        sharpe_ratio = self._calculate_sharpe_ratio()
        sharpe_component = np.clip(sharpe_ratio / 10, -1, 1)  # Normalize to [-1, 1]
        
        # Update and get max drawdown penalty
        drawdown_penalty = -self._update_max_drawdown()
        
        # Get volatility penalty
        volatility_penalty = -self._calculate_volatility_penalty()
        
        # Add penalty for excessive trading
        action_penalty = 0
        if self.portfolio.last_action != Action.HOLD:
            action_penalty = -self.transaction_fee
        
        # Window-based reward component
        window_reward = 0
        if len(self.portfolio_history) >= self.reward_window:
            window_start_value = self.portfolio_history[-self.reward_window].total_value_in_erg
            window_return = (self.portfolio.total_value_in_erg - window_start_value) / window_start_value
            window_reward = window_return
        
        # Combine all components
        total_reward = (
            portfolio_return * 0.4 +    # Base portfolio performance
            window_reward * 0.2 +       # Medium-term performance
            sharpe_component * 0.2 +    # Risk-adjusted performance
            drawdown_penalty * 0.1 +    # Drawdown penalty
            volatility_penalty * 0.1 +  # Volatility penalty
            action_penalty              # Trading cost penalty
        )
        
        return float(total_reward)

    def _execute_trade(self, action: Action) -> None:
        """Execute trading action and update portfolio with position limits"""
        current_data = self.data.iloc[self.current_step]
        price = current_data.price
        
        if action == Action.BUY and self.portfolio.erg_balance >= self.trade_size:
            # Check position limit
            potential_token_value = (
                (self.portfolio.token_balance * price) + 
                (self.trade_size * (1 - self.transaction_fee))
            )
            
            if potential_token_value <= self.max_position:
                # Calculate tokens to receive
                tokens_to_receive = (self.trade_size * (1 - self.transaction_fee)) / price
                
                # Update portfolio
                self.portfolio.erg_balance -= self.trade_size
                self.portfolio.token_balance += tokens_to_receive
            
        elif action == Action.SELL and self.portfolio.token_balance > 0:
            # Calculate ERG to receive
            tokens_to_sell = min(
                self.portfolio.token_balance,
                self.trade_size / price
            )
            erg_to_receive = tokens_to_sell * price * (1 - self.transaction_fee)
            
            # Update portfolio
            self.portfolio.token_balance -= tokens_to_sell
            self.portfolio.erg_balance += erg_to_receive
            
        elif action == Action.REBALANCE:
            # Simple 50-50 rebalancing with position limit
            total_value = self.portfolio.total_value_in_erg
            target_erg = total_value / 2
            
            # Ensure target position doesn't exceed max_position
            target_erg = min(target_erg, self.max_position)
            
            current_erg_value = self.portfolio.erg_balance
            current_token_value = self.portfolio.token_balance * price
            
            if current_erg_value > target_erg:
                # Buy tokens
                erg_to_trade = (current_erg_value - target_erg) * (1 - self.transaction_fee)
                tokens_to_receive = erg_to_trade / price
                
                self.portfolio.erg_balance -= erg_to_trade
                self.portfolio.token_balance += tokens_to_receive
                
            elif current_token_value > target_erg:
                # Sell tokens
                tokens_to_sell = (current_token_value - target_erg) / price
                erg_to_receive = tokens_to_sell * price * (1 - self.transaction_fee)
                
                self.portfolio.token_balance -= tokens_to_sell
                self.portfolio.erg_balance += erg_to_receive
        
        # Update portfolio state
        self.portfolio.current_price = price
        self.portfolio.token_value_in_erg = self.portfolio.token_balance * price
        self.portfolio.total_value_in_erg = (
            self.portfolio.erg_balance + self.portfolio.token_value_in_erg
        )
        self.portfolio.last_action = action
        
        # Record portfolio state
        self.portfolio_history.append(PortfolioState(**vars(self.portfolio)))

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        action = Action(action)
        
        # Execute trade
        self._execute_trade(action)
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Move to next step
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        truncated = False
        
        # Get new observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio.total_value_in_erg,
            'current_price': self.portfolio.current_price,
            'last_action': action.name,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self.max_drawdown,
            'volatility_penalty': self._calculate_volatility_penalty()
        }
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.returns_history.clear()
        self.peak_value = self.initial_balance
        self.max_drawdown = 0.0
        
        self.portfolio = PortfolioState(
            erg_balance=self.initial_balance,
            token_balance=0.0,
            token_value_in_erg=0.0,
            total_value_in_erg=self.initial_balance,
            last_action=Action.HOLD,
            current_price=self.data.iloc[0].price
        )
        
        self.portfolio_history = [PortfolioState(**vars(self.portfolio))]
        
        # Additional info dictionary
        info = {
            'initial_portfolio_value': self.initial_balance,
            'initial_price': self.data.iloc[0].price
        }
        
        return self._get_observation(), info

    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            print(f"\nStep: {self.current_step}")
            print(f"Portfolio Value: {self.portfolio.total_value_in_erg:.4f} ERG")
            print(f"ERG Balance: {self.portfolio.erg_balance:.4f}")
            print(f"Token Balance: {self.portfolio.token_balance:.4f}")
            print(f"Current Price: {self.portfolio.current_price:.6f}")
            print(f"Last Action: {self.portfolio.last_action.name}")
            print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.4f}")
            print(f"Max Drawdown: {self.max_drawdown:.4%}")
            print(f"Volatility Penalty: {self._calculate_volatility_penalty():.4f}")
            
    def close(self):
        pass