import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional

from portfolio import PortfolioManager
from trading_actions import TradingExecutor, Action
from reward_function import RewardCalculator
from state_encoder import StateEncoder
from risk_metrics import RiskMetricsCalculator
from env_monitor import EnvironmentMonitor

class DEXTradingEnv(gym.Env):
    """DEX Trading Environment that implements the OpenAI Gym interface.
    
    This environment simulates a DEX trading scenario where an agent can execute
    trades using ERG and tokens. It provides a realistic trading experience with
    comprehensive state information, risk management, and performance monitoring.
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 10.0,
                 transaction_fee: float = 0.002,
                 max_position: float = 25.0,
                 reward_scaling: float = 1.0,
                 risk_free_rate: float = 0.02/365):
        """Initialize the trading environment."""
        super().__init__()
        
        # Set up logging
        self.logger = logging.getLogger("DEXTradingEnv")
        
        # Store configuration
        self.data = data
        self.current_step = 0
        
        # Initialize core components
        self.portfolio = PortfolioManager(initial_balance=initial_balance)
        self.trading_executor = TradingExecutor(
            portfolio_manager=self.portfolio,
            transaction_fee=transaction_fee,
            max_trade_size=0.25  # Limit single trade to 25% of portfolio
        )
        
        self.reward_calculator = RewardCalculator(
            portfolio_manager=self.portfolio,
            base_transaction_fee=transaction_fee,
            reward_scaling=reward_scaling
        )
        
        self.state_encoder = StateEncoder(
            portfolio_manager=self.portfolio,
            price_feature_window=20,
            volume_feature_window=5
        )
        
        self.risk_calculator = RiskMetricsCalculator(
            portfolio_manager=self.portfolio,
            risk_free_rate=risk_free_rate
        )
        
        self.monitor = EnvironmentMonitor(
            max_portfolio_value=initial_balance * 100,  # Set reasonable limits
            min_portfolio_value=initial_balance * 0.1,
            max_position_size=0.95,
            max_drawdown=0.5
        )
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Box(
            low=-5,
            high=5,
            shape=(self.state_encoder.get_observation_dim(),),
            dtype=np.float32
        )
        
        # Store last info for monitoring
        self.last_info = None
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        try:
            # Get current market data
            current_data = self.data.iloc[self.current_step]
            
            # Execute trading action
            success, message = self.trading_executor.execute_trade(
                Action(action),
                current_data['price']
            )
            
            # Calculate reward
            reward = self.reward_calculator.calculate_reward(Action(action))
            
            # Update risk metrics
            risk_metrics = self.risk_calculator.update_metrics(current_data['price'])
            
            # Get new state observation
            observation = self.state_encoder.encode_state(current_data)
            
            # Check environment state
            is_valid, monitor_message = self.monitor.check_state(
                self.portfolio.state.total_value_in_erg,
                self.portfolio.get_position_size(),
                action,
                reward
            )
            
            # Determine termination conditions
            terminated = False
            if not is_valid:
                terminated = True
                reward = -1.0  # Penalty for invalid state
            elif not success:
                terminated = True
                reward = -1.0  # Penalty for failed trade
            elif self.current_step >= len(self.data) - 1:
                terminated = True
            
            # Move to next step
            self.current_step += 1
            
            # Prepare info dict
            info = {
                'portfolio_value': self.portfolio.state.total_value_in_erg,
                'current_price': current_data['price'],
                'last_action': Action(action).name,
                'position_size': self.portfolio.get_position_size(),
                'success': success,
                'message': message,
                'monitor_message': monitor_message,
                'risk_metrics': risk_metrics,
                'monitoring_stats': self.monitor.get_statistics()
            }
            
            self.last_info = info
            return observation, reward, terminated, False, info
            
        except Exception as e:
            self.logger.error(f"Error in environment step: {e}")
            return self.state_encoder.encode_state(self.data.iloc[self.current_step]), -1.0, True, False, {
                'error': str(e),
                'portfolio_value': self.portfolio.state.total_value_in_erg
            }
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        
        # Reset all components
        self.portfolio = PortfolioManager(initial_balance=self.portfolio.initial_balance)
        self.reward_calculator = RewardCalculator(
            portfolio_manager=self.portfolio,
            base_transaction_fee=self.trading_executor.transaction_fee
        )
        
        # Get initial state
        initial_data = self.data.iloc[0]
        observation = self.state_encoder.encode_state(initial_data)
        
        info = {
            'portfolio_value': self.portfolio.state.total_value_in_erg,
            'current_price': initial_data['price'],
            'reset_message': 'Environment reset successfully'
        }
        
        self.last_info = info
        return observation, info
        
    def render(self):
        """Render the current state of the environment."""
        if self.render_mode == "human":
            print("\nEnvironment State:")
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: {self.portfolio.state.total_value_in_erg:.4f} ERG")
            print(f"Position Size: {self.portfolio.get_position_size():.2%}")
            if self.last_info:
                print(f"Last Action: {self.last_info['last_action']}")
                print(f"Current Price: {self.last_info['current_price']:.6f}")
                
                if 'risk_metrics' in self.last_info:
                    print("\nRisk Metrics:")
                    metrics = self.last_info['risk_metrics']
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
                    print(f"  Current Drawdown: {metrics['current_drawdown']:.2%}")
                    print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
                    print(f"  Volatility: {metrics['volatility']:.4f}")
                    print(f"  Win Rate: {metrics['win_rate']:.2%}")
                    
                if 'monitoring_stats' in self.last_info:
                    print("\nMonitoring Stats:")
                    stats = self.last_info['monitoring_stats']
                    for key, value in stats.items():
                        if isinstance(value, dict):
                            print(f"  {key}:")
                            for subkey, subvalue in value.items():
                                print(f"    {subkey}: {subvalue}")
                        else:
                            print(f"  {key}: {value}")
    
    def get_valid_actions(self) -> np.ndarray:
        """Return mask of valid actions based on current state."""
        valid_actions = np.ones(self.action_space.n, dtype=np.int8)
        
        # Check if we can buy (need sufficient ERG balance)
        if self.portfolio.state.erg_balance < self.trading_executor.transaction_fee:
            valid_actions[Action.BUY.value] = 0
        
        # Check if we can sell (need tokens to sell)
        if self.portfolio.state.token_balance <= 0:
            valid_actions[Action.SELL.value] = 0
        
        # Check if rebalancing is valid (need sufficient total value)
        if self.portfolio.state.total_value_in_erg < self.portfolio.initial_balance * 0.5:
            valid_actions[Action.REBALANCE.value] = 0
        
        return valid_actions
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get comprehensive portfolio statistics."""
        metrics = self.risk_calculator.get_current_metrics_summary()
        
        stats = {
            'total_value': self.portfolio.state.total_value_in_erg,
            'erg_balance': self.portfolio.state.erg_balance,
            'token_balance': self.portfolio.state.token_balance,
            'position_size': self.portfolio.get_position_size(),
            'total_return': (self.portfolio.state.total_value_in_erg / 
                           self.portfolio.initial_balance - 1),
            'current_price': self.portfolio.state.current_price
        }
        
        # Add risk metrics
        stats.update(metrics)
        
        return stats
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get the names and current values of all observation features."""
        current_observation = self.state_encoder.encode_state(
            self.data.iloc[self.current_step]
        )
        
        feature_names = self.state_encoder.get_feature_names()
        return dict(zip(feature_names, current_observation))
    
    def close(self):
        """Clean up resources."""
        pass
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        if seed is not None:
            np.random.seed(seed)
            
    def _validate_data(self) -> bool:
        """Validate that the input data has all required columns."""
        required_columns = {'price', 'volume_erg'}
        return all(col in self.data.columns for col in required_columns)