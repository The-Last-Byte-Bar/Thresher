from typing import Optional, Dict
import numpy as np
import logging
from portfolio import PortfolioManager
from trading_actions import Action

class RewardCalculator:
    """Calculates trading rewards with multiple components to encourage balanced behavior.
    
    This class implements a sophisticated reward function that considers multiple aspects
    of trading performance including portfolio returns, position management, risk control,
    and trading costs. Each component is weighted to create a balanced incentive structure.
    """
    
    def __init__(self,
                 portfolio_manager: PortfolioManager,
                 base_transaction_fee: float = 0.002,
                 target_position_size: float = 0.5,
                 max_position_penalty: float = 0.1,
                 reward_scaling: float = 1.0,
                 risk_free_rate: float = 0.02/365):  # Daily risk-free rate
        
        self.portfolio = portfolio_manager
        self.transaction_fee = base_transaction_fee
        self.target_position = target_position_size
        self.max_position_penalty = max_position_penalty
        self.reward_scaling = reward_scaling
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger("RewardCalculator")
        
        # State for tracking returns and calculating metrics
        self.previous_portfolio_value = portfolio_manager.state.total_value_in_erg
        self.previous_price = portfolio_manager.state.current_price
        self.returns_history = []
        
    def calculate_reward(self, action: Action) -> float:
        """Calculate the total reward combining multiple trading performance factors."""
        try:
            # Calculate base portfolio return
            portfolio_return = self._calculate_portfolio_return()
            
            # Calculate trading action reward
            action_reward = self._calculate_action_reward(action)
            
            # Calculate position management reward/penalty
            position_reward = self._calculate_position_reward()
            
            # Calculate risk-adjusted components
            risk_penalty = self._calculate_risk_penalty()
            
            # Combine all components with appropriate weights
            total_reward = (
                portfolio_return * 0.3 +          # Long-term portfolio growth
                action_reward * 0.3 +             # Immediate trading decisions
                position_reward * 0.2 +           # Position management
                risk_penalty * 0.2                # Risk control
            )
            
            # Scale the final reward
            scaled_reward = total_reward * self.reward_scaling
            
            # Store return for historical tracking
            self.returns_history.append(portfolio_return)
            if len(self.returns_history) > 100:  # Keep last 100 returns
                self.returns_history.pop(0)
            
            # Update previous values for next calculation
            self.previous_portfolio_value = self.portfolio.state.total_value_in_erg
            self.previous_price = self.portfolio.state.current_price
            
            return float(scaled_reward)
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {e}")
            return 0.0
    
    def _calculate_portfolio_return(self) -> float:
        """Calculate the basic portfolio return component."""
        if self.previous_portfolio_value <= 0:
            return 0.0
            
        return (
            self.portfolio.state.total_value_in_erg - self.previous_portfolio_value
        ) / self.previous_portfolio_value
    
    def _calculate_action_reward(self, action: Action) -> float:
        """Calculate reward/penalty based on the specific trading action taken."""
        if action == Action.HOLD:
            return self._calculate_hold_reward()
            
        # Calculate immediate profit/loss from the trade
        trade_pnl = self.portfolio.state.total_value_in_erg - self.previous_portfolio_value
        position_size = self.portfolio.get_position_size()
        
        if action in [Action.BUY, Action.SELL]:
            # Scale reward based on position size and profit
            if trade_pnl > 0:
                # Reward profitable trades less as position size grows
                immediate_reward = trade_pnl * (1.0 - position_size)
            else:
                # Penalize losses more for larger positions
                immediate_reward = trade_pnl * (1.0 + position_size)
                
            # Apply transaction cost penalty
            immediate_reward -= self.transaction_fee * self.previous_portfolio_value
            
            return immediate_reward
            
        elif action == Action.REBALANCE:
            # Higher fee for rebalancing
            rebalance_fee = self.transaction_fee * 1.5
            return trade_pnl - (rebalance_fee * self.previous_portfolio_value)
            
        return 0.0
    
    def _calculate_hold_reward(self) -> float:
        """Calculate reward/penalty for holding a position."""
        if self.previous_price <= 0:
            return 0.0
            
        # Calculate price movement
        price_change = (
            self.portfolio.state.current_price / self.previous_price - 1
        )
        
        # Small reward for holding during low volatility
        if abs(price_change) < 0.01:  # 1% threshold
            return 0.001  # Small positive reward
        else:
            # Small penalty for not acting during high volatility
            return -0.001 * abs(price_change)
    
    def _calculate_position_reward(self) -> float:
        """Calculate reward/penalty based on position management."""
        current_position = self.portfolio.get_position_size()
        
        # Penalize deviation from target position size
        position_penalty = -abs(current_position - self.target_position)
        
        # Additional penalty for very large positions
        if current_position > 0.8:  # 80% threshold
            position_penalty *= 2  # Double penalty for over-concentration
            
        return position_penalty * self.max_position_penalty
    
    def _calculate_risk_penalty(self) -> float:
        """Calculate risk-adjusted penalty based on portfolio metrics."""
        # Get current drawdown
        drawdown = self.portfolio.get_drawdown()
        
        # Calculate rolling volatility if enough history
        if len(self.returns_history) >= 20:
            volatility = np.std(self.returns_history[-20:])
        else:
            volatility = 0.0
            
        # Combine risk metrics
        risk_penalty = -(drawdown * 0.5 + volatility * 0.5)
        
        return risk_penalty
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current reward-related metrics for monitoring."""
        return {
            'portfolio_return': self._calculate_portfolio_return(),
            'drawdown': self.portfolio.get_drawdown(),
            'volatility': np.std(self.returns_history[-20:]) if len(self.returns_history) >= 20 else 0.0,
            'position_size': self.portfolio.get_position_size()
        }