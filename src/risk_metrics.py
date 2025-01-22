from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import deque
import logging
from portfolio import PortfolioManager

class RiskMetricsCalculator:
    """Calculates and tracks trading risk metrics and performance indicators.
    
    This class provides comprehensive risk analysis including Sharpe ratio,
    drawdown, Value at Risk (VaR), and various other risk-adjusted performance
    metrics that help evaluate trading strategy effectiveness.
    """
    
    def __init__(self,
                 portfolio_manager: PortfolioManager,
                 window_size: int = 100,
                 risk_free_rate: float = 0.02/365,  # Daily risk-free rate
                 var_confidence: float = 0.95):
                 
        self.portfolio = portfolio_manager
        self.window_size = window_size
        self.risk_free_rate = risk_free_rate
        self.var_confidence = var_confidence
        self.logger = logging.getLogger("RiskMetrics")
        
        # Historical tracking
        self.returns = deque(maxlen=window_size)
        self.portfolio_values = deque(maxlen=window_size)
        self.drawdowns = deque(maxlen=window_size)
        self.position_sizes = deque(maxlen=window_size)
        
        # Peak tracking for drawdown calculation
        self.peak_value = portfolio_manager.initial_balance
        
    def update_metrics(self, current_price: float) -> Dict[str, float]:
        """Update all risk metrics with current portfolio state."""
        try:
            # Get current portfolio state
            current_value = self.portfolio.state.total_value_in_erg
            
            # Calculate return
            if self.portfolio_values:
                current_return = (current_value - self.portfolio_values[-1]) / self.portfolio_values[-1]
            else:
                current_return = 0.0
                
            # Update peak value and calculate drawdown
            self.peak_value = max(self.peak_value, current_value)
            current_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0.0
            
            # Update historical trackers
            self.returns.append(current_return)
            self.portfolio_values.append(current_value)
            self.drawdowns.append(current_drawdown)
            self.position_sizes.append(self.portfolio.get_position_size())
            
            # Calculate comprehensive metrics
            metrics = {
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'sortino_ratio': self._calculate_sortino_ratio(),
                'max_drawdown': max(self.drawdowns) if self.drawdowns else 0.0,
                'value_at_risk': self._calculate_var(),
                'volatility': self._calculate_volatility(),
                'avg_position_size': np.mean(self.position_sizes) if self.position_sizes else 0.0,
                'position_size_volatility': np.std(self.position_sizes) if len(self.position_sizes) > 1 else 0.0,
                'current_drawdown': current_drawdown,
                'profit_factor': self._calculate_profit_factor(),
                'win_rate': self._calculate_win_rate()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
            return {}
            
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio using return history."""
        if len(self.returns) < 2:
            return 0.0
            
        excess_returns = np.array(self.returns) - self.risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
            
        # Annualize Sharpe ratio
        sharpe = np.sqrt(365) * (np.mean(excess_returns) / np.std(excess_returns))
        return float(np.clip(sharpe, -100, 100))
        
    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio focusing on downside deviation."""
        if len(self.returns) < 2:
            return 0.0
            
        excess_returns = np.array(self.returns) - self.risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
            
        # Annualize Sortino ratio
        sortino = np.sqrt(365) * (np.mean(excess_returns) / np.std(downside_returns))
        return float(np.clip(sortino, -100, 100))
        
    def _calculate_volatility(self) -> float:
        """Calculate return volatility."""
        if len(self.returns) < 2:
            return 0.0
        return float(np.std(self.returns))
        
    def _calculate_var(self) -> float:
        """Calculate Value at Risk at specified confidence level."""
        if len(self.returns) < 2:
            return 0.0
            
        # Use historical VaR calculation
        sorted_returns = np.sort(self.returns)
        index = int((1 - self.var_confidence) * len(sorted_returns))
        return float(abs(sorted_returns[index]))
        
    def _calculate_profit_factor(self) -> float:
        """Calculate ratio of gross profits to gross losses."""
        if not self.returns:
            return 0.0
            
        profits = sum(r for r in self.returns if r > 0)
        losses = abs(sum(r for r in self.returns if r < 0))
        
        return float(profits / losses) if losses > 0 else 0.0
        
    def _calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades."""
        if not self.returns:
            return 0.0
            
        profitable_trades = sum(1 for r in self.returns if r > 0)
        return float(profitable_trades / len(self.returns))
        
    def get_current_metrics_summary(self) -> Dict[str, float]:
        """Get a summary of current risk metrics suitable for logging."""
        metrics = {
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'current_drawdown': self.drawdowns[-1] if self.drawdowns else 0.0,
            'max_drawdown': max(self.drawdowns) if self.drawdowns else 0.0,
            'volatility': self._calculate_volatility(),
            'win_rate': self._calculate_win_rate()
        }
        return metrics
        
    def validate_risk_limits(self, 
                           max_drawdown: float = 0.5,
                           max_volatility: float = 0.1) -> Tuple[bool, str]:
        """Check if current risk metrics are within acceptable limits."""
        current_metrics = self.get_current_metrics_summary()
        
        if current_metrics['max_drawdown'] > max_drawdown:
            return False, f"Max drawdown ({current_metrics['max_drawdown']:.2%}) exceeded limit ({max_drawdown:.2%})"
            
        if current_metrics['volatility'] > max_volatility:
            return False, f"Volatility ({current_metrics['volatility']:.2%}) exceeded limit ({max_volatility:.2%})"
            
        return True, "Risk metrics within limits"