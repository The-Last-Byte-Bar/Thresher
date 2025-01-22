from typing import Dict, List, Tuple, Any, Optional
from collections import deque, Counter
import numpy as np
import logging
from datetime import datetime
import warnings

# Suppress numpy warnings about overflow, which we'll handle explicitly
warnings.filterwarnings('ignore', category=RuntimeWarning, message='overflow')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class EnvironmentMonitor:
    """Monitors environment state for numerical stability and suspicious behavior."""
    
    def __init__(self, 
                 max_portfolio_value: float = 1e6,
                 min_portfolio_value: float = 1.0,
                 max_position_size: float = 0.95,
                 max_drawdown: float = 0.5,
                 window_size: int = 100):
        """Initialize the environment monitor with safety thresholds."""
        self.max_portfolio_value = max_portfolio_value
        self.min_portfolio_value = min_portfolio_value
        self.max_position_size = max_position_size
        self.max_drawdown_threshold = max_drawdown
        self.window_size = window_size
        
        # Initialize monitoring state with deque for efficient rolling window
        self.portfolio_values = deque(maxlen=window_size)
        self.action_history = deque(maxlen=window_size)
        self.position_sizes = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.peak_value = 0.0
        
        # Track different types of anomalies
        self.anomaly_counts = {
            'portfolio_value': 0,
            'position_size': 0,
            'action_repeat': 0,
            'reward_spike': 0
        }
        
        self.logger = logging.getLogger("EnvironmentMonitor")

    def check_state(self, 
                   portfolio_value: float,
                   position_size: float,
                   action: int,
                   reward: float) -> Tuple[bool, str]:
        """
        Check if the current state is valid and within acceptable bounds.
        
        Args:
            portfolio_value: Current total portfolio value
            position_size: Current position size as fraction
            action: Last action taken
            reward: Last reward received
        
        Returns:
            Tuple of (is_valid, message)
        """
        # Update tracking deques
        self.portfolio_values.append(portfolio_value)
        self.position_sizes.append(position_size)
        self.action_history.append(action)
        self.rewards.append(reward)
        
        # Update peak value for drawdown calculation
        self.peak_value = max(self.peak_value, portfolio_value)
        
        # Check portfolio value bounds
        if portfolio_value > self.max_portfolio_value:
            self.anomaly_counts['portfolio_value'] += 1
            return False, f"Portfolio value {portfolio_value:.2f} exceeds maximum {self.max_portfolio_value:.2f}"
            
        if portfolio_value < self.min_portfolio_value:
            self.anomaly_counts['portfolio_value'] += 1
            return False, f"Portfolio value {portfolio_value:.2f} below minimum {self.min_portfolio_value:.2f}"
            
        # Check position size
        if position_size > self.max_position_size:
            self.anomaly_counts['position_size'] += 1
            return False, f"Position size {position_size:.2%} exceeds maximum {self.max_position_size:.2%}"
            
        # Check drawdown
        if len(self.portfolio_values) > 0:
            current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
            if current_drawdown > self.max_drawdown_threshold:
                return False, f"Drawdown {current_drawdown:.2%} exceeds maximum {self.max_drawdown_threshold:.2%}"
        
        # Check for action repeating
        if len(self.action_history) >= 10:
            recent_actions = list(self.action_history)[-10:]
            if len(set(recent_actions)) == 1:  # All actions are the same
                self.anomaly_counts['action_repeat'] += 1
                return False, "Same action repeated 10 times in a row"
        
        # Check for reward spikes
        if len(self.rewards) > 1:
            avg_reward = np.mean(list(self.rewards)[:-1])
            std_reward = np.std(list(self.rewards)[:-1]) if len(self.rewards) > 2 else 1.0
            if abs(reward - avg_reward) > 5 * std_reward:  # More than 5 standard deviations
                self.anomaly_counts['reward_spike'] += 1
                
        return True, "Valid state"

    def get_statistics(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = {
            'anomalies': self.anomaly_counts.copy(),
            'portfolio_stats': {
                'current': self.portfolio_values[-1] if self.portfolio_values else 0.0,
                'peak': self.peak_value,
                'drawdown': (self.peak_value - self.portfolio_values[-1]) / self.peak_value 
                           if self.portfolio_values and self.peak_value > 0 else 0.0
            },
            'position_stats': {
                'current': self.position_sizes[-1] if self.position_sizes else 0.0,
                'average': np.mean(list(self.position_sizes)) if self.position_sizes else 0.0,
                'max': max(self.position_sizes) if self.position_sizes else 0.0
            },
            'action_stats': {
                'unique_actions': len(set(self.action_history)) if self.action_history else 0,
                'last_action': self.action_history[-1] if self.action_history else None
            },
            'reward_stats': {
                'recent_avg': np.mean(list(self.rewards)[-10:]) if len(self.rewards) >= 10 else 0.0,
                'total_avg': np.mean(list(self.rewards)) if self.rewards else 0.0,
                'std': np.std(list(self.rewards)) if len(self.rewards) > 1 else 0.0
            }
        }
        return stats

    def reset(self):
        """Reset monitoring state."""
        self.portfolio_values.clear()
        self.action_history.clear()
        self.position_sizes.clear()
        self.rewards.clear()
        self.peak_value = 0.0
        self.anomaly_counts = {k: 0 for k in self.anomaly_counts}