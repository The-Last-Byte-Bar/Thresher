from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from portfolio import PortfolioManager
from trading_actions import Action

class StateEncoder:
    """Encodes trading environment state into a format suitable for the RL agent.
    
    This class handles the conversion of raw market data and portfolio state into
    a normalized observation vector that can be used by the RL agent. It includes
    price features, volume metrics, technical indicators, and portfolio state.
    """
    
    def __init__(self,
                 portfolio_manager: PortfolioManager,
                 price_feature_window: int = 20,
                 volume_feature_window: int = 5,
                 clip_range: Tuple[float, float] = (-5, 5)):
        
        self.portfolio = portfolio_manager
        self.price_window = price_feature_window
        self.volume_window = volume_feature_window
        self.clip_range = clip_range
        self.logger = logging.getLogger("StateEncoder")
        
        # Initialize feature tracking
        self.price_history = []
        self.volume_history = []
        
    def encode_state(self, current_data: pd.Series) -> np.ndarray:
        """Convert current market data and portfolio state into an observation vector."""
        try:
            # Update historical data
            self._update_history(current_data)
            
            # Calculate different feature groups
            price_features = self._encode_price_features(current_data)
            volume_features = self._encode_volume_features(current_data)
            technical_features = self._encode_technical_features(current_data)
            portfolio_features = self._encode_portfolio_features()
            
            # Combine all features
            observation = np.concatenate([
                price_features,
                volume_features,
                technical_features,
                portfolio_features
            ])
            
            # Ensure numerical stability
            observation = np.clip(observation, self.clip_range[0], self.clip_range[1])
            observation = np.nan_to_num(observation, nan=0.0, posinf=self.clip_range[1], neginf=self.clip_range[0])
            
            return observation.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error encoding state: {e}")
            # Return zero vector of appropriate size in case of error
            return np.zeros(self.get_observation_dim(), dtype=np.float32)
    
    def _update_history(self, current_data: pd.Series) -> None:
        """Update price and volume history for feature calculation."""
        self.price_history.append(current_data['price'])
        self.volume_history.append(current_data['volume_erg'])
        
        # Maintain window sizes
        if len(self.price_history) > self.price_window:
            self.price_history.pop(0)
        if len(self.volume_history) > self.volume_window:
            self.volume_history.pop(0)
    
    def _encode_price_features(self, current_data: pd.Series) -> np.ndarray:
        """Calculate price-based features."""
        # Get moving averages from data if available, otherwise calculate
        ma_5 = current_data.get('price_ma_5', 
                              np.mean(self.price_history[-5:]) if len(self.price_history) >= 5 else current_data['price'])
        ma_20 = current_data.get('price_ma_20',
                               np.mean(self.price_history) if self.price_history else current_data['price'])
        
        # Calculate normalized price features
        current_price = current_data['price']
        
        features = [
            self._safe_normalize(current_price, ma_5),    # Price relative to short MA
            self._safe_normalize(current_price, ma_20),   # Price relative to long MA
            self._calculate_momentum()                    # Price momentum
        ]
        
        return np.array(features)
    
    def _encode_volume_features(self, current_data: pd.Series) -> np.ndarray:
        """Calculate volume-based features."""
        current_volume = current_data['volume_erg']
        
        # Calculate volume moving average
        volume_ma = current_data.get('volume_ma_5',
                                   np.mean(self.volume_history) if self.volume_history else current_volume)
        
        features = [
            self._safe_normalize(current_volume, volume_ma),   # Volume relative to MA
            np.log1p(current_volume) / 10.0                    # Log-scaled volume
        ]
        
        return np.array(features)
    
    def _encode_technical_features(self, current_data: pd.Series) -> np.ndarray:
        """Calculate technical indicators."""
        # Normalize RSI to [-1, 1]
        rsi = current_data.get('rsi_14', 50.0)
        rsi_normalized = (rsi / 100.0) * 2 - 1
        
        # Bollinger Band position
        bb_position = current_data.get('bb_position_20', 0.5)
        bb_normalized = (bb_position - 0.5) * 2  # Center around 0
        
        # Volatility
        volatility = current_data.get('volatility', self._calculate_volatility())
        volatility_normalized = np.clip(volatility, 0, 1)
        
        features = [
            rsi_normalized,
            bb_normalized,
            volatility_normalized
        ]
        
        return np.array(features)
    
    def _encode_portfolio_features(self) -> np.ndarray:
        """Encode current portfolio state."""
        state = self.portfolio.state
        
        # Calculate position features
        position_size = self.portfolio.get_position_size()
        balance_ratio = state.erg_balance / self.portfolio.initial_balance
        
        # Calculate returns and drawdown
        portfolio_return = (state.total_value_in_erg / self.portfolio.initial_balance) - 1
        drawdown = self.portfolio.get_drawdown()
        
        features = [
            position_size,                # Current position size [0, 1]
            self._safe_normalize(balance_ratio, 1.0),  # Balance relative to initial
            portfolio_return,             # Total return so far
            drawdown                      # Current drawdown
        ]
        
        return np.array(features)
    
    def _safe_normalize(self, value: float, base: float, epsilon: float = 1e-8) -> float:
        """Safely normalize a value relative to a base value."""
        if abs(base) < epsilon:
            return 0.0
        return (value / base) - 1.0
    
    def _calculate_momentum(self) -> float:
        """Calculate price momentum indicator."""
        if len(self.price_history) < 2:
            return 0.0
            
        return (self.price_history[-1] / self.price_history[0] - 1) if self.price_history[0] > 0 else 0.0
    
    def _calculate_volatility(self) -> float:
        """Calculate price volatility."""
        if len(self.price_history) < 2:
            return 0.0
            
        returns = np.diff(self.price_history) / self.price_history[:-1]
        return np.std(returns)
    
    def get_observation_dim(self) -> int:
        """Get the dimension of the observation space."""
        return (
            3 +  # Price features
            2 +  # Volume features
            3 +  # Technical features
            4    # Portfolio features
        )
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the observation vector."""
        return [
            # Price features
            'price_to_ma5',
            'price_to_ma20',
            'momentum',
            
            # Volume features
            'volume_to_ma',
            'log_volume',
            
            # Technical features
            'rsi_normalized',
            'bb_position',
            'volatility',
            
            # Portfolio features
            'position_size',
            'balance_ratio',
            'portfolio_return',
            'drawdown'
        ]