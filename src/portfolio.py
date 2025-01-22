from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import logging

@dataclass
class PortfolioState:
    """Represents the current state of the trading portfolio."""
    erg_balance: float
    token_balance: float
    token_value_in_erg: float
    total_value_in_erg: float
    current_price: float
    last_action: 'Action'  # Forward reference
    
    def to_array(self) -> np.ndarray:
        """Convert portfolio state to numpy array for the RL agent."""
        return np.array([
            self.erg_balance,
            self.token_balance,
            self.token_value_in_erg,
            self.total_value_in_erg,
            self.current_price
        ], dtype=np.float32)

class PortfolioManager:
    """Manages portfolio state and operations."""
    
    def __init__(self, initial_balance: float = 10.0):
        self.initial_balance = initial_balance
        self.logger = logging.getLogger("PortfolioManager")
        self.state = self._initialize_portfolio()
        self.peak_value = initial_balance
        
    def _initialize_portfolio(self) -> PortfolioState:
        """Initialize a new portfolio with starting balance."""
        return PortfolioState(
            erg_balance=self.initial_balance,
            token_balance=0.0,
            token_value_in_erg=0.0,
            total_value_in_erg=self.initial_balance,
            current_price=0.0,
            last_action=None  # Will be set on first action
        )
        
    def update_state(self, 
                    new_erg_balance: float,
                    new_token_balance: float,
                    current_price: float,
                    action: 'Action') -> None:
        """Update portfolio state with new values."""
        token_value = new_token_balance * current_price
        total_value = new_erg_balance + token_value
        
        # Update peak value for drawdown calculations
        self.peak_value = max(self.peak_value, total_value)
        
        self.state = PortfolioState(
            erg_balance=new_erg_balance,
            token_balance=new_token_balance,
            token_value_in_erg=token_value,
            total_value_in_erg=total_value,
            current_price=current_price,
            last_action=action
        )
        
    def get_position_size(self) -> float:
        """Calculate current position size as a fraction of total portfolio."""
        if self.state.total_value_in_erg <= 0:
            return 0.0
        return self.state.token_value_in_erg / self.state.total_value_in_erg
        
    def get_drawdown(self) -> float:
        """Calculate current drawdown from peak value."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.state.total_value_in_erg) / self.peak_value
        
    def validate_state(self) -> Tuple[bool, str]:
        """Validate portfolio state for numerical stability and constraints."""
        if self.state.total_value_in_erg < 0:
            return False, "Negative portfolio value"
            
        if self.state.total_value_in_erg > 1e6:  # Arbitrary large value threshold
            return False, "Unrealistic portfolio value"
            
        if self.state.erg_balance < 0:
            return False, "Negative ERG balance"
            
        if self.state.token_balance < 0:
            return False, "Negative token balance"
            
        return True, "Valid portfolio state"