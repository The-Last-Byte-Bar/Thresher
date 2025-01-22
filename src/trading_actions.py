from enum import Enum
from typing import Tuple, Optional
import logging
from portfolio import PortfolioManager

class Action(Enum):
    """Available trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    REBALANCE = 3

class TradingExecutor:
    """Handles execution of trading actions with safety checks."""
    
    def __init__(self, 
                 portfolio_manager: PortfolioManager,
                 transaction_fee: float = 0.002,
                 max_trade_size: float = 0.25,
                 rebalance_fee_multiplier: float = 1.5):
        self.portfolio = portfolio_manager
        self.transaction_fee = transaction_fee
        self.max_trade_size = max_trade_size
        self.rebalance_fee_multiplier = rebalance_fee_multiplier
        self.logger = logging.getLogger("TradingExecutor")
        
    def execute_trade(self, action: Action, current_price: float) -> Tuple[bool, str]:
        """Execute a trading action safely."""
        try:
            if action == Action.HOLD:
                return True, "Hold position"
                
            elif action == Action.BUY:
                return self._execute_buy(current_price)
                
            elif action == Action.SELL:
                return self._execute_sell(current_price)
                
            elif action == Action.REBALANCE:
                return self._execute_rebalance(current_price)
                
            return False, f"Unknown action: {action}"
            
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
            return False, str(e)
            
    def _execute_buy(self, current_price: float) -> Tuple[bool, str]:
        """Execute a buy order with safety checks."""
        state = self.portfolio.state
        
        # Calculate maximum trade size
        max_erg_trade = min(
            state.erg_balance,
            state.total_value_in_erg * self.max_trade_size
        )
        
        if max_erg_trade <= 0:
            return False, "Insufficient ERG balance for buy"
            
        # Calculate trade amounts with fees
        erg_to_spend = max_erg_trade * (1 - self.transaction_fee)
        tokens_to_receive = erg_to_spend / current_price
        
        # Update portfolio state
        new_erg_balance = state.erg_balance - max_erg_trade
        new_token_balance = state.token_balance + tokens_to_receive
        
        self.portfolio.update_state(
            new_erg_balance,
            new_token_balance,
            current_price,
            Action.BUY
        )
        
        return True, f"Bought {tokens_to_receive:.6f} tokens for {erg_to_spend:.6f} ERG"
        
    def _execute_sell(self, current_price: float) -> Tuple[bool, str]:
        """Execute a sell order with safety checks."""
        state = self.portfolio.state
        
        # Calculate maximum trade size
        max_token_trade = min(
            state.token_balance,
            (state.total_value_in_erg * self.max_trade_size) / current_price
        )
        
        if max_token_trade <= 0:
            return False, "Insufficient token balance for sell"
            
        # Calculate trade amounts with fees
        erg_to_receive = max_token_trade * current_price * (1 - self.transaction_fee)
        
        # Update portfolio state
        new_erg_balance = state.erg_balance + erg_to_receive
        new_token_balance = state.token_balance - max_token_trade
        
        self.portfolio.update_state(
            new_erg_balance,
            new_token_balance,
            current_price,
            Action.SELL
        )
        
        return True, f"Sold {max_token_trade:.6f} tokens for {erg_to_receive:.6f} ERG"
        
    def _execute_rebalance(self, current_price: float) -> Tuple[bool, str]:
        """Execute portfolio rebalancing with increased fees."""
        state = self.portfolio.state
        
        # Calculate target allocation (50/50)
        target_erg = state.total_value_in_erg * 0.5
        current_erg = state.erg_balance
        
        # Determine trade direction and size
        if current_erg > target_erg:
            # Need to buy tokens
            amount_to_trade = min(
                current_erg - target_erg,
                state.total_value_in_erg * self.max_trade_size
            )
            return self._execute_buy_portion(amount_to_trade, current_price)
        else:
            # Need to sell tokens
            amount_to_trade = min(
                target_erg - current_erg,
                state.total_value_in_erg * self.max_trade_size
            )
            return self._execute_sell_portion(amount_to_trade, current_price)
            
    def _execute_buy_portion(self, amount: float, current_price: float) -> Tuple[bool, str]:
        """Execute a portion of a buy rebalance."""
        rebalance_fee = self.transaction_fee * self.rebalance_fee_multiplier
        erg_to_spend = amount * (1 - rebalance_fee)
        tokens_to_receive = erg_to_spend / current_price
        
        state = self.portfolio.state
        self.portfolio.update_state(
            state.erg_balance - amount,
            state.token_balance + tokens_to_receive,
            current_price,
            Action.REBALANCE
        )
        
        return True, f"Rebalance: Bought {tokens_to_receive:.6f} tokens"
        
    def _execute_sell_portion(self, amount: float, current_price: float) -> Tuple[bool, str]:
        """Execute a portion of a sell rebalance."""
        rebalance_fee = self.transaction_fee * self.rebalance_fee_multiplier
        tokens_to_sell = amount / current_price
        erg_to_receive = amount * (1 - rebalance_fee)
        
        state = self.portfolio.state
        self.portfolio.update_state(
            state.erg_balance + erg_to_receive,
            state.token_balance - tokens_to_sell,
            current_price,
            Action.REBALANCE
        )
        
        return True, f"Rebalance: Sold {tokens_to_sell:.6f} tokens"