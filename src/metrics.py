from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class TokenMetrics:
    token_name: str
    current_price: float
    price_ma_5: float
    price_ma_20: float
    volume_ma_5: float
    price_volatility: float
    recent_trades: int
    buy_pressure: float
    last_trade_timestamp: int
    erg_liquidity: float
    token_liquidity: float
    
    # Advanced metrics
    hour: int
    day_of_week: int
    time_since_last_trade: float
    trade_density_1h: float
    trade_density_24h: float
    price_momentum: Dict[str, float]
    price_acceleration: float
    rsi_14: float
    bb_position: Dict[str, float]
    volume_imbalance: Dict[str, float]
    liquidity_imbalance: float
    market_impact: float
    trade_sign_imbalance: Dict[str, float]

class RLMetricsCalculator:
    def __init__(self, 
                 window_sizes: List[int] = [5, 20, 50],
                 export_dir: str = "data"):
        self.window_sizes = window_sizes
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        
        self.token_data: Dict[str, List] = {}
        self.current_metrics: Dict[str, TokenMetrics] = {}
        
    def update_with_swap(self, swap: Dict) -> Dict[str, TokenMetrics]:
        """Update metrics with a new swap event"""
        # Get token info
        token_name = swap['output_token_name'] if swap['input_token'] == 'ERG' else swap['input_token_name']
        if token_name == 'ERG':
            return self.current_metrics
            
        if token_name not in self.token_data:
            self.token_data[token_name] = []
            
        # Calculate volume in ERG terms
        volume_erg = (
            swap['input_amount_adjusted'] if swap['input_token'] == 'ERG'
            else swap['output_amount_adjusted'] if swap['output_token'] == 'ERG'
            else swap['input_amount_adjusted'] * swap['price']
        )
            
        # Add swap data with timestamp
        swap_data = {
            'timestamp': swap['timestamp'],
            'datetime': datetime.fromtimestamp(swap['timestamp'] / 1000),
            'is_buy': swap['input_token'] == 'ERG',
            'input_amount_adjusted': swap['input_amount_adjusted'],
            'output_amount_adjusted': swap['output_amount_adjusted'],
            'price': swap['price'],
            'erg_liquidity': swap.get('erg_liquidity', 0),
            'token_liquidity': swap.get('token_liquidity', 0),
            'volume_erg': volume_erg,
            'tx_id': swap['tx_id']
        }
        
        self.token_data[token_name].append(swap_data)
        self._update_token_metrics(token_name)
        
        return self.current_metrics
        
    def _update_token_metrics(self, token_name: str):
        """Calculate comprehensive metrics for a token"""
        swaps = self.token_data[token_name]
        if not swaps:
            return
            
        # Convert to DataFrame for calculations
        df = pd.DataFrame(swaps)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Basic metrics from original implementation
        current_price = df['price'].iloc[-1]
        price_mas = {size: df['price'].tail(size).mean() for size in self.window_sizes}
        volume_ma_5 = df['volume_erg'].tail(5).mean()
        
        # Volatility and buy pressure
        returns = df['price'].pct_change()
        volatility = returns.tail(20).std() if len(df) > 20 else 0
        recent_trades = df.tail(20)
        buy_pressure = len(recent_trades[recent_trades['is_buy']]) / len(recent_trades) if len(recent_trades) > 0 else 0
        
        # Latest liquidity values
        erg_liquidity = df['erg_liquidity'].iloc[-1]
        token_liquidity = df['token_liquidity'].iloc[-1]
        
        # Advanced time-based features
        latest_time = df['datetime'].iloc[-1]
        hour = latest_time.hour
        day_of_week = latest_time.dayofweek
        time_since_last_trade = (df['datetime'].iloc[-1] - df['datetime'].iloc[-2]).total_seconds() / 60 if len(df) > 1 else 0
        
        # Trade density
        trade_density_1h = len(df[df['datetime'] >= latest_time - pd.Timedelta(hours=1)])
        trade_density_24h = len(df[df['datetime'] >= latest_time - pd.Timedelta(days=1)])
        
        # Price momentum and acceleration
        price_momentum = {
            f'{period}': df['price'].pct_change(period).iloc[-1]
            for period in self.window_sizes
        }
        price_acceleration = df['price'].pct_change().diff().iloc[-1]
        
        # RSI calculation
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_14 = 100 - (100 / (1 + rs)).iloc[-1]
        
        # Bollinger Bands position
        bb_position = {}
        for period in self.window_sizes:
            roll = df['price'].rolling(period)
            middle = roll.mean().iloc[-1]
            std = roll.std().iloc[-1]
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            bb_position[f'{period}'] = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
        # Volume imbalance
        df['buy_volume'] = df['volume_erg'].where(df['is_buy'], 0)
        df['sell_volume'] = df['volume_erg'].where(~df['is_buy'], 0)
        volume_imbalance = {}
        for period in self.window_sizes:
            buy_sum = df['buy_volume'].tail(period).sum()
            sell_sum = df['sell_volume'].tail(period).sum()
            volume_imbalance[f'{period}'] = (buy_sum - sell_sum) / (buy_sum + sell_sum) if (buy_sum + sell_sum) > 0 else 0
            
        # Market microstructure
        liquidity_imbalance = erg_liquidity / (token_liquidity * current_price) if token_liquidity > 0 else 0
        market_impact = abs(df['price'].pct_change().iloc[-1]) / df['volume_erg'].iloc[-1] if df['volume_erg'].iloc[-1] > 0 else 0
        
        # Trade sign imbalance
        trade_sign_imbalance = {
            f'{period}': (df['is_buy'].tail(period).mean() - 0.5) * 2
            for period in self.window_sizes
        }
        
        # Update metrics
        self.current_metrics[token_name] = TokenMetrics(
            token_name=token_name,
            current_price=current_price,
            price_ma_5=price_mas[5],
            price_ma_20=price_mas[20],
            volume_ma_5=volume_ma_5,
            price_volatility=volatility,
            recent_trades=len(recent_trades),
            buy_pressure=buy_pressure,
            last_trade_timestamp=int(latest_time.timestamp()),
            erg_liquidity=erg_liquidity,
            token_liquidity=token_liquidity,
            
            # Advanced metrics
            hour=hour,
            day_of_week=day_of_week,
            time_since_last_trade=time_since_last_trade,
            trade_density_1h=trade_density_1h,
            trade_density_24h=trade_density_24h,
            price_momentum=price_momentum,
            price_acceleration=price_acceleration,
            rsi_14=rsi_14,
            bb_position=bb_position,
            volume_imbalance=volume_imbalance,
            liquidity_imbalance=liquidity_imbalance,
            market_impact=market_impact,
            trade_sign_imbalance=trade_sign_imbalance
        )
        
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to be safe for all operating systems"""
        # Replace problematic characters with underscores
        invalid_chars = '<>:"/\\|?*#%&{}'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace spaces with underscores
        filename = filename.replace(' ', '_')
        
        # Remove any remaining problematic characters
        filename = ''.join(c for c in filename if c.isalnum() or c in '_-.')
        
        # Ensure filename isn't too long
        if len(filename) > 200:
            filename = filename[:200]
            
        return filename
        
    def export_metrics(self):
        """Export detailed metrics to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export detailed swap data with advanced metrics
        for token_name, swaps in self.token_data.items():
            if not swaps:
                continue
                
            df = pd.DataFrame(swaps)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Calculate all features
            df = self._calculate_all_features(df)
            
            # Create sanitized filename
            safe_token_name = self.sanitize_filename(token_name)
            filename = f"{safe_token_name}_swaps_{timestamp}.csv"
            
            try:
                # Save to CSV
                filepath = self.export_dir / filename
                df.to_csv(filepath, index=False)
            except Exception as e:
                logging.error(f"Error saving metrics for {token_name}: {e}")
            
        # Export current metrics summary
        metrics_data = []
        for token_name, metrics in self.current_metrics.items():
            metrics_dict = {
                'token_name': metrics.token_name,
                'current_price': metrics.current_price,
                'price_ma_5': metrics.price_ma_5,
                'price_ma_20': metrics.price_ma_20,
                'volume_ma_5': metrics.volume_ma_5,
                'price_volatility': metrics.price_volatility,
                'recent_trades': metrics.recent_trades,
                'buy_pressure': metrics.buy_pressure,
                'last_trade': datetime.fromtimestamp(metrics.last_trade_timestamp),
                'erg_liquidity': metrics.erg_liquidity,
                'token_liquidity': metrics.token_liquidity,
                'rsi_14': metrics.rsi_14,
                'market_impact': metrics.market_impact,
                'liquidity_imbalance': metrics.liquidity_imbalance
            }
            
            # Add window-based metrics
            for period in self.window_sizes:
                metrics_dict.update({
                    f'price_momentum_{period}': metrics.price_momentum[f'{period}'],
                    f'bb_position_{period}': metrics.bb_position[f'{period}'],
                    f'volume_imbalance_{period}': metrics.volume_imbalance[f'{period}'],
                    f'trade_sign_imbalance_{period}': metrics.trade_sign_imbalance[f'{period}']
                })
                
            metrics_data.append(metrics_dict)
            
        if metrics_data:
            try:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_filepath = self.export_dir / f"token_metrics_summary_{timestamp}.csv"
                metrics_df.to_csv(metrics_filepath, index=False)
            except Exception as e:
                logging.error(f"Error saving metrics summary: {e}")
            
    def _calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features for the dataframe"""
        # Add basic metrics
        df['price_ma_5'] = df['price'].rolling(5).mean()
        df['price_ma_20'] = df['price'].rolling(20).mean()
        df['volume_ma_5'] = df['volume_erg'].rolling(5).mean()
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Add advanced metrics
        for period in self.window_sizes:
            df[f'price_momentum_{period}'] = df['price'].pct_change(period)
            df[f'volume_momentum_{period}'] = df['volume_erg'].pct_change(period)
            
            # Bollinger Bands
            roll = df['price'].rolling(period)
            df[f'bb_middle_{period}'] = roll.mean()
            df[f'bb_std_{period}'] = roll.std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (df[f'bb_std_{period}'] * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (df[f'bb_std_{period}'] * 2)
            df[f'bb_position_{period}'] = (df['price'] - df[f'bb_lower_{period}']) / (
                df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']
            )
        
        return df
        
    def get_potential_signals(self, token_name: str) -> Dict[str, float]:
        """Get comprehensive trading signals based on current metrics"""
        if token_name not in self.current_metrics:
            return {}
            
        metrics = self.current_metrics[token_name]
        
        signals = {
            'ma_crossover': (metrics.price_ma_5 / metrics.price_ma_20) - 1,
            'price_trend': (metrics.current_price / metrics.price_ma_5) - 1,
            'buy_pressure': metrics.buy_pressure - 0.5,
            'volatility_signal': -1 if metrics.price_volatility > 0.1 else 1,
            'rsi_signal': (metrics.rsi_14 - 50) / 50,  # Normalized around 0
            'liquidity_signal': -1 if metrics.liquidity_imbalance > 2 or metrics.liquidity_imbalance < 0.5 else 1,
            'market_impact_signal': -1 if metrics.market_impact > 0.01 else 1
        }
        
        # Add window-based signals
        for period in self.window_sizes:
            signals.update({
                f'momentum_{period}': metrics.price_momentum[f'{period}'],
                f'bb_signal_{period}': metrics.bb_position[f'{period}'] - 0.5,  # Normalized around 0
                f'volume_signal_{period}': metrics.volume_imbalance[f'{period}'],
                f'trade_pressure_{period}': metrics.trade_sign_imbalance[f'{period}']
            })
            
        return signals