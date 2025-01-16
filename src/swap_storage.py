import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import asdict
from dex_analyzer import SwapEvent

class SwapStorage:
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
    def save_swaps(self, swaps: List[SwapEvent], filename: Optional[str] = None):
        """Save swap events to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"swaps_{timestamp}.json"
            
        filepath = self.storage_dir / filename
        
        # Convert swap events to dictionaries
        swap_data = [asdict(swap) for swap in swaps]
        
        # Add metadata
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_swaps": len(swaps),
            "swaps": swap_data
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def analyze_swaps(self, swaps: List[SwapEvent]) -> Dict:
        """Generate detailed analytics from swap events"""
        token_volumes = {}
        pair_metrics = {}
        lp_actions = []
        
        for swap in swaps:
            # Track volumes per token
            if swap.input_token not in token_volumes:
                token_volumes[swap.input_token] = {
                    "name": swap.input_token_name,
                    "total_amount": 0,
                    "total_amount_adjusted": 0,
                    "times_as_input": 0,
                    "times_as_output": 0
                }
            if swap.output_token not in token_volumes:
                token_volumes[swap.output_token] = {
                    "name": swap.output_token_name,
                    "total_amount": 0,
                    "total_amount_adjusted": 0,
                    "times_as_input": 0,
                    "times_as_output": 0
                }
                
            # Update token volumes
            token_volumes[swap.input_token]["total_amount"] += swap.input_amount
            token_volumes[swap.input_token]["total_amount_adjusted"] += swap.input_amount_adjusted
            token_volumes[swap.input_token]["times_as_input"] += 1
            
            token_volumes[swap.output_token]["total_amount"] += swap.output_amount
            token_volumes[swap.output_token]["total_amount_adjusted"] += swap.output_amount_adjusted
            token_volumes[swap.output_token]["times_as_output"] += 1
            
            # Track pair metrics
            pair = f"{swap.input_token_name}-{swap.output_token_name}"
            if pair not in pair_metrics:
                pair_metrics[pair] = {
                    "swap_count": 0,
                    "prices": [],
                    "total_input_adjusted": 0,
                    "total_output_adjusted": 0
                }
            
            pair_metrics[pair]["swap_count"] += 1
            pair_metrics[pair]["prices"].append(swap.price)
            pair_metrics[pair]["total_input_adjusted"] += swap.input_amount_adjusted
            pair_metrics[pair]["total_output_adjusted"] += swap.output_amount_adjusted
            
            # Track LP actions
            if swap.is_lp_action:
                lp_actions.append({
                    "tx_id": swap.tx_id,
                    "timestamp": swap.timestamp,
                    "action_type": "add" if swap.input_token_decimals > swap.output_token_decimals else "remove",
                    "tokens": [swap.input_token_name, swap.output_token_name]
                })
        
        # Calculate additional pair metrics
        for pair in pair_metrics:
            metrics = pair_metrics[pair]
            prices = metrics["prices"]
            
            if prices:
                metrics["avg_price"] = sum(prices) / len(prices)
                metrics["min_price"] = min(prices)
                metrics["max_price"] = max(prices)
                if len(prices) > 1:
                    metrics["price_volatility"] = calculate_volatility(prices)
        
        return {
            "token_volumes": token_volumes,
            "pair_metrics": pair_metrics,
            "lp_actions": lp_actions,
            "total_swaps": len(swaps),
            "total_lp_actions": len(lp_actions)
        }

def calculate_volatility(prices: List[float]) -> float:
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return (variance ** 0.5) / mean if mean != 0 else 0