import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import asdict

from client import NodeClient
from dex_analyzer import DEXAnalyzer, SwapEvent, TokenRegistry
from swap_storage import SwapStorage
from metrics import RLMetricsCalculator

class DEXMonitor:
    def __init__(self, node_client: NodeClient, dex_address: str, node_url: str, min_trades: int = 20):
        self.node_client = node_client
        self.dex_address = dex_address
        self.analyzer = DEXAnalyzer(dex_address, node_url, min_trades=min_trades)
        self.storage = SwapStorage()
        self.metrics_calculator = RLMetricsCalculator(
            window_sizes=[5, 20, 50],
            export_dir="data/metrics"  # Specify default metrics export directory
        )
        self.logger = logging.getLogger(__name__)
        
    @classmethod
    async def create(cls, node_url: str, dex_address: str, min_trades: int = 20) -> 'DEXMonitor':
        """Async factory method"""
        node_client = await NodeClient.create(node_url)
        return cls(node_client, dex_address, node_url, min_trades)
        
    async def fetch_time_range(self, 
                             days_back: int = 30,
                             end_date: Optional[datetime] = None) -> List[Dict]:
        """Fetch transactions for a specific time range"""
        end_date = end_date or datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Fetching transactions from {start_date} to {end_date}")
        
        # Get height range for the time period
        start_height, end_height = await self.node_client.get_height_range_for_dates(
            self.dex_address, start_date, end_date
        )
        
        if not start_height or not end_height:
            raise Exception("Could not determine block height range")
            
        self.logger.info(f"Block height range: {start_height} to {end_height}")
            
        # Fetch transactions
        transactions = await self.node_client.get_transactions_range(
            self.dex_address,
            start_height=start_height,
            end_height=end_height
        )
        
        return transactions
        
    async def analyze_transactions(self, transactions: List[Dict]) -> List[SwapEvent]:
        """Analyze transactions and return swap events"""
        swaps = []
        for tx in transactions:
            swap = await self.analyzer.analyze_transaction(tx)
            if swap:
                swaps.append(swap)
                # Update metrics with each swap
                self.metrics_calculator.update_with_swap(asdict(swap))
                
        # Sort swaps by timestamp to ensure chronological order
        swaps.sort(key=lambda x: x.timestamp)
        return swaps
        
    def save_and_analyze_swaps(self, swaps: List[SwapEvent], export_dir: Optional[str] = None) -> Dict:
        """Save swaps to disk and return analytics"""
        # Save raw swap data
        self.storage.save_swaps(swaps)
        
        # Update metrics calculator export directory if specified
        if export_dir:
            self.metrics_calculator.export_dir = Path(export_dir)
            self.metrics_calculator.export_dir.mkdir(exist_ok=True)
        
        # Export metrics
        self.metrics_calculator.export_metrics()
        
        # Get analytics
        analytics = self.storage.analyze_swaps(swaps)
        
        # Add additional metrics from the calculator
        analytics["token_metrics"] = {}
        for token_name, metrics in self.metrics_calculator.current_metrics.items():
            analytics["token_metrics"][token_name] = asdict(metrics)
        
        return analytics
    
    def get_token_metrics(self, token_name: str) -> Optional[Dict]:
        """Get current metrics for a specific token"""
        metrics = self.metrics_calculator.current_metrics.get(token_name)
        if metrics:
            # Include potential trading signals
            metrics_dict = asdict(metrics)
            signals = self.metrics_calculator.get_potential_signals(token_name)
            metrics_dict["trading_signals"] = signals
            return metrics_dict
        return None
        
    async def close(self):
        """Clean up resources"""
        try:
            if self.node_client:
                await self.node_client.close_session()
            if self.analyzer:
                await self.analyzer.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Configuration
    NODE_URL = "http://0.0.0.0:9053"
    DEX_ADDRESS = "5vSUZRZbdVbnk4sJWjg2uhL94VZWRg4iatK9VgMChufzUgdihgvhR8yWSUEJKszzV7Vmi6K8hCyKTNhUaiP8p5ko6YEU9yfHpjVuXdQ4i5p4cRCzch6ZiqWrNukYjv7Vs5jvBwqg5hcEJ8u1eerr537YLWUoxxi1M4vQxuaCihzPKMt8NDXP4WcbN6mfNxxLZeGBvsHVvVmina5THaECosCWozKJFBnscjhpr3AJsdaL8evXAvPfEjGhVMoTKXAb2ZGGRmR8g1eZshaHmgTg2imSiaoXU5eiF3HvBnDuawaCtt674ikZ3oZdekqswcVPGMwqqUKVsGY4QuFeQoGwRkMqEYTdV2UDMMsfrjrBYQYKUBFMwsQGMNBL1VoY78aotXzdeqJCBVKbQdD3ZZWvukhSe4xrz8tcF3PoxpysDLt89boMqZJtGEHTV9UBTBEac6sDyQP693qT3nKaErN8TCXrJBUmHPqKozAg9bwxTqMYkpmb9iVKLSoJxG7MjAj72SRbcqQfNCVTztSwN3cRxSrVtz4p87jNFbVtFzhPg7UqDwNFTaasySCqM"
    OUTPUT_DIR = "data/metrics"  # Directory for exported metrics
    
    try:
        # Initialize monitor with minimum trade threshold
        monitor = await DEXMonitor.create(NODE_URL, DEX_ADDRESS, min_trades=100)
        
        # Fetch last 60 days of transactions
        logger.info("Starting transaction fetch...")
        transactions = await monitor.fetch_time_range(days_back=10)
        logger.info(f"Fetched {len(transactions)} transactions")
        
        # Analyze swaps
        logger.info("Analyzing swaps...")
        swaps = await monitor.analyze_transactions(transactions)
        logger.info(f"Found {len(swaps)} swap events")
        
        # Save and analyze swaps, export metrics
        analytics = monitor.save_and_analyze_swaps(swaps, export_dir=OUTPUT_DIR)
        
        # Print recent swaps summary with enhanced metrics
        print("\nRecent swaps (last 5):")
        for swap in swaps[-5:]:
            token_name = swap.output_token_name if swap.input_token == 'ERG' else swap.input_token_name
            metrics = monitor.get_token_metrics(token_name)
            
            print(f"\nTransaction: {swap.tx_id}")
            print(f"Time: {datetime.fromtimestamp(swap.timestamp/1000)}")
            print(f"Swap: {swap.input_amount_adjusted:.6f} {swap.input_token_name} -> "
                  f"{swap.output_amount_adjusted:.6f} {swap.output_token_name}")
            print(f"Price: {swap.price:.6f}")
            
            if metrics:
                print(f"\nMetrics for {token_name}:")
                print(f"- Current Price: {metrics['current_price']:.6f}")
                print(f"- Volume MA5: {metrics['volume_ma_5']:.2f} ERG")
                print(f"- Buy Pressure: {metrics['buy_pressure']:.2%}")
                print(f"- RSI: {metrics['rsi_14']:.1f}")
                
                # Print key trading signals
                signals = metrics.get('trading_signals', {})
                if signals:
                    print("\nTrading Signals:")
                    print(f"- MA Crossover: {signals['ma_crossover']:.2%}")
                    print(f"- RSI Signal: {signals['rsi_signal']:.2f}")
                    print(f"- Volume Signal: {signals['volume_signal_5']:.2f}")
        
        # Print analytics summary
        print("\nDEX Analytics:")
        print(json.dumps({
            "total_swaps": analytics["total_swaps"],
            "unique_tokens": len(analytics["token_volumes"]),
            "unique_pairs": len(analytics["pair_metrics"]),
            "tokens_with_metrics": len(analytics["token_metrics"])
        }, indent=2))
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise
        
    finally:
        if 'monitor' in locals():
            await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())