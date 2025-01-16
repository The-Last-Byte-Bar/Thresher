import asyncio
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional
from dataclasses import asdict
import logging
from functools import partial

class ParallelDEXMonitor:
    def __init__(self, node_client, dex_address: str, node_url: str, min_trades: int = 20):
        self.node_client = node_client
        self.dex_address = dex_address
        self.analyzer = DEXAnalyzer(dex_address, node_url, min_trades=min_trades)
        self.storage = SwapStorage()
        self.metrics_calculator = RLMetricsCalculator(
            window_sizes=[5, 20, 50],
            export_dir="data/metrics"
        )
        self.logger = logging.getLogger(__name__)
        self.num_processes = mp.cpu_count()  # Use all available CPU cores
        
    @staticmethod
    async def _analyze_transaction_chunk(chunk: List[Dict], dex_address: str, node_url: str) -> List[Optional[SwapEvent]]:
        """Process a chunk of transactions in parallel"""
        analyzer = DEXAnalyzer(dex_address, node_url)
        results = []
        
        for tx in chunk:
            try:
                swap = await analyzer.analyze_transaction(tx)
                if swap:
                    results.append(swap)
            except Exception as e:
                logging.error(f"Error analyzing transaction {tx.get('id')}: {e}")
                continue
                
        await analyzer.close()
        return results

    async def analyze_transactions_parallel(self, transactions: List[Dict]) -> List[SwapEvent]:
        """Analyze transactions using parallel processing"""
        if not transactions:
            return []

        self.logger.info(f"Processing {len(transactions)} transactions using {self.num_processes} processes")
        
        # Calculate chunk size - ensure each process gets a meaningful amount of work
        chunk_size = max(100, len(transactions) // (self.num_processes * 2))
        chunks = [transactions[i:i + chunk_size] for i in range(0, len(transactions), chunk_size)]
        
        # Create partial function with fixed arguments
        analyze_func = partial(
            self._analyze_transaction_chunk,
            dex_address=self.dex_address,
            node_url=self.node_url
        )
        
        # Process chunks in parallel using ProcessPoolExecutor
        all_swaps = []
        async with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Create list of tasks
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(executor, analyze_func, chunk)
                for chunk in chunks
            ]
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Combine results
            for chunk_results in results:
                if chunk_results:
                    all_swaps.extend(chunk_results)

        # Sort swaps by timestamp to maintain chronological order
        all_swaps.sort(key=lambda x: x.timestamp)
        
        # Update metrics for all swaps
        for swap in all_swaps:
            self.metrics_calculator.update_with_swap(asdict(swap))
            
        self.logger.info(f"Processed {len(all_swaps)} valid swap events")
        return all_swaps

    async def process_time_range(self, days_back: int = 30, end_date: Optional[datetime] = None) -> Dict:
        """Process transactions for a time range using parallel processing"""
        try:
            # Fetch transactions
            transactions = await self.fetch_time_range(days_back, end_date)
            self.logger.info(f"Fetched {len(transactions)} transactions")
            
            # Analyze transactions in parallel
            swaps = await self.analyze_transactions_parallel(transactions)
            self.logger.info(f"Found {len(swaps)} swap events")
            
            # Save and analyze swaps
            analytics = self.save_and_analyze_swaps(swaps)
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error processing time range: {e}")
            raise

async def main():
    # Configuration
    NODE_URL = "http://0.0.0.0:9053"
    DEX_ADDRESS = "5vSUZRZbdVbnk4sJWjg2uhL94VZWRg4iatK9VgMChufzUgdihgvhR8yWSUEJKszzV7Vmi6K8hCyKTNhUaiP8p5ko6YEU9yfHpjVuXdQ4i5p4cRCzch6ZiqWrNukYjv7Vs5jvBwqg5hcEJ8u1eerr537YLWUoxxi1M4vQxuaCihzPKMt8NDXP4WcbN6mfNxxLZeGBvsHVvVmina5THaECosCWozKJFBnscjhpr3AJsdaL8evXAvPfEjGhVMoTKXAb2ZGGRmR8g1eZshaHmgTg2imSiaoXU5eiF3HvBnDuawaCtt674ikZ3oZdekqswcVPGMwqqUKVsGY4QuFeQoGwRkMqEYTdV2UDMMsfrjrBYQYKUBFMwsQGMNBL1VoY78aotXzdeqJCBVKbQdD3ZZWvukhSe4xrz8tcF3PoxpysDLt89boMqZJtGEHTV9UBTBEac6sDyQP693qT3nKaErN8TCXrJBUmHPqKozAg9bwxTqMYkpmb9iVKLSoJxG7MjAj72SRbcqQfNCVTztSwN3cRxSrVtz4p87jNFbVtFzhPg7UqDwNFTaasySCqM"
    
    # Initialize monitor
    node_client = await NodeClient.create(NODE_URL)
    monitor = ParallelDEXMonitor(node_client, DEX_ADDRESS, NODE_URL)
    
    try:
        # Process last 30 days of transactions
        analytics = await monitor.process_time_range(days_back=30)
        print(f"Processed {analytics['total_swaps']} swap events")
        
    finally:
        await monitor.close()

if __name__ == "__main__":
    asyncio.run(main())