import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict

from client import NodeClient
from dex_analyzer import DEXAnalyzer
from metrics import RLMetricsCalculator

async def load_and_export_metrics(
    node_url: str,
    dex_address: str,
    days_back: int = 30,
    output_dir: str = "data"
) -> None:
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:
        # Initialize components
        node_client = await NodeClient.create(node_url)
        analyzer = DEXAnalyzer(dex_address, node_url)
        calculator = RLMetricsCalculator()

        logger.info(f"Fetching {days_back} days of transaction data...")
        
        # Get time range for fetching
        end_date = datetime.now()
        
        # Get height range for the time period
        start_height, end_height = await node_client.get_height_range_for_dates(
            dex_address, 
            end_date - timedelta(days=days_back),
            end_date
        )
        
        if not start_height or not end_height:
            raise Exception("Could not determine block height range")
            
        logger.info(f"Block height range: {start_height} to {end_height}")
            
        # Fetch transactions
        transactions = await node_client.get_transactions_range(
            dex_address,
            start_height=start_height,
            end_height=end_height
        )
        
        logger.info(f"Processing {len(transactions)} transactions...")

        # Process transactions and update metrics
        swap_count = 0
        for tx in transactions:
            swap = await analyzer.analyze_transaction(tx)
            if swap:
                calculator.update_with_swap(swap)
                swap_count += 1
                
        logger.info(f"Processed {swap_count} swap events")

        # Export metrics
        logger.info("Exporting metrics to CSV...")
        output_path = Path(output_dir)
        calculator.export_metrics(output_dir)
        
        # Print summary of exported files
        csv_files = list(output_path.glob("*.csv"))
        logger.info(f"\nExported {len(csv_files)} files:")
        for file in csv_files:
            logger.info(f"- {file.name}")

    except Exception as e:
        logger.error(f"Error processing transactions: {e}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        await node_client.close_session()
        await analyzer.close()

if __name__ == "__main__":
    # Configuration
    NODE_URL = "http://0.0.0.0:9053"  # Update with your node URL
    DEX_ADDRESS = "5vSUZRZbdVbnk4sJWjg2uhL94VZWRg4iatK9VgMChufzUgdihgvhR8yWSUEJKszzV7Vmi6K8hCyKTNhUaiP8p5ko6YEU9yfHpjVuXdQ4i5p4cRCzch6ZiqWrNukYjv7Vs5jvBwqg5hcEJ8u1eerr537YLWUoxxi1M4vQxuaCihzPKMt8NDXP4WcbN6mfNxxLZeGBvsHVvVmina5THaECosCWozKJFBnscjhpr3AJsdaL8evXAvPfEjGhVMoTKXAb2ZGGRmR8g1eZshaHmgTg2imSiaoXU5eiF3HvBnDuawaCtt674ikZ3oZdekqswcVPGMwqqUKVsGY4QuFeQoGwRkMqEYTdV2UDMMsfrjrBYQYKUBFMwsQGMNBL1VoY78aotXzdeqJCBVKbQdD3ZZWvukhSe4xrz8tcF3PoxpysDLt89boMqZJtGEHTV9UBTBEac6sDyQP693qT3nKaErN8TCXrJBUmHPqKozAg9bwxTqMYkpmb9iVKLSoJxG7MjAj72SRbcqQfNCVTztSwN3cRxSrVtz4p87jNFbVtFzhPg7UqDwNFTaasySCqM"
    OUTPUT_DIR = "data/metrics"
    DAYS_BACK = 30

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Run the async function
    asyncio.run(load_and_export_metrics(
        node_url=NODE_URL,
        dex_address=DEX_ADDRESS,
        days_back=DAYS_BACK,
        output_dir=OUTPUT_DIR
    ))