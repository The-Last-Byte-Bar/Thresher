from __future__ import annotations
import aiohttp
import logging
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime

class NodeClient:
    def __init__(self, node_url: str, api_key: Optional[str] = None):
        self.node_url = node_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self.max_batch_size = 100  # Reduced batch size for better control

    @classmethod
    async def create(cls, node_url: str, api_key: Optional[str] = None) -> NodeClient:
        client = cls(node_url, api_key)
        await client.init_session()
        return client

    async def init_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_current_height(self) -> Optional[int]:
        """Get current blockchain height"""
        try:
            url = f"{self.node_url}/blocks/lastHeaders/1"
            async with self.session.get(url) as response:
                if response.status == 200:
                    headers = await response.json()
                    if headers and len(headers) > 0:
                        return headers[0]['height']
        except Exception as e:
            self.logger.error(f"Error getting current height: {e}")
        return None

    async def get_transactions_range(self, 
                                   address: str,
                                   start_height: Optional[int] = None,
                                   end_height: Optional[int] = None) -> List[Dict]:
        """Get transactions within a specific block height range"""
        if not self.session:
            await self.init_session()

        current_height = await self.get_current_height()
        if not current_height:
            self.logger.error("Could not determine current height")
            return []

        # Validate/adjust height range
        end_height = min(end_height or current_height, current_height)
        start_height = max(start_height or 0, 0)
        
        if start_height >= end_height:
            self.logger.error(f"Invalid height range: {start_height} to {end_height}")
            return []

        self.logger.info(f"Fetching transactions between blocks {start_height} and {end_height}")
        
        all_transactions = []
        offset = 0
        
        while True:
            batch = await self._make_request(address, offset, self.max_batch_size)
            if not batch or 'items' not in batch:
                break

            # Filter transactions by height
            valid_txs = [
                tx for tx in batch['items']
                if start_height <= tx['inclusionHeight'] <= end_height
            ]
            
            if valid_txs:
                all_transactions.extend(valid_txs)
                
            # Check if we've reached transactions before our start height
            oldest_height = min(tx['inclusionHeight'] for tx in batch['items'])
            if oldest_height < start_height:
                break
                
            if len(batch['items']) < self.max_batch_size:
                break
                
            offset += self.max_batch_size
            
            # Log progress
            # self.logger.info(f"Fetched {len(all_transactions)} transactions in range so far...")

        self.logger.info(f"Found {len(all_transactions)} transactions in specified height range")
        return sorted(all_transactions, key=lambda x: x['inclusionHeight'])

    async def _make_request(self, 
                          address: str, 
                          offset: int = 0, 
                          limit: int = 100) -> Optional[Dict]:
        """Make a single request to the node API"""
        try:
            url = f"{self.node_url}/blockchain/transaction/byAddress"
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            if self.api_key:
                headers['api_key'] = self.api_key

            params = {
                'offset': offset,
                'limit': limit
            }
            
            self.logger.debug(f"Fetching transactions {offset} to {offset + limit}")
                
            async with self.session.post(url, 
                                       json=address,
                                       params=params, 
                                       headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    response_text = await response.text()
                    self.logger.error(f"Node API error: {response.status}")
                    self.logger.error(f"Response body: {response_text}")
                    return None

        except Exception as e:
            self.logger.error(f"Request error: {str(e)}")
            return None

    async def get_height_range_for_dates(self, 
                                       address: str, 
                                       start_date: datetime, 
                                       end_date: datetime) -> Tuple[Optional[int], Optional[int]]:
        """Find block heights corresponding to date range"""
        try:
            # Get current chain info
            current_height = await self.get_current_height()
            if not current_height:
                return None, None
    
            self.logger.info(f"Current blockchain height: {current_height}")
            
            # Calculate blocks based on time difference
            seconds_per_block = 120  # 2 minutes per block
            end_timestamp = end_date.timestamp()
            start_timestamp = start_date.timestamp()
            time_diff_seconds = end_timestamp - start_timestamp
            
            # Calculate number of blocks in the time range
            blocks_in_range = int(time_diff_seconds / seconds_per_block)
            self.logger.info(f"Time range represents approximately {blocks_in_range} blocks")
            
            # Calculate end height (use current height or estimate from end date)
            end_height = min(current_height, current_height - int((datetime.now().timestamp() - end_timestamp) / seconds_per_block))
            
            # Calculate start height by going back the required number of blocks
            start_height = max(0, end_height - blocks_in_range)
            
            self.logger.info(f"Calculated height range: {start_height} to {end_height}")
            self.logger.info(f"Range spans {end_height - start_height} blocks")
            
            return start_height, end_height
    
        except Exception as e:
            self.logger.error(f"Error calculating height range: {e}")
            return None, None