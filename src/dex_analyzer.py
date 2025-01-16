import re
from dataclasses import dataclass
import aiohttp
import json
from datetime import datetime
import logging
from typing import Dict, Optional, List, Set, Tuple
import asyncio

@dataclass
class SwapEvent:
    tx_id: str
    timestamp: int
    height: int
    input_token: str
    input_token_name: str
    input_token_decimals: int
    output_token: str
    output_token_name: str
    output_token_decimals: int
    input_amount: float  # Raw amount
    output_amount: float  # Raw amount
    input_amount_adjusted: float  # Adjusted for decimals
    output_amount_adjusted: float  # Adjusted for decimals
    price: float  # Using adjusted amounts
    fee_erg: float
    is_lp_action: bool
    erg_liquidity: float = 0.0  # Current ERG in pool after swap
    token_liquidity: float = 0.0  # Current token amount in pool after swap
    price_impact: float = 0.0  # Estimated price impact of this swap


@dataclass
class TokenInfo:
    id: str
    name: str
    decimals: int
    is_lp: bool = False
    trade_count: int = 0
    volume_erg: float = 0.0

class TokenRegistry:
    def __init__(self, node_url: str):
        self.node_url = node_url
        self.tokens: Dict[str, TokenInfo] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        self.logger = logging.getLogger(__name__)
        self.lp_patterns = [
            r'.*_LP$',
            r'.*LP_.*',
            r'.*[\s_-]LP[\s_-].*',
            r'LP[\s_-].*',
            r'.*Liquidity.*',
            r'Pool.*Token',
        ]

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    def is_lp_token(self, token_name: str) -> bool:
        """Enhanced LP token detection using multiple patterns"""
        token_name = token_name.strip()
        return any(re.match(pattern, token_name, re.IGNORECASE) for pattern in self.lp_patterns)

    async def get_token_info(self, token_id: str) -> Optional[TokenInfo]:
        """Fetch token information from the node or return predefined info for ERG."""
        if token_id == "ERG":
            return TokenInfo(
                id="ERG",
                name="ERG",
                decimals=9,
                is_lp=False
            )
    
        if token_id in self.tokens:
            return self.tokens[token_id]
    
        try:
            if not self.session:
                await self.init_session()
    
            url = f"{self.node_url}/blockchain/token/byId/{token_id}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    name = data.get('name', 'Unknown')
                    decimals = data.get('decimals', 0)
                    is_lp = self.is_lp_token(name)
    
                    token_info = TokenInfo(
                        id=token_id,
                        name=name,
                        decimals=decimals,
                        is_lp=is_lp
                    )
                    self.tokens[token_id] = token_info
                    return token_info
                else:
                    self.logger.error(f"Failed to fetch token info for {token_id}: {response.status}")
                    return None
        except Exception as e:
            self.logger.error(f"Error fetching token info: {e}")
            return None


class DEXAnalyzer:
    def __init__(self, dex_address: str, node_url: str, min_trades: int = 10):
        self.dex_address = dex_address
        self.token_registry = TokenRegistry(node_url)
        self.logger = logging.getLogger(__name__)
        self.min_trades = min_trades
        self.token_trade_counts = {}

    async def _get_pool_liquidity(self, tx: Dict, trading_token_id: str) -> Tuple[float, float]:
        """Get current pool liquidity from first output box for specific token"""
        # First output box contains pool state after swap
        pool_box = tx['outputs'][0]
        
        # Get ERG amount (convert nanoERG to ERG)
        erg_liquidity = pool_box['value'] / 1e9
        
        # Get token amount for the specific trading token
        token_liquidity = 0
        if pool_box.get('assets'):
            for asset in pool_box['assets']:
                if asset['tokenId'] == trading_token_id:
                    token_liquidity = asset['amount']
                    break
                    
        return erg_liquidity, token_liquidity
        
    async def analyze_transaction(self, tx: Dict) -> Optional[SwapEvent]:
        try:
            if not self._is_dex_transaction(tx):
                return None
    
            # Calculate token changes
            changes = self._calculate_token_changes(tx)
            if len(changes) < 2:
                return None  # A swap requires at least two assets to change
    
            # Identify the LP's output and the trader's received asset
            input_token, output_token = None, None
            input_amount, output_amount = 0, 0
    
            for token_id, net_change in changes.items():
                if net_change < 0:
                    input_token = token_id
                    input_amount = abs(net_change)  # The LP is losing this asset
                elif net_change > 0:
                    output_token = token_id
                    output_amount = net_change  # The LP is providing this asset
    
            if not (input_token and output_token):
                return None  # Ensure both input and output tokens are identified
    
            # Fetch token information
            input_token_info = await self.token_registry.get_token_info(input_token)
            output_token_info = await self.token_registry.get_token_info(output_token)
    
            if not input_token_info or not output_token_info:
                return None
                
            # Skip if either token is an LP token
            if input_token_info.is_lp or output_token_info.is_lp:
                return None
    
            # Adjust amounts based on decimals
            input_amount_adjusted = input_amount / (10 ** input_token_info.decimals)
            output_amount_adjusted = output_amount / (10 ** output_token_info.decimals)
            price = output_amount_adjusted / input_amount_adjusted if input_amount_adjusted > 0 else 0
            
            # Determine which token to track liquidity for (non-ERG token)
            trading_token_id = output_token if input_token == 'ERG' else input_token
            
            # Get current pool liquidity
            erg_liquidity, token_liquidity = await self._get_pool_liquidity(tx, trading_token_id)
    
            # Calculate price impact
            price_impact = 0.0
            if input_token == 'ERG':
                old_price = (erg_liquidity - input_amount_adjusted) / (token_liquidity + output_amount_adjusted)
                new_price = erg_liquidity / token_liquidity
                price_impact = (new_price - old_price) / old_price if old_price > 0 else 0
    
            return SwapEvent(
                tx_id=tx['id'],
                timestamp=tx['timestamp'],
                height=tx['inclusionHeight'],
                input_token=input_token,
                input_token_name=input_token_info.name,
                input_token_decimals=input_token_info.decimals,
                output_token=output_token,
                output_token_name=output_token_info.name,
                output_token_decimals=output_token_info.decimals,
                input_amount=input_amount,
                output_amount=output_amount,
                input_amount_adjusted=input_amount_adjusted,
                output_amount_adjusted=output_amount_adjusted,
                price=price,
                fee_erg=self._calculate_fee(tx),
                is_lp_action=False,  # We're filtering out LP actions above
                erg_liquidity=erg_liquidity,
                token_liquidity=token_liquidity,
                price_impact=price_impact
            )
        except Exception as e:
            self.logger.error(f"Error analyzing transaction {tx.get('id')}: {e}")
            return None

    async def close(self):
        """Clean up resources"""
        await self.token_registry.close()

    def _is_dex_transaction(self, tx: Dict) -> bool:
        """Check if transaction involves the DEX address"""
        return any(box.get('address') == self.dex_address 
                  for box in tx['inputs'] + tx['outputs'])
        
    def _calculate_token_changes(self, tx: Dict) -> Dict[str, float]:
        """Calculate net token changes for the LP."""
        changes: Dict[str, float] = {}
    
        # Track LP-related token changes
        for box in tx['outputs']:
            if box.get('address') == self.dex_address:
                for asset in box.get('assets', []):
                    token_id = asset['tokenId']
                    amount = asset['amount']
                    changes[token_id] = changes.get(token_id, 0) + amount
                changes['ERG'] = changes.get('ERG', 0) + box.get('value', 0)
    
        for box in tx['inputs']:
            if box.get('address') == self.dex_address:
                for asset in box.get('assets', []):
                    token_id = asset['tokenId']
                    amount = asset['amount']
                    changes[token_id] = changes.get(token_id, 0) - amount
                changes['ERG'] = changes.get('ERG', 0) - box.get('value', 0)
    
        return changes
        
    def _calculate_fee(self, tx: Dict) -> float:
        """Calculate the transaction fee in ERG"""
        input_erg = sum(box.get('value', 0) for box in tx['inputs'])
        output_erg = sum(box.get('value', 0) for box in tx['outputs'])
        return (input_erg - output_erg) / 1e9