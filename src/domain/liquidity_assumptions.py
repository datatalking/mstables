from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

class LiquidityAnalyzer:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.metrics: Dict[str, pd.DataFrame] = {}

    def calculate_amihud_illiquidity(self, returns_col: str, volume_col: str, window: int = 21) -> pd.Series:
        """Calculate Amihud illiquidity ratio."""
        daily_illiq = abs(self.data[returns_col]) / (self.data[volume_col] * self.data[returns_col])
        return daily_illiq.rolling(window=window).mean()

    def analyze_bid_ask_spread(self, bid_col: str, ask_col: str, window: int = 21) -> pd.DataFrame:
        """Analyze bid-ask spread patterns."""
        spreads = (self.data[ask_col] - self.data[bid_col]) / ((self.data[ask_col] + self.data[bid_col]) / 2)

        spread_metrics = pd.DataFrame({
            'spread': spreads,
            'rolling_mean': spreads.rolling(window=window).mean(),
            'rolling_std': spreads.rolling(window=window).std()
        })

        self.metrics['spreads'] = spread_metrics
        return spread_metrics

    def estimate_market_impact(self, volume_col: str, price_col: str, trade_size: float) -> pd.Series:
        """Estimate market impact for given trade size."""
        daily_volume = self.data[volume_col]
        price = self.data[price_col]

        # Simple square root market impact model
        participation_rate = trade_size / daily_volume
        market_impact = 0.1 * price * np.sqrt(participation_rate)

        return market_impact

    def detect_liquidity_regimes(self, volume_col: str, window: int = 63) -> pd.DataFrame:
        """Detect changes in liquidity regimes."""
        volume = self.data[volume_col]

        liquidity_metrics = pd.DataFrame({
            'volume': volume,
            'rolling_mean': volume.rolling(window=window).mean(),
            'rolling_std': volume.rolling(window=window).std(),
            'z_score': (volume - volume.rolling(window=window).mean()) / volume.rolling(window=window).std()
        })

        # Define liquidity regimes
        liquidity_metrics['regime'] = pd.cut(
            liquidity_metrics['z_score'],
            bins=[-np.inf, -2, -1, 1, 2, np.inf],
            labels=['Very Low', 'Low', 'Normal', 'High', 'Very High']
        )

        self.metrics['liquidity_regimes'] = liquidity_metrics
        return liquidity_metrics