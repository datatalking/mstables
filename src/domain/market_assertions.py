from typing import Dict, List, Optional, Union
import numpy as np
from scipy import stats
import pandas as pd

class MarketAssertions:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.results: Dict[str, Dict] = {}

    def test_normality(self, column: str) -> Dict[str, float]:
        """Test if returns follow normal distribution."""
        data = self.data[column].dropna()
        stat, p_value = stats.normaltest(data)
        self.results[f"{column}_normality"] = {
            "statistic": stat,
            "p_value": p_value,
            "is_normal": p_value > 0.05
        }
        return self.results[f"{column}_normality"]

    def test_tail_risk(self, column: str, confidence: float = 0.99) -> Dict[str, float]:
        """Compare empirical vs theoretical tail risks."""
        data = self.data[column].dropna()
        empirical_var = np.percentile(data, (1-confidence)*100)
        theoretical_var = stats.norm.ppf(1-confidence, np.mean(data), np.std(data))

        self.results[f"{column}_tail_risk"] = {
            "empirical_var": empirical_var,
            "theoretical_var": theoretical_var,
            "tail_risk_ratio": abs(empirical_var/theoretical_var)
        }
        return self.results[f"{column}_tail_risk"]

    def test_correlation_stability(self, col1: str, col2: str, window: int = 252) -> pd.Series:
        """Test if correlations are stable over time."""
        rolling_corr = self.data[col1].rolling(window).corr(self.data[col2])
        self.results[f"corr_stability_{col1}_{col2}"] = {
            "mean": rolling_corr.mean(),
            "std": rolling_corr.std(),
            "stability_score": 1 / rolling_corr.std() if rolling_corr.std() != 0 else float('inf')
        }
        return rolling_corr