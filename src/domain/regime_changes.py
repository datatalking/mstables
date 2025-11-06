from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pandas as pd

class RegimeDetector:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = StandardScaler()
        self.regimes: Dict[str, np.ndarray] = {}
        self.models: Dict[str, GaussianMixture] = {}

    def detect_regimes(self, features: List[str], n_regimes: int = 2) -> np.ndarray:
        """Detect market regimes using Gaussian Mixture Models."""
        X = self.data[features].copy()
        X_scaled = self.scaler.fit_transform(X)

        model = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = model.fit_predict(X_scaled)

        regime_key = "_".join(features)
        self.regimes[regime_key] = regimes
        self.models[regime_key] = model

        return regimes

    def get_regime_characteristics(self, features: List[str]) -> Dict[str, pd.DataFrame]:
        """Get statistical characteristics of each regime."""
        regime_key = "_".join(features)
        if regime_key not in self.regimes:
            raise KeyError("Regimes not detected yet. Run detect_regimes first.")

        characteristics = {}
        for regime in np.unique(self.regimes[regime_key]):
            mask = self.regimes[regime_key] == regime
            regime_data = self.data[features][mask]
            characteristics[f"regime_{regime}"] = pd.DataFrame({
                "mean": regime_data.mean(),
                "std": regime_data.std(),
                "skew": regime_data.skew(),
                "kurtosis": regime_data.kurtosis()
            })
            
        return characteristics

    def predict_regime_probability(self, new_data: pd.DataFrame) -> np.ndarray:
        """Predict probability of being in each regime."""
        regime_key = "_".join(new_data.columns)
        if regime_key not in self.models:
            raise KeyError("Model not trained for these features.")

        X_scaled = self.scaler.transform(new_data)
        return self.models[regime_key].predict_proba(X_scaled)