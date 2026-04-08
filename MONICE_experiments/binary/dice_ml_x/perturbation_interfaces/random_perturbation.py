"""
Random perturbation implementation for counterfactual instances.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import pandas as pd
import numpy as np


class RandomPerturbation(_BasePerturbation):
    """
    Implements random perturbation strategy for counterfactual instances.

    Perturbs the continuous features randomly within a given range and modifies
    the categorical features.
    """
    def __init__(self, continuous_features: list = [], categorical_features: dict = {}, feature_ranges: dict = {}) -> None:
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.feature_ranges = feature_ranges

    def generate(self, c_i: pd.DataFrame) -> pd.DataFrame:
        """
        Generates random perturbations for both continuous and categorical features.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance to be perturbed.
            continuous_features (list): List of continuous features.
            categorical_features (dict): Categorical features with their possible
                values.
            feature_ranges (dict): Ranges for continuous features as {feature: (min, max)}.
        """
        c_i_prime = c_i.copy()

        for feature in self.continuous_features:
            if feature in c_i.columns:
                low, high = self.feature_ranges.get(feature, (0, 1))
                c_i_prime[feature] = np.random.uniform(low, high)

        
        for cat_feature, cat_vals in self.categorical_features.items():
            if cat_feature in c_i.columns:
                c_i_prime[cat_feature] = np.random.choice(cat_vals)

        return c_i_prime