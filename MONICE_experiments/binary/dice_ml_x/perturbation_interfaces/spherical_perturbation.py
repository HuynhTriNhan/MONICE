"""
Spherical perturbation implementation.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import pandas as pd
import numpy as np

class SphericalPerturbation(_BasePerturbation):
    """
    Implements spherical perturbation for counterfactual instances.

    Generates perturbations within a spherical boundary constructed around the
    given counterfactual instance.

    Attributes:
        radius (float): The radius of the sphere that will be generated around the data point.
        continuous_features (list): List of continuous feature names.
        categorical_features (dict): Dictionary of categorical features and their possible values.
        feature_ranges (dict): Dictionary of feature ranges.
    """

    def __init__(self, radius: float = 1.0, continuous_features: list = [],
                 categorical_features: list = [], feature_ranges: dict = {}) -> None:
        """
        Initializes the required attributes for the generation and validation processes.

        Args:
            radius (float): The radius of the sphere that will be generated around the data point.
            continuous_features (list): List of continuous feature names.
            categorical_features (dict): Dictionary of categorical features and their possible values.
            feature_ranges (dict): Dictionary of feature ranges.
        
        Returns:
            None: Nothing is returned.
        """
        self.radius = radius
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.feature_ranges = feature_ranges

    def generate(self, c_i: pd.DataFrame) -> pd.DataFrame:
        """
        Generates perturbations within a spherical boundary around the given counterfactual instance.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance that will be perturbed.
            radius (float): Radius of the sphere that will be constructed around the 
                given counterfactual instance.
            continuous_features (list): List of continuous features.
        Returns:
            pandas.DataFrame: A perturbed version of the given counterfactual explanation.
        """
        c_i_prime = c_i.copy().reset_index(drop=True)

        for feature in self.continuous_features:
            if feature in c_i.columns:
                feature_min, feature_max = self.feature_ranges.get(feature, (0, 1))
                scaled_radius = self.radius * (feature_max - feature_min)
                perturbation = np.random.uniform(-scaled_radius, scaled_radius)
                c_i_prime[feature] += perturbation
                c_i_prime[feature] = np.clip(c_i_prime[feature], feature_min, feature_max)

        for cat_feat, cat_vals in self.categorical_features.items():
            if cat_feat in c_i.columns:
                current_val = c_i[cat_feat].values[0]
                valid_vals = list(set(cat_vals) - set([current_val]))
                if valid_vals and np.random.rand() < 0.5:
                    c_i_prime.at[0, cat_feat] = np.random.choice(valid_vals)
        return c_i_prime