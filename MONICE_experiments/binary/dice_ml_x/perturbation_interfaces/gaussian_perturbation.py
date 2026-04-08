"""
Gaussian perturbation implementation.
"""

from dice_ml_x.perturbation_interfaces.base_perturbation import _BasePerturbation
import numpy as np
import pandas as pd
import torch

class GaussianPerturbation(_BasePerturbation):
    """
    Implements Gaussian perturbation strategy for counterfactual instances.
    
    Generates perturbations by adding noise to continuous features and randomly
    changes the categorical features.

    Attributes:
        std_dev (float): Standard deviation for Gaussian noise that will be added
                to the continuous features.
        continuous_features (list): List of continuous feature names.
        categorical_features (dict): Dictionary of categorical features and their possible values.
    """

    def __init__(self, std_dev=0.1, continuous_features: list = [], categorical_features: dict = {}, categorical_indices: dict = {}) -> None:
        """
        Initializes the GaussianPerturbation with features given.

        Args:
            std_dev (float): Standard deviation for Gaussian noise that will be added
                to the continuous features.
            continuous_features (list): List of continuous feature names.
            categorical_features (dict): Dictionary of categorical features and their possible values.

        Returns:
            None
        """
        self.std_dev = std_dev
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features

    def generate(self, c_i: pd.DataFrame) -> pd.DataFrame:
        """
        Generates perturbations for continuous and categorical features.

        Args:
            c_i (pandas.DataFrame): The counterfactual instance that will be perturbed.
            std_dev (float): Standard deviation for Gaussian noise that will be added
                to the continuous features.

        Returns:
            pandas.DataFrame: A perturbed version of the given counterfactual explanation.
        """
        c_i_prime = c_i.copy().reset_index(drop=True)
        # Handle the continuous features
        for feature in self.continuous_features:
            if feature in c_i.columns:
                c_i_prime[feature] += np.random.normal(0, self.std_dev)

        # Handle the categorical features
        for cat_feature, cats in self.categorical_features.items():
            if cat_feature in c_i.columns:
                current_val = c_i[cat_feature].values[0]
                valid_vals = [cat for cat in cats if cat != current_val]
                if valid_vals:
                    c_i_prime.at[0, cat_feature] = np.random.choice(valid_vals)

        c_i_prime = c_i_prime.iloc[:1].reset_index(drop=True)

        return c_i_prime