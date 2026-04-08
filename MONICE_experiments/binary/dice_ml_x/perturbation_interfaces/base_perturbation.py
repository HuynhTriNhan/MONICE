"""
Abstract base class for various perturbation methods.

`_BasePerturbation` class is defined in the module as a common interface
for implementing various perturbation strategies.
"""

from abc import abstractmethod, ABC
import pandas as pd
import numpy as np

class _BasePerturbation(ABC):
    """
    Abstract base class for different perturbation methods
    
    All strategies that will inherit this class is enforced to implement
    `generate` and `validate` methods.
    """

    @abstractmethod
    def generate(self, c_i: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generates perturbations for the given counterfactual c_i

        Args:
            c_i (pd.DataFrame): The counterfactual instance that will be perturbed.
            **kwargs: Additional arguments that will be used to specify the 
                strategy to generate perturbations.
        
        Returns: 
            pandas.DataFrame: The perturbed version of the given counterfactual explanation.
        """
        pass


    def validate(self, c_i: pd.DataFrame, c_i_prime: pd.DataFrame, target_class: int, predict_fn: callable, tol: float) -> bool:
        """
        Validates that the model outcomes the same output both for c_i the
        counterfactual instance and c_i_prime the perturbed counterfactual.

        Args:
            c_i (pandas.DataFrame): The original counterfactual instance.
            c_i_prime (pandas.DataFrame): The perturbed counterfactual instance.
            target_class (int): The target class
            predict_fn (callable): The prediction function of the model.
            tol (float): Tolerance for difference between two predictions
        Returns:
            bool: Boolean that indicates the validity of the perturbed counterfactual.
        """
        try:
            pred_c_i = predict_fn(c_i)
            pred_c_i_prime = predict_fn(c_i_prime)
            pred_c_i = pred_c_i[0][target_class]
            pred_c_i_prime = pred_c_i_prime[0][target_class]

            return np.abs(pred_c_i - pred_c_i_prime) <= tol
        except Exception as e:
            print(f"An exception occurred: {e}, for c_i:\n {c_i.head()}\n, c_i_prime:\n {c_i_prime.head()}")
            raise