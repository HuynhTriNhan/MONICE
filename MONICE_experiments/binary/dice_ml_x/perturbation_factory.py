"""
This module creates perturbation instances by using factory pattern.

Different perturbation classes that are implemented in the module can be instantiated
dynamically.

Classes:
    PertubationFactory: A factory class to create perturbation strategy instances.
"""

from importlib import import_module

class PerturbationFactory:
    """
    Factory class to create perturbation classes instances dynamically.

    This class allows users to create perturbation intances depending on the method
    specified. Supported methods are `gaussian`, `random`, `spherical`.

    Methods:
        get_perturbation(method: str, **kwargs):
            Creates the desired instance with the specified arguments.

    Example:
        pertubation = PerturbationFactory.get_perturbation("gaussian", std_dev=0.1)
    """

    @staticmethod
    def get_perturbation(method: str, **kwargs):
        """
        Creates and instance of the perturbation instance based on the method.

        Args:
            method (str): The perturbation method. Supported methods:
                - "gaussian" (GaussianPerturbation)
                - "random" (RandomPerturbation)
                - "spherical" (SphericalPerturbation)
            **kwargs: Additional keyword arguments that will be passed to the perturbation
                class.
            
        Returns:
            An instance of the specified class.

        Raises:
            ValueError: If the method name is not one of the supported methods.
            ImportError: If the specified module can not be imported.
        """

        strategy_map = {
            "gaussian": "dice_ml_x.perturbation_interfaces.gaussian_perturbation.GaussianPerturbation",
            "random": "dice_ml_x.perturbation_interfaces.random_perturbation.RandomPerturbation",
            "spherical": "dice_ml_x.perturbation_interfaces.spherical_perturbation.SphericalPerturbation"
        }

        if method not in strategy_map.keys():
            ValueError(f"Unsupported method: {method}. Supported methods are: {list(strategy_map.keys())}")

        class_path = strategy_map[method]
        module_name, class_name = class_path.rsplit(".", 1)

        try:
            module = import_module(module_name)
            perturbation_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {class_path}. Make sure that the module and the class exist") from e
        
        return perturbation_class(**kwargs)
