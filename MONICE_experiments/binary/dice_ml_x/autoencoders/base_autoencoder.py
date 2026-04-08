"""
Abstract base class for auto encoders.

`_BaseAutoEncoder` class is defined in the module as a common interface
for implementing various auto encoder algorithms.
"""

import torch
from torch import nn
from abc import ABC, abstractmethod
import numpy as np
import json
from torch.utils.data import DataLoader

class _BaseAutoEncoder(nn.Module, ABC):
    """
    Abstract base class for auto encoders.

    All auto encoder classes that will inherit this class is enforced to implement
    `forward` and `train_autoencoder` methods.
    """
    def __init__(self) -> None:
        """
        Initializes the _BaseAutoEncoder class.

        Returns:
            None
        """
        super(_BaseAutoEncoder, self).__init__()
        self._history: dict = {'loss': []}

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Implements forward pass through the auto encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the auto encoder (decoded).
        """
        raise NotImplementedError("forward() method must be implemented in the subclass")
    
    @abstractmethod
    def train_autoencoder(self, X: torch.Tensor | np.ndarray | DataLoader,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        epsilon: float = 1.0,
        verbose: bool = True,
        device: torch.device | str = "cpu",
        save_model: bool = False,
        save_interval: int = 5,) -> None:
        """
        Trains the autoencoder with given parameters.

        Args:
            X (torch.Tensor): Input data.
            epochs (int): Number of epochs for training.
            batch_size (int): Number of samples in a batch.
            learning_rate (float): Learning rate.
            verbose (bool): Prints extra information if set to `True`.

        Returns:
            None
        """
        raise NotImplementedError("train_autoencoder() method must be implemented in the subclass")
    
    def save_model(self, model_dir: str="autoencoder_model") -> None:
        """
        Saves the trained model and the history of the training.

        Args:
            model_dir (str): Directory to save the model.

        Returns:
            None
        """
        torch.save(self.state_dict(), f"{model_dir}.pth")
        with open(f"{model_dir}.json", "w") as history_file:
            json.dump(self._history, history_file)
        print(f"Model and history saved to {model_dir}.pth and {model_dir}.json")
    
    def load_model(self, model_dir: str) -> None:
        """
        Loads a model from the given directory.

        Args:
            model_dir (str): Directory to load the model.

        Returns:
            None
        """
        self.load_state_dict(torch.load(model_dir))
        try:
            with open(f"{model_dir}.json", "r") as history_file:
                self._history = json.load(history_file)
        except FileNotFoundError:
            print("No training history found!")
        print("Model has been loaded successfully!")
