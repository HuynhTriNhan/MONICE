from dice_ml_x.autoencoders.base_autoencoder import _BaseAutoEncoder
from torch import nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import numpy as np
from tqdm import tqdm


class StandardAutoEncoder(_BaseAutoEncoder):
    """
    Implements a standard auto encoder.

    Attributes:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimension of the latent space.
        hidden_dim (int): Dimension of the hidden layer.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int) -> None:
        """
        Initializes the StandardAutoEncoder class with given arguments.

        Args:
            input_dim (int): Dimension of the input data.
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer.

        Returns:
            None
        """
        super(StandardAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple:     # type: ignore
        """
        Forward pass through the auto encoder.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple: Tuple containing the decoded data and the encoded data respectively.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def train_autoencoder(self, X: torch.Tensor, epochs: int=10, batch_size: int=16, learning_rate: float=1e-3,
                          verbose: bool=True, save_model: bool=False, save_interval: int=5) -> None:
        """
        Trains the auto encoder with given parameters.

        Args:
            X (torch.Tensor): Input data.
            epochs (int): Number of epochs for training.
            batch_size (int): Number of samples in a batch.
            learning_rate (float): Learning rate.

        Returns:
            None
        """
        optimizer = Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if isinstance(X, DataLoader):
            dataloader = X

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch}", unit="batch")     # type: ignore
            
            for batch in progress_bar:
                x_batch = batch[0]
                optimizer.zero_grad()

                x_decoded, _ = self.forward(x_batch)
                loss = criterion(x_decoded, x_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
            
            average_loss = total_loss / len(dataloader)     # type: ignore
            self._history['loss'].append(average_loss)

            if save_model and epoch % save_interval == 0:
                self.save_model(f"autoencoder_model_{epoch}")

            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Overall Loss: {average_loss:.4f}")
        print("Training finished!")