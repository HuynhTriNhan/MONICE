from dice_ml_x.autoencoders.base_autoencoder import _BaseAutoEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.adam import Adam
import numpy as np
from tqdm import tqdm


class VariationalAutoEncoder(_BaseAutoEncoder):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, beta: float=1.0) -> None:
        super(VariationalAutoEncoder, self).__init__()
        self.beta = beta
        self._history['kl_loss'] = []
        self._history['reconstruction_loss'] = []
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim),
            nn.GELU()
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, input_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def reparam(self, mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        s_dev = torch.exp(0.5 * log_variance)
        epsilon = torch.randn_like(s_dev)
        return mu + s_dev * epsilon
    
    def forward(self, x: torch.Tensor) -> tuple:
        hidden_activation = torch.relu(self.encoder(x))
        mu = self.fc_mu(hidden_activation)
        log_variance = self.fc_log_var(hidden_activation)
        z = self.reparam(mu, log_variance)
        decoded = self.decoder(z)
        return decoded, mu, log_variance
    
    def _compute_vae_loss(self, x: torch.Tensor, decoded_x: torch.Tensor, mu: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        
        reconstruction_loss = nn.functional.mse_loss(decoded_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(torch.sum(1 + log_variance - mu.pow(2) - log_variance.exp(), dim=1))
        total_loss = reconstruction_loss + self.beta * kl_loss
        return kl_loss, reconstruction_loss, total_loss
    
    def train_autoencoder(self, X, epochs=10, batch_size=16, learning_rate=0.001, verbose=True, save_model=False, save_interval=5):
        optimizer = Adam(self.parameters(), lr=learning_rate)

        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        
            dataset = TensorDataset(X)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if isinstance(X, DataLoader):
            dataloader = X

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            total_kl_loss = 0
            total_recons_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
            
            for batch in progress_bar:
                x_batch = batch[0]
                optimizer.zero_grad()

                x_decoded, mu, log_variance = self.forward(x_batch)
                kl_loss, reconstruction_loss, loss = self._compute_vae_loss(x_batch, x_decoded, mu, log_variance)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_kl_loss += kl_loss.item()
                total_recons_loss += reconstruction_loss.item()
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            average_loss = total_loss / len(dataloader)
            average_kl_loss = total_kl_loss / len(dataloader)
            average_recons_loss = total_recons_loss / len(dataloader)
            self._history['loss'].append(average_loss)
            self._history['kl_loss'].append(average_kl_loss)
            self._history['reconstruction_loss'].append(average_recons_loss)

            if save_model and epoch % save_interval == 0:
                self.save_model(f"autoencoder_model_{epoch}")

            if verbose:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}")
        print("Training finished!")



        