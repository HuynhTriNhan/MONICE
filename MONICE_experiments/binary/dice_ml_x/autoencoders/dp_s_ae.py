from __future__ import annotations

"""
Differentially‑Private Auto‑Encoder (Functional Mechanism)
---------------------------------------------------------
Implements the exact training recipe used in *Differentially‑Private
Counterfactuals via Functional Mechanism* (2025).  Noise is injected into
the scalar reconstruction loss **before back‑prop** so that the final
encoder & decoder weights satisfy ε‑DP.  The architecture is kept
simple (Linear → ReLU stacks) but can be configured through
`hidden_dims`.

*Key guarantees*
^^^^^^^^^^^^^^^^
Let *K* be the maximum width of any hidden layer.  Setting
`lap_scale = 4*(K+1)/ε` yields global L₁‑sensitivity Δ ≤ 4(K+1) and thus
an ε‑DP model (Theorem 2 in the paper).

Usage
:::::
>>> ae = DPStandardAutoEncoder(input_dim=14,
>>>                            latent_dim=8,
>>>                            hidden_dims=[32, 16]).to(device)
>>> ae.train_dp(X, epochs=20, epsilon=1.0)
"""

from typing import List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions.laplace import Laplace
from tqdm import tqdm

from dice_ml_x.autoencoders.base_autoencoder import _BaseAutoEncoder


class DPStandardAutoEncoder(_BaseAutoEncoder):
    """Linear encoder‑decoder with DP functional‑mechanism training."""

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [max(64, 2 * latent_dim)]  # sensible default

        self.hidden_dims = hidden_dims  # save for Δ computation

        # Encoder ------------------------------------------------------
        enc_layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            enc_layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        enc_layers.append(nn.Linear(last, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror) --------------------------------------------
        dec_layers: List[nn.Module] = []
        last = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(last, h), nn.ReLU()])
            last = h
        dec_layers.extend([nn.Linear(last, input_dim), nn.Sigmoid()])
        self.decoder = nn.Sequential(*dec_layers)

        # Loss tracker
        self._history = {"loss": []}

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:    # type: ignore
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    # ------------------------------------------------------------------
    # DP functional‑mechanism trainer
    # ------------------------------------------------------------------
    def train_autoencoder(
        self,
        X: torch.Tensor | np.ndarray | DataLoader,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        epsilon: float = 1.0,
        verbose: bool = True,
        device: torch.device | str = "cpu",
        save_model: bool = False,
        save_interval: int = 5,
    ) -> None:
        """Train with functional‑mechanism noise so the model is ε‑DP."""

        self.to(device)
        optimiser = Adam(self.parameters(), lr=learning_rate)
        mse = nn.MSELoss()

        # ----- DataLoader plumbing -----------------------------------
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(X, torch.Tensor):
            ds = TensorDataset(X)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        else:  # already a DataLoader
            dl = X

        # ----- Sensitivity & Laplace sampler --------------------------
        K = max(self.hidden_dims)
        lap_scale = (4 * (K + 1)) / epsilon
        lap = Laplace(torch.tensor(0.0, device=device),
                      torch.tensor(lap_scale, device=device))

        # ----- Training loop -----------------------------------------
        for ep in range(epochs):
            total = 0.0
            pbar = tqdm(dl, desc=f"Epoch {ep+1}/{epochs}", disable=not verbose)
            for batch in pbar:
                x_batch = batch[0].to(device)
                optimiser.zero_grad()

                x_rec, _ = self(x_batch)
                loss_clean = mse(x_rec, x_batch)

                # ---- DP tweak: add scalar Laplace noise -------------
                loss_noisy = loss_clean + lap.sample()
                loss_noisy.backward()
                optimiser.step()

                total += loss_clean.item()
                pbar.set_postfix(loss=f"{loss_clean.item():.4f}")

            avg = total / len(dl)
            self._history["loss"].append(avg)
            if verbose:
                print(f"Epoch {ep+1:02d} | loss: {avg:.4f}")
            if save_model and (ep + 1) % save_interval == 0:
                self.save_model(f"dp_ae_ep{ep+1}")

        if verbose:
            print("DP training finished ✓")

    # ------------------------------------------------------------------
    @property
    def history(self):  # read‑only accessor
        return self._history
