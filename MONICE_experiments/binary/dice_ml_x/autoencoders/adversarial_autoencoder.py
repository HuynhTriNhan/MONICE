from dice_ml_x.autoencoders.base_autoencoder import _BaseAutoEncoder
from torch import nn

class AdversarialAutoEncoder(_BaseAutoEncoder):
    def __init__(self, input_dim, latent_dim, hidden_dim) -> None:
        super(AdversarialAutoEncoder, self).__init__()