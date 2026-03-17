import numpy as np
from typing import Callable
from sklearn.metrics import mean_squared_error


class AutoEncoderWrapper:
    def __init__(self, autoencoder_trainer):
        self.autoencoder_trainer = autoencoder_trainer
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Transform input using the preprocessor
        x_preprocessed = self.autoencoder_trainer.preprocessor.transform(x)
        
        # Get reconstruction from autoencoder
        x_reconstructed = self.autoencoder_trainer.default_autoencoder.predict(x_preprocessed)
        
        # Calculate reconstruction error for each sample
        reconstruction_errors = []
        for i in range(x_preprocessed.shape[0]):
            error = mean_squared_error(
                x_preprocessed[i:i+1], 
                x_reconstructed[i:i+1]
            )
            reconstruction_errors.append(error)
        
        return np.array(reconstruction_errors)