"""Preprocessing utilities and autoencoder implementation."""

import numpy as np
from math import ceil
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from monice import AutoencoderPlausibility


class MLPAutoencoder(AutoencoderPlausibility):
    """
    MLP-based autoencoder for plausibility modeling.
    
    Uses reconstruction error as a measure of data plausibility.
    """

    def __init__(self, X_train, grid_params, preprocessor=None, criterion='lower'):
        """
        Initialize MLP autoencoder.
        
        Args:
            X_train: Training data
            grid_params: Parameters for grid search
            preprocessor: Optional preprocessor for data
            criterion: Optimization criterion ('lower' or 'higher')
        """
        self.X_train_raw = X_train 
        self.preprocessor = preprocessor 
        self.grid_params = grid_params
        self.ae = None
        
        # Process data
        if self.preprocessor is not None:
            self.X_train = self.preprocessor.transform(X_train)
        else:
            self.X_train = np.asarray(X_train)
            
        if len(self.X_train.shape) != 2:
            raise ValueError("X_train must be a 2D array")
            
        self.input_dim = self.X_train.shape[1]
        super().__init__(self._predict_wrapper, criterion)

    def fit(self):
        """Train the autoencoder using grid search."""
        # Design architecture
        latent_dim = 2 if ceil(self.input_dim / 4) < 4 else 4
        hidden_layers = (
            ceil(self.input_dim / 2),
            ceil(self.input_dim / 4),
            latent_dim,
            ceil(self.input_dim / 4),
            ceil(self.input_dim / 2),
        )
        print(f"Autoencoder hidden layers: {hidden_layers}")

        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='tanh',
            solver='adam',
            learning_rate_init=0.005,
            learning_rate='adaptive',
            max_iter=100,
            tol=1e-5,
            verbose=0,
            validation_fraction=0.2,
            early_stopping=True,
            n_iter_no_change=5
        )
        
        # Grid search for hyperparameters
        grid_search = GridSearchCV(model, self.grid_params, cv=3, n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.X_train)
    
        self.ae = grid_search.best_estimator_
        print(f"Best autoencoder params: {grid_search.best_params_}")

    def _predict_wrapper(self, X):
        if self.ae is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
            
        # Process data
        if self.preprocessor is not None:
            X_processed = self.preprocessor.transform(X)
        else:
            X_processed = np.asarray(X)
            
        # Calculate reconstruction error
        X_recon = self.ae.predict(X_processed)
        rmse = np.sqrt(((X_processed - X_recon) ** 2).mean(axis=1))
        
        return rmse


def preprocessing_pipeline(num_idx, cat_idx):
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_idx),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_idx)
    ])
    
    return preprocessor 