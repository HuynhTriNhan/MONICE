import pickle
from math import ceil
import numpy as np
import pandas as pd
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from experiments.core.path_utils import path_generator
from experiments.core.data_loader import TabularDataLoader
from experiments.core.data_preprocessing import PreprocessorBuilder

AE_PARAM_GRID = {
    'learning_rate_init': [0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
}


class AutoencoderTrainer:
    def __init__(self, dataset_name: str):
        self.paths = path_generator(dataset_name)
        self.preprocessor = PreprocessorBuilder(dataset_name).load()
        self.dataset = TabularDataLoader(self.paths['dataset'])

        self.X_train_transformed = self.preprocessor.transform(self.dataset.X_train)
        self.input_dim = self.X_train_transformed.shape[1]
        
        self.class_autoencoders = {}
        self.default_autoencoder = None
        self.training_stats = {}

    def build_autoencoder(self):
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
        return model

    def train_default_autoencoder(self, param_grid=AE_PARAM_GRID):
        start_time = time.time()
        ae_model = self.build_autoencoder()
        gs = GridSearchCV(ae_model, param_grid, n_jobs=-2, verbose=0)
        gs.fit(self.X_train_transformed, self.X_train_transformed)
        self.default_autoencoder = gs.best_estimator_
        self.default_autoencoder.fit(self.X_train_transformed, self.X_train_transformed)
        training_time = time.time() - start_time
        
        # Save training stats
        self.training_stats['default'] = {
            'best_params': gs.best_params_,
            'training_time': training_time,
            'n_samples': len(self.X_train_transformed),
            'hidden_layers': self.default_autoencoder.hidden_layer_sizes
        }
        
        with open(self.paths['autoencoder'], 'wb') as f:
            pickle.dump(self.default_autoencoder, f)
        
    def train_class_autoencoders(self, param_grid=AE_PARAM_GRID):
        """Train separate autoencoders for each class (0 and 1)"""
        y_train = self.dataset.y_train
        
        # binary classification
        for class_label in [0, 1]:
            start_time = time.time()
            # Filter
            class_indices = np.where(y_train == class_label)[0]
            X_class = self.X_train_transformed[class_indices]
            
            if len(X_class) == 0:
                continue
                
            ae_model = self.build_autoencoder()
            gs = GridSearchCV(ae_model, param_grid, n_jobs=-2, verbose=0)
            gs.fit(X_class, X_class)
            
            self.class_autoencoders[class_label] = gs.best_estimator_
            self.class_autoencoders[class_label].fit(X_class, X_class)
            training_time = time.time() - start_time
            
            # Save training stats
            self.training_stats[f'class_{class_label}'] = {
                'best_params': gs.best_params_,
                'training_time': training_time,
                'n_samples': len(X_class),
                'hidden_layers': self.class_autoencoders[class_label].hidden_layer_sizes
            }
            
            with open(self.paths[f'autoencoder_{class_label}'], 'wb') as f:
                pickle.dump(self.class_autoencoders[class_label], f)


    def reconstruction_error(self, X, class_label: int = None):
        """
        Calculate reconstruction error using class-specific autoencoder if class_label is provided,
        otherwise use default autoencoder behavior (__call__)
        
        Args:
            X: Input data
            class_label: Class label (0 or 1) to specify which autoencoder to use
                         If None, uses the default __call__ behavior
        
        Returns:
            Reconstruction error(s) as numpy array
        """
        if class_label is None:
            return self.__call__(X)
        
        if class_label not in [0, 1]:
            raise ValueError(f"Invalid class_label: {class_label}. Must be 0 or 1.")
        
        if class_label not in self.class_autoencoders:
            try:
                with open(self.paths[f'autoencoder_{class_label}'], 'rb') as f:
                    self.class_autoencoders[class_label] = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Autoencoder for class {class_label} not found. Train it first with train_class_autoencoders().")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_preprocessed = self.preprocessor.transform(X)
        
        X_pred = self.class_autoencoders[class_label].predict(X_preprocessed)
        
        error = np.mean((X_preprocessed - X_pred)**2, axis=1)
        return error

    def predict(self, X, class_label: int = None):
        if class_label is None:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            X_preprocessed = self.preprocessor.transform(X)
            return self.default_autoencoder.predict(X_preprocessed)
        
        if class_label not in [0, 1]:
            raise ValueError(f"Invalid class_label: {class_label}. Must be 0 or 1.")
        
        # Load class autoencoder if not already loaded
        if class_label not in self.class_autoencoders:
            try:
                with open(self.paths[f'autoencoder_{class_label}'], 'rb') as f:
                    self.class_autoencoders[class_label] = pickle.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"Autoencoder for class {class_label} not found. Train it first with train_class_autoencoders().")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_preprocessed = self.preprocessor.transform(X)
        X_pred = self.class_autoencoders[class_label].predict(X_preprocessed)
        return X_pred
    
    def __call__(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_preprocessed = self.preprocessor.transform(X)
        X_pred = self.default_autoencoder.predict(X_preprocessed)

        error =  np.mean((X_preprocessed - X_pred)**2, axis=1)
        return error
    
    def statistics(self):

        stats = {}
        y_train = self.dataset.y_train
        
        default_errors = self.__call__(self.dataset.X_train)
        stats['default_avg_error'] = np.mean(default_errors)
        stats['default_median_error'] = np.median(default_errors)
        stats['default_std_error'] = np.std(default_errors)
        
        for class_label in [0, 1]:
            if class_label in self.class_autoencoders:
                class_indices = np.where(y_train == class_label)[0]
                X_class = self.dataset.X_train[class_indices]
                
                # Error of class autoencoder on its own class
                class_errors = self.reconstruction_error(X_class, class_label)
                stats[f'class_{class_label}_avg_error'] = np.mean(class_errors)
                stats[f'class_{class_label}_median_error'] = np.median(class_errors)
                stats[f'class_{class_label}_std_error'] = np.std(class_errors)
                
                # Error of default autoencoder on this class
                default_class_errors = self.__call__(X_class)
                stats[f'default_on_class_{class_label}_avg_error'] = np.mean(default_class_errors)
                
                # Cross-evaluation: other class autoencoder on this class
                other_class = 1 if class_label == 0 else 0
                if other_class in self.class_autoencoders:
                    cross_errors = self.reconstruction_error(X_class, other_class)
                    stats[f'class_{other_class}_on_class_{class_label}_avg_error'] = np.mean(cross_errors)
        
        # Add training information
        for key, value in self.training_stats.items():
            for k, v in value.items():
                stats[f'{key}_{k}'] = v
        
        df = pd.DataFrame([stats])
        
        df.to_csv(self.paths['autoencoder_stats'], index=False)
        # print(f"Autoencoder statistics saved to {self.paths['autoencoder_stats']}")
        
        return stats
        
    def fit(self, param_grid=AE_PARAM_GRID):
        # print("Training default autoencoder...")
        self.train_default_autoencoder(param_grid)
        
        # print("Training class-specific autoencoders...")
        self.train_class_autoencoders(param_grid)
        
        # print("Computing and saving statistics...")
        self.statistics()
            
    def load(self):
        try:
            with open(self.paths['autoencoder'], 'rb') as f:
                self.default_autoencoder = pickle.load(f)
            return self.default_autoencoder
        except FileNotFoundError:
            raise FileNotFoundError(f"Autoencoder not found at {self.paths['autoencoder']}")
    
    def load_class_autoencoder(self, class_label: int):
        if class_label not in [0, 1]:
            raise ValueError(f"Invalid class_label: {class_label}. Must be 0 or 1.")
            
        try:
            with open(self.paths[f'autoencoder_{class_label}'], 'rb') as f:
                self.class_autoencoders[class_label] = pickle.load(f)
            return self.class_autoencoders[class_label]
        except FileNotFoundError:
            raise FileNotFoundError(f"Autoencoder for class {class_label} not found at {self.paths[f'autoencoder_{class_label}']}")

        
