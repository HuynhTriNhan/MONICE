import numpy as np
from abc import ABC, abstractmethod
from experiments.models.classification_trainer import SklearnTabularModeler
from experiments.core.data_loader import TabularDataLoader
from experiments.models.autoencoder_trainer import AutoencoderTrainer

class DistanceMetric(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        pass

class GowerDistanceMetric(DistanceMetric):
    def __init__(self, X_train: np.ndarray, num_feats: list, cat_feats: list, eps: float = 1e-6):
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.eps = eps
        if num_feats:
            self.scale = X_train[:, num_feats].max(axis=0) - X_train[:, num_feats].min(axis=0)
            self.scale[self.scale < eps] = eps
        else:
            self.scale = np.array([])
        self.X_train = X_train

    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = np.zeros(X2.shape[0])
        count = 0
        if self.num_feats:
            num_diff = np.abs(X2[:, self.num_feats] - X1[0, self.num_feats]) / self.scale
            dist += np.sum(num_diff, axis=1)
            count += len(self.num_feats)
        if self.cat_feats:
            cat_diff = X2[:, self.cat_feats] != X1[0, self.cat_feats]
            dist += np.sum(cat_diff, axis=1)
            count += len(self.cat_feats)
        return dist / count if count > 0 else dist

class HEOMDistance(DistanceMetric):
    def __init__(self, X_train: np.ndarray, num_feats: list, cat_feats: list, eps: float = 1e-6):
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.eps = eps
        if num_feats:
            self.scale = X_train[:, num_feats].max(axis=0) - X_train[:, num_feats].min(axis=0)
            self.scale[self.scale < eps] = eps
        else:
            self.scale = np.array([])
        self.X_train = X_train
        
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        dist = np.zeros(X2.shape[0])
        if self.num_feats:
            num_diff = ((X2[:, self.num_feats] - X1[0, self.num_feats]) / self.scale) ** 2
            dist += np.sum(num_diff, axis=1)
        if self.cat_feats:
            cat_diff = X2[:, self.cat_feats] != X1[0, self.cat_feats]
            dist += np.sum(cat_diff, axis=1)
        return np.sqrt(dist)

class NearestNeighborFinder:
    """Find nearest neighbors from target class."""
    
    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric
    
    def sorted_nearest_neighbor(self, X: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Find sorted nearest neighbors."""
        if len(candidates) == 0:
            return np.array([])
        
        distances = self.distance_metric.distance(X, candidates)
        sorted_indices = np.argsort(distances)
        return candidates[sorted_indices]
    
    def avg_distance_5NN(self, X: np.ndarray) -> float:
        """Find average distance of 5 nearest neighbors."""
        
        distances = self.distance_metric.distance(X, self.distance_metric.X_train)
        sorted_indices = np.argsort(distances)
        return np.mean(distances[sorted_indices[:5]])

class InterpretabilityMetric:
    def __init__(self, model: SklearnTabularModeler, autoencoder: AutoencoderTrainer, preprocessor, eps = 1e-12):   
        self.model = model
        self.autoencoder = autoencoder 
        self.preprocessor = preprocessor
        self.eps = eps
        
    def IM1(self, X: np.ndarray) -> float:
        # expected x_cf close to class i 
        # IM1(AE_i, AE_t0, x_cf) = L2(x_cf - AE_i(x_cf))^2/ L2(x_cf - AE_t0(x_cf))^2 + eps
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        cf_class = self.model.predict_proba(X).argmax(axis=1)
        
        results = []
        for i, x in enumerate(X):
            x_reshaped = x.reshape(1, -1)
            x_preprocessed = self.preprocessor.transform(x_reshaped)
            ae_i = self.autoencoder.predict(x_reshaped, cf_class[i])
            ae_t0 = self.autoencoder.predict(x_reshaped, (1-cf_class[i]))
            
            numerator = np.linalg.norm(x_preprocessed - ae_i, ord=2) ** 2
            denominator = np.linalg.norm(x_preprocessed - ae_t0, ord=2) ** 2 + self.eps
            im1 = numerator / denominator
            results.append(im1)
        
        return np.array(results)
    
    def IM2(self, X: np.ndarray) -> float:
        # expected x_cf close to class i 
        # IM2(AE_i, AE_t0, x_cf) = L2(AE_i(x_cf) - AE(x_cf))/ (L1(x_cf) + eps)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        cf_class = self.model.predict_proba(X).argmax(axis=1)
        
        results = []
        for i, x in enumerate(X):
            x_reshaped = x.reshape(1, -1)
            x_preprocessed = self.preprocessor.transform(x_reshaped)
            ae_i = self.autoencoder.predict(x_reshaped, cf_class[i])
            ae = self.autoencoder.predict(x_reshaped)
            
            numerator = np.linalg.norm(ae_i - ae, ord=2) ** 2
            denominator = np.linalg.norm(x_preprocessed, ord=1) + self.eps
            im2 = numerator / denominator
            results.append(im2)
        
        return np.array(results)
