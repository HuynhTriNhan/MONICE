import pickle
from typing import List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from experiments.core.path_utils import path_generator
from experiments.core.data_loader import TabularDataLoader

class Preprocessor:
    """
    A wrapper module for mixed-type preprocessing using MinMaxScaler and OneHotEncoder.
    Can be saved and reused with pickle.
    """
    def __init__(self, categorical_indices: Optional[List[int]] = None, continuous_indices: Optional[List[int]] = None):
        self.categorical_indices = categorical_indices or []
        self.continuous_indices = continuous_indices or []
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> ColumnTransformer:
        transformers = []

        if self.continuous_indices:
            transformers.append(
                ("minmax", MinMaxScaler(feature_range=(-1, 1)), self.continuous_indices)
            )

        if self.categorical_indices:
            transformers.append(
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_indices)
            )

        return ColumnTransformer(transformers)

    def fit(self, X):
        self.pipeline.fit(X)

    def transform(self, X):
        return self.pipeline.transform(X)

    def fit_transform(self, X):
        return self.pipeline.fit_transform(X)

    def inverse_transform(self, X):
        return self.pipeline.inverse_transform(X)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)
        
class PreprocessorBuilder:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.paths = path_generator(dataset_name)
        self.data = TabularDataLoader(self.paths['dataset'])
        self.preprocessor = Preprocessor(self.data.categorical_indices, self.data.continuous_indices)

    def build(self):
        self.preprocessor.fit(self.data.X_train)
        self.preprocessor.save(self.paths['preprocessor'])
        
    def load(self):
        try: 
            preprocessor = Preprocessor.load(self.paths['preprocessor'])
        except FileNotFoundError:
            self.build()
            preprocessor = Preprocessor.load(self.paths['preprocessor'])
        return preprocessor