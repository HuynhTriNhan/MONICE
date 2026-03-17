import pickle
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from experiments.core.path_utils import path_generator
from experiments.core.data_loader import TabularDataLoader
from experiments.core.data_preprocessing import PreprocessorBuilder

MODELS = {
    'RF': RandomForestClassifier,
    'ANN': MLPClassifier
}

STATIC_PARAMS = {
    'RF': {'class_weight': 'balanced'},
    'ANN': {
        'solver': 'lbfgs',
        'learning_rate': 'adaptive',
        'early_stopping': True,
        'max_iter': 1000,
        'tol': 0.0001
    }
}


class Modeler(ABC):
    @abstractmethod
    def grid_search(self):
        pass

    @abstractmethod
    def save(self):
        pass
    @abstractmethod
    def load(self):
        pass


class SklearnTabularModeler(Modeler):
    def __init__(self, dataset_name: str, model: str):
        self.dataset_name = dataset_name
        self.model_name = model
        self.paths = path_generator(self.dataset_name, model)
        self.model = MODELS[model](**STATIC_PARAMS[model])
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.preprocessor = PreprocessorBuilder(self.dataset_name).load()
        self.num_classes = len(np.unique(self.dataset.y_train))
        self.best_model = None

    def grid_search(self, grid):
        X_train_preprocessed = self.preprocessor.transform(self.dataset.X_train)
        gs = GridSearchCV(self.model, grid, verbose=0, cv=5, scoring='roc_auc_ovr', n_jobs=-1)
        gs.fit(X_train_preprocessed, self.dataset.y_train)
        self.best_model = gs.best_estimator_
        self.best_model.fit(X_train_preprocessed, self.dataset.y_train)

    def predict(self, X):
        if self.best_model is None:
            self.load()
        X_preprocessed = self.preprocessor.transform(X)
        return self.best_model.predict(X_preprocessed)

    def predict_proba(self, X):
        if self.best_model is None:
            self.load()
        X_preprocessed = self.preprocessor.transform(X)
        return self.best_model.predict_proba(X_preprocessed)
    
    def save_stats(self):
        stats = pd.DataFrame()
        X_test_preprocessed = self.preprocessor.transform(self.dataset.X_test)
        X_train_preprocessed = self.preprocessor.transform(self.dataset.X_train)

        if self.num_classes == 2:
            y_score = self.best_model.predict_proba(X_test_preprocessed)[:, 1]
            auc = roc_auc_score(self.dataset.y_test, y_score)
            average_type = 'binary'
        else:
            y_score = self.best_model.predict_proba(X_test_preprocessed)
            auc = roc_auc_score(self.dataset.y_test, y_score, multi_class='ovr', average='macro')
            average_type = 'macro'

        y_pred_test = self.best_model.predict(X_test_preprocessed)
        y_pred_train = self.best_model.predict(X_train_preprocessed)

        stats.loc['Imbalance', self.dataset_name] = np.bincount(self.dataset.y_train).min() / len(self.dataset.y_train)
        stats.loc['Train size', self.dataset_name] = self.dataset.X_train.shape[0]
        stats.loc['Test size', self.dataset_name] = self.dataset.X_test.shape[0]
        stats.loc['n features', self.dataset_name] = self.dataset.X_test.shape[1]
        stats.loc['n cat features', self.dataset_name] = len(self.dataset.categorical_indices)
        stats.loc['n con features', self.dataset_name] = len(self.dataset.continuous_indices)

        stats.loc['auc', self.dataset_name] = auc
        stats.loc['Accuracy', self.dataset_name] = accuracy_score(self.dataset.y_test, y_pred_test)
        stats.loc['Precision', self.dataset_name] = precision_score(self.dataset.y_test, y_pred_test, average=average_type)
        stats.loc['F1', self.dataset_name] = f1_score(self.dataset.y_test, y_pred_test, average=average_type)
        stats.loc['Recall', self.dataset_name] = recall_score(self.dataset.y_test, y_pred_test, average=average_type)

        cf_test = confusion_matrix(self.dataset.y_test, y_pred_test, normalize='all')
        for i in range(min(2, self.num_classes)):
            for j in range(min(2, self.num_classes)):
                stats.loc[f'C{i}{j} test', self.dataset_name] = cf_test[i, j]

        cf_train = confusion_matrix(self.dataset.y_train, y_pred_train, normalize='all')
        for i in range(min(2, self.num_classes)):
            for j in range(min(2, self.num_classes)):
                stats.loc[f'C{i}{j} train', self.dataset_name] = cf_train[i, j]

        for param, value in self.best_model.get_params().items():
            if isinstance(value, (bool, str)):
                value = str(value)
            stats.loc[param, self.dataset_name] = value

        stats.to_csv(self.paths['model_stats'])
        
    def display_stats(self):
        stats = pd.read_csv(self.paths['model_stats'], index_col=0)
        print(stats)
    
    def save(self):
        if self.best_model is None:
            return
        with open(self.paths['model'], 'wb') as f:
            pickle.dump(self.best_model, f)

    def load(self):
        try:
            with open(self.paths['model'], 'rb') as f:
                self.best_model = pickle.load(f)
            return self.best_model
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found at {self.paths['model']}")
    
    def get_model_info(self):
        """Get basic information about the model"""
        return {
            'dataset': self.dataset_name,
            'model_type': self.model_name,
            'num_classes': self.num_classes,
            'model_path': self.paths['model'],
            'stats_path': self.paths['model_stats'],
            'is_trained': self.best_model is not None
        }

def dynamic_MLP_layers(dataset_name, n_options, max_scale):
    preprocessor = PreprocessorBuilder(dataset_name).load()
    dataset = TabularDataLoader(path_generator(dataset_name)['dataset'])

    input_size = preprocessor.transform(dataset.X_train).shape[1]
    max_size = int(input_size * max_scale)
    step = int(np.ceil((max_size - 2) / n_options))
    grid = list(range(2, max_size, step))
    return [(i,) for i in grid]
