import os

import torch.utils
import torch.utils.data
import dice_ml_x
from dice_ml_x.utils import helpers, neuralnetworks
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from time import time
import torch

from collections import OrderedDict
from typing import List
from torcheval.metrics.functional import multiclass_f1_score, multiclass_recall, multiclass_precision, multiclass_auroc
from keras.metrics import F1Score, Precision, Recall, AUC
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, roc_auc_score
from scipy.spatial.distance import cdist

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

class Benchmarking:
    def __init__(self, datasets: List[tuple], backends: list, metrics=None):
        self.datasets = datasets
        self.backends = backends
        self.metrics = metrics if metrics else ["fidelity", "proximity", "diversity", "robustness"]
        self.models = {}
        self.results = {}
        self.sklearn_pipeline = None
        
    def split_data(self, df, target_col, test_size=0.2, random_state=42):
        target = df[target_col]
        train_dataset, test_dataset, y_train, y_test = train_test_split(df,
                                                                target,
                                                                test_size=test_size,
                                                                random_state=random_state,
                                                                stratify=target)
        return train_dataset, test_dataset, y_train, y_test
      
        
    def preprocess_data(self, backend: str, df: pd.DataFrame, continuous_features: list,
                        target_name: str, batch_size: int, fitted_pipeline=None,
                        pyt_scaler=None, pyt_encoder=None, pyt_label_encoder=None,
                        test_size=0.2):
        if test_size != 0.0:
            train_dataset, test_dataset, y_train, y_test = self.split_data(df, target_name, test_size=test_size)
            
            x_test = test_dataset.drop(target_name, axis=1)
        else:
            train_dataset = df
            y_train = df[target_name]
            x_test = None
            y_test = None
            test_dataset = None

        x_train = train_dataset.drop(target_name, axis=1)

        if backend == "sklearn":
            categorical = x_train.columns.difference(continuous_features)

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))])

            transformations = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical)])

            self.sklearn_pipeline = Pipeline(steps=[('preprocessor', transformations),
                                ('classifier', RandomForestClassifier())])
            return x_train, x_test, train_dataset, test_dataset, y_train, y_test, None, None, None, None
        elif backend == "PYT":
            if pyt_scaler:
                pyt_train_dataset = neuralnetworks.PYTDataset(df, scaler=pyt_scaler,
                                                              encoder=pyt_encoder,
                                                              target_encoder=pyt_label_encoder,
                                                              target_column=target_name, train=True)
                scaler = pyt_train_dataset.scaler
                encoder = pyt_train_dataset.encoder
                target_encoder = pyt_train_dataset.target_encoder
                if test_dataset is not None:
                    pyt_test_dataset = neuralnetworks.PYTDataset(df, scaler=scaler,
                                                                encoder=encoder,
                                                                target_encoder=target_encoder,
                                                                target_column=target_name, train=False)
                else:
                    pyt_test_dataset = None
            else:
                pyt_train_dataset = neuralnetworks.PYTDataset(df, target_column=target_name, train=True)
                if test_dataset is not None:
                    pyt_test_dataset = neuralnetworks.PYTDataset(df, target_column=target_name, train=False)
                else:
                    pyt_test_dataset = None

            train_df = pyt_train_dataset.train_dataset_df
            test_df = pyt_train_dataset.test_dataset_df if test_dataset is not None else None
            y_train_df = pyt_train_dataset.y_train_df
            y_test_df = pyt_train_dataset.y_test_df if test_dataset is not None else None
            scaler = pyt_train_dataset.scaler
            encoder = pyt_train_dataset.encoder
            target_encoder = pyt_train_dataset.target_encoder
            pyt_train_dataloader = DataLoader(pyt_train_dataset, batch_size=batch_size, shuffle=True)
            pyt_test_dataloader = DataLoader(pyt_test_dataset, batch_size=batch_size // 4, shuffle=False) if test_dataset is not None else None
            
            return pyt_train_dataloader, pyt_test_dataloader, train_df, test_df, y_train_df, y_test_df, None, scaler, encoder, target_encoder
        elif backend == "TF2":
            categorical = x_train.columns.difference(continuous_features)

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
        
            transformations = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical),
                    ('num', StandardScaler(), continuous_features)
                ],
                sparse_threshold=0
            )

            if fitted_pipeline == None:
                transformation_pipeline = transformations.fit(x_train)
            else:
                transformation_pipeline = fitted_pipeline



            x_train_transformed_data = transformation_pipeline.transform(x_train)
            x_test_transformed_data = transformation_pipeline.transform(x_test) if test_dataset is not None else None

            tf_train_dataset = tf.data.Dataset.from_tensor_slices((x_train_transformed_data, y_train.values))
            tf_test_dataset = tf.data.Dataset.from_tensor_slices((x_test_transformed_data, y_test.values)) if test_dataset is not None else None

            tf_train_dataset = tf_train_dataset.shuffle(len(x_train)).batch(batch_size)
            tf_test_dataset = tf_test_dataset.batch(batch_size=batch_size) if test_dataset is not None else None

            return tf_train_dataset, tf_test_dataset, train_dataset, test_dataset, y_train, y_test, transformation_pipeline, None, None, None
    
    def train_random_forest(self, x_train, y_train) -> Pipeline:
        self.sklearn_pipeline.fit(x_train, y_train)
        return self.sklearn_pipeline
    

    def compute_RF_metrics(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        print(y_pred == y_test)
        y_proba = model.predict_proba(x_test)[:, 1]  # only for binary classification

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='binary'),
            'recall': recall_score(y_test, y_pred, average='binary'),
            'precision': precision_score(y_test, y_pred, average='binary'),
            'auc': roc_auc_score(y_test, y_proba)
        }
    
    
    def train_pytorch_model(self, train_dataloader: DataLoader, test_dataloader, model_save_dir, epochs=20):

        dummy_inputs, _ = next(iter(train_dataloader))
        in_features = dummy_inputs.shape[1]
        n_classes = len(train_dataloader.dataset.y_train_df.unique())
        trainer = neuralnetworks.PYTModel(in_features, model_save_dir)
        trainer.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader, epochs=epochs)
        return trainer, trainer.model
    
    def compute_pytorch_metrics(self, model, test_dataloader: torch.utils.data.DataLoader) -> dict:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.eval()

        all_outputs = []
        all_labels = []
        correct_test_preds = 0

        with torch.no_grad():
            for test_features, test_labels in test_dataloader:

                test_features = test_features.to(device)
                test_labels = test_labels.long().to(device)

                test_outputs = model(test_features)         
                test_outputs = test_outputs.squeeze(dim=1)  
                
                test_preds = (test_outputs > 0.5).long()
                correct_test_preds += (test_preds == test_labels).sum().item()

                all_outputs.append(test_outputs.cpu())
                all_labels.append(test_labels.cpu())

        all_outputs = torch.cat(all_outputs, dim=0)  # shape [total_samples]
        all_labels = torch.cat(all_labels, dim=0)    # shape [total_samples]

        total_samples = len(test_dataloader.dataset)
        accuracy = correct_test_preds / total_samples
        p_class1 = all_outputs
        p_class0 = 1.0 - p_class1
        probs_2d = torch.stack([p_class0, p_class1], dim=1)  # shape [total_samples, 2]

        f1_score_val = multiclass_f1_score(probs_2d, all_labels, num_classes=2, average='macro').item()
        recall_val   = multiclass_recall(probs_2d, all_labels, num_classes=2, average='macro').item()
        precision_val= multiclass_precision(probs_2d, all_labels, num_classes=2, average='macro').item()
        auc_val      = multiclass_auroc(probs_2d, all_labels, num_classes=2, average='macro').item()

        return {
            'accuracy': accuracy,
            'f1_score': f1_score_val,
            'recall': recall_val,
            'precision': precision_val,
            'auc': auc_val
        }

    
    def train_keras_model(self, train_dataset, test_dataset, epochs=10):
        model = neuralnetworks.TF2Model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        loss_fn = tf.keras.losses.BinaryCrossentropy()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
        model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=0)
        return model


    def compute_keras_metrics(self, model, tf_test_dataset, y_test):
        _, accuracy = model.evaluate(tf_test_dataset, verbose=0)
        from keras.metrics import F1Score
        f1_metric = F1Score(threshold=0.5)
        y_pred = model.predict(tf_test_dataset)
        y_true = y_test.values.reshape(y_test.shape[0], 1)
        f1_metric.update_state(y_true, y_pred)
        f1_result = f1_metric.result()
        f1_score = f1_result.numpy()[0]

        recall_metric = Recall(thresholds=[0.5])
        recall_metric.update_state(y_true, y_pred)
        recall_result = recall_metric.result()
        recall = recall_result.numpy()

        precision_metric = Precision(thresholds=[0.5])
        precision_metric.update_state(y_true, y_pred)
        precision_result = precision_metric.result()
        precision = precision_result.numpy()

        auc_metric = AUC(num_thresholds=3)
        auc_metric.update_state(y_true, y_pred)
        auc_result = auc_metric.result()
        auc = auc_result.numpy()

        return {
            'accuracy': accuracy,
            'f1_score': f1_score,
            'recall': recall,
            'precision': precision,
            'auc': auc
        }
    

    def train_model(self, backend, X, x_test, y, epochs=10):
        if backend == 'sklearn':
            return self.train_random_forest(X, y)
        elif backend == 'PYT':
            return self.train_pytorch_model(X, x_test)
        elif backend == 'TF2':
            return self.train_keras_model(X, x_test, epochs)
        
    def compute_stability_pytorch(self, C_set_1: torch.Tensor, C_set_2: torch.Tensor, p: int=2) -> float:
        return torch.mean(torch.cdist(C_set_1, C_set_2, p=2)).item()
        
    def compute_proximity_pytorch(self, original_instance: torch.Tensor, C: torch.Tensor) -> float:
        return torch.mean(torch.cdist(original_instance, C, p=1)).item()

    def compute_diversity_pytorch(self, C_ohe: torch.Tensor):
        k = C_ohe.shape[0]
        pairwise_dist = torch.cdist(C_ohe, C_ohe, p=2)
        diversity = (torch.sum(pairwise_dist) - torch.sum(torch.diagonal(pairwise_dist))) / (k * (k - 1))
        return diversity.item()
    
    def compute_validity(self, CFs: pd.DataFrame) -> float:
        uniqe_rows, _ = CFs.drop_duplicates().shape
        rows, _ = CFs.shape
        return float(uniqe_rows) / float(rows)
    
    def do_perturbation(self, x: pd.DataFrame, data_class):
        x_tensor = torch.tensor(x.values, dtype=torch.float32)
        cat_cols = data_class.get_encoded_categorical_feature_indexes()
        cat_cols = [col for group in cat_cols for col in group]
        continuous_feature_indexes = list(set(list(range(len(x.columns)))) - set(cat_cols))
        categorical_feature_indexes = data_class.get_encoded_categorical_feature_indexes()
        if continuous_feature_indexes:
            
            continuous_slice = x_tensor[:, continuous_feature_indexes]
            noise = continuous_slice * 0.1
            noise_mask = torch.zeros_like(x_tensor)
            noise_mask[:, continuous_feature_indexes] = noise
            x_tensor = x_tensor + noise_mask

        if categorical_feature_indexes:
            for cat_cols in categorical_feature_indexes:
                cat_slice = x_tensor[:, cat_cols]
                sample_size = cat_slice.shape[0]
                num_cats = cat_slice.shape[1]

                rand_idx = torch.randint(low=0, high=num_cats, size=(sample_size, ))

                cat_slice_perturbed = torch.nn.functional.one_hot(rand_idx, num_classes=num_cats).float()

                cat_mask = torch.zeros_like(x_tensor)
                cat_mask[:, cat_cols] = cat_slice_perturbed
                x_perturbed = x_tensor + cat_mask
        return x_perturbed

    def generate_perturbations_pytorch(self, x_ohe: pd.DataFrame, data_class,
                               model: any, max_iter=100, tol=1e-3, gamma=1e-2):
        x_ohe_tensor = torch.tensor(x_ohe.values, dtype=torch.float32, requires_grad=True)
        x_perturbed = self.do_perturbation(x_ohe, data_class)
        perturbation_optimizer = torch.optim.Adam([x_perturbed], lr=1e-3)

        prev_loss = np.inf
        for _ in range(max_iter):
            with torch.no_grad():
                model.eval()
                pred_i = model(x_ohe_tensor)
                pred_i_prime = model(x_perturbed)
            class_loss = torch.mean((pred_i - pred_i_prime) ** 2)
            distance = torch.norm(x_perturbed - x_ohe_tensor, p=2)
            loss = class_loss + gamma * distance


            perturbation_optimizer.zero_grad()
            loss.backward()

            perturbation_optimizer.step()
            if abs(loss.item() - prev_loss) < tol:
                break
            prev_loss = loss.item()
        return x_perturbed.detach()

    
    def load_and_train(self, batch_size, artefact_path=None):
        num_processes = len(self.datasets) * len(self.backends)
        
        if artefact_path is None:
            artefact_path = "benchmarking_artefact"
        if not os.path.isdir(artefact_path):
            os.mkdir(artefact_path)
        with tqdm(total=num_processes, desc="Benchmarking", leave=True) as d_pbar:
            for df, target_column, dataset_name in self.datasets:

                continuous_features = df.select_dtypes(include=[np.number]).columns.to_list()
                continuous_features.remove(target_column)
                self.results[dataset_name] = {}
                for backend in self.backends:
            
                    x_train_transformed, x_test_transformed, train_df, test_df, y_train, y_test = self.preprocess_data(backend=backend,
                                                                                                                    df=df,
                                                                                                                    continuous_features=continuous_features,
                                                                                                                    target_name=target_column,
                                                                                                                    batch_size=batch_size)
                    
                    model = self.train_model(backend, x_train_transformed, x_test_transformed, y_train)
                    
                    if backend == "sklearn":
                        model_metrics = self.compute_RF_metrics(model, x_test_transformed, y_test)
                    elif backend == "PYT":
                        model_metrics = self.compute_pytorch_metrics(model, x_test_transformed)
                    elif backend == "TF2":
                        model_metrics = self.compute_keras_metrics(model, x_test_transformed, y_test)
                    
                    backend_results = {
                        'model_metrics': model_metrics,
                        'cfs': {},
                        'input_instance': {},
                        'time': {},
                        'exp_history': {},
                        'metrics': {}
                    }

                    if backend == "PYT":
                        model_dir = os.path.join(artefact_path, dataset_name)
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, f"{backend}_model.pt")

                        torch.save(model.state_dict(), model_path)
                        backend_results['model_path'] = model_path
                    elif backend == "TF2":
                        model_dir = os.path.join(artefact_path, dataset_name)
                        os.makedirs(model_dir, exist_ok=True)
                        model_path = os.path.join(model_dir, f"{backend}_model")
                        model.save_weights(model_path, save_format='tf')    
                        backend_results['model_path'] = model_path
                    else:  # sklearn
                        backend_results['model'] = model

                    print(f"the dataset is : {dataset_name}, the backend is : {backend}")
                        
                    cfs, input_instance, generation_time, exp_loss_history, metrics = self.generate_cfs(df,
                                                                                continuous_features,
                                                                                model,
                                                                                backend,
                                                                                target_column)
                    backend_results['cfs'] = cfs
                    backend_results['input_instance'] = input_instance
                    backend_results['time'] = generation_time
                    backend_results['exp_history'] = exp_loss_history
                    backend_results['metrics'] = metrics
                    self.results[dataset_name][backend] = backend_results
                    d_pbar.set_postfix(OrderedDict(
                            dataset=dataset_name,
                            backend=backend
                    ))
                    d_pbar.update(1)

    
    def generate_cfs(self, dataset: pd.DataFrame,
                     continuous_features: list,
                     model: any, model_backend: str,
                     target_name: str, total_CFs=10, proximity_weight=0.5,
                     diversity_weight=1.0, robustness_weight=0.4,
                     algorithm="DiverceCF", desired_class="opposite"):
        train_dataset, test_dataset, _, _ = self.split_data(dataset, target_name)

        x_train = train_dataset.drop(target_name, axis=1)
        x_test = test_dataset.drop(columns=[target_name])

        numerical = continuous_features
        categorical = dataset.columns.difference(list(numerical))

        cat_features = {}
        for col in categorical:
            if col in dataset.columns:
                cat_features[col] = dataset[col].unique().tolist()

        if model_backend == "sklearn":
            exp_method = 'genetic'
            m = dice_ml_x.Model(model=model, backend=model_backend)
        else:
            exp_method = 'gradient'
            m = dice_ml_x.Model(model=model, backend=model_backend, func='ohe-min-max')
        
        d = dice_ml_x.Data(dataframe=train_dataset, continuous_features=list(numerical), outcome_name=target_name)

        exp = dice_ml_x.Dice(d, m, method=exp_method)
        
        kwargs = {
            'gaussian': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'std_dev': 0.3
            },
            'random': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'feature_ranges': exp.data_interface.get_features_range_float()[1]
            },
            'spherical': {
                'continuous_features': continuous_features,
                'categorical_features': cat_features,
                'feature_ranges': exp.data_interface.get_features_range_float()[1] 
            }
        } 
        start = time()
        explainer_options = OrderedDict(query_instances=x_test[1:2], total_CFs=total_CFs,
                                        perturbation_method='gaussian', desired_class=desired_class, proximity_weight=proximity_weight,
                                        diversity_weight=diversity_weight, robustness_weight=robustness_weight,
                                        algorithm=algorithm, **kwargs['gaussian'])
        dice_exp = exp.generate_counterfactuals(**explainer_options)
        end = time()
        generation_time = (end-start)
        CFs_df = dice_exp.to_dataframe()
        if model_backend == "PYT":
            metrics_dict = self.compute_metrics_pytorch(d, CFs_df, model, test_dataset[1:2],
                                            target_name, exp, explainer_options)
        elif model_backend == "TF2":
            metrics_dict = self.compute_metrics_tf(d, CFs_df, model, test_dataset[1:2],
                                                   target_name, exp, explainer_options)
        elif model_backend == "sklearn":
            metrics_dict = self.compute_metrics_sklearn(d, CFs_df, model, test_dataset[1:2],
                                                        target_name, exp, explainer_options)
        return CFs_df, x_test[1:2], generation_time, exp.loss_history, metrics_dict
    
    def compute_metrics_pytorch(self, data_class, C: pd.DataFrame,
                        model: any, original_instance: pd.DataFrame,
                        target_name: str, explainer: dice_ml_x.Dice, explainer_options: OrderedDict) -> dict:
        x_ohe = data_class.get_ohe_min_max_normalized_data(original_instance)
        x_ohe_tensor = torch.tensor(x_ohe.values, dtype=torch.float32)
        x_ohe_tensor_targetless = data_class.get_ohe_min_max_normalized_data(original_instance.drop(columns=[target_name]))
        C_ohe = data_class.get_ohe_min_max_normalized_data(C)
        C_ohe_tensor = torch.tensor(C_ohe.values, dtype=torch.float32)
        proximity = self.compute_proximity_pytorch(x_ohe_tensor, C_ohe_tensor)
        diversity = self.compute_diversity_pytorch(C_ohe_tensor)
        x_ohe_prime_tensor = self.generate_perturbations_pytorch(x_ohe_tensor_targetless, data_class, model)
        x_ohe_prime_decoded = data_class.get_decoded_data(x_ohe_prime_tensor.numpy())
        explainer_options['query_instances'] = data_class.get_inverse_ohe_min_max_normalized_data(x_ohe_prime_decoded)
        C_prime = explainer.generate_counterfactuals(**explainer_options)
        C_prime = C_prime.to_dataframe()
        na_cols = C_prime.columns[C_prime.isna().any().tolist()].tolist()
        C_prime[na_cols] = C_prime[na_cols].fillna(explainer_options['query_instances'][na_cols].iloc[0])
        C_prime[target_name] = (C_prime[target_name] >= 0.5).astype(int)
        C_prime_ohe = data_class.get_ohe_min_max_normalized_data(C_prime)
        C_prime_ohe_tensor = torch.tensor(C_prime_ohe.values, dtype=torch.float32)
        robustness = self.compute_stability_pytorch(C_ohe_tensor, C_prime_ohe_tensor)
        num_features = x_ohe_tensor_targetless.shape[1]
        return {
            'min_max_proximity':[0.0, float(num_features)],
            'proximity': proximity,
            'min_max_diversity': [0.0, torch.sqrt(torch.tensor(num_features, dtype=torch.float32)) / 2.0],
            'diversity': diversity,
            'min_max_robustness': [0.0, torch.sqrt(torch.tensor(num_features, dtype=torch.float32))],
            'robustness': robustness,
            'num_features': num_features
        }
    
    def tf_cdist(self, x: tf.Tensor, y: tf.Tensor, p=2) -> tf.Tensor:
        """
        Approximates PyTorch's cdist in TensorFlow.
        x: shape [N, D]
        y: shape [M, D]
        Returns pairwise distances of shape [N, M].
        """
        # Expand to [N, 1, D] and [1, M, D]
        x_expanded = tf.expand_dims(x, 1)  # [N, 1, D]
        y_expanded = tf.expand_dims(y, 0)  # [1, M, D]
        diff = x_expanded - y_expanded     # [N, M, D]
        
        if p == 1:
            dist = tf.norm(diff, ord=1, axis=-1)   # shape [N, M]
        else:  # p==2
            dist = tf.norm(diff, ord='euclidean', axis=-1)
        return dist
    
    def compute_proximity_tf(self, original_instance: tf.Tensor, C: tf.Tensor) -> float:
        """
        Approximates the PyTorch-based compute_proximity using p=1 distance in TF.
        """
        dist_matrix = self.tf_cdist(original_instance, C, p=1)  # shape [N, M]
        return tf.reduce_mean(dist_matrix).numpy()
    
    def compute_stability_tf(self, C_set_1: tf.Tensor, C_set_2: tf.Tensor, p: int=2) -> float:
        """
        Approximates the PyTorch-based compute_stability using TensorFlow.
        """
        dist_matrix = self.tf_cdist(C_set_1, C_set_2, p=p)  # shape [N, M]
        return tf.reduce_mean(dist_matrix).numpy()
    
    def compute_diversity_tf(self, C_ohe: tf.Tensor) -> float:
        """
        Approximates the PyTorch-based compute_diversity using pairwise distances in TF.
        """
        k = tf.shape(C_ohe)[0]
        # Pairwise distance of shape [k, k]
        dist_matrix = self.tf_cdist(C_ohe, C_ohe, p=2)
        sum_all = tf.reduce_sum(dist_matrix)
        diag_sum = tf.reduce_sum(tf.linalg.diag_part(dist_matrix))
        # (sum_of_all - sum_of_diagonal) / (k*(k-1))
        denom = tf.cast(k * (k - 1), tf.float32)
        diversity = (sum_all - diag_sum) / denom
        return diversity.numpy()
    
    def do_perturbation_tf(self, x: pd.DataFrame, data_class) -> tf.Tensor:
        """
        Replaces the PyTorch approach with a TF-based approach for 'do_perturbation'.
        x: a pandas DataFrame
        """
        # Convert DataFrame to TF tensor
        x_tensor = tf.constant(x.values, dtype=tf.float32)

        # Identify categorical vs continuous columns
        cat_cols = data_class.get_encoded_categorical_feature_indexes()
        cat_cols = [col for group in cat_cols for col in group]
        continuous_feature_indexes = list(set(range(x_tensor.shape[1])) - set(cat_cols))
        
        # 1. Add noise to continuous columns
        if continuous_feature_indexes:
            continuous_slice = tf.gather(x_tensor, continuous_feature_indexes, axis=1)
            noise = continuous_slice * 0.1  # ~10% noise
            noise_mask = tf.scatter_nd(
                indices=[[0, 0]],  # This is conceptual; you need a proper mask shape
                updates=[0.0], 
                shape=x_tensor.shape
            )
            # Instead, simpler might be: build a full zero matrix, then place noise in continuous columns.
            # For clarity:
            noise_mask_full = tf.zeros_like(x_tensor)
            noise_mask_full_continuous = tf.tensor_scatter_nd_update(
                noise_mask_full,
                indices=[[i, j] for i in range(x_tensor.shape[0]) for j in continuous_feature_indexes],
                updates=tf.reshape(noise, [-1])
            )
            x_tensor = x_tensor + noise_mask_full_continuous

        # 2. Randomly flip categorical columns
        if cat_cols:
            # For each group in cat_cols, pick random index => one-hot
            # This is more complicated in TF if you want in-batch random categories, but conceptually:
            for cat_cols_group in data_class.get_encoded_categorical_feature_indexes():
                cat_slice = tf.gather(x_tensor, cat_cols_group, axis=1)  # shape [batch_size, cat_size]
                sample_size = tf.shape(cat_slice)[0]
                num_cats = tf.shape(cat_slice)[1]

                rand_idx = tf.random.uniform(shape=(sample_size,), minval=0, maxval=num_cats, dtype=tf.int32)
                cat_slice_perturbed = tf.one_hot(rand_idx, depth=num_cats, dtype=tf.float32)

                # Build a mask for updating cat_slice in x_tensor
                zero_mask = tf.zeros_like(x_tensor)
                # Similar scatter approach as above for all the cat_cols_group columns...
                # For brevity, assume you do a gather -> update -> scatter back. 
                # Then set x_perturbed = x_tensor + cat_mask, etc.
                # ...
        return x_tensor  # or x_perturbed
    
    def generate_perturbations_tf(self, x_ohe: pd.DataFrame, data_class,
                              model: tf.keras.Model, max_iter=100, tol=1e-3, gamma=1e-2):
        """
        Conceptual TF version of generate_perturbations.
        """
        # Convert DataFrame to a TF variable so we can track gradients.
        x_ohe_var = tf.Variable(x_ohe.values, dtype=tf.float32)

        # Create the initial perturbation
        x_perturbed = self.do_perturbation_tf(x_ohe, data_class)
        x_perturbed_var = tf.Variable(x_perturbed, dtype=tf.float32)

        prev_loss = np.inf

        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                # We'll watch x_perturbed_var
                tape.watch(x_perturbed_var)

                # Forward pass
                pred_i = model(x_ohe_var, training=False)
                pred_i_prime = model(x_perturbed_var, training=False)

                # Similar to PyTorch: (pred_i - pred_i_prime) ** 2
                class_loss = tf.reduce_mean(tf.square(pred_i - pred_i_prime))

                # L2 distance
                distance = tf.norm(x_perturbed_var - x_ohe_var, ord=2)
                loss = class_loss + gamma * distance

            grads = tape.gradient(loss, [x_perturbed_var])
            # Basic gradient descent step
            for g, var in zip(grads, [x_perturbed_var]):
                var.assign_sub(1e-3 * g)

            if abs(loss.numpy() - prev_loss) < tol:
                break
            prev_loss = loss.numpy()

        return x_perturbed_var.value()
    
    def compute_metrics_tf(
        self, data_class, C: pd.DataFrame, 
        model: tf.keras.Model, 
        original_instance: pd.DataFrame,
        target_name: str, explainer: dice_ml_x.Dice, 
        explainer_options: OrderedDict
    ) -> dict:
        """
        TensorFlow-based analog to compute_metrics_pytorch.
        """
        # 1) Convert data to TF Tensors
        x_ohe = data_class.get_ohe_min_max_normalized_data(original_instance)
        x_ohe_tf = tf.constant(x_ohe.values, dtype=tf.float32)

        x_ohe_targetless = data_class.get_ohe_min_max_normalized_data(original_instance.drop(columns=[target_name]))
        x_ohe_targetless_tf = tf.constant(x_ohe_targetless.values, dtype=tf.float32)

        # 2) Counterfactual data
        C_ohe = data_class.get_ohe_min_max_normalized_data(C)
        C_ohe_tf = tf.constant(C_ohe.values, dtype=tf.float32)

        # 3) Metrics
        proximity = self.compute_proximity_tf(x_ohe_tf, C_ohe_tf)
        diversity = self.compute_diversity_tf(C_ohe_tf)
        
        # 4) Generate perturbations in TF
        x_ohe_prime_tf = self.generate_perturbations_tf(x_ohe_targetless, data_class, model)

        # 5) Decode back
        x_ohe_prime_decoded = data_class.get_decoded_data(x_ohe_prime_tf.numpy())
        # Convert back to original scale
        explainer_options['query_instances'] = data_class.get_inverse_ohe_min_max_normalized_data(x_ohe_prime_decoded)
        
        # 6) Build new CFs from the explainer
        C_prime = explainer.generate_counterfactuals(**explainer_options)
        C_prime = C_prime.to_dataframe()

        # Fill NAs
        na_cols = C_prime.columns[C_prime.isna().any()].tolist()
        if na_cols:
            C_prime[na_cols] = C_prime[na_cols].fillna(explainer_options['query_instances'][na_cols].iloc[0])

        # Binarize the target
        C_prime[target_name] = (C_prime[target_name] >= 0.5).astype(int)

        # 7) Convert new CFs to TF
        C_prime_ohe = data_class.get_ohe_min_max_normalized_data(C_prime)
        C_prime_ohe_tf = tf.constant(C_prime_ohe.values, dtype=tf.float32)

        # 8) Compute stability
        robustness = self.compute_stability_tf(C_ohe_tf, C_prime_ohe_tf)
        num_features = x_ohe_targetless_tf.shape[1]
        return {
            "min_max_proximity": [0.0, float(num_features)],
            "proximity": proximity,
            "min_max_diversity": [0.0, (tf.sqrt(tf.cast(num_features, tf.float32)) / 2.0).numpy()],
            "diversity": diversity,
            "min_max_robustness": [0, tf.sqrt(tf.cast(num_features, tf.float32)).numpy()],
            "robustness": robustness,
            "num_features": num_features
        }
    
    def numpy_cdist(self, x: np.ndarray, y: np.ndarray, p=2) -> np.ndarray:
        """
        Approximate PyTorch/TensorFlow cdist using scipy.spatial.distance.cdist.
        x: shape [N, D]
        y: shape [M, D]
        Returns pairwise distances of shape [N, M].
        """
        if p == 1:
            return cdist(x, y, metric='cityblock')  # L1 distance
        else:
            return cdist(x, y, metric='euclidean')  # L2 distance
        

    def compute_proximity_sklearn(self, original_instance: np.ndarray, C: np.ndarray, p=1) -> float:
        """
        L1 (p=1) or L2 (p=2) distance between original_instance(s) and CFs, averaged.
        """
        dist_matrix = self.numpy_cdist(original_instance, C, p=p)  # shape [N, M]
        return dist_matrix.mean()

    def compute_stability_sklearn(self, C_set_1: np.ndarray, C_set_2: np.ndarray, p=2) -> float:
        """
        Mean pairwise distance across cross-sets: shape [N, M].
        """
        dist_matrix = self.numpy_cdist(C_set_1, C_set_2, p=p)
        return dist_matrix.mean()

    def compute_diversity_sklearn(self, C_ohe: np.ndarray) -> float:
        """
        Average pairwise L2 distance among points in C_ohe, ignoring diagonal.
        """
        k = C_ohe.shape[0]
        if k < 2:
            return 0.0  # if only one CF or none, diversity = 0

        dist_matrix = self.numpy_cdist(C_ohe, C_ohe, p=2)  # shape [k, k]
        sum_all = dist_matrix.sum()
        # diag is zero for L2( point, same_point ) anyway, but let's be explicit
        diag_sum = np.diagonal(dist_matrix).sum()

        denom = float(k * (k - 1))  # number of off-diagonal pairs
        diversity = (sum_all - diag_sum) / denom
        return diversity
    
    def do_perturbation_sklearn(self, x_df: pd.DataFrame, data_class) -> np.ndarray:
        """
        Model-agnostic: add ~10% noise to continuous columns and randomly flip
        categorical columns. Returns a NumPy array.
        """
        x_array = x_df.copy()  # shape [N, D]
        data_class.create_ohe_params(x_array)
        cat_cols = data_class.get_encoded_categorical_feature_indexes()
        cat_cols = [col for group in cat_cols for col in group]
        continuous_feature_indexes = list(set(range(x_array.shape[1])) - set(cat_cols))

        # 1) Add noise to continuous
        if continuous_feature_indexes:
            continuous_slice = x_array.iloc[:, continuous_feature_indexes]
            noise = continuous_slice * 0.1  # 10% noise
            x_array.iloc[:, continuous_feature_indexes] += noise

        # 2) Randomly “flip” categorical
        # For demonstration, we treat each group of one-hot columns as we did in TF
        for cat_group in data_class.get_encoded_categorical_feature_indexes():
            # shape [N, group_size]
            cat_slice = x_array.iloc[:, cat_group]
            sample_size, num_cats = cat_slice.shape

            # pick a random index in [0, num_cats)
            rand_idx = np.random.randint(low=0, high=num_cats, size=(sample_size,))
            cat_slice_perturbed = np.zeros_like(cat_slice)
            # set the “chosen category” to 1
            for i in range(sample_size):
                cat_slice_perturbed[i, rand_idx[i]] = 1.0

            x_array.iloc[:, cat_group] = cat_slice_perturbed

        return x_array
    

    def generate_perturbations_sklearn(
        self,
        x_df: pd.DataFrame,
        data_class,
        model: any,
        max_iter=50,
        gamma=1e-2
    ) -> np.ndarray:
        """
        Simple model-agnostic approach:
        1) convert x_df -> x_array
        2) create random perturbations
        3) pick the best perturbation that minimizes [ (pred_i - pred_i_prime)**2 + gamma*distance ]
        """
        # Convert the original input to NumPy
        x_orig = data_class.get_ohe_min_max_normalized_data(x_df)
        x_orig_numpy = x_orig.values.astype(np.float32)  # shape [N, D]

        # We'll assume there's 1 row, or maybe you handle multiple rows with a loop
        if x_orig.shape[0] != 1:
            raise ValueError("For demonstration, we assume x_df is a single instance. Adjust if multiple.")
        
        # pred_i is the model’s output on the original
        # If it’s a classifier, you might do model.predict_proba(...) or model.predict(...) => your choice
        pred_i = model.predict(x_df)  # shape [1] or [1, #classes]
        # We might interpret this as the logit or probability for class=1 in a binary scenario. 
        # In real usage, define how you interpret "pred_i" vs "pred_i_prime" carefully.

        best_loss = float('inf')
        best_perturbation = x_orig_numpy.copy()

        # Distance between x_orig and x_pert is L2
        def l2_distance(a, b):
            return np.linalg.norm(a - b, ord=2)

        for _ in range(max_iter):
            # 1) do_perturbation
            x_pert = self.do_perturbation_sklearn(x_orig, data_class)  # shape [N, D], N=1 here
            x_pert_df = data_class.get_inverse_ohe_min_max_normalized_data(x_pert)
            # 2) predict with model
            pred_i_prime = model.predict(x_pert_df)
            # 3) compute class_loss ~ mean((pred_i - pred_i_prime)**2)
            # for demonstration, assume they're floats
            class_loss = ((pred_i - pred_i_prime)**2).mean()
            # 4) distance
            dist = l2_distance(x_orig_numpy, x_pert)
            # 5) total loss
            total_loss = class_loss + gamma * dist

            if total_loss < best_loss:
                best_loss = total_loss
                best_perturbation = x_pert

        # best_perturbation is shape [1, D]
        return best_perturbation
    
    def compute_metrics_sklearn(
        self,
        data_class,
        C: pd.DataFrame,
        model: any,
        original_instance: pd.DataFrame,
        target_name: str,
        explainer: any,             # or some CF generator
        explainer_options: dict
    ) -> dict:
        """
        Example scikit‐learn version, conceptually similar to your TF function.
        """
        # 1) Convert data to NumPy
        x_ohe = data_class.get_ohe_min_max_normalized_data(original_instance)
        x_ohe_array = x_ohe.values.astype(np.float32)  # shape [N, D]

        x_ohe_targetless = data_class.get_ohe_min_max_normalized_data(
            original_instance.drop(columns=[target_name])
        )
        x_ohe_targetless_array = x_ohe_targetless.values.astype(np.float32)

        # 2) Convert CFs
        C_ohe = data_class.get_ohe_min_max_normalized_data(C)
        C_ohe_array = C_ohe.values.astype(np.float32)

        # 3) Proximity, Diversity (assuming p=1 for proximity, L2 for diversity)
        proximity_val = self.compute_proximity_sklearn(x_ohe_array, C_ohe_array, p=1)
        diversity_val = self.compute_diversity_sklearn(C_ohe_array)

        # 4) Generate perturbations (random approach)
        # or do some black‐box approach 
        x_ohe_prime_array = self.generate_perturbations_sklearn(
            original_instance, data_class, model
        )

        # 5) Decode back to original scale if needed
        x_ohe_prime_decoded = data_class.get_decoded_data(x_ohe_prime_array)
        x_ohe_prime_orig = data_class.get_inverse_ohe_min_max_normalized_data(x_ohe_prime_decoded)

        # 6) Build new CF from your CF explainer/generator
        explainer_options['query_instances'] = x_ohe_prime_orig
        C_prime = explainer.generate_counterfactuals(**explainer_options)
        C_prime = C_prime.to_dataframe()

        # fill NAs, binarize target, etc. as needed
        na_cols = C_prime.columns[C_prime.isna().any()].tolist()
        if na_cols:
            C_prime[na_cols] = C_prime[na_cols].fillna(
                explainer_options['query_instances'][na_cols].iloc[0]
            )
        C_prime[target_name] = (C_prime[target_name] >= 0.5).astype(int)

        # 7) Convert new CF to NumPy
        C_prime_ohe = data_class.get_ohe_min_max_normalized_data(C_prime)
        C_prime_ohe_array = C_prime_ohe.values.astype(np.float32)

        # 8) Compute stability
        robustness_val = self.compute_stability_sklearn(C_ohe_array, C_prime_ohe_array, p=2)

        # optional: min/max values for interpretability
        num_features = x_ohe_targetless_array.shape[1]

        return {
            "min_max_proximity": [0.0, float(num_features)],
            "proximity": proximity_val,
            "min_max_diversity": [0.0, (np.sqrt(num_features) / 2.0)],
            "diversity": diversity_val,
            "min_max_robustness": [0.0, np.sqrt(num_features)],
            "robustness": robustness_val,
            "num_features": num_features
        }
