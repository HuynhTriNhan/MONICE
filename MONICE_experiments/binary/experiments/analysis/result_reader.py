import pickle
import pandas as pd
import numpy as np
from experiments.core.path_utils import path_generator
from experiments.analysis.helpers import GowerDistanceMetric, HEOMDistance, NearestNeighborFinder, InterpretabilityMetric
from experiments.models.autoencoder_trainer import AutoencoderTrainer
from experiments.core.data_loader import TabularDataLoader
from experiments.models.classification_trainer import SklearnTabularModeler
from experiments.core.data_preprocessing import PreprocessorBuilder
import os
from typing import List, Dict, Any, Tuple
from joblib import Parallel, delayed

N_CF = {
    'monicemultiobjectivegower': 6,
    'monicemultiobjectiveheom': 6,
    'monicesparsprox': 6,
    'moniceproxplaus': 6,
    'monicesparsplaus': 6,
        
    'nicenone': 1,
    'nicespars': 1,
    'niceprox': 1,
    'niceplaus': 1,
    
    'moc': 6,
    'dicerandom': 6,
    'diceextended': 6,
    'geco': 6,
    'lime': 1,
    
    'cfproto': 1, 
}

class ResultReader:
    def __init__(self, dataset_names: List[str], models: List[str], cf_names: List[str]):
        self.dataset_names = dataset_names
        self.models = models
        self.cf_names = cf_names
        self._distance_metrics = {}
        self._datasets = {} 
        self.all_results = self.load_all_results()

    def _get_distance_metric(self, dataset_name: str, metric_type: str):
        """Get or create distance metric for dataset"""
        key = f"{dataset_name}_{metric_type}"
        if key not in self._distance_metrics:
            # Load dataset info
            if dataset_name not in self._datasets:
                paths = path_generator(dataset_name)
                self._datasets[dataset_name] = TabularDataLoader(paths['dataset'])
            
            dataset = self._datasets[dataset_name]
            
            if metric_type == 'gower':
                self._distance_metrics[key] = GowerDistanceMetric(
                    dataset.X_train, 
                    dataset.continuous_indices, 
                    dataset.categorical_indices
                )
            elif metric_type == 'heom':
                self._distance_metrics[key] = HEOMDistance(
                    dataset.X_train, 
                    dataset.continuous_indices, 
                    dataset.categorical_indices
                )
        
        return self._distance_metrics[key]

    def _is_valid_cf(self, cf: np.ndarray) -> bool:
        """Check if a single counterfactual is valid (not NaN)"""
        if cf is None:
            return False
        return not np.all(np.isnan(cf))

    def _is_valid_prediction(self, original: np.ndarray, cf: np.ndarray, modeler) -> bool:
        """Check if counterfactual has different prediction than original"""
        try:
            orig_pred = modeler.predict_proba(original.reshape(1, -1)).argmax()
            cf_pred = modeler.predict_proba(cf.reshape(1, -1)).argmax()
            return orig_pred != cf_pred
        except:
            return False

    def _preprocess_result(self, result: Dict[str, Any], modeler=None) -> Dict[str, Any]:
        """Preprocess a single result to extract valid counterfactuals"""
        original = result.get('original')
        cf = result.get('cf')
        time_taken = result.get('time', 0)
        
        if original is None or cf is None:
            return {
                'original': original,
                'valid_cfs': [],
                'time': time_taken,
                'total_cfs': 0,
                'valid_cfs_count': 0
            }
        
        valid_cfs = []
        total_cfs = 0
        
        # Handle both single CF and multiple CFs
        if len(cf.shape) == 1:
            # Single CF case
            total_cfs = 1
            if self._is_valid_cf(cf):
                if modeler is None or self._is_valid_prediction(original, cf, modeler):
                    valid_cfs.append(cf)
        else:
            # Multiple CFs case
            total_cfs = cf.shape[0]
            for cf_instance in cf:
                if self._is_valid_cf(cf_instance):
                    if modeler is None or self._is_valid_prediction(original, cf_instance, modeler):
                        valid_cfs.append(cf_instance)
                        
        return {
            'original': original,
            'valid_cfs': valid_cfs,
            'time': time_taken,
            'total_cfs': total_cfs,
            'valid_cfs_count': len(valid_cfs)
        }

    def load_results(self, dataset_name: str, model: str, cf_name: str, preprocess: bool = True) -> List[Dict[str, Any]]:
        """Load results and optionally preprocess them"""
        paths = path_generator(dataset_name, model, cf_name)
        results_path = paths['results']
        
        if not os.path.isfile(results_path):
            print(f"Cannot find results file: {results_path}")
            return []
        
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        if not preprocess:
            print(f"Loaded {len(results)} results from {results_path}")
            return results
        
        # Preprocess results
        modeler = None
        if dataset_name != "mushroom" or cf_name not in ["geco", "cfproto", "diceextended"]:
            try:
                modeler = SklearnTabularModeler(dataset_name, model)
                modeler.load()
            except:
                print(f"Warning: Could not load model for {dataset_name}-{model}")
        
        preprocessed_results = []
        for i,result in enumerate(results):
            preprocessed_result = self._preprocess_result(result, modeler)
            preprocessed_results.append(preprocessed_result)
        
        print(f"Loaded and preprocessed {len(preprocessed_results)} results from {results_path}")
        return preprocessed_results
    
    def load_all_results(self):
        """Load all results for all combinations of dataset, model, cf_name"""
        all_results = {}
        
        for dataset in self.dataset_names:
            all_results[dataset] = {}
            for model in self.models:
                all_results[dataset][model] = {}
                for cf_name in self.cf_names:
                    results = self.load_results(dataset, model, cf_name, preprocess=True)
                    if results:
                        all_results[dataset][model][cf_name] = results
                        
        return all_results
        
    def calculate_coverage(self, results: List[Dict[str, Any]], dataset_name: str = None, cf_name: str = None) -> float:
        """Calculate coverage percentage (non-NaN counterfactuals)"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        valid_count = sum(1 for result in results if result['valid_cfs_count'] > 0)
        return (valid_count / len(results)) * 100 if results else 0
    
    def calculate_time(self, results: List[Dict[str, Any]], dataset_name: str = None, cf_name: str = None) -> float:
        """Calculate average time for cases with valid counterfactuals"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        times = [result['time'] for result in results if result['valid_cfs_count'] > 0]
        return np.mean(times) if times else 0
    
    def calculate_valid_cf(self, results: List[Dict[str, Any]], dataset_name: str = None, cf_name: str = None) -> float:
        """Calculate percentage of valid cf"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        expected_n_cf = N_CF[cf_name] * len(results)
        
        valid_cfs = sum(result['valid_cfs_count'] for result in results)
        return (valid_cfs * 100.0 / expected_n_cf) if expected_n_cf > 0 else 0
    
    def calculate_sparsity(self, results: List[Dict[str, Any]], dataset_name: str = None, cf_name: str = None) -> float:
        """Calculate average L0 distance (number of changed features)"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        sparsities = []
        for result in results:
            original = result['original']
            for cf in result['valid_cfs']:
                sparsity = np.sum(original != cf, axis=1)
                sparsities.append(sparsity)
        return np.mean(sparsities) if sparsities else 0
    
    def calculate_gower_distance(self, results: List[Dict[str, Any]], dataset_name: str, cf_name: str) -> float:
        """Calculate average Gower (L1) distance"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        distances = []
        distance_metric = self._get_distance_metric(dataset_name, 'gower')
        
        for result in results:
            original = result['original']
            for cf in result['valid_cfs']:
                dist = distance_metric.distance(original, cf.reshape(1, -1))
                distances.append(np.mean(dist))
        return np.mean(distances) if distances else 0
    
    def calculate_heom_distance(self, results: List[Dict[str, Any]], dataset_name: str, cf_name: str) -> float:
        """Calculate average HEOM (L2) distance"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        distances = []
        distance_metric = self._get_distance_metric(dataset_name, 'heom')
        
        for result in results:
            original = result['original']
            for cf in result['valid_cfs']:
                dist = distance_metric.distance(original, cf.reshape(1, -1))
                distances.append(np.mean(dist))
        return np.mean(distances) if distances else 0
    
    def calculate_ae_loss(self, results: List[Dict[str, Any]], dataset_name: str, cf_name: str) -> float:
        """Calculate average autoencoder loss"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        autoencoder = AutoencoderTrainer(dataset_name)
        autoencoder.load()
        losses = []
        for result in results:
            for cf in result['valid_cfs']:
                loss = np.mean(autoencoder(cf.reshape(1, -1)))
                losses.append(loss)
        return np.mean(losses) if losses else 0
            
    def calculate_5NN_distance(self, results: List[Dict[str, Any]], dataset_name: str, cf_name: str) -> float:
        """Calculate average distance to 5 nearest neighbors"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        distances = []
        distance_metric = self._get_distance_metric(dataset_name, 'gower')
        nn_finder = NearestNeighborFinder(distance_metric)
        
        for result in results:
            for cf in result['valid_cfs']:
                avg_dist = nn_finder.avg_distance_5NN(cf.reshape(1, -1))
                distances.append(avg_dist)
        return np.mean(distances) if distances else 0

    def calculate_IM1(self, results: List[Dict[str, Any]], dataset_name: str, model: str, cf_name: str) -> float:
        """Calculate IM1 interpretability metric"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        modeler = SklearnTabularModeler(dataset_name, model)
        modeler.load()
        
        autoencoder = AutoencoderTrainer(dataset_name)
        autoencoder.load()
        autoencoder.load_class_autoencoder(0)
        autoencoder.load_class_autoencoder(1)
        
        im_metric = InterpretabilityMetric(modeler, autoencoder, autoencoder.preprocessor)
        
        im1_values = []
        for result in results:
            for cf in result['valid_cfs']:
                im1 = im_metric.IM1(cf.reshape(1, -1))
                im1_values.extend(im1)
        return np.mean(im1_values) if im1_values else 0

    def calculate_IM2(self, results: List[Dict[str, Any]], dataset_name: str, model: str, cf_name: str) -> float:
        """Calculate IM2 interpretability metric"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        modeler = SklearnTabularModeler(dataset_name, model)
        modeler.load()
        
        autoencoder = AutoencoderTrainer(dataset_name)
        autoencoder.load()
        autoencoder.load_class_autoencoder(0)
        autoencoder.load_class_autoencoder(1)
        
        im_metric = InterpretabilityMetric(modeler, autoencoder, autoencoder.preprocessor)
        
        im2_values = []
        for result in results:
            for cf in result['valid_cfs']:
                im2 = im_metric.IM2(cf.reshape(1, -1))
                im2_values.extend(im2)
        return np.mean(im2_values) if im2_values else 0

    def calculate_diversity(self, results: List[Dict[str, Any]], dataset_name: str, cf_name: str) -> float:
        """Calculate diversity between counterfactuals"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        diversities = []
        distance_metric = self._get_distance_metric(dataset_name, 'gower')
        
        for result in results:
            valid_cfs = result['valid_cfs']
            if len(valid_cfs) > 1:
                for i in range(len(valid_cfs)):
                    for j in range(i+1, len(valid_cfs)):
                        dist = distance_metric.distance(
                            valid_cfs[i].reshape(1, -1), 
                            valid_cfs[j].reshape(1, -1)
                        )
                        diversities.append(dist[0])
        return np.mean(diversities) if diversities else 0

    def calculate_confidence(self, results: List[Dict[str, Any]], dataset_name: str, model: str, cf_name: str) -> float:
        """Calculate average confidence (max probability) of counterfactual predictions"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        modeler = SklearnTabularModeler(dataset_name, model)
        modeler.load()
        
        confidences = []
        for result in results:
            for cf in result['valid_cfs']:
                try:
                    proba = modeler.predict_proba(cf.reshape(1, -1))
                    confidence = np.max(proba)
                    confidences.append(confidence)
                except:
                    continue
        return np.mean(confidences) if confidences else 0

    def calculate_entropy(self, results: List[Dict[str, Any]], dataset_name: str, model: str, cf_name: str) -> float:
        """Calculate average entropy of counterfactual prediction probability distributions"""
        if dataset_name == "mushroom" and cf_name in ["geco", "cfproto", "diceextended"]:
            return np.nan
            
        modeler = SklearnTabularModeler(dataset_name, model)
        modeler.load()
        
        entropies = []
        for result in results:
            for cf in result['valid_cfs']:
                try:
                    proba = modeler.predict_proba(cf.reshape(1, -1))[0]
                    # Add small epsilon to avoid log(0)
                    proba = np.clip(proba, 1e-15, 1.0)
                    entropy = -np.sum(proba * np.log2(proba))
                    entropies.append(entropy)
                except:
                    continue
        return np.mean(entropies) if entropies else 0

    def generate_metric_table(self, all_results: Dict, model: str, metric_name: str, metric_func) -> pd.DataFrame:
        """Generate a table for a specific metric using parallel processing"""
        
        def compute_metric_entry(dataset: str, cf_name: str) -> Tuple[str, str, Any]:
            """Compute metric value for one (dataset, cf_name) entry"""
            if (dataset in all_results and 
                model in all_results[dataset] and 
                cf_name in all_results[dataset][model]):

                results = all_results[dataset][model][cf_name]
                try:
                    if metric_name in ['sparsity', 'coverage', 'time', 'validity', 'ae_loss', 'gower_proximity', 'heom_proximity', 'diversity', '5NN_distance']:
                        value = metric_func(results, dataset, cf_name)
                    elif metric_name in ['IM1', 'IM2', 'confidence', 'entropy']:
                        value = metric_func(results, dataset, model, cf_name)
                    else:
                        value = metric_func(results)
                except Exception as e:
                    print(f"Error computing {metric_name} for {dataset}-{cf_name}: {e}")
                    value = np.nan
            else:
                value = np.nan
            return (dataset, cf_name, value)

        #  Parallel to compute all entries
        results_list = Parallel(n_jobs=os.cpu_count() - 1)(
            delayed(compute_metric_entry)(dataset, cf_name)
            for dataset in self.dataset_names
            for cf_name in self.cf_names
        )

        # Fill result table
        table = pd.DataFrame(index=self.dataset_names, columns=self.cf_names)
        for dataset, cf_name, value in results_list:
            table.loc[dataset, cf_name] = value

        return table

    def analyze_results(self) -> Dict[str, Any]:
        """Main analysis function"""
        
        print("Generating analysis tables...")
        analysis = {}
        
        # Define metrics
        metrics = {
            'coverage': self.calculate_coverage,
            'validity': self.calculate_valid_cf,
            'time': self.calculate_time,
            'sparsity': self.calculate_sparsity,
            'confidence': self.calculate_confidence,
            'entropy': self.calculate_entropy,
            'gower_proximity': self.calculate_gower_distance,
            'heom_proximity': self.calculate_heom_distance,
            'ae_loss': self.calculate_ae_loss,
            'diversity': self.calculate_diversity,
            '5NN_distance': self.calculate_5NN_distance,
            'IM1': self.calculate_IM1,
            'IM2': self.calculate_IM2
        }
        
        # Create tables directory if not exists
        os.makedirs('./tables', exist_ok=True)
        
        # Generate tables for each model and metric
        for model in self.models:
            analysis[model] = {}
            for metric_name, metric_func in metrics.items():
                print(f"Processing {model} - {metric_name}")
                table = self.generate_metric_table(self.all_results, model, metric_name, metric_func)
                analysis[model][metric_name] = table
                
                # Save to CSV
                filename = f'./tables/{model}_{metric_name}.csv'
                table.to_csv(filename)
                print(f"Saved {filename}")
        
        return analysis

    def statistical_analysis(self):
        """Generate feature change analysis by algorithm for each dataset and model"""
        print("Starting feature change analysis...")
        
        # Create stats directory
        os.makedirs('./stats', exist_ok=True)
        
        # Load dataset info to get feature names
        dataset_info = {}
        for dataset_name in self.dataset_names:
            paths = path_generator(dataset_name)
            dataset = TabularDataLoader(paths['dataset'])
            dataset_info[dataset_name] = {
                'feature_names': dataset.feature_names,
                'continuous_indices': dataset.continuous_indices,
                'categorical_indices': dataset.categorical_indices,
                'n_features': len(dataset.feature_names)
            }
        
        for dataset_name in self.dataset_names:
            for model in self.models:
                print(f"Analyzing {dataset_name} - {model}...")
                
                # Load model
                try:
                    modeler = SklearnTabularModeler(dataset_name, model)
                    modeler.load()
                except:
                    print(f"Could not load model for {dataset_name}-{model}")
                    continue
                
                # class transitions
                class_transitions = {}  # {(class_in, class_out): data}
                
                for cf_name in self.cf_names:
                    if (dataset_name in self.all_results and 
                        model in self.all_results[dataset_name] and 
                        cf_name in self.all_results[dataset_name][model]):
                        
                        results = self.all_results[dataset_name][model][cf_name]
                        
                        for result in results:
                            if result['valid_cfs_count'] == 0:
                                continue
                                
                            original = result['original']
                            try:
                                orig_pred = modeler.predict_proba(original.reshape(1, -1)).argmax()
                            except:
                                continue
                                
                            for cf in result['valid_cfs']:
                                try:
                                    cf_pred = modeler.predict_proba(cf.reshape(1, -1)).argmax()
                                    
                                    # Record class transition
                                    transition = (orig_pred, cf_pred)
                                    if transition not in class_transitions:
                                        class_transitions[transition] = {}
                                    
                                    if cf_name not in class_transitions[transition]:
                                        class_transitions[transition][cf_name] = []
                                    
                                    # Calculate feature changes
                                    feature_changes = (cf.flatten() != original.flatten()).astype(int)
                                    class_transitions[transition][cf_name].append(feature_changes)
                                except:
                                    continue
                
                # Generate markdown reports for each class transition
                for (class_in, class_out), transition_data in class_transitions.items():
                    filename = f"./stats/{model}-{dataset_name}-{class_in}-{class_out}.md"
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"# Feature Change Analysis: {model.upper()} - {dataset_name}\n")
                        f.write(f"## Class Transition: {class_in} -> {class_out}\n\n")
                        
                        # Get dataset info
                        feature_names = dataset_info[dataset_name]['feature_names']
                        n_features = dataset_info[dataset_name]['n_features']
                        

                        # Analyze each algorithm first
                        algorithm_stats = {}
                        for cf_name, cf_list in transition_data.items():
                            if not cf_list:
                                continue
                                
                            # Calculate feature changes for this algorithm
                            algo_feature_changes = np.zeros(n_features)
                            for feature_changes in cf_list:
                                # Ensure feature_changes is 1D and matches expected size
                                feature_changes = np.array(feature_changes).flatten()
                                if feature_changes.shape[0] == n_features:
                                    algo_feature_changes += feature_changes
                                else:
                                    print(f"Warning: Algorithm {cf_name} feature changes shape mismatch: expected {n_features}, got {feature_changes.shape}")
                                    continue
                            
                            # Calculate rates
                            algo_cf_count = len(cf_list)
                            algo_change_rates = [(feature_names[i], int(algo_feature_changes[i]),
                                                (algo_feature_changes[i] / algo_cf_count * 100) if algo_cf_count > 0 else 0)
                                               for i in range(n_features)]
                            # Sort by rate descending
                            algo_change_rates.sort(key=lambda x: x[2], reverse=True)
                            
                            algorithm_stats[cf_name] = {
                                'cf_count': algo_cf_count,
                                'feature_tuples': algo_change_rates,
                                'avg_features_changed': np.mean([np.sum(fc) for fc in cf_list]),
                                'total_changes': np.sum(algo_feature_changes)
                            }
                        
                        f.write("## Feature Change Analysis by Algorithm\n")
                        f.write("Each column shows (Feature, Times Changed, Rate%) sorted by rate descending:\n\n")
                        
                        # Create algorithm columns table
                        algo_names = list(algorithm_stats.keys())
                        if algo_names:
                            # Table header
                            header = "| Rank |" + "".join(f" {algo.upper()} |" for algo in algo_names)
                            f.write(header + "\n")
                            
                            # Table separator
                            separator = "|------|" + "".join("------|" for _ in algo_names)
                            f.write(separator + "\n")
                            
                            # Find max features to show (up to 15)
                            max_features = min(15, max(len(stats['feature_tuples']) for stats in algorithm_stats.values()))
                            
                            # Table rows
                            for rank in range(max_features):
                                row = f"| {rank+1} |"
                                for algo in algo_names:
                                    tuples = algorithm_stats[algo]['feature_tuples']
                                    if rank < len(tuples):
                                        feature, times, rate = tuples[rank]
                                        if rate > 0:  # Only show if there are changes
                                            row += f" {feature} ({times}, {rate:.1f}%) |"
                                        else:
                                            row += " - |"
                                    else:
                                        row += " - |"
                                f.write(row + "\n")
                        
                        f.write(f"\n---\n*{model.upper()} - {dataset_name} - Class transition: {class_in} -> {class_out}*\n")
                    
                    print(f"Generated: {filename}")
        
        print("Feature change analysis completed!")