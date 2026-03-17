"""
===================================================================
MONICE: Multi-Objective Nearest Instance Counterfactual Explanation
===================================================================
"""

import numpy as np
import time
from typing import Optional, List, Dict, Callable, Union, Any, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings("ignore")
MAX_OFFSPRING = 100

# ============================================================================
# RESULT STRUCTURES
# ============================================================================

@dataclass
class CounterfactualResult:
    """
    Result container for counterfactual explanations.
    
    Attributes:
        counterfactual: Generated counterfactual instances (n_cfs, n_features)
        target_class: Target class for counterfactuals
        original_class: Original predicted class
        original_instance: Original instance to explain
        k_nearest: Number of nearest neighbors used
        n_cfs: Number of counterfactuals generated
        quality_metrics: Dict containing various quality metrics
        computation_time: Time taken to generate counterfactuals
        constraints_satisfied: List indicating constraint satisfaction for each CF
        nearest_neighbors: K-nearest neighbors from target class 2D array
        optimization_strategy: List of optimization objectives used
        numerical_steps: Numerical interpolation steps used
    """
    counterfactual: np.ndarray
    target_class: int
    original_class: int
    original_instance: np.ndarray
    k_nearest: int
    n_cfs: int
    quality_metrics: Dict[str, List[float]]
    computation_time: float
    constraints_satisfied: List[bool]
    nearest_neighbors: np.ndarray
    optimization_strategy: List[str]
    numerical_steps: List[float]


# ============================================================================
# PLAUSIBILITY MODELS ABSTRACT CLASS
# ============================================================================

class PlausibilityModel(ABC):
    """
    Abstract base class for plausibility models.
    
    All plausibility models must implement the score() method 
    and get_criterion() method.
    """
    
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate plausibility scores for instances.
        
        Args:
            X: Input instances (n_samples, n_features)
            
        Returns:
            Array of plausibility scores (n_samples,)
        """
        pass
    
    @abstractmethod
    def get_criterion(self) -> str:
        """
        Return optimization criterion for plausibility scores.
        
        Returns:
            'higher' if higher scores are better, 'lower' if lower scores are better
        """
        pass


class AutoencoderPlausibility(PlausibilityModel):
    """
    Autoencoder-based plausibility model.
    
    Uses reconstruction error as a measure of plausibility.
    Lower reconstruction error indicates higher plausibility.
    """
    
    def __init__(self, model: Callable, criterion: str = 'lower'):
        self.model = model
        self.criterion = criterion
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error as plausibility score."""
        return self.model(X)
    
    def get_criterion(self) -> str:
        """Return optimization criterion."""
        return self.criterion


# ============================================================================
# DISTANCE METRICS
# ============================================================================
class DistanceMetric(ABC):
    """Abstract base class for distance metrics."""
    
    @abstractmethod
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Calculate distance between X1 and X2.
        Args:
            X1: First instance(s) (1, n_features)
            X2: Second instance(s) (n_samples, n_features)
        Returns:
            Distance array (n_samples,)
        """
        pass

class GowerDistance(DistanceMetric):
    """
    Gower distance metric for mixed-type data.
    
    Handles both numerical and categorical features with proper scaling.
    """
    
    def __init__(self, 
                 X_train: np.ndarray,
                 num_feats: list,
                 cat_feats: list,
                 eps: float = 1e-6,
                 cost_weights: Optional[Dict[int, float]] = None):
        
        # Use a copy to avoid mutating the caller's array
        self.X_train = X_train.copy()
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.cost_weights = cost_weights
        self.eps = eps

        if self.num_feats:
            self.X_train[:, self.num_feats] = self.X_train[:, self.num_feats].astype(np.float64)

        # Calculate scaling factors for numerical features
        if self.num_feats:
            num_data = self.X_train[:, num_feats].astype(np.float64)
            self.scale = (num_data.max(axis=0) - num_data.min(axis=0)).astype(np.float64)
            self.scale[self.scale < eps] = eps
            # Pre-compute weight arrays for numerical features
            self._num_weights = np.array(
                [cost_weights.get(f, 1.0) for f in num_feats], dtype=np.float64
            ) if cost_weights else np.ones(len(num_feats), dtype=np.float64)
            self._total_num_weight = float(np.sum(self._num_weights))
        else:
            self.scale = np.array([])
            self._num_weights = np.array([])
            self._total_num_weight = 0.0

        if self.cat_feats:
            # Pre-compute weight arrays for categorical features
            self._cat_weights = np.array(
                [cost_weights.get(f, 1.0) for f in cat_feats], dtype=np.float64
            ) if cost_weights else np.ones(len(cat_feats), dtype=np.float64)
            self._total_cat_weight = float(np.sum(self._cat_weights))
        else:
            self._cat_weights = np.array([])
            self._total_cat_weight = 0.0

        self._total_weight = self._total_num_weight + self._total_cat_weight
    
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        total_distance = np.zeros(X2.shape[0], dtype=np.float64)

        # Numerical features
        if self.num_feats:
            num_diff = np.abs(
                (X2[:, self.num_feats].astype(np.float64) -
                 X1[0, self.num_feats].astype(np.float64)) / self.scale
            )
            total_distance += (num_diff * self._num_weights).sum(axis=1)

        # Categorical features
        if self.cat_feats:
            cat_diff = (X2[:, self.cat_feats] != X1[0, self.cat_feats]).astype(np.float64)
            total_distance += (cat_diff * self._cat_weights).sum(axis=1)
        
        if self._total_weight > 0:
            return total_distance / self._total_weight
        else:
            return np.zeros(X2.shape[0], dtype=np.float64)

class HEOMDistance(DistanceMetric):
    """
    HEOM distance metric for mixed-type data.
    
    Handles both numerical and categorical features with proper scaling.
    """
    def __init__(self, X_train: np.ndarray, num_feats: list, cat_feats: list, eps: float = 1e-6):
        # Use a copy to avoid mutating the caller's array
        self.X_train = X_train.copy()
        self.num_feats = num_feats
        self.cat_feats = cat_feats
        self.eps = eps
        
        if self.num_feats:
            self.X_train[:, self.num_feats] = self.X_train[:, self.num_feats].astype(np.float64)
        
        self.cost_weights = None
        # Calculate scaling factors for numerical features
        if self.num_feats:
            num_data = self.X_train[:, num_feats].astype(np.float64)
            self.scale = (num_data.max(axis=0) - num_data.min(axis=0)).astype(np.float64)
            self.scale[self.scale < eps] = eps
        else:
            self.scale = np.array([])
            
    def distance(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        distance = np.zeros(X2.shape[0], dtype=np.float64)
        if self.num_feats:
            num_1 = X1[0, self.num_feats].astype(np.float64)
            num_2 = X2[:, self.num_feats].astype(np.float64)
            num_diff = (num_2 - num_1) / self.scale
            num_squared = np.square(num_diff).astype(np.float64)
            distance += np.sum(num_squared, axis=1)
            
        if self.cat_feats:
            cat_1 = X1[0, self.cat_feats]
            cat_2 = X2[:, self.cat_feats]
            cat_diff = cat_2 != cat_1
            distance += np.sum(cat_diff, axis=1).astype(np.float64) 
        return np.sqrt(distance)

class NearestNeighborFinder:
    """Find nearest neighbors from target class."""
    
    def __init__(self, distance_metric: DistanceMetric):
        self.distance_metric = distance_metric
    
    def sorted_nearest_neighbor(self, X: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Find sorted nearest neighbors.
        
        Args:
            X: Query instance (1, n_features)
            candidates: Candidate instances (n_candidates, n_features)
            
        Returns:
            Sorted candidates by distance (n_candidates, n_features)
        """
        if len(candidates) == 0:
            return np.array([])
        
        distances = self.distance_metric.distance(X, candidates)
        sorted_indices = np.argsort(distances)
        return candidates[sorted_indices]


# ============================================================================
# NSGA-II MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================

class NSGAIIOptimizer:
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) optimizer.
    
    Implements fast non-dominated sorting and crowding distance calculation
    for multi-objective optimization.
    """
    
    def __init__(self, 
                 objectives: List[str] = ['robustness', 'sparsity', 'proximity', 'plausibility'],
                 maximize: List[bool] = [False, False, False, False],
                 verbose: bool = False):
        """
        Initialize NSGA-II optimizer.
        
        Args:
            objectives: List of objective names
            maximize: List indicating whether to maximize (True) or minimize (False) each objective
            verbose: Whether to print debugging information
        """
        self.objectives = objectives
        self.maximize = maximize
        self.verbose = verbose
        # Pre-compute sign vector: +1 for maximize, -1 for minimize
        # Negating minimization objectives lets us always "maximize" for domination checks
        self._sign = np.array([1.0 if m else -1.0 for m in maximize], dtype=np.float64)
    
    def fast_non_dominated_sort(self, objectives_matrix: np.ndarray) -> List[List[int]]:
        """
        Perform fast non-dominated sorting on candidates.
        Vectorized implementation: O(n^2 * m) with NumPy broadcasting 
        
        Args:
            objectives_matrix: Matrix of objective values (n_candidates, n_objectives)
            
        Returns:
            List of Pareto fronts, each containing indices of solutions
        """
        n = len(objectives_matrix)
        if n == 0:
            return [[]]

        # Flip signs so that "better" always means larger
        # Shape: (n, m)
        signed = objectives_matrix * self._sign

        # Vectorized domination: signed[i] dominates signed[j] iff
        #   signed[i] >= signed[j] in all objectives  AND  > in at least one
        # Broadcasting: (n, 1, m) vs (1, n, m)
        better_or_equal = signed[:, None, :] >= signed[None, :, :]  # (n, n, m)
        strictly_better = signed[:, None, :] >  signed[None, :, :]  # (n, n, m)

        dominates_matrix = better_or_equal.all(axis=2) & strictly_better.any(axis=2)  # (n, n)
        np.fill_diagonal(dominates_matrix, False)

        domination_count = dominates_matrix.sum(axis=0).tolist()          # how many dominate i
        dominated_solutions = [list(np.where(dominates_matrix[i])[0]) for i in range(n)]

        fronts = [[]]
        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        # Build subsequent fronts
        front_idx = 0
        while front_idx < len(fronts) and fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front_idx += 1
            
        if self.verbose:
            print(f"Number of Pareto fronts: {len(fronts)}")
        
        return fronts

    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2."""
        better_in_one = False
        for i in range(len(obj1)):
            if self.maximize[i]:
                if obj1[i] < obj2[i]:
                    return False
                if obj1[i] > obj2[i]:
                    better_in_one = True
            else:
                if obj1[i] > obj2[i]:
                    return False
                if obj1[i] < obj2[i]:
                    better_in_one = True
        return better_in_one

    def crowding_distance(self, front: List[int], objectives_matrix: np.ndarray) -> Dict[int, float]:
        """Compute normalized crowding distances for solutions in a front."""
        if len(front) <= 2:
            return {idx: float('inf') for idx in front}

        distances = {idx: 0.0 for idx in front}
        n_objectives = objectives_matrix.shape[1]

        # Normalize objective to calculate crowding distance
        norm_objectives = objectives_matrix.copy().astype(np.float64)
        for i in range(n_objectives):
            col = norm_objectives[:, i]
            min_val, max_val = np.min(col), np.max(col)
            range_val = max_val - min_val
            if range_val < 1e-6:
                range_val = 1e-6  # avoid divide-by-zero
            norm_objectives[:, i] = (col - min_val) / range_val

        # Compute distances
        for obj_idx in range(n_objectives):
            reverse = self.maximize[obj_idx]
            front_sorted = sorted(front, key=lambda x: norm_objectives[x][obj_idx], reverse=reverse)

            distances[front_sorted[0]] = float('inf')
            distances[front_sorted[-1]] = float('inf')

            for i in range(1, len(front_sorted) - 1):
                prev_val = norm_objectives[front_sorted[i - 1]][obj_idx]
                next_val = norm_objectives[front_sorted[i + 1]][obj_idx]
                distances[front_sorted[i]] += (next_val - prev_val)

        return distances


    def select_best_candidates(self, objectives_matrix: np.ndarray, n_solutions: int) -> np.ndarray:
        """Select best candidates using NSGA-II selection. """
        
        # Perform non-dominated sorting
        fronts = self.fast_non_dominated_sort(objectives_matrix)
        
        selected_indices = []
        
        # Add solutions from fronts
        for front in fronts:
            if len(selected_indices) + len(front) <= n_solutions:
                selected_indices += front
            else:
                # Use crowding distance for partial selection
                remaining_slots = n_solutions - len(selected_indices)
                
                if remaining_slots > 0:
                    distances = self.crowding_distance(front, objectives_matrix)
                    front_sorted = sorted(front, key=lambda x: distances[x], reverse=True)
                    selected_indices += front_sorted[:remaining_slots]
                break
    
        return np.array(selected_indices[:n_solutions])


# ============================================================================
# CONSTRAINED MULTI-OBJECTIVE OPTIMIZER
# ============================================================================

class ConstrainedMultiObjectiveOptimizer:
    """
    Constrained multi-objective optimizer for counterfactual generation.
    
    Uses NSGA-II for selecting the best candidates at each generation.
    """
    
    def __init__(self, 
                 monice_instance,
                 population_size: int = 8,
                 max_generations: int = 15,
                 objectives: List[str] = ['robustness', 'sparsity', 'proximity', 'plausibility'],
                 maximize: List[bool] = [False, False, False, False],
                 numerical_steps: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                 eps: float = 1e-6,
                 early_stopping_rounds: int = 2,
                 rng: Optional[np.random.RandomState] = None,
                 verbose: bool = False):
        """
        Initialize Constrained multi-objective optimizer.
        
        Args:
            monice_instance: Reference to MONICE instance
            population_size: Number of candidates to maintain 
            max_generations: Maximum number of generations
            objectives: List of objectives to optimize
            maximize: List indicating whether to maximize each objective
            numerical_steps: Interpolation steps for numerical features
            eps: Epsilon for numerical stability
            early_stopping_rounds: Generations without improvement before stopping
            rng: Random state for reproducibility
            verbose: Whether to print debugging information
        """
        self.monice = monice_instance
        self.population_size = population_size
        self.max_generations = max_generations
        self.objectives = objectives
        self.maximize = maximize
        self.numerical_steps = np.array(numerical_steps)
        self.eps = eps
        self.early_stopping_rounds = early_stopping_rounds
        self.rng = rng if rng is not None else np.random.RandomState()
        self.verbose = verbose
        
        # Initialize NSGA-II optimizer
        self.nsga_optimizer = NSGAIIOptimizer(
            objectives=self.objectives,
            maximize=self.maximize,
            verbose=verbose
        )
    
    def optimize(self, X_original: np.ndarray, nearest_neighbors: np.ndarray, 
                 target_class: int, n_cfs: int) -> np.ndarray:
        """Run constrained multi-objective optimization."""
        if self.verbose:
            print("\nCONSTRAINED MULTI-OBJECTIVE OPTIMIZATION")
            print("-" * 50)
            print(f"Target class: {target_class}")
            print(f"Population size: {self.population_size}")
            print(f"Max generations: {self.max_generations}")
            print(f"Objectives: {self.objectives}")
            print("-" * 50)
        
        # Initialize population with nearest neighbors and their crossovers
        population = self._initialize_population(nearest_neighbors, X_original, target_class)

        if self.verbose:
            print(f"Initial population size: {len(population)}")
        
        # Select initial population, allow for more candidates to be selected
        population = self._select_best_cfs(population, X_original, target_class, self.population_size * 2)
        previous_objectives = None
        no_improvement_count = 0

        if self.verbose:
            print(f"Initial population size: {len(population)}")

        # Evolution loop
        for generation in range(self.max_generations):
            if self.verbose:
                print(f"\nGeneration {generation + 1}/{self.max_generations}")
                print(f"  Current population size: {len(population)}")
                
            # Generate offspring
            offspring = self._crossover(population, X_original)
            offspring = self._select_valid_counterfactual(offspring, X_original, target_class)
 
            if len(offspring) == 0:    
                if self.verbose:
                    print(f" No valid offspring generated")
                break
            if offspring.shape[0] > MAX_OFFSPRING:
                selected_indices = self.rng.choice(offspring.shape[0], MAX_OFFSPRING, replace=False)
                offspring = offspring[selected_indices]
                
            if self.verbose:
                print(f"Generated {offspring.shape[0]} valid offspring")
            
            # Combine populations
            combined_population = np.concatenate([population, offspring], axis=0)
            
            # Select next generation
            population = self._select_next_population(combined_population, X_original, target_class)
            
            if population.shape[0] == 0:
                if self.verbose:
                    raise ValueError("Warning: No valid counterfactuals found in current generation")
                break
            
            current_objectives = self._calculate_objectives(population, X_original, target_class)
            if previous_objectives is not None:
                if current_objectives.shape[0] != previous_objectives.shape[0]:
                    previous_objectives = current_objectives
                    continue
                improvement = not np.allclose(current_objectives, previous_objectives, rtol=1e-5, atol=1e-4)
                
                if not improvement:
                    no_improvement_count += 1
                    if self.verbose:
                        print(f"No improvement in {no_improvement_count} generations")
                else:
                    no_improvement_count = 0
                if no_improvement_count >= self.early_stopping_rounds:
                    if self.verbose:
                        print("Early stopping triggered")
                    break
            previous_objectives = current_objectives
            
        best_cfs = self._select_best_cfs(population, X_original, target_class, n_cfs)

        if self.verbose:
            print(f"\nConstrained multi-objective optimization completed successfully")
            print(f"Final selection: {best_cfs.shape[0]} counterfactuals")
            print("-" * 50)
            
        return best_cfs
        
    def _initialize_population(self, nearest_neighbors: np.ndarray, 
                              X_original: np.ndarray, target_class: int) -> np.ndarray:
        """Initialize population with nearest neighbors and their crossovers.
        
        Creates initial population using:
        1. Original nearest neighbors
        2. Uniform crossover between nearest neighbors (p=0.5)
        3. Interpolation-based offspring with numerical steps of 0.5 and 1.0
        """
        population = []
        n = nearest_neighbors.shape[0]
        
        # Nearest neighbors are always kept and appended at the end
        NN_population = nearest_neighbors.copy()
        
        # Uniform crossover between nearest neighbors (p=0.5) and
        # interpolation-based offspring with numerical steps of 0.5 and 1.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                parent_1 = nearest_neighbors[i:i+1, :].copy()
                parent_2 = nearest_neighbors[j:j+1, :].copy()
                
                # Uniform crossover
                uniform_offspring = parent_1.copy()
                mask = self.rng.random(parent_1.shape[1]) < 0.5
                uniform_offspring[0, mask] = parent_2[0, mask]
                population.append(uniform_offspring)

                # Interpolation-based offspring
                generated_offspring = self._generate_offspring(parent_1, parent_2, np.array([0.5, 1.0]))
                if generated_offspring.size > 0:
                    population.append(generated_offspring)

        if len(population) == 0:
            # No pairs to cross (single NUN) — just use it directly
            return NN_population

        population_arr = np.concatenate(population, axis=0)
        population_arr = self._select_valid_counterfactual(population_arr, X_original, target_class)
                     
        if population_arr.shape[0] > MAX_OFFSPRING:
            selected_indices = self.rng.choice(population_arr.shape[0], MAX_OFFSPRING, replace=False)
            population_arr = population_arr[selected_indices]
        
        population_arr = np.concatenate([population_arr, NN_population], axis=0)
        population_arr = self._remove_duplicates(population_arr)
        
        return population_arr
    
    def _crossover(self, population: np.ndarray, X_original: np.ndarray) -> np.ndarray:
        """Generate offspring by crossover between population members and original instance."""
        offspring = []
        
        for i in range(population.shape[0]):
            parent = population[i:i+1, :]
            parent_offspring = self._generate_offspring(parent, X_original, self.numerical_steps)
            
            if parent_offspring.size > 0:
                offspring.append(parent_offspring)

        if len(offspring) == 0:
            return np.array([])
            
        offspring = np.concatenate(offspring, axis=0)
        offspring = self._remove_duplicates(offspring)
        return offspring

    def _generate_offspring(self, parent_1: np.ndarray, parent_2: np.ndarray, numerical_steps: np.ndarray) -> np.ndarray:
        """Generate offspring through interpolation and feature exchange."""
        offspring = []
        
        # Find differing features
        diff_mask = parent_1[0] != parent_2[0]
        diff_features = np.where(diff_mask)[0]
        
        if len(diff_features) == 0:
            return np.array([])
        
        # Separate features by type
        num_set = set(self.monice.num_feats)
        cat_set = set(self.monice.cat_feats)
        num_diff_features = [f for f in diff_features if f in num_set]
        cat_diff_features = [f for f in diff_features if f in cat_set]
        
        # Generate numerical offspring through interpolation
        if num_diff_features:
            num_offspring = self._generate_numerical_offspring(parent_1, parent_2, num_diff_features, numerical_steps)
            if len(num_offspring) > 0:
                offspring.append(num_offspring)

        # Generate categorical offspring through direct replacement
        if cat_diff_features:
            cat_offspring = self._generate_categorical_offspring(parent_1, parent_2, cat_diff_features)
            if len(cat_offspring) > 0:
                offspring.append(cat_offspring)
        
        if len(offspring) == 0:
            return np.array([])
        
        return np.concatenate(offspring, axis=0)
    
    def _generate_numerical_offspring(self, parent_1: np.ndarray, parent_2: np.ndarray, 
                                     num_diff_features: List[int], steps: np.ndarray) -> np.ndarray:
        """Generate offspring by interpolating numerical features."""
        offspring = []

        for feat_idx in num_diff_features:
            current_val = parent_1[0, feat_idx]
            target_val = parent_2[0, feat_idx]
            
            if np.abs(current_val - target_val) <= self.eps:
                continue
            # Interpolate values
            interpolated_vals = current_val + steps * (target_val - current_val)
            
            # Handle integer features
            if self.monice.integer_feats and feat_idx in self.monice.integer_feats:
                interpolated_vals = np.round(interpolated_vals)

            # Create offspring for each interpolation step
            batch_offspring = np.tile(parent_1, (len(steps), 1))    
            batch_offspring[:, feat_idx] = interpolated_vals
            
            offspring.append(batch_offspring)

        if len(offspring) == 0:
            return np.array([])
        
        return np.concatenate(offspring, axis=0)
    
    def _generate_categorical_offspring(self, parent_1: np.ndarray, parent_2: np.ndarray, 
                                       cat_diff_features: List[int]) -> np.ndarray:
        """Generate offspring by changing categorical features."""
        n = len(cat_diff_features)
        batch_offspring = np.tile(parent_1, (n, 1))
        
        # Change one categorical feature at a time
        for i, feat_idx in enumerate(cat_diff_features):
            batch_offspring[i, feat_idx] = parent_2[0, feat_idx]
        
        return batch_offspring
    
    def _calculate_objectives(self, X_cfs: np.ndarray, X_original: np.ndarray, 
                             target_class: int) -> np.ndarray:
        """Calculate objective values for counterfactual candidates.
        
        Batches all predict_fn calls to avoid redundant inference.
        """
        objectives_matrix = np.zeros((X_cfs.shape[0], len(self.objectives)))

        # Cache predict_fn result if robustness is needed (avoid calling twice)
        probs_cache = None
        if 'robustness' in self.objectives:
            probs_cache = self.monice.predict_fn(X_cfs)

        for i, objective in enumerate(self.objectives):
            if objective == 'robustness':
                #  entropy as robustness measure (lower is better)
                probs = probs_cache
                entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
                objectives_matrix[:, i] = entropy
                
            elif objective == 'sparsity':
                # Number of changed features (lower is better)
                objectives_matrix[:, i] = np.sum(X_cfs != X_original[0], axis=1)
                
            elif objective == 'proximity':
                # Distance from original (lower is better)
                objectives_matrix[:, i] = self.monice.distance_metric.distance(X_original, X_cfs)
                
            elif objective == 'plausibility':
                # Plausibility score
                objectives_matrix[:, i] = self.monice.plausibility_model.score(X_cfs)
                
        return objectives_matrix

    def _select_next_population(self, combined_population: np.ndarray, X_original: np.ndarray, 
                         target_class: int) -> np.ndarray:
        """Select next generation population using NSGA-II."""
        # Remove duplicates
        population = self._remove_duplicates(combined_population)
        
        # Calculate objectives
        objectives_matrix = self._calculate_objectives(population, X_original, target_class)
        
        # NSGA-II selection
        selected_indices = self.nsga_optimizer.select_best_candidates(
            objectives_matrix, min(self.population_size, len(population))
        )
        
        return population[selected_indices]
    
    def _select_valid_counterfactual(self, candidates: np.ndarray, X_original: np.ndarray, 
                                    target_class: int) -> np.ndarray:
        """Filter candidates that satisfy constraints and predict target class."""
        if candidates.size == 0:
            return candidates
            
        # Check constraints (vectorized batch call)
        constraints_mask = self.monice._check_constraints_satisfied(candidates, X_original)
        
        # Check prediction (single batched call)
        predict_mask = self.monice.predict_fn(candidates).argmax(axis=1) == target_class
        
        # Combine masks
        valid_mask = constraints_mask & predict_mask
        
        return candidates[valid_mask]
    
    def _select_best_cfs(self, valid_cfs: np.ndarray, X_original: np.ndarray, 
                        target_class: int, n_cfs: int) -> np.ndarray:
        """Select final best counterfactuals using NSGA-II."""
        if valid_cfs.shape[0] <= n_cfs:
            return valid_cfs
        
        objectives_matrix = self._calculate_objectives(valid_cfs, X_original, target_class)
        selected_indices = self.nsga_optimizer.select_best_candidates(objectives_matrix, min(n_cfs, valid_cfs.shape[0]))
        
        return valid_cfs[selected_indices]
    
    def _remove_duplicates(self, candidates: np.ndarray) -> np.ndarray:
        """Remove duplicate candidates."""
        if len(candidates.shape) == 1:
            candidates = candidates.reshape(1, -1)
        
        unique_candidates = np.unique(candidates, axis=0)
        
        return self.monice.num_as_float(unique_candidates)
        

# ============================================================================
#  MONICE ALGORITHM
# ============================================================================

class MONICE:
    """
    Multi-Objective Nearest Instance Counterfactual Explanation.
    
    Generates diverse counterfactual explanations using NSGA-II multi-objective optimization.
    """
    
    def __init__(self, 
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 predict_fn: Callable,
                 plausibility_model: Callable,
                 cat_feats: List[int] = [],
                 num_feats: List[int] = [],
                 integer_feats: List[int] = [],
                 cost_weights: Optional[Dict[int, float]] = None,
                 immutable_features: Optional[List[int]] = None,
                 feature_bounds: Optional[Dict[int, Union[Tuple[float, float], List[Any]]]] = None,
                 monotonic_constraints: Optional[Dict[int, str]] = None,
                 distance_metric: str = 'gower',
                 justified_cf: bool = True,
                 eps: float = 1e-6,
                 random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize MONICE.
        
        Args:
            X_train: Training data (n_samples, n_features)
            y_train: Training labels (n_samples,)
            predict_fn: Function that returns class probabilities
            plausibility_model: Plausibility model (required)
            cat_feats: List of categorical feature indices
            num_feats: List of numerical feature indices
            integer_feats: List of integer feature indices (subset of numerical features)
            cost_weights: Optional dict mapping feature indices to cost weights
            immutable_features: List of feature indices that cannot be changed
            feature_bounds: Dict mapping feature indices to bounds.
            monotonic_constraints: Dict mapping feature indices to 'increasing' or 'decreasing'
            distance_metric: Distance metric to use ('gower' or 'heom')
            justified_cf: Whether to use only correctly classified training instances
            eps: Small value for numerical stability
            random_state: Seed for reproducibility (replaces global np.random.seed)
            verbose: Whether to print debugging information
            
        """
        
        self._rng = np.random.RandomState(random_state)

        self.X_train = X_train
        self.y_train = y_train
        self.cat_feats = cat_feats or []
        self.num_feats = num_feats or []
        self.integer_feats = integer_feats or []
        
        # Initialize cost weights with default value of 1.0 for all features
        self.cost_weights = {}
        for feat_idx in self.num_feats + self.cat_feats:
            self.cost_weights[feat_idx] = 1.0
        
        # Override with user-provided cost weights
        if cost_weights is not None:
            for feat_idx, weight in cost_weights.items():
                if feat_idx in self.num_feats + self.cat_feats:
                    self.cost_weights[feat_idx] = weight
                else:
                    warnings.warn(f"Feature index {feat_idx} not found in num_feats or cat_feats. Ignoring.")

        self.predict_fn = predict_fn
        self.plausibility_model = plausibility_model
        self.immutable_features = immutable_features or []
        self.monotonic_constraints = monotonic_constraints or {}
        self.feature_bounds = feature_bounds or {}
        self.justified_cf = justified_cf
        self.eps = eps
        self.verbose = verbose
        
        # Auto-detect integer features if not specified
        if not self.integer_feats and self.num_feats:
            self.integer_feats = self._auto_detect_integer_features()
            if self.verbose and self.integer_feats:
                print(f"Auto-detected {len(self.integer_feats)} integer features: {self.integer_feats}")

        # Pre-compute integer_feats as a set for O(1) lookup in hot paths
        self._integer_feats_set = set(self.integer_feats)
        
        # Convert numerical features to float
        self.X_train = self.num_as_float(self.X_train)
        
        # Setup distance metric
        if distance_metric.lower() == 'gower':
            self.distance_metric = GowerDistance(
                self.X_train, self.num_feats, self.cat_feats, self.eps, self.cost_weights
            )
        elif distance_metric.lower() == 'heom':
            self.distance_metric = HEOMDistance(
                self.X_train, self.num_feats, self.cat_feats, self.eps
            )
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}. Supported: 'gower', 'heom'")
        
        # Setup nearest neighbor finder
        self.nn_finder = NearestNeighborFinder(self.distance_metric)

        # Get model information
        self.train_proba = self.predict_fn(self.X_train)
        self.n_classes = self.train_proba.shape[1]
        self.X_train_class = np.argmax(self.train_proba, axis=1)
        
        # Create candidates mask
        if self.justified_cf:
            self.candidates_mask = self.y_train == self.X_train_class
        else:
            self.candidates_mask = np.ones(self.X_train.shape[0], dtype=bool)
        
        # Initialize feature bounds
        self._initialize_feature_bounds()
        
        
    def num_as_float(self, X: np.ndarray) -> np.ndarray:
        """Convert numerical features to float type."""
        X_copy = X.copy()
        if self.num_feats:
            X_copy[:, self.num_feats] = X_copy[:, self.num_feats].astype(np.float64)
        return X_copy
    
    def _auto_detect_integer_features(self) -> List[int]:
        detected_integer_feats = []
        
        for feat_idx in self.num_feats:
            values = self.X_train[:, feat_idx]
            
            rounded_values = np.round(values)
            
            diff = np.abs(values - rounded_values)
            
            # Check if all differences are zero (or very close to zero)
            if np.max(diff) == 0 or np.allclose(diff, 0, atol=self.eps):
                detected_integer_feats.append(feat_idx)
                if self.verbose:
                    print(f"Auto-detected integer feature: {feat_idx}")
        
        return detected_integer_feats
    
    def _initialize_feature_bounds(self):

        # Auto-initialize bounds for numerical features
        for feat_idx in self.num_feats:
            if feat_idx not in self.feature_bounds:
                X_filtered = self.X_train[self.candidates_mask]
                min_val = float(X_filtered[:, feat_idx].min())
                max_val = float(X_filtered[:, feat_idx].max())
                self.feature_bounds[feat_idx] = (min_val, max_val)
                if self.verbose:
                    print(f"Bounds for numerical feature {feat_idx}: ({min_val}, {max_val})")
        
        # Auto-initialize bounds for categorical features (as allowed values)
        for feat_idx in self.cat_feats:
            if feat_idx not in self.feature_bounds:
                unique_values = list(np.unique(self.X_train[:, feat_idx]))
                self.feature_bounds[feat_idx] = unique_values
                if self.verbose:
                    print(f"Allowed values for categorical feature {feat_idx}: {unique_values}")
    
    def _calculate_quality_metrics(self, X_cfs: np.ndarray, X_original: np.ndarray, 
                                  target_class: int) -> Dict[str, List[float]]:
        """Calculate quality metrics for counterfactuals."""
        if len(X_cfs.shape) == 1:
            X_cfs = X_cfs.reshape(1, -1)
            
        n_cfs = X_cfs.shape[0]
        metrics = {}
        
        # Get predictions (single batched call)
        full_probs = self.predict_fn(X_cfs)
        
        # Probability of target class
        prob_target_class = full_probs[:, target_class]
        metrics['prob_target_class'] = prob_target_class.tolist()
        
        # Validity check
        predicted_classes = full_probs.argmax(axis=1)
        valid_cfs = (predicted_classes == target_class).tolist()
        metrics['valid_cf'] = valid_cfs
        
        # Robustness (entropy - lower is better)
        entropy = -np.sum(full_probs * np.log(full_probs + 1e-12), axis=1)
        metrics['robustness'] = entropy.tolist()
        
        # Sparsity (number of changed features - lower is better)
        sparsity_scores = np.sum(X_cfs != X_original[0], axis=1).tolist()
        metrics['sparsity'] = sparsity_scores
        
        # Proximity (distance - lower is better)
        proximity_scores = self.distance_metric.distance(X_original, X_cfs).tolist()
        metrics['proximity'] = proximity_scores
        
        # Plausibility 
        plausibility_scores = self.plausibility_model.score(X_cfs).tolist()
        metrics['plausibility'] = plausibility_scores
        
        # Cost (weighted feature changes - lower is better)
        # Vectorized: diff mask (n_cfs, n_feats), then apply weights per feature type
        diff_mask = X_cfs != X_original[0]  # (n_cfs, n_feats)

        cost_arr = np.zeros(n_cfs, dtype=np.float64)
        for feat_idx in range(X_cfs.shape[1]):
            if feat_idx not in self.cost_weights:
                continue
            weight = self.cost_weights[feat_idx]
            changed = diff_mask[:, feat_idx]
            if feat_idx in self.cat_feats:
                cost_arr += changed.astype(np.float64) * weight
            else:
                cost_arr += changed.astype(np.float64) * weight * np.abs(
                    X_cfs[:, feat_idx].astype(np.float64) - float(X_original[0, feat_idx])
                )

        metrics['cost'] = cost_arr.tolist()
        
        return metrics
    
    def _check_constraints_satisfied(self, X_candidates: np.ndarray, 
                                   X_original: np.ndarray) -> np.ndarray:
        """Check if constraints are satisfied for candidates."""

        if len(X_candidates.shape) == 1:
            X_candidates = X_candidates.reshape(1, -1)
            
        n_candidates = X_candidates.shape[0]
        satisfied = np.ones(n_candidates, dtype=bool)
        
        # Check immutable features
        for feat_idx in self.immutable_features:
            violated = (X_candidates[:, feat_idx] != X_original[0, feat_idx])
            satisfied &= ~violated
        
        # Check feature bounds (both numerical and categorical)
        for feat_idx, bounds in self.feature_bounds.items():
            if isinstance(bounds, tuple) and len(bounds) == 2:
                # Numerical bounds: (min_val, max_val)
                min_val, max_val = bounds
                violated = (X_candidates[:, feat_idx] < min_val) | (X_candidates[:, feat_idx] > max_val)
                satisfied &= ~violated
            elif isinstance(bounds, list):
                # Categorical allowed values
                violated = ~np.isin(X_candidates[:, feat_idx], bounds)
                satisfied &= ~violated
        
        # Check monotonic constraints
        for feat_idx, direction in self.monotonic_constraints.items():
            original_val = X_original[0, feat_idx]
            candidate_vals = X_candidates[:, feat_idx]
            
            if direction.lower() == 'increasing':
                # Counterfactual value must be >= original value
                violated = candidate_vals < original_val
            elif direction.lower() == 'decreasing':
                # Counterfactual value must be <= original value
                violated = candidate_vals > original_val
            else:
                # Invalid direction, treat as violated
                violated = np.ones(n_candidates, dtype=bool)
                raise ValueError(f"Invalid monotonic constraint direction '{direction}' for feature {feat_idx}. "
                                 f"Must be 'increasing' or 'decreasing'.")
            
            satisfied &= ~violated
        
        return satisfied

    def explain(self, 
                X: np.ndarray, 
                target_classes: Union[List[int], str] = 'other',
                optimization: List[str] = ['robustness', 'sparsity', 'proximity', 'plausibility'],
                k_nearest: int = 1,
                n_cfs: int = 1,
                numerical_steps: List[float] = [0.2, 0.4, 0.6, 0.8, 1.0],
                max_generations: int = None,
                population_size: int = None) -> Dict[int, CounterfactualResult]:
        """
        Generate counterfactual explanations.
        
        Args:
            X: Instance to explain (1D or 2D array)
            target_classes: Target classes ('other', int, or list of ints)
            optimization: List of objectives to optimize
            k_nearest: Number of nearest neighbors to find
            n_cfs: Number of counterfactuals to generate
            numerical_steps: Interpolation steps for numerical features
            max_generations: Maximum number of generations for optimization
            population_size: Size of population for optimization
        Returns:
            Dictionary mapping target classes to CounterfactualResult objects
        """
        if max_generations is None:
            max_generations = 3*len(self.num_feats) + len(self.cat_feats) - len(self.immutable_features)
        if population_size is None:
            population_size = 3*n_cfs

        if self.verbose:
            print("\n" + "="*60)
            print("MONICE COUNTERFACTUAL EXPLANATION")
            print("="*60)
            print(f"Optimization objectives: {optimization}")
            print(f"K-nearest neighbors: {k_nearest}")
            print(f"Number of counterfactuals: {n_cfs}")
            print(f"Population size: {population_size}")
            print(f"Max generations: {max_generations}")
            print(f"Numerical steps: {numerical_steps}")
            print("="*60)

        # Set up optimizer
        maximize = []
        for obj in optimization:
            if obj in ['robustness', 'sparsity', 'proximity']:
                maximize.append(False)  # minimize
            elif obj == 'plausibility':
                if self.plausibility_model.get_criterion() == 'lower':
                    maximize.append(False)
                else:
                    maximize.append(True)
            else:
                raise ValueError(f"Unknown objective: {obj}")
        
        optimizer = ConstrainedMultiObjectiveOptimizer(
            monice_instance=self,
            objectives=optimization, 
            maximize=maximize,
            numerical_steps=numerical_steps,
            population_size=population_size,
            max_generations=max_generations,
            eps=self.eps,
            rng=self._rng,
            verbose=self.verbose
        )
                      
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X = self.num_as_float(X)
        X_proba = self.predict_fn(X)
        original_class = int(np.argmax(X_proba, axis=1)[0])
        
        # Determine target classes
        if target_classes == 'other':
            target_classes = [i for i in range(self.n_classes) if i != original_class]
        elif isinstance(target_classes, int):
            target_classes = [target_classes]
        
        results = {}
        
        # Generate counterfactuals for each target class
        for target_class in target_classes:
            if self.verbose:
                print(f"\nProcessing target class: {target_class}")
                print("-" * 30)
            
            start_time = time.time()
            
            # Find candidates
            class_mask = (self.X_train_class == target_class) & self.candidates_mask
            
            if not np.any(class_mask):
                warnings.warn(f"No training data available for target class {target_class}")
                continue
            
            class_candidates = self.X_train[class_mask, :]
            if self.verbose:
                print(f"Found {len(class_candidates)} candidates")
                
            # Find nearest neighbors
            sorted_nearest_neighbor = self.nn_finder.sorted_nearest_neighbor(X, class_candidates)
            nearest_neighbors = []
            
            for i in range(len(sorted_nearest_neighbor)):
                candidate = sorted_nearest_neighbor[i:i+1, :].copy()
                
                # Replace immutable features
                if self.immutable_features:
                    candidate[:, self.immutable_features] = X[0, self.immutable_features]
                
                # Check constraints
                if not self._check_constraints_satisfied(candidate, X)[0]:
                    continue
                
                # Check prediction
                if self.predict_fn(candidate).argmax() == target_class:
                    nearest_neighbors.append(candidate)
                    if len(nearest_neighbors) == k_nearest:
                        break
            
            if len(nearest_neighbors) < k_nearest:
                warnings.warn(f"Not enough valid nearest neighbors for target class {target_class}")
                if len(nearest_neighbors) == 0:
                    continue
            
            nearest_neighbors = np.vstack(nearest_neighbors)
            if self.verbose:
                print(f"Selected {len(nearest_neighbors)} valid nearest neighbors")
            
            # Optimize counterfactuals
            counterfactual = optimizer.optimize(X, nearest_neighbors, target_class, n_cfs)
            
            end_time = time.time()
            
            
            computation_time = end_time - start_time
            
            if self.verbose:
                print(f"Optimization completed in {computation_time:.3f} seconds")
                print(f"Generated {counterfactual.shape[0]} counterfactuals")
            
            # Calculate metrics
            quality_metrics = self._calculate_quality_metrics(counterfactual, X, target_class)
            
            # Check constraints — vectorized batch call instead of per-CF loop
            constraints_satisfied = self._check_constraints_satisfied(counterfactual, X).tolist()
            
            # Create result
            result = CounterfactualResult(
                counterfactual=counterfactual,
                target_class=target_class,
                original_class=original_class,
                original_instance=X,
                k_nearest=k_nearest,
                n_cfs=counterfactual.shape[0],
                quality_metrics=quality_metrics,
                computation_time=computation_time,
                constraints_satisfied=constraints_satisfied,
                nearest_neighbors=nearest_neighbors,
                optimization_strategy=optimization,
                numerical_steps=numerical_steps,
            )
            
            results[target_class] = result
        
        return results