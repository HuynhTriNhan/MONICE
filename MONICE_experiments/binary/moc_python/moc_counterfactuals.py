"""
Multi-Objective Counterfactuals (MOC) Implementation in Python

This module implements the MOC algorithm described in:
Dandl, S., Molnar, C., Binder, M., Bischl, B. (2020).
Multi-Objective Counterfactual Explanations.

The algorithm uses NSGA-II with mixed integer evolutionary strategies
to generate counterfactual explanations that optimize multiple objectives.
"""

import pickle
import warnings
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import gower
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm


class MOCCounterfactuals:
    """
    Multi-Objective Counterfactuals using NSGA-II genetic algorithm.

    This class generates counterfactual explanations by optimizing multiple
    objectives simultaneously:
    1. Distance to target prediction
    2. Distance to original instance (Gower distance)
    3. Number of changed features
    4. Distance to training data (optional)
    """

    def __init__(
        self,
        predictor: BaseEstimator,
        x_interest: pd.DataFrame,
        target: Union[float, List[float]],
        data: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        epsilon: Optional[float] = None,
        fixed_features: Optional[List[str]] = None,
        max_changed: Optional[int] = None,
        population_size: int = 50,
        generations: int = 50,
        p_crossover: float = 0.6,
        p_mutation: float = 0.7,
        p_mutation_gen: float = 0.25,
        p_mutation_use_orig: float = 0.2,
        p_crossover_gen: float = 0.6,
        p_crossover_use_orig: float = 0.7,
        initialization: str = "random",
        track_feasibility: bool = True,
        k_neighbors: int = 1,
        neighbor_weights: Optional[List[float]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize MOC Counterfactuals generator.

        Parameters:
        -----------
        predictor : BaseEstimator
            Trained ML model with predict or predict_proba method
        x_interest : pd.DataFrame
            Single row DataFrame with instance to explain
        target : float or list of float
            Desired prediction value or [min, max] interval
        data : pd.DataFrame
            Training data for constraints and initialization
        categorical_features : list of str, optional
            Names of categorical features
        epsilon : float, optional
            Soft constraint for prediction distance
        fixed_features : list of str, optional
            Features that cannot be changed
        max_changed : int, optional
            Maximum number of features that can be changed
        population_size : int
            Population size for genetic algorithm
        generations : int
            Number of generations to run
        p_crossover : float
            Probability of crossover
        p_mutation : float
            Probability of mutation
        p_mutation_gen : float
            Probability of gene mutation
        p_mutation_use_orig : float
            Probability of using original value mutation
        p_crossover_gen : float
            Probability of gene crossover
        p_crossover_use_orig : float
            Probability of using original value crossover
        initialization : str
            Initialization strategy: 'random', 'sd', or 'traindata'
        track_feasibility : bool
            Whether to track distance to training data
        k_neighbors : int
            Number of nearest neighbors for feasibility tracking
        neighbor_weights : list of float, optional
            Weights for k nearest neighbors
        feature_ranges : dict, optional
            Custom ranges for numerical features
        random_state : int, optional
            Random seed for reproducibility
        """

        self.predictor = predictor
        self.x_interest = x_interest.copy()
        self.target = target if isinstance(target, list) else [target]
        self.data = data.copy()
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.epsilon = epsilon
        self.fixed_features = fixed_features or []
        self.max_changed = max_changed

        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_mutation_gen = p_mutation_gen
        self.p_mutation_use_orig = p_mutation_use_orig
        self.p_crossover_gen = p_crossover_gen
        self.p_crossover_use_orig = p_crossover_use_orig

        self.initialization = initialization
        self.track_feasibility = track_feasibility
        self.k_neighbors = k_neighbors
        self.neighbor_weights = neighbor_weights
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

        # Validation
        self._validate_inputs()

        # Setup feature information
        self._setup_features(feature_ranges)

        # Setup DEAP framework
        self._setup_deap()

        # Results storage
        self.results = None
        self.log = []
        self.population_history = []

    def _validate_inputs(self):
        """Validate input parameters."""

        # Validate x_interest
        if len(self.x_interest) != 1:
            raise ValueError("x_interest must contain exactly one row")

        # Validate target
        if isinstance(self.target, list) and len(self.target) == 2:
            if self.target[0] > self.target[1]:
                raise ValueError("Target range minimum must be <= maximum")

        # Validate feature consistency
        missing_features = set(self.x_interest.columns) - set(self.data.columns)
        if missing_features:
            raise ValueError(
                f"Features in x_interest not found in data: {missing_features}"
            )

    def _setup_features(
        self, feature_ranges: Optional[Dict[str, Tuple[float, float]]]
    ):
        """Setup feature types, ranges, and encoders."""

        self.feature_names = list(self.x_interest.columns)
        self.n_features = len(self.feature_names)

        # Setup feature ranges
        self.feature_ranges = {}
        self.feature_values = {}

        for col in self.feature_names:
            if col in self.numerical_features:
                if feature_ranges and col in feature_ranges:
                    self.feature_ranges[col] = feature_ranges[col]
                else:
                    self.feature_ranges[col] = (
                        float(self.data[col].min()),
                        float(self.data[col].max()),
                    )
            else:
                # Categorical feature
                self.feature_values[col] = list(self.data[col].unique())

        # Compute Gower distance ranges
        self._compute_gower_ranges()

    def _compute_gower_ranges(self):
        """Compute ranges for Gower distance calculation."""
        ranges = []
        for col in self.feature_names:
            if col in self.numerical_features:
                col_range = (
                    self.feature_ranges[col][1] - self.feature_ranges[col][0]
                )
                ranges.append(col_range if col_range > 0 else 1.0)
            else:
                ranges.append(np.nan)  # Categorical features
        self.gower_ranges = np.array(ranges)

    def _setup_deap(self):
        """Setup DEAP evolutionary framework."""

        # Create fitness class (minimizing all objectives)
        if not hasattr(creator, "FitnessMin"):
            creator.create(
                "FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0)
            )

        # Create individual class
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Register individual creation
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        # Register genetic operators
        self.toolbox.register("mate", self._crossover)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", self._select_nsga2)
        self.toolbox.register("evaluate", self._evaluate_individual)

    def _create_individual(self):
        """Create a single individual for the population."""

        individual = []
        use_orig = []

        # Initialize each feature
        for i, col in enumerate(self.feature_names):

            if col in self.fixed_features:
                # Fixed features always use original value
                individual.append(self.x_interest[col].iloc[0])
                use_orig.append(True)
            elif col in self.numerical_features:
                # Numerical feature
                if self.initialization == "random":
                    if np.random.random() < 0.5:
                        # Use original value
                        individual.append(self.x_interest[col].iloc[0])
                        use_orig.append(True)
                    else:
                        # Random value in range
                        low, high = self.feature_ranges[col]
                        individual.append(np.random.uniform(low, high))
                        use_orig.append(False)
                elif self.initialization == "sd":
                    # Use standard deviation based initialization
                    orig_val = self.x_interest[col].iloc[0]
                    std = self.data[col].std()
                    if np.random.random() < 0.5:
                        individual.append(orig_val)
                        use_orig.append(True)
                    else:
                        low, high = self.feature_ranges[col]
                        val = np.random.normal(orig_val, std)
                        val = np.clip(val, low, high)
                        individual.append(val)
                        use_orig.append(False)
                else:  # traindata initialization
                    if np.random.random() < 0.5:
                        individual.append(self.x_interest[col].iloc[0])
                        use_orig.append(True)
                    else:
                        # Sample from training data
                        individual.append(
                            np.random.choice(self.data[col].values)
                        )
                        use_orig.append(False)
            else:
                # Categorical feature
                if np.random.random() < 0.5:
                    individual.append(self.x_interest[col].iloc[0])
                    use_orig.append(True)
                else:
                    individual.append(
                        np.random.choice(self.feature_values[col])
                    )
                    use_orig.append(False)

        # Apply max_changed constraint
        if self.max_changed is not None:
            n_changed = sum(not uo for uo in use_orig)
            if n_changed > self.max_changed:
                # Randomly select features to revert to original
                changed_indices = [i for i, uo in enumerate(use_orig) if not uo]
                to_revert = np.random.choice(
                    changed_indices,
                    size=n_changed - self.max_changed,
                    replace=False,
                )
                for idx in to_revert:
                    individual[idx] = self.x_interest.iloc[0, idx]
                    use_orig[idx] = True

        # Store use_orig information in individual
        ind = creator.Individual(individual)
        ind.use_orig = use_orig
        return ind

    def _crossover(self, ind1, ind2):
        """Crossover operation between two individuals."""

        child1 = creator.Individual(ind1[:])
        child2 = creator.Individual(ind2[:])
        child1.use_orig = ind1.use_orig[:]
        child2.use_orig = ind2.use_orig[:]

        for i in range(self.n_features):
            col = self.feature_names[i]

            if col in self.fixed_features:
                continue

            # Crossover use_orig flags
            if np.random.random() < self.p_crossover_use_orig:
                child1.use_orig[i], child2.use_orig[i] = (
                    child2.use_orig[i],
                    child1.use_orig[i],
                )

            # Crossover feature values
            if np.random.random() < self.p_crossover_gen:
                if col in self.numerical_features:
                    # SBX crossover for numerical features
                    eta = 20.0  # Distribution index
                    if abs(ind1[i] - ind2[i]) > 1e-14:
                        low, high = self.feature_ranges[col]
                        xl, xu = low, high

                        x1, x2 = min(ind1[i], ind2[i]), max(ind1[i], ind2[i])
                        rand = np.random.random()

                        if rand <= 0.5:
                            beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - rand))) ** (
                                1.0 / (eta + 1.0)
                            )

                        child1[i] = 0.5 * ((x1 + x2) - beta * abs(x2 - x1))
                        child2[i] = 0.5 * ((x1 + x2) + beta * abs(x2 - x1))

                        # Clip to bounds
                        child1[i] = np.clip(child1[i], xl, xu)
                        child2[i] = np.clip(child2[i], xl, xu)
                else:
                    # Uniform crossover for categorical features
                    child1[i], child2[i] = child2[i], child1[i]

        # Apply constraints
        self._apply_constraints(child1)
        self._apply_constraints(child2)

        return child1, child2

    def _mutate(self, individual):
        """Mutation operation on an individual."""

        mutant = creator.Individual(individual[:])
        mutant.use_orig = individual.use_orig[:]

        for i in range(self.n_features):
            col = self.feature_names[i]

            if col in self.fixed_features:
                continue

            # Mutate use_orig flag
            if np.random.random() < self.p_mutation_use_orig:
                mutant.use_orig[i] = not mutant.use_orig[i]

            # Mutate feature value
            if (
                np.random.random() < self.p_mutation_gen
                and not mutant.use_orig[i]
            ):
                if col in self.numerical_features:
                    # Gaussian mutation
                    low, high = self.feature_ranges[col]
                    std = (
                        high - low
                    ) * 0.1  # 10% of range as standard deviation
                    mutant[i] = np.random.normal(mutant[i], std)
                    mutant[i] = np.clip(mutant[i], low, high)
                else:
                    # Random choice for categorical
                    mutant[i] = np.random.choice(self.feature_values[col])

        # Apply constraints
        self._apply_constraints(mutant)

        return (mutant,)

    def _apply_constraints(self, individual):
        """Apply constraints to an individual."""

        # Apply use_orig constraint
        for i in range(self.n_features):
            if individual.use_orig[i]:
                individual[i] = self.x_interest.iloc[0, i]

        # Apply max_changed constraint
        if self.max_changed is not None:
            n_changed = sum(not uo for uo in individual.use_orig)
            if n_changed > self.max_changed:
                # Randomly select features to revert to original
                changed_indices = [
                    i
                    for i, uo in enumerate(individual.use_orig)
                    if not uo
                    and self.feature_names[i] not in self.fixed_features
                ]
                if len(changed_indices) > 0:
                    to_revert = np.random.choice(
                        changed_indices,
                        size=min(
                            n_changed - self.max_changed, len(changed_indices)
                        ),
                        replace=False,
                    )
                    for idx in to_revert:
                        individual[idx] = self.x_interest.iloc[0, idx]
                        individual.use_orig[idx] = True

    def _evaluate_individual(self, individual):
        """Evaluate fitness of an individual."""

        # Create DataFrame from individual
        x_candidate = pd.DataFrame([individual], columns=self.feature_names)

        # Objective 1: Distance to target prediction
        try:
            if hasattr(self.predictor, "predict_proba"):
                pred = self.predictor.predict_proba(x_candidate)[0, 1]
            else:
                pred = self.predictor.predict(x_candidate)[0]
        except:
            pred = self.predictor.predict(x_candidate)[0]

        if len(self.target) == 1:
            dist_target = abs(pred - self.target[0])
        else:
            # Target is an interval [min, max]
            if self.target[0] <= pred <= self.target[1]:
                dist_target = 0.0
            else:
                dist_target = min(
                    abs(pred - self.target[0]), abs(pred - self.target[1])
                )

        # Objective 2: Gower distance to x_interest
        try:
            dist_x_interest = gower.gower_matrix(
                self.x_interest.values,
                x_candidate.values,
                cat_features=[
                    i
                    for i, col in enumerate(self.feature_names)
                    if col in self.categorical_features
                ],
            )[0, 1]
        except:
            # Fallback to simple distance calculation
            dist_x_interest = self._compute_gower_distance(
                self.x_interest.iloc[0].values, individual
            )

        # Objective 3: Number of changed features
        nr_changed = sum(
            1
            for i in range(self.n_features)
            if not individual.use_orig[i]
            and individual[i] != self.x_interest.iloc[0, i]
        )

        # Objective 4: Distance to training data (if tracking feasibility)
        dist_train = 0.0
        if self.track_feasibility:
            try:
                dist_matrix = gower.gower_matrix(
                    self.data[self.feature_names].values,
                    x_candidate.values,
                    cat_features=[
                        i
                        for i, col in enumerate(self.feature_names)
                        if col in self.categorical_features
                    ],
                )
                distances = dist_matrix[:, 0]
                # Get k nearest neighbors
                k_nearest = np.partition(
                    distances, min(self.k_neighbors - 1, len(distances) - 1)
                )[: self.k_neighbors]

                if self.neighbor_weights is not None:
                    dist_train = np.average(
                        k_nearest,
                        weights=self.neighbor_weights[: len(k_nearest)],
                    )
                else:
                    dist_train = np.mean(k_nearest)
            except:
                dist_train = 1.0  # Penalty if calculation fails

        return (dist_target, dist_x_interest, float(nr_changed), dist_train)

    def _compute_gower_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Gower distance between two instances."""

        distances = []

        for i in range(len(x1)):
            col = self.feature_names[i]

            if col in self.categorical_features:
                # Categorical feature: 0 if same, 1 if different
                distances.append(0.0 if x1[i] == x2[i] else 1.0)
            else:
                # Numerical feature: normalized absolute difference
                col_range = self.gower_ranges[i]
                if col_range > 0:
                    distances.append(abs(x1[i] - x2[i]) / col_range)
                else:
                    distances.append(0.0 if x1[i] == x2[i] else 1.0)

        return np.mean(distances)

    def _select_nsga2(self, population, k: int):
        """NSGA-II selection with crowding distance."""

        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(population)

        selected = []
        front_idx = 0

        while len(selected) + len(fronts[front_idx]) <= k:
            # Add entire front
            selected.extend(fronts[front_idx])
            front_idx += 1
            if front_idx >= len(fronts):
                break

        # Fill remaining slots using crowding distance
        if len(selected) < k and front_idx < len(fronts):
            remaining = k - len(selected)
            front = fronts[front_idx]

            # Calculate crowding distance
            crowding_distances = self._calculate_crowding_distance(front)

            # Sort by crowding distance (descending)
            front_with_distance = list(zip(front, crowding_distances))
            front_with_distance.sort(key=lambda x: x[1], reverse=True)

            # Add individuals with highest crowding distance
            selected.extend([ind for ind, _ in front_with_distance[:remaining]])

        return selected

    def _fast_non_dominated_sort(self, population):
        """Fast non-dominated sorting algorithm."""

        fronts = [[]]
        domination_count = {}
        dominated_solutions = {}

        for i, ind1 in enumerate(population):
            domination_count[i] = 0
            dominated_solutions[i] = []

            for j, ind2 in enumerate(population):
                if i == j:
                    continue

                if self._dominates(ind1, ind2):
                    dominated_solutions[i].append(j)
                elif self._dominates(ind2, ind1):
                    domination_count[i] += 1

            if domination_count[i] == 0:
                fronts[0].append(ind1)

        front_idx = 0
        while len(fronts[front_idx]) > 0:
            next_front = []
            for i, ind in enumerate(population):
                if ind in fronts[front_idx]:
                    for j in dominated_solutions[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(population[j])

            if len(next_front) > 0:
                fronts.append(next_front)
                front_idx += 1
            else:
                break

        return fronts

    def _dominates(self, ind1, ind2) -> bool:
        """Check if ind1 dominates ind2."""

        fitness1 = ind1.fitness.values
        fitness2 = ind2.fitness.values

        # For minimization: ind1 dominates ind2 if ind1 is better in at least one objective
        # and not worse in any objective
        at_least_one_better = False

        for f1, f2 in zip(fitness1, fitness2):
            if f1 > f2:  # ind1 is worse in this objective
                return False
            elif f1 < f2:  # ind1 is better in this objective
                at_least_one_better = True

        return at_least_one_better

    def _calculate_crowding_distance(self, front) -> List[float]:
        """Calculate crowding distance for individuals in a front."""

        if len(front) <= 2:
            return [float("inf")] * len(front)

        distances = [0.0] * len(front)
        n_objectives = len(front[0].fitness.values)

        for obj_idx in range(n_objectives):
            # Sort by objective value
            front_with_idx = [(ind, i) for i, ind in enumerate(front)]
            front_with_idx.sort(key=lambda x: x[0].fitness.values[obj_idx])

            # Boundary points get infinite distance
            distances[front_with_idx[0][1]] = float("inf")
            distances[front_with_idx[-1][1]] = float("inf")

            # Calculate range for normalization
            obj_range = (
                front_with_idx[-1][0].fitness.values[obj_idx]
                - front_with_idx[0][0].fitness.values[obj_idx]
            )

            if obj_range == 0:
                continue

            # Calculate crowding distance for intermediate points
            for i in range(1, len(front_with_idx) - 1):
                distances[front_with_idx[i][1]] += (
                    front_with_idx[i + 1][0].fitness.values[obj_idx]
                    - front_with_idx[i - 1][0].fitness.values[obj_idx]
                ) / obj_range

        return distances

    def generate_counterfactuals(self, verbose: bool = True) -> Dict:
        """
        Generate counterfactual explanations using NSGA-II.

        Parameters:
        -----------
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        dict : Results containing counterfactuals and statistics
        """

        if verbose:
            print(f"Generating counterfactuals with NSGA-II...")
            print(f"Population size: {self.population_size}")
            print(f"Generations: {self.generations}")
            print(f"Target: {self.target}")

        # Initialize population
        population = self.toolbox.population(n=self.population_size)

        # Evaluate initial population
        fitnesses = [self.toolbox.evaluate(ind) for ind in population]
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Statistics tracking
        stats = {
            "generation": [],
            "min_dist_target": [],
            "mean_dist_target": [],
            "min_dist_x_interest": [],
            "mean_dist_x_interest": [],
            "min_nr_changed": [],
            "mean_nr_changed": [],
            "min_dist_train": [],
            "mean_dist_train": [],
            "hypervolume": [],
        }

        # Evolution loop
        progress_bar = (
            tqdm(range(self.generations), desc="Evolution")
            if verbose
            else range(self.generations)
        )

        for gen in progress_bar:
            # Select parents
            offspring = algorithms.varAnd(
                population, self.toolbox, self.p_crossover, self.p_mutation
            )

            # Evaluate offspring
            fitnesses = [self.toolbox.evaluate(ind) for ind in offspring]
            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            # Select next generation
            population = self.toolbox.select(
                population + offspring, self.population_size
            )

            # Record statistics
            fitness_values = np.array(
                [ind.fitness.values for ind in population]
            )
            stats["generation"].append(gen)
            stats["min_dist_target"].append(np.min(fitness_values[:, 0]))
            stats["mean_dist_target"].append(np.mean(fitness_values[:, 0]))
            stats["min_dist_x_interest"].append(np.min(fitness_values[:, 1]))
            stats["mean_dist_x_interest"].append(np.mean(fitness_values[:, 1]))
            stats["min_nr_changed"].append(np.min(fitness_values[:, 2]))
            stats["mean_nr_changed"].append(np.mean(fitness_values[:, 2]))
            stats["min_dist_train"].append(np.min(fitness_values[:, 3]))
            stats["mean_dist_train"].append(np.mean(fitness_values[:, 3]))

            # Calculate hypervolume (simplified)
            hv = self._calculate_hypervolume(fitness_values)
            stats["hypervolume"].append(hv)

            # Store population history
            self.population_history.append([ind[:] for ind in population])

            if verbose:
                progress_bar.set_postfix(
                    {
                        "Min Target Dist": f"{stats['min_dist_target'][-1]:.4f}",
                        "HV": f"{hv:.4f}",
                    }
                )

        # Store final results
        self.log = pd.DataFrame(stats)

        # Extract final counterfactuals
        final_population = population
        counterfactuals = []

        for ind in final_population:
            cf_data = dict(zip(self.feature_names, ind))
            cf_data.update(
                {
                    "dist_target": ind.fitness.values[0],
                    "dist_x_interest": ind.fitness.values[1],
                    "nr_changed": ind.fitness.values[2],
                    "dist_train": ind.fitness.values[3],
                }
            )
            counterfactuals.append(cf_data)

        counterfactuals_df = pd.DataFrame(counterfactuals)

        # Calculate differences from x_interest
        differences = {}
        for col in self.feature_names:
            if col in self.numerical_features:
                differences[col] = (
                    counterfactuals_df[col] - self.x_interest[col].iloc[0]
                )
            else:
                differences[col] = counterfactuals_df[col].apply(
                    lambda x: x if x != self.x_interest[col].iloc[0] else 0
                )

        differences_df = pd.DataFrame(differences)

        self.results = {
            "counterfactuals": counterfactuals_df,
            "differences": differences_df,
            "x_interest": self.x_interest,
            "target": self.target,
            "statistics": stats,
        }

        if verbose:
            print(f"\nGenerated {len(counterfactuals_df)} counterfactuals")
            feasible = len(
                counterfactuals_df[
                    counterfactuals_df["dist_target"] <= (self.epsilon or 0.1)
                ]
            )
            print(f"Feasible solutions: {feasible}")

        return self.results

    def _calculate_hypervolume(self, fitness_values: np.ndarray) -> float:
        """Calculate hypervolume indicator (simplified version)."""

        # Use reference point as maximum values + 1
        ref_point = np.max(fitness_values, axis=0) + 1

        # For simplicity, we'll use a basic approximation
        # In practice, you might want to use a dedicated HV library
        pareto_front = self._get_pareto_front(fitness_values)

        if len(pareto_front) == 0:
            return 0.0

        # Calculate dominated volume (simplified)
        volumes = []
        for point in pareto_front:
            volume = np.prod(ref_point - point)
            volumes.append(volume)

        return np.sum(volumes)

    def _get_pareto_front(self, fitness_values: np.ndarray) -> np.ndarray:
        """Extract Pareto front from fitness values."""

        is_efficient = np.ones(len(fitness_values), dtype=bool)

        for i, point in enumerate(fitness_values):
            if is_efficient[i]:
                # Remove dominated points
                dominated = np.all(fitness_values <= point, axis=1) & np.any(
                    fitness_values < point, axis=1
                )
                is_efficient = is_efficient & ~dominated
                is_efficient[i] = True  # Keep the current point

        return fitness_values[is_efficient]

    def get_best_counterfactuals(
        self, n: int = 10, criteria: str = "dist_target"
    ) -> pd.DataFrame:
        """Get n best counterfactuals based on specified criteria."""

        if self.results is None:
            raise ValueError(
                "No results available. Run generate_counterfactuals() first."
            )

        df = self.results["counterfactuals"].copy()
        return df.nsmallest(n, criteria)
