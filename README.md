# MONICE: Multi-Objective Nearest Instance Counterfactual Explanation

MONICE is a novel algorithm for generating counterfactual explanations using evolutionary computation combined with NSGA-II multi-objective optimization.

## Overview

MONICE addresses the critical need for interpretable machine learning by generating actionable counterfactual explanations. It answers "what-if" questions about model predictions, such as:

- "What changes are needed for this loan application to be approved?"
- "What actions should this patient take to reduce their risk score?"
- "Which features need to be modified to achieve a different prediction outcome?"

## Algorithm

MONICE operates through a three-stage process:

1. **Nearest Unlike Neighbors (NUNs) Discovery**: Identifies the closest instances from the target class
2. **Population Initialization**: Creates an initial population through interpolation of NUNs
3. **Multi-Objective Optimization**: Uses NSGA-II to evolve counterfactuals that balance multiple objectives

## Key Features

- **Multi-Objective Optimization**: Balances sparsity, proximity, robustness, and plausibility
- **Evolutionary Search**: Employs NSGA-II for efficient Pareto-optimal solution discovery
- **Constraint Handling**: Supports immutable features, monotonic constraints, and feature bounds
- **Mixed Data Types**: Handles both numerical and categorical features seamlessly

## Installation

```bash
pip install numpy pandas scikit-learn ucimlrepo
```

## Quick Start

### Step 1: Data Preparation

- Prepare heterogeneous tabular data (numerical, categorical, integer)
- Apply label encoding to convert all features to numerical format for computational efficiency
- If using mixed data types, uncomment the `remove_duplicates` section in `examples/monice.py`

### Step 2: Train Models

```python
# Train classifier
predict_fn = lambda x: model.predict_proba(x)

# Train plausibility model
plausibility_model = MLPAutoencoder(X_train, grid_params, preprocessor=preprocessor)
plausibility_model.fit()
```

### Step 3: Initialize MONICE

```python
monice = MONICE(
    X_train=X_train,
    y_train=y_train,
    predict_fn=predict_fn,
    plausibility_model=plausibility_model,
    cat_feats=cat_idx,              # Categorical feature indices
    num_feats=num_idx,              # Numerical feature indices
    integer_feats=integer_features, # Integer feature indices (auto-detected)
    cost_weights=cost_weights,      # Optional: feature change costs
    immutable_features=immutable_features,  # Optional: unchangeable features
    monotonic_constraints=monotonic_constraints, # Optional: constraints
    distance_metric='gower',        # 'gower' or 'heom'
    verbose=True
)
```

### Step 4: Generate Counterfactuals

```python
results = monice.explain(
    X=test_instance,
    target_classes='other',         # Generate for all other classes
    k_nearest=3,                    # Number of nearest neighbors (default: 1)
    n_cfs=3,                       # Number of counterfactuals to generate (default: 1)
    numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0],  # Interpolation steps (default)
    # max_generations and population_size are auto-calculated if not specified
    # max_generations = 3*len(num_feats) + len(cat_feats) - len(immutable_features)
    # population_size = 3*n_cfs
)
```

**For complete examples, see `examples/examples.ipynb`**

## Parameters

### MONICE Constructor

- `X_train`, `y_train`: Training data and labels (required)
- `predict_fn`: Function returning class probabilities (required)
- `plausibility_model`: Model for evaluating counterfactual plausibility (required)
- `cat_feats`: List of categorical feature indices (default: [])
- `num_feats`: List of numerical feature indices (default: [])
- `integer_feats`: List of integer feature indices, auto-detected if not specified (default: [])
- `cost_weights`: Dict mapping feature indices to cost weights (default: None, all features have weight 1.0)
- `immutable_features`: List of feature indices that cannot be changed (default: None)
- `feature_bounds`: Dict mapping feature indices to bounds, auto-initialized if not specified (default: None)
- `monotonic_constraints`: Dict mapping feature indices to 'increasing' or 'decreasing' (default: None)
- `distance_metric`: Distance metric to use - 'gower' or 'heom' (default: 'gower')
- `justified_cf`: Use only correctly classified training instances (default: True)
- `eps`: Small value for numerical stability (default: 1e-6)
- `verbose`: Print debugging information (default: False)

### explain() Method

- `X`: Instance to explain (1D or 2D array) (required)
- `target_classes`: Target classes for counterfactuals - 'other', int, or list of ints (default: 'other')
- `optimization`: List of optimization objectives (default: ['robustness', 'sparsity', 'proximity', 'plausibility'])
- `k_nearest`: Number of nearest neighbors to use (default: 1)
- `n_cfs`: Number of counterfactuals to generate (default: 1)
- `numerical_steps`: Interpolation steps for numerical features (default: [0.2, 0.4, 0.6, 0.8, 1.0])
- `max_generations`: Maximum optimization generations (default: None, auto-calculated as `3*len(num_feats) + len(cat_feats) - len(immutable_features)`)
- `population_size`: Population size for genetic algorithm (default: None, auto-calculated as `3*n_cfs`)

## Optimization Objectives

1. **Robustness**: Lower prediction uncertainty (entropy) is better
2. **Sparsity**: Fewer feature changes are better
3. **Proximity**: Closer to original instance is better (Gower distance)
4. **Plausibility**: More realistic according to your model is better (Autoencoder)

## Constraints and Bounds

### Feature Bounds

```python
feature_bounds = {
    2: (0, 100),           # Numerical feature: must be between 0-100
    3: ['A', 'B', 'C']     # Categorical feature: only allows A, B, or C
}
```

### Monotonic Constraints

```python
monotonic_constraints = {
    4: 'increasing',  # Feature 4 can only increase
    5: 'decreasing'   # Feature 5 can only decrease
}
```

### Cost Weights

```python
cost_weights = {
    0: 3.0,  # Feature 0 costs 3x to change
    1: 1.5,  # Feature 1 costs 1.5x to change
    2: 1.0   # Feature 2 has standard cost
}
```

## Results: CounterfactualResult

Each target class returns a `CounterfactualResult` containing:

- `counterfactual`: Generated counterfactual instances (n_cfs, n_features)
- `target_class`: Target class for counterfactuals
- `original_class`: Originally predicted class
- `original_instance`: Original instance to explain
- `k_nearest`: Number of nearest neighbors used
- `n_cfs`: Number of counterfactuals generated
- `quality_metrics`: Dictionary with quality scores:
  - `prob_target_class`: Probability of target class
  - `valid_cf`: Whether counterfactual is valid
  - `robustness`: Entropy score (lower is better)
  - `sparsity`: Number of changed features (lower is better)
  - `proximity`: Distance from original instance (lower is better)
  - `plausibility`: Plausibility score
  - `cost`: Weighted cost of changes (lower is better)
- `computation_time`: Time taken to generate counterfactuals (seconds)
- `constraints_satisfied`: List indicating constraint satisfaction for each counterfactual
- `nearest_neighbors`: K-nearest neighbors from target class used for initialization
- `optimization_strategy`: List of optimization objectives used
- `numerical_steps`: Numerical interpolation steps used

## Complete Example

See `examples/examples.ipynb` for a complete example including:

- Loading and preprocessing cirrhosis data
- Training Random Forest classifier
- Training MLP autoencoder for plausibility
- Generating counterfactuals with various constraints
- Displaying and analyzing results
