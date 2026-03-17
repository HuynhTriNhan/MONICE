# Multi-Objective Counterfactuals (MOC) – Multi-Objective Explainable AI Algorithm

## About This Implementation

This is a **Python reimplementation** of the Multi-Objective Counterfactuals (MOC) method. The original work and reference implementation are as follows:

- **Original paper:**  
  Dandl, S., Molnar, C., Binder, M., Bischl, B. (2020). *Multi-Objective Counterfactual Explanations.*  
  [arXiv:2004.11165](https://arxiv.org/abs/2004.11165)

- **Original implementation (R):**  
  The official code from the paper is written in **R** and available at:  
  **https://github.com/dandls/moc**

This repository provides a Python port of that algorithm for users who prefer or require a Python workflow. The implementation in `moc_counterfactuals.py` follows the algorithm described in the paper and uses NSGA-II with mixed-type (numerical and categorical) support.

---

## Overview

**MOC (Multi-Objective Counterfactuals)** is an algorithm for generating **counterfactual explanations** for machine learning models. It uses **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) to optimize multiple objectives at once, producing high-quality, realistic explanations.

### What Are Counterfactual Explanations?

Counterfactual explanations answer the question: *"What would need to change to get a different prediction?"*

**Example:** If someone is denied a loan due to low income, a counterfactual might say: *"If income increased to $50,000, the loan would be approved."*

## MOC Algorithm Architecture

### Four Optimization Objectives

MOC optimizes four objectives simultaneously:

1. **Distance to Target Prediction**
   - Distance between the current prediction and the desired prediction
   - e.g. From "denied" (0.2) to "approved" (0.8)

2. **Distance to Original Instance**
   - Gower distance between the original instance and the counterfactual
   - Keeps the counterfactual close to the original input

3. **Number of Changed Features**
   - How many features must change
   - Favors solutions with fewer changes (sparsity / feasibility)

4. **Distance to Training Data**
   - Distance to the nearest training point
   - Encourages counterfactuals to stay within the data distribution

### NSGA-II Genetic Algorithm

MOC uses **NSGA-II**, a multi-objective evolutionary algorithm:

```
Initialize population
├── Create random individuals
├── Apply constraints (fixed features, max_changed)
└── Evaluate fitness for 4 objectives

Evolution loop (generations):
├── 1. Selection: Select best parents
├── 2. Crossover: Create offspring
├── 3. Mutation: Random mutation
├── 4. Evaluation: Evaluate new offspring
├── 5. Non-dominated Sorting: Pareto ranking
└── 6. Crowding Distance: Favor diversity

Result: Set of Pareto-optimal counterfactuals
```

## Implementation Details

### Population Initialization

MOC supports three initialization strategies:

1. **Random**: Random values within allowed bounds
2. **Standard Deviation**: Normal distribution around the original value
3. **Training Data**: Sample from the training set

```python
# Example initialization for feature "income"
if initialization == "random":
    income = random.uniform(min_income, max_income)
elif initialization == "sd":
    income = normal(original_income, std_income)
else:  # traindata
    income = random.choice(training_incomes)
```

### Genetic Operators

#### 1. **Crossover**
- **SBX Crossover** for numerical features
- **Uniform Crossover** for categorical features
- Controlled by: `p_crossover_gen`, `p_crossover_use_orig`

```python
# SBX Crossover for numerical features
eta = 20.0  # Distribution index
beta = calculate_beta(rand, eta)
child1 = 0.5 * ((parent1 + parent2) - beta * abs(parent2 - parent1))
child2 = 0.5 * ((parent1 + parent2) + beta * abs(parent2 - parent1))
```

#### 2. **Mutation**
- **Gaussian Mutation** for numerical features
- **Random Choice** for categorical features
- Controlled by: `p_mutation_gen`, `p_mutation_use_orig`

```python
# Gaussian Mutation
std = (feature_max - feature_min) * 0.1  # 10% of range
new_value = normal(current_value, std)
new_value = clip(new_value, feature_min, feature_max)
```

### Non-dominated Sorting and Crowding Distance

#### Fast Non-dominated Sorting

Individuals are ranked into **Pareto fronts**:

```
Front 0: Non-dominated solutions (best)
Front 1: Dominated only by Front 0
Front 2: Dominated only by Front 0 and 1
...
```

#### Crowding Distance

Measures diversity within a front:
- **Boundary points**: distance = ∞
- **Internal points**: distance based on neighbors

### Gower Distance

Measures distance for mixed-type data (numerical + categorical):

```python
def gower_distance(x1, x2):
    distances = []
    for i, feature in enumerate(features):
        if feature is categorical:
            # 0 if same, 1 if different
            distances.append(0 if x1[i] == x2[i] else 1)
        else:
            # Normalized absolute difference
            range_i = feature_max[i] - feature_min[i]
            distances.append(abs(x1[i] - x2[i]) / range_i)

    return mean(distances)
```

## Usage

### Install Dependencies

```bash
pip install pandas numpy scikit-learn deap gower tqdm
```

### Basic Example

```python
from moc_counterfactuals import MOCCounterfactuals
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Prepare data
data = pd.read_csv("dataset.csv")
X = data.drop('target', axis=1)
y = data['target']

# 2. Train model
model = RandomForestClassifier()
model.fit(X, y)

# 3. Select instance to explain
x_interest = X.iloc[[0]]  # First row

# 4. Create MOC Counterfactuals
moc = MOCCounterfactuals(
    predictor=model,
    x_interest=x_interest,
    target=[0.7, 1.0],  # Want prediction between 0.7 and 1.0
    data=X,
    categorical_features=['gender', 'education'],
    numerical_features=['age', 'income'],
    population_size=100,
    generations=50
)

# 5. Generate counterfactuals
results = moc.generate_counterfactuals(verbose=True)

# 6. Get best results
best_cfs = moc.get_best_counterfactuals(n=5, criteria='dist_target')
print(best_cfs)
```

### Advanced Example with Constraints

```python
moc_advanced = MOCCounterfactuals(
    predictor=model,
    x_interest=x_interest,
    target=0.8,
    data=X,
    categorical_features=['gender', 'education'],
    numerical_features=['age', 'income'],

    # Constraints
    fixed_features=['gender'],  # Do not change gender
    max_changed=3,              # At most 3 features may change

    # Custom ranges
    feature_ranges={
        'age': (18, 65),           # Age 18–65
        'income': (20000, 150000)   # Income 20k–150k
    },

    # GA parameters
    population_size=200,
    generations=100,
    p_crossover=0.7,
    p_mutation=0.8,

    # Smarter initialization
    initialization='traindata',

    # Feasibility tracking
    track_feasibility=True,
    k_neighbors=5,

    # Reproducibility
    random_state=42
)

results = moc_advanced.generate_counterfactuals()
```

## Analyzing Results

### Result Structure

```python
results = {
    'counterfactuals': DataFrame,   # Counterfactual solutions
    'differences': DataFrame,       # Differences from x_interest
    'x_interest': DataFrame,        # Original instance
    'target': list,                # Target prediction
    'statistics': dict             # Evolution statistics
}
```

### Quality Evaluation

```python
# Best counterfactuals by criterion
best_target = moc.get_best_counterfactuals(n=5, criteria='dist_target')
best_proximity = moc.get_best_counterfactuals(n=5, criteria='dist_x_interest')
best_sparse = moc.get_best_counterfactuals(n=5, criteria='nr_changed')
best_feasible = moc.get_best_counterfactuals(n=5, criteria='dist_train')

# Pareto front analysis
pareto_front = results['counterfactuals']
print(f"Number of solutions: {len(pareto_front)}")
print(f"Feasible solutions: {len(pareto_front[pareto_front['dist_target'] <= 0.1])}")

# Visualize trade-offs
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.scatter(pareto_front['dist_target'], pareto_front['dist_x_interest'])
plt.xlabel('Distance to Target')
plt.ylabel('Distance to Original')
plt.title('Target vs Proximity Trade-off')

plt.subplot(2, 2, 2)
plt.scatter(pareto_front['nr_changed'], pareto_front['dist_x_interest'])
plt.xlabel('Number of Changes')
plt.ylabel('Distance to Original')
plt.title('Sparsity vs Proximity Trade-off')

plt.tight_layout()
plt.show()
```

## Key Parameters

### Genetic Algorithm Parameters

| Parameter          | Description           | Default | Suggestion                    |
| ------------------ | --------------------- | ------- | ----------------------------- |
| `population_size`  | Population size       | 50      | 100–500 for harder problems   |
| `generations`      | Number of generations | 50      | 100–200 for better results    |
| `p_crossover`      | Crossover probability | 0.6     | 0.6–0.9                       |
| `p_mutation`       | Mutation probability  | 0.7     | 0.5–0.9                       |
| `p_mutation_gen`   | Per-gene mutation     | 0.25    | 0.1–0.5                       |

### Domain-specific Parameters

| Parameter         | Description                  | Example                 |
| ----------------- | ---------------------------- | ----------------------- |
| `fixed_features`  | Features that must not change| `['gender', 'race']`    |
| `max_changed`     | Max number of features to change | `3`                 |
| `feature_ranges`  | Allowed value ranges         | `{'age': (18, 65)}`     |
| `epsilon`         | Acceptable distance to target| `0.1`                   |

## Advantages of MOC

### 1. **Multi-objective Optimization**
- Optimizes several criteria at once
- No need for predefined weights (unlike weighted sum)
- Produces a diverse Pareto front

### 2. **Flexibility**
- Mixed-type data (numerical + categorical)
- Rich constraint support
- Multiple initialization strategies

### 3. **Robustness**
- NSGA-II is well-established
- Crowding distance promotes diversity
- Handles local optima reasonably well

### 4. **Interpretability**
- Results are easy to interpret and explain
- Clear trade-offs between objectives
- Supports decision-making

## Extensions and Customization

### Custom Objectives

```python
def custom_objective(individual, x_interest, data):
    """Add a custom objective"""
    # e.g. Penalize extreme values
    extreme_penalty = sum(
        1 for val in individual
        if val > percentile_95 or val < percentile_5
    )
    return extreme_penalty

# Modify MOC class to add this objective
```

### Custom Constraints

```python
def apply_business_constraints(individual):
    """Apply domain constraints"""
    # e.g. Income cannot drop by more than 20%
    original_income = x_interest['income'].iloc[0]
    if individual[income_idx] < 0.8 * original_income:
        individual[income_idx] = 0.8 * original_income

    return individual
```

## References

### Papers

1. **Dandl, S., Molnar, C., Binder, M., Bischl, B. (2020)**  
   *Multi-Objective Counterfactual Explanations.*  
   [arXiv:2004.11165](https://arxiv.org/abs/2004.11165)

2. **Deb, K., Pratap, A., Agarwal, S., Meyarivan, T. (2002)**  
   *A fast and elitist multiobjective genetic algorithm: NSGA-II.*  
   IEEE Transactions on Evolutionary Computation.

### Original Code (R)

- **MOC (R):** [https://github.com/dandls/moc](https://github.com/dandls/moc) — official R implementation from the paper. This Python package is a reimplementation of that algorithm.

### Libraries

- **DEAP**: Distributed Evolutionary Algorithms in Python  
- **Gower**: Gower distance implementation  
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm  

## Tips and Best Practices

### 1. **Choosing Parameters**

```python
# Simple problem
population_size = 50
generations = 30

# Complex problem
population_size = 200
generations = 100

# High-dimensional data
p_mutation = 0.8  # More exploration
```

### 2. **Handling Categorical Features**

```python
# Encode before running MOC
from sklearn.preprocessing import LabelEncoder

for col in categorical_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    x_interest[col] = le.transform(x_interest[col])
```

### 3. **Monitoring Convergence**

```python
# Plot evolution statistics
stats = results['statistics']
plt.plot(stats['min_dist_target'], label='Min Target Distance')
plt.plot(stats['hypervolume'], label='Hypervolume')
plt.legend()
plt.show()
```

### 4. **Handling Large Datasets**

```python
# Sample subset for training data distance
if len(data) > 10000:
    sample_data = data.sample(n=5000, random_state=42)
    moc = MOCCounterfactuals(..., data=sample_data, ...)
```

---

## Conclusion

MOC is a strong choice for counterfactual explanations:

- **Multi-objective**: Optimizes several criteria at once  
- **Flexible**: Supports mixed data types and many constraints  
- **Efficient**: Built on the well-studied NSGA-II  
- **Practical**: Produces feasible, interpretable counterfactuals  

It is especially useful in **interpretable AI** settings (e.g. finance, healthcare, legal) where understanding *"why"* and *"what to change"* matters.
