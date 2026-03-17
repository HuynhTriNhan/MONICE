# MONICE Experiments

This directory contains experimental evaluations of MONICE on binary classification datasets. The experiments compare MONICE against state-of-the-art counterfactual explanation methods across multiple datasets and machine learning models.

## Data Splitting Strategy

Datasets are partitioned into three distinct subsets to ensure proper evaluation:

1. **Training Set (80%)**: `X_train`, `y_train`

   - Used for training classification models and autoencoders
   - Serves as reference data for MONICE's nearest neighbor search
   - Maintains class distribution through stratified sampling

2. **Test Set (20%)**: `X_test`, `y_test`

   - Used for evaluating model performance
   - Derived from the original dataset with stratified sampling to preserve class distribution
   - Ensures unbiased evaluation of generalization capability

3. **Explanation Set (~200 samples)**: `X_explain`, `y_explain`
   - Subset extracted from the test set
   - Used exclusively for generating counterfactual explanations
   - If test set contains fewer than 200 samples, the entire test set is used
   - Maintains class distribution through stratified sampling

### Implementation

```python
# Initial train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    features, labels,
    stratify=labels,
    test_size=0.2,
    random_state=42
)

# Extract explanation subset from test set
if X_test.shape[0] > 200:
    _, X_explain, _, y_explain = train_test_split(
        X_test, y_test,
        stratify=y_test,
        test_size=200/X_test.shape[0],
        random_state=42
    )
else:
    X_explain = X_test.copy()
    y_explain = y_test.copy()
```

## Experimental Workflow

### 1. Data Acquisition and Preprocessing

```python
from experiments.core.data_fetcher import OpenMLFetcher

# Fetch dataset from OpenML repository
fetcher = OpenMLFetcher('adult', dataset_id=45068)
fetcher.save()  # Saves to data/{dataset_name}/data.pkl
```

The data fetcher performs automatic preprocessing including:

- Missing value handling
- Feature type identification (categorical vs. continuous)
- Label encoding for categorical features
- Removal of high-cardinality string features and date features

### 2. Model Training

```python
from experiments.models.classification_trainer import SklearnTabularModeler

# Train Random Forest or Artificial Neural Network
modeler = SklearnTabularModeler('adult', 'RF')
modeler.grid_search(grid_params)
modeler.save()
```

Models are trained using grid search with cross-validation to select optimal hyperparameters. Both Random Forest (RF) and Artificial Neural Network (ANN) classifiers are evaluated.

### 3. Autoencoder Training

```python
from experiments.models.autoencoder_trainer import AutoencoderTrainer

ae_trainer = AutoencoderTrainer('adult')
ae_trainer.fit(ae_grid)
```

Autoencoders are trained to assess counterfactual plausibility. The reconstruction error serves as a plausibility metric, with lower errors indicating more realistic instances.

### 4. Counterfactual Generation

```python
from experiments.counterfactuals.cf_generator import CounterfactualGenerator

experiment = CounterfactualGenerator('adult', 'RF', 'monicemultiobjectivegower')
experiment.run_experiment()
```

Counterfactuals are generated for each instance in the explanation set using various methods. Results are saved for subsequent analysis.

### 5. Results Analysis

```python
from experiments.analysis.result_reader import ResultReader

reader = ResultReader(
    dataset_names=['adult', 'pima_diabetes', ...],
    models=['RF', 'ANN'],
    cf_names=['monicemultiobjectivegower', 'nice', ...]
)
reader.analyze_results()
reader.statistical_analysis()
```

Statistical analysis is performed to compare methods across datasets and models, including significance testing and effect size calculations.

## Datasets

The following binary classification datasets from OpenML are utilized:

- **adult** (ID: 45068): Income prediction (>$50K)
- **GiveMeSomeCredit** (ID: 46929): Credit default prediction
- **Bank_Customer_Churn** (ID: 46911): Customer churn prediction
- **compas-two-years** (ID: 42192): Recidivism prediction within two years
- **pima_diabetes** (ID: 37): Diabetes diagnosis prediction
- **Heart-Disease-Dataset-(Comprehensive)** (ID: 43672): Heart disease prediction
- **mushroom** (ID: 24): Mushroom edibility classification
- **phoneme** (ID: 1489): Phoneme classification
- **tokyo1** (ID: 40705): Tokyo dataset

These datasets exhibit diverse characteristics including varying numbers of features, sample sizes, and class distributions, providing comprehensive evaluation coverage.

## Counterfactual Methods

The following counterfactual explanation methods are evaluated:

### MONICE Variants

- **monicemultiobjectivegower**: MONICE with Gower distance metric
- **monicemultiobjectiveheom**: MONICE with HEOM (Heterogeneous Euclidean-Overlap Metric) distance
- **monicesparsprox**: MONICE optimizing sparsity and proximity objectives
- **moniceproxplaus**: MONICE optimizing proximity and plausibility objectives
- **monicesparsplaus**: MONICE optimizing sparsity and plausibility objectives

### Baseline Methods

- **NICE variants**: `nicenone`, `nicespars`, `niceprox`, `niceplaus`
- **dicerandom**: DiCE with random sampling strategy
- **moc**: Model-Oblivious Counterfactuals
- **geco**: GECO (Genetic Counterfactual Explanations)
- **lime**: LIME (Local Interpretable Model-agnostic Explanations)

## Evaluation Metrics

Counterfactual explanations are evaluated using the following metrics:

- **Validity**: Proportion of counterfactuals that successfully achieve the target class prediction
- **Sparsity**: Number of features modified in the counterfactual explanation
- **Proximity**: Distance between original instance and counterfactual (measured using Gower and HEOM distances)
- **Robustness**: Prediction entropy, quantifying prediction uncertainty
- **Plausibility**: Autoencoder reconstruction error, measuring how realistic the counterfactual is
- **Coverage**: Proportion of instances for which valid counterfactuals can be generated
- **Diversity**: Diversity of generated counterfactuals for a given instance
- **Time**: Computational time required for counterfactual generation

Results are aggregated and stored in CSV format in the `tables/` directory, organized by model type and metric.

## Running Experiments

The complete experimental pipeline can be executed using `main.ipynb`:

1. **Data Preparation**: Fetch and preprocess all datasets from OpenML
2. **Model Training**: Train classification models (RF and ANN) with hyperparameter optimization
3. **Autoencoder Training**: Train autoencoders for plausibility assessment
4. **Counterfactual Generation**: Generate counterfactuals using all methods for all dataset-model combinations
5. **Analysis**: Perform statistical analysis and generate visualizations

The experiments are designed to run in parallel where possible to reduce computational time. Results are automatically saved and can be analyzed using the provided analysis tools.
