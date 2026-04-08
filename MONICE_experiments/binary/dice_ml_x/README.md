
# DiCE-Extended: Ensemble Selection of Diverse Counterfactual Explanations Using Continuous Optimization

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python Versions](https://img.shields.io/pypi/pyversions/dice-ml)

## Overview

[Mothilal et al. (2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) introduce their method of generating counterfactual explanations considering _feasibility_, and _diversity_. [Guidotti and Ruggieri (2021)](https://link.springer.com/chapter/10.1007/978-3-030-88942-5_28), claim counterfactual explanations to be robust they should be similar for similar instances when they explain. In this study, in a search to improve the quality and reliability of the counterfactual explanations _robustness_ is found to be helpful and it also introduced in the optimization function.

DiCE-Extended is built upon the [DiCE (Diverse Counterfactual Explanations)](https://github.com/interpretml/DiCE) [(Mothilal et al. 2020)](https://dl.acm.org/doi/10.1145/3351095.3372850) framework by introducing a robustness term in the optimization function.

## Installation

DiCE-Extended requires Python 3.6 or higher. Install it using pip:

```bash
pip install dice-extended
```

Alternatively, install via conda:

```bash
conda install -c conda-forge dice-extended
```

## Getting Started

The following code piece provides an idea on how to generate counterfactual explanations with dice-extended (The code will be updated):

```python
import dice_ml_x
from dice_ml_x.utils import helpers
from sklearn.model_selection import train_test_split

# Load dataset
dataset = helpers.load_adult_income_dataset()
target = dataset["income"]
train_dataset, test_dataset, _, _ = train_test_split(
    dataset, target, test_size=0.2, random_state=0, stratify=target
)

# Initialize Data and Model
d = dice_ml_x.Data(
    dataframe=train_dataset,
    continuous_features=['age', 'hours_per_week'],
    outcome_name='income'
)
m = dice_ml_x.Model(
    model_path=dice_ml.utils.helpers.get_adult_income_modelpath(),
    backend='TF2',
    func="ohe-min-max"
)

# Generate Counterfactual Explanations
exp = dice_ml_x.DiceX(d, m)
query_instance = test_dataset.drop(columns="income")[0:1]
dice_exp = exp.generate_counterfactuals(
    query_instance, total_CFs=4, desired_class="opposite"
)
dice_exp.visualize_as_dataframe()
```

## Manipulated Optimization Function

The core enhancement in DiCE-Extended is the manipulated optimization function, designed to balance proximity, diversity, and feasibility of counterfactuals. The function is formulated as:

<!-- 
$$ C(x) = \operatorname*{arg\,min}_{c_1, ... , c_k} \frac{1}{2} \sum_{i}^{k}yloss(f (c_i ), y) + \frac{\lambda_1}{k}\sum_{i}^{k}dist(c_i , x) - \lambda_2*dpp\_diversity(c_1, ... ,c_k) - \frac{\lambda_3}{k}\sum_{i}^{k}robustness(c_i, c_i') $$
-->

$$
C(x) = \underset{c_1, ..., c_k}{\text{arg min}}
\frac{1}{2} \sum_{i=1}^{k} yloss(f(c_i), y) +
\frac{\lambda_1}{k} \sum_{i=1}^{k} dist(c_i, x) -
\lambda_2 \cdot dpp\_diversity(c_1, ..., c_k) -
\frac{\lambda_3}{k} \sum_{i=1}^{k} robustness(c_i, c_i')
$$

- **Proximity Loss**: The first term that averages the distance between generated counterfactuals and the original input ensure the counterfactuals to be as close as possible to the original input.
- **Diversity Loss**: Diversity of the counterfactual explanations is aquired by determinental point process of which loss is represented by the second term and it ensures that _k_ number of counterfactual explanations are generated.
- **Robustness Loss**: [Guidotti (2024)](https://link.springer.com/article/10.1007/s10618-022-00831-6) defines robustness as necessity of similar instances being explained by similar counterfactual explanations such that if $b(x_1)=b(x_2)=y$ then an explainer $f$ should generate counterfactuals $c_1$ and $c_2$ that are similar and can explain $x_1$ and $x_2$. The robustness term that is based on [Dice-Sørensen Coefficient](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient), is adopted from [Bonasera and Carrizosa (2024)](
https://doi.org/10.48550/arXiv.2407.00843).

$$Robustness(c_i, c_i') = \frac{2 * \lvert c_i \cap c_i' \rvert}{\lvert c_i \rvert + \lvert c_i' \rvert} $$

By adjusting the weights $\lambda_1$, $\lambda_2$, $\lambda_3$ counterfactual explanations can be customised by specific needs.

## Evaluation

For experimental purposes, $\lambda_3$, the coefficient of the robustness loss, is set to `0.2`. For counterfactual generation by considering the robustness **genetic algorithm** is adapted to the modified loss function given above.

### Preliminary Evaluation

Since computing robustness loss requires generating perturbed version of the counterfactual instance generated for a given original instance, a perturbation module is integrated to the project. In the module, `gaussian`, `spherical`, and `random` perturbation strategies are introduced. The Gaussian perturbation generation strategy generates perturbed counterfactual instances by uniformly perturbing the continuous features and randomly perturbing the categorical features. To guarantee the convergence `L-BFGS-B`*(Limited Broyden–Fletcher–Goldfarb–Shanno Bounded)* optimization algorithm and brute-force method are combined. For the remaining strategies brute-force approach is adopted. The outputs and the time required for generation of each strategies are given below.

### 1. Counterfactual Generation with Gaussian Perturbation Strategy

#### Original Instance

|   | age | workclass | education | marital_status | occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|------------|------|--------|----------------|--------|
| 0 | 29  | Private   | HS-grad   | Married        | Blue-Collar | White | Female | 38           | 0      |

#### Generated Counterfactuals

|   | age | workclass | education | marital_status | occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|------------|------|--------|----------------|--------|
| 0 |	- |	- |	Assoc |	- |	Service |	- |	- |	- |	1 |
| 1 |	30 |	- |	Bachelors |	- |	Professional |	- |	- |	- |	1 |
| 2	| 27 |	- |	Assoc | 	- |	Service |	- |	- |	- |	1 |
| 3	| -| 	- |	Assoc |	- |	Service |	- |	- |	40 |	1 |
| 4 |	- |	- |	Bachelors |	- |	- |	- |	Male |	40 |	1 |

#### Required Time(s)

100%|██████████| 1/1 [01:02<00:00, 62.95s/it]

### 2. Counterfactual Generation with Random Perturbation Strategy

#### Original Instance

|   | age | workclass | education | marital_status | occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|------------|------|--------|----------------|--------|
| 0 | 29  | Private   | HS-grad   | Married        | Blue-Collar | White | Female | 38           | 0      |

#### Generated Counterfactuals

|   | age | workclass | education | marital_status | occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|------------|------|--------|----------------|--------|
| 0 |	30 |	- |	Bachelors |	- |	Professional |	- |	- |	- |	1 |
| 1 |	27 |	- |	Assoc |	- |	Service |	- |	- |	- |	1 |
| 2 |	- |	- |	Bachelors |	- |	- |	- |	Male |	40 |	1 |
| 3 |	- |	- |	Assoc |	- |	White-Collar |	- |	- |	40 |	1 |
| 4 |	- |	- |	Bachelors |	- |	Sales |	- |	- |	40 |	1 |

#### Required Time(s)

100%|██████████| 1/1 [00:59<00:00, 59.46s/it]

### 3. Counterfactual Generation with Spherical Perturbation Strategy

#### Original Instance

|   | age | workclass | education | marital_status | occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|------------|------|--------|----------------|--------|
| 0 | 29  | Private   | HS-grad   | Married        | Blue-Collar | White | Female | 38           | 0      |

#### Generated Counterfactuals

|   | age | workclass | education | marital_status |   occupation | race | gender | hours_per_week | income |
|---|-----|-----------|-----------|----------------|--------------|------|--------|----------------|--------|
| 0 |  30 |         - | Bachelors |              - | Professional |    - |      - |              - |      1 |
| 0 |  27 |         - |     Assoc |              - |      Service |    - |      - |              - |      1 |
| 0 |   - |         - | Bachelors |              - |            - |    - |   Male |             40 |      1 |
| 0 |   - |         - |     Assoc |              - |      Service |    - |      - |             40 |      1 |
| 0 |  30 |         - |   Masters |              - | Professional |    - |   Male |              - |      1 |

#### Required Time(s)

100%|██████████| 1/1 [01:13<00:00, 73.96s/it]

## Contributing

We welcome contributions to DiCE-Extended. Please refer to our [Contributing Guidelines](CONTRIBUTING.md) for more information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Citing

In case you may found useful this work for your research cite the original dice paper and this study's paper as well.

```bibtex
@inproceedings{mothilal2020dice,
  title={Explaining machine learning classifiers through diverse counterfactual explanations},
  author={Mothilal, Ramaravind K and Sharma, Amit and Tan, Chenhao},
  booktitle={Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency},
  pages={607--617},
  year={2020}
}

@InProceedings{10.1007/978-3-032-08384-5_25,
author="Bakir, Volkan
and Goktas, Polat
and Aky{\"u}z, S{\"u}reyya",
editor="Le Thi, Hoai An
and Pham Dinh, Tao
and Le, Hoai Minh",
title="DiCE-Extended: A Robust Approach to Counterfactual Explanations in Machine Learning",
booktitle="Modelling, Computation and Optimization in Information Systems and Management Sciences",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="299--310",
abstract="Explainable artificial intelligence (XAI) has become increasingly important in decision-critical domains such as healthcare, finance, and law. Counterfactual (CF) explanations, a key approach in XAI, provide users with actionable insights by suggesting minimal modifications to input features that lead to different model outcomes. Despite significant advancements, existing CF generation methods often struggle to balance proximity, diversity, and robustness, limiting their real-world applicability. A widely adopted framework, Diverse Counterfactual Explanations (DiCE), emphasizes diversity but lacks robustness, making CF explanations sensitive to perturbations and domain constraints. To address these challenges, we introduce DiCE-Extended, an enhanced CF explanation framework that integrates multi-objective optimization techniques to improve robustness while maintaining interpretability. Our approach introduces a novel robustness metric based on the Dice-S{\o}rensen coefficient, enabling stability under small input variations. Additionally, we refine CF generation using weighted loss components ({\$}{\$}{\backslash}lambda {\_}p{\$}{\$}$\lambda$p, {\$}{\$}{\backslash}lambda {\_}d{\$}{\$}$\lambda$d, {\$}{\$}{\backslash}lambda {\_}r{\$}{\$}$\lambda$r) to balance proximity, diversity, and robustness. We empirically validate DiCE-Extended on benchmark datasets (COMPAS, Lending Club, German Credit, Adult Income) across multiple ML backends (Scikit-learn, PyTorch, TensorFlow). Results demonstrate improved CF validity, stability, and alignment with decision boundaries compared to standard DiCE-generated explanations. Our findings highlight the potential of DiCE-Extended in generating more reliable and interpretable CFs for high-stakes applications. Future work could explore adaptive optimization techniques and domain-specific constraints to further enhance CF generation in real-world scenarios.",
isbn="978-3-032-08384-5"
}
```

## Acknowledgments

We extend our gratitude to the authors of [DiCE](https://github.com/interpretml/DiCE) for their foundational work in counterfactual explanations generation.
