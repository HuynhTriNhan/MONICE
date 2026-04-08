# Modifications to `dice_ml_x`

This document records the changes made to the `dice_ml_x` directory as part of the MONICE experiments.

---

## Bug Fix: `explainer_interfaces/explainer_base.py`

### Method: `predict_fn_for_sparsity`

**File:** `dice_ml_x/explainer_interfaces/explainer_base.py`

### Problem

The original implementation converted the input instance to a 1D NumPy array before calling the model:

```python
def predict_fn_for_sparsity(self, input_instance):
    input_instance_np = self.model.transformer.transform(input_instance).to_numpy(dtype=np.float32)[0]
    input_instance_np = input_instance_np.astype(np.float32)
    return self.model.get_output(input_instance_np)
```

The call `.to_numpy(dtype=np.float32)[0]` first converts the single-row DataFrame `(1, n_features)` to a 2D NumPy array, then indexes with `[0]` to produce a **1D array of shape `(n_features,)`**.

This 1D array is then passed to `model.get_output()` → `model.predict_proba()` → `preprocessor.transform()`, where sklearn's `ColumnTransformer` receives a 1D input. Since `ColumnTransformer` expects a 2D array `(n_samples, n_features)`, it raises:

```
ValueError: X does not contain any features, but ColumnTransformer is expecting N features
```

This error is raised during the **post-hoc sparsity enhancement** step (i.e., after counterfactuals are found), causing the entire `generate_counterfactuals` call to fail and return NaN results.

### Root Cause

`predict_fn_for_sparsity` is called inside `do_posthoc_sparsity_enhancement` with a single-row DataFrame slice. The `[0]` index collapses the 2D structure into 1D, which is incompatible with sklearn's `ColumnTransformer`.

The reference implementation in `dice_ml` (the standard version) does not have this issue:

```python
# dice_ml (correct)
def predict_fn_for_sparsity(self, input_instance):
    return self.model.get_output(input_instance)
```

### Fix

Removed the erroneous NumPy conversion and aligned the implementation with `dice_ml`:

```python
def predict_fn_for_sparsity(self, input_instance):
    """prediction function for sparsity correction"""
    return self.model.get_output(input_instance)
```

### Impact

- All other code paths (`predict_fn`, training, evaluation) are unaffected.
- The fix allows posthoc sparsity enhancement to complete successfully, enabling `DiceExtendedWrapper` to return valid counterfactuals instead of NaN.

---

## Design Decision: `DiceExtendedWrapper` uses `method='genetic'`

**File:** `MONICE_experiments/binary/experiments/counterfactuals/cf_wrapper.py`

### Motivation

`dice_ml_x` extends the original `dice_ml` library by adding a **robustness term** to the counterfactual generation objective. This robustness term is only available in the `DiceGenetic` explainer - not in `DiceRandom` or `DiceKD`.

`DiceExtendedWrapper` is therefore configured with `method='genetic'` to take advantage of this feature:

```python
self.explainer = dice_ml_x.Dice(self.d, self.m, method='genetic')
```

### How Robustness Works in `DiceGenetic`

The genetic algorithm optimizes a combined loss:

```
total_loss = y_loss
           + proximity_weight  * proximity_loss
           + sparsity_weight   * sparsity_loss
           + robustness_weight * robustness_loss
```

The `robustness_loss` measures how stable the counterfactuals are under small perturbations of the input. At each iteration, the current candidate CFs are perturbed (via Gaussian noise on continuous features and random flips on categorical features), and the distance between the original CFs and their perturbed counterparts is computed using the **Dice-Sørensen coefficient**.

A lower `robustness_loss` means the CFs remain valid even when the input shifts slightly, which is desirable for real-world actionability.

### Default Hyperparameters

| Parameter | Default | Note |
|---|---|---|
| `robustness_weight` | `0.4` | Weight of robustness term in total loss |
| `robustness_type` | `DICE_SORENSEN` | Distance metric for robustness |
| `perturbation_method` | `"gaussian"` | How CFs are perturbed internally |

These defaults are applied automatically when calling `generate_counterfactuals` without extra arguments, so no additional configuration is needed in `DiceExtendedWrapper.explain()`.

### Why Not `method='random'`?

`DiceRandom` does not include any robustness term in its objective. Its `_generate_counterfactuals` signature does not accept `robustness_weight`, and any such argument passed via `**kwargs` is silently ignored. Using `method='random'` with `dice_ml_x` produces results **identical** to the standard `dice_ml`, making `DiceExtendedWrapper` indistinguishable from `DiceRandomWrapper`.
