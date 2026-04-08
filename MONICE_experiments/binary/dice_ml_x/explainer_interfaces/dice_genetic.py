"""
Module to generate diverse counterfactual explanations based on genetic algorithm
This code is similar to 'GeCo: Quality Counterfactual Explanations in Real Time': https://arxiv.org/pdf/2101.01292.pdf
"""
import copy
import random
import timeit

import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException
import scipy.optimize as opt
from sklearn.preprocessing import OneHotEncoder

from dice_ml_x import diverse_counterfactuals as exp
from dice_ml_x.constants import ModelTypes, RobustnessType
from dice_ml_x.explainer_interfaces.explainer_base import ExplainerBase

from dice_ml_x.perturbation_factory import PerturbationFactory
from typing import Optional
from enum import Enum


class DiceGenetic(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method

        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.
        """
        super().__init__(data_interface, model_interface)  # initiating data related parameters
        self.num_output_nodes = None

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.loss_history = {
            "iterations": [],
            "y_loss": [],
            "sparsity_loss": [],
            "proximity_loss": [],
            "robustness_loss": [],
            "total_loss": []
        }
        self.original_instance: pd.DataFrame = pd.DataFrame()

        # Initializing a label encoder to obtain label-encoded values for categorical variables
        self.labelencoder = self.data_interface.fit_label_encoders()
        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'

    def update_hyperparameters(self, proximity_weight, sparsity_weight,
                               diversity_weight, robustness_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.robustness_weight = robustness_weight
        self.categorical_penalty = categorical_penalty

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights,
                                encoding='one-hot'):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]
        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type
        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=False)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            feature_weights_list = []
            if encoding == 'one-hot':
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            elif encoding == 'label':
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # the weight is inversely proportional to max value
                        feature_weights_list.append(round(1 / self.feature_range[feature].max(), 2))
            self.feature_weights_list = [feature_weights_list]

    def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
        remaining_cfs = np.zeros((num_inits, self.data_interface.number_of_features))
        # kx is the number of valid inits found so far
        kx = 0
        precisions = self.data_interface.get_decimal_precisions()
        while kx < num_inits:
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature in features_to_vary:
                    if feature in self.data_interface.continuous_feature_names:
                        one_init[jx] = np.round(np.random.uniform(
                            self.feature_range[feature][0], self.feature_range[feature][1]), precisions[jx])
                    else:
                        one_init[jx] = np.random.choice(self.feature_range[feature])
                else:
                    one_init[jx] = query_instance[jx]
            if self.is_cf_valid(self.predict_fn_scores(one_init)):
                remaining_cfs[kx] = one_init
                kx += 1
        return remaining_cfs

    def do_KD_init(self, features_to_vary, query_instance, cfs, desired_class, desired_range):
        cfs = self.label_encode(cfs)
        cfs = cfs.reset_index(drop=True)

        self.cfs = np.zeros((self.population_size, self.data_interface.number_of_features))
        for kx in range(self.population_size):
            if kx >= len(cfs):
                break
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = (query_instance[jx])
                else:
                    if feature in self.data_interface.continuous_feature_names:
                        if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if self.feature_range[feature][0] <= query_instance[jx] <= self.feature_range[feature][1]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.uniform(
                                    self.feature_range[feature][0], self.feature_range[feature][1])
                    else:
                        if cfs.iat[kx, jx] in self.feature_range[feature]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if query_instance[jx] in self.feature_range[feature]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.choice(self.feature_range[feature])
            self.cfs[kx] = one_init
            kx += 1

        new_array = [tuple(row) for row in self.cfs]
        uniques = np.unique(new_array, axis=0)

        if len(uniques) != self.population_size:
            remaining_cfs = self.do_random_init(
                self.population_size - len(uniques), features_to_vary, query_instance, desired_class, desired_range)
            self.cfs = np.concatenate([uniques, remaining_cfs])

    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, desired_range,
                              desired_class,
                              query_instance, query_instance_df_dummies, verbose):
        """Intializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1  # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        self.features_to_vary = features_to_vary

        # CF initialization
        self.cfs = []
        if initialization == 'random':
            self.cfs = self.do_random_init(
                self.population_size, features_to_vary, query_instance, desired_class, desired_range)

        elif initialization == 'kdtree':
            # Partitioned dataset and KD Tree for each class (binary) of the dataset
            self.dataset_with_predictions, self.KD_tree, self.predictions = \
                self.build_KD_tree(self.data_interface.data_df.copy(),
                                   desired_range, desired_class, self.predicted_outcome_name)
            if self.KD_tree is None:
                self.cfs = self.do_random_init(
                    self.population_size, features_to_vary, query_instance, desired_class, desired_range)

            else:
                num_queries = min(len(self.dataset_with_predictions), self.population_size * self.total_CFs)
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output, desired_class, desired_range)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class,
                                 query_instance, query_instance_df_dummies, algorithm, features_to_vary,
                                 permitted_range, yloss_type, diversity_loss_type, feature_weights,
                                 proximity_weight, sparsity_weight, diversity_weight, robustness_weight,
                                 categorical_penalty, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        if len(self.cfs) != total_CFs:
            self.do_cf_initializations(
                total_CFs, initialization, algorithm, features_to_vary, desired_range, desired_class,
                query_instance, query_instance_df_dummies, verbose)

        else:
            self.total_CFs = total_CFs
        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, sparsity_weight, diversity_weight, robustness_weight, categorical_penalty)

    def _generate_counterfactuals(self, query_instance, total_CFs, perturbation_method="gaussian", initialization="kdtree",
                                  desired_range=None, desired_class="opposite", proximity_weight=0.2,
                                  sparsity_weight=0.2, diversity_weight=4.0, robustness_weight=0.4,
                                  categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all",
                                  permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="binary", maxiterations=500, thresh=1e-2, verbose=False,
                                  preprocessing_bins=10, robustness_type=RobustnessType.DICE_SORENSEN,
                                  separate_features: bool | None=None, **kwargs):
        """Generates diverse counterfactual explanations

        :param query_instance: A dictionary of feature names and values. Test point of interest.
        :param total_CFs: Total number of counterfactuals required.
        :param initialization: Method to use to initialize the population of the genetic algorithm
        :param desired_range: For regression problems. Contains the outcome range to generate counterfactuals in.
        :param desired_class: For classification problems. Desired counterfactual class - can take 0 or 1.
                              Default value is "opposite" to the outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param sparsity_weight: A positive float. Larger this weight, less features are changed from the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                                Defaults to the range inferred from training data. If None, uses the parameters initialized
                                in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and
                                corresponding weights as values. Default option is "inverse_mad" where the
                                weight for a continuous feature is the inverse of the Median Absolute Devidation (MAD)
                                of the feature's values in the training set; the weight for a categorical feature is
                                equal to 1 by default.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param maxiterations: Maximum iterations to run the genetic algorithm for.
        :param thresh: The genetic algorithm stops when the difference between the previous best loss and current
                       best loss is less than thresh
        :param verbose: Parameter to determine whether to print 'Diverse Counterfactuals found!'

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                 (see diverse_counterfactuals.py).
        """

        if not hasattr(self.data_interface, 'data_df') and initialization == "kdtree":
            raise UserConfigValidationException(
                    "kd-tree initialization is not supported for private data"
                    " interface because training data to build kd-tree is not available.")

        self.population_size = 10 * total_CFs

        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        # Prepares user defined query_instance for DiCE.
        query_instance_orig = query_instance
        query_instance_orig = self.data_interface.prepare_query_instance(
                query_instance=query_instance_orig)
        query_instance = self.data_interface.prepare_query_instance(
                query_instance=query_instance)
        # number of output nodes of ML model
        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(query_instance)
        self.original_instance = query_instance_orig.copy()
        query_instance = self.label_encode(query_instance)
        query_instance = np.array(query_instance.values[0])
        self.x1 = query_instance

        # find the predicted value of query_instance
        test_pred = self.predict_fn_scores(query_instance)

        self.test_pred = test_pred
        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred[0])

        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in self.data_interface.get_all_dummy_colnames():
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                      query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                      yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                      sparsity_weight, diversity_weight, robustness_weight, categorical_penalty, verbose)

        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, perturbation_method,
                                                      features_to_vary, maxiterations, thresh,
                                                      verbose, preprocessing_bins, robustness_type,
                                                      separate_features=separate_features, **kwargs)

        desired_class_param = self.decode_model_output(pd.Series(self.target_cf_class[0]))[0] \
            if hasattr(self, 'target_cf_class') else desired_class
        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class_param,
                                          model_type=self.model.model_type)

    def predict_fn_scores(self, input_instance):
        """Returns prediction scores."""
        if not isinstance(input_instance, pd.DataFrame):
            input_instance = self.label_decode(input_instance)
        out = self.model.get_output(input_instance)
        if self.model.model_type == ModelTypes.Classifier and out.shape[1] == 1:
            # DL models return only 1 for binary classification
            out = np.hstack((1-out, out))
        return out

    def predict_fn(self, input_instance):
        """Returns actual prediction."""
        if not isinstance(input_instance, pd.DataFrame):
            input_instance = self.label_decode(input_instance)
        
        preds = self.model.get_output(input_instance, model_score=False)
        return preds

    def _predict_fn_custom(self, input_instance, desired_class):
        """Checks that the maximum predicted score lies in the desired class."""
        """The reason we do so can be illustrated by
        this example: If the predict probabilities are [0, 0.5, 0,5], the computed yloss is 0 as class 2 has the same
        value as the maximum score. sklearn's usual predict function, which implements argmax, returns class 1 instead
        of 2. This is why we need a custom predict function that returns the desired class if the maximum predict
        probability is the same as the probability of the desired class."""

        input_instance = self.label_decode(input_instance)
        output = self.model.get_output(input_instance, model_score=True)
        if self.model.model_type == ModelTypes.Classifier and np.array(output).shape[1] == 1:
            # DL models return only 1 for binary classification
            output = np.hstack((1-output, output))
        desired_class = int(desired_class)
        maxvalues = np.max(output, 1)
        predicted_values = np.argmax(output, 1)

        # We iterate through output as we often call _predict_fn_custom for multiple inputs at once
        for i in range(len(output)):
            if output[i][desired_class] == maxvalues[i]:
                predicted_values[i] = desired_class

        return predicted_values
    
    def generate_perturbations_hybrid(self, input_instance: pd.DataFrame, method: str, max_iter=500,
                               tol=1e-1, max_attempts: int=20, **kwargs) -> pd.DataFrame:
        """
        Generates perturbations for given counterfactual instances.

        Args:
            input_instance (pandas.DataFrame): Counterfactual instance that will be perturbed.
            method (str): Perturbation strategy. Supported methods:
                - "gaussian" (GaussianPerturbation)
                - "random" (RandomPerturbation)
                - "spherical" (SphericalPerturbation)
            max_iter (int): Maximum number of iteration for optimization process.
            tol (float): Tolerance for convergence.
            max_attempts (int): Maximum number of attempts if the optimization fails.
            **kwargs (dict): Additional arguments that will be passed to the perturbation generation method.
            
        Returns:
            pandas.DataFrame: Perturbed counterfactuals as pandas.DataFrame.
        """
        perturbation_instance = PerturbationFactory.get_perturbation(method, **kwargs)
        perturbed_cfs = []

        for idx, c_i in input_instance.iterrows():
            c_i_df = pd.DataFrame([c_i])
            c_i_numerical = self.label_encode(c_i_df.copy())
            
            pred_c_i_df = self.predict_fn_scores(c_i_df)
            target_class = np.argmax(pred_c_i_df)

            attempt_cnt = 0
            valid_perturbation_found = False
            while not valid_perturbation_found:
                attempt_cnt += 1
                initial_c_i_prime = perturbation_instance.generate(c_i=c_i_df)
                initial_guess_df = self.label_encode(initial_c_i_prime)
                initial_guess = initial_guess_df.iloc[0, :].values

                bounds = []
                ranges = self.data_interface.get_features_range_float()[1]

                for feature in c_i_df.columns:
                    feat_min, feat_max = ranges[feature]
                    bounds.append((feat_min, feat_max))

                categorical_ranges = {category: rng for category, rng in self.data_interface.get_features_range()[1].items()
                                    if category in self.data_interface.categorical_feature_names}      
                  
                def l2_objective(perturbed_values: np.ndarray):
                    """Computes the l2 loss for predictions"""
                    perturbed_vals_df = pd.DataFrame([perturbed_values], columns=c_i_df.columns)
                    
                    for cat in self.data_interface.categorical_feature_names:
                        if cat in perturbed_vals_df.columns:
                            category_idx = int(round(perturbed_vals_df.at[0, cat]))
                            category_idx = np.clip(category_idx, 0, len(categorical_ranges[cat]))
                            perturbed_vals_df.at[0, cat] = categorical_ranges[cat][category_idx]

                    pred_c_i_proba = self.predict_fn_scores(c_i_df)
                    pred_c_i_prime_proba = self.predict_fn_scores(perturbed_vals_df)

                    pred_c_i_proba_class = pred_c_i_proba[0][target_class]
                    pred_c_i_prime_proba_class = pred_c_i_prime_proba[0][target_class]
                    loss = (pred_c_i_proba_class - pred_c_i_prime_proba_class) ** 2
                    return loss
                
                def jacobian_objective(perturbed_values: np.ndarray) -> np.ndarray:
                    """Computes the Jacobian matrix for trust region"""
                    perturbed_vals_df = pd.DataFrame([perturbed_values], columns=c_i_df.columns)
                    
                    for cat in self.data_interface.categorical_feature_names:
                        if cat in perturbed_vals_df.columns:
                            category_idx = int(round(perturbed_vals_df.at[0, cat]))
                            category_idx = np.clip(category_idx, 0, len(categorical_ranges[cat]))
                            perturbed_vals_df.at[0, cat] = categorical_ranges[cat][category_idx]

                    pred_c_i_proba = self.predict_fn_scores(c_i_df)
                    pred_c_i_prime_proba = self.predict_fn_scores(perturbed_vals_df)

                    pred_c_i_proba_class = pred_c_i_proba[0][target_class]
                    pred_c_i_prime_proba_class = pred_c_i_prime_proba[0][target_class]
                    grad = 2 * (pred_c_i_proba_class - pred_c_i_prime_proba_class) * np.sign(perturbed_values - c_i_numerical.iloc[0].values)
                    return grad
                

                def hessian_objective(perturbed_values: np.ndarray) -> np.ndarray:
                    """Computes the Hessian matrix for trust region"""
                    return 2 * np.eye(len(perturbed_values))
                    
            
                result = opt.minimize(
                    fun=l2_objective,
                    x0=initial_guess,
                    bounds=bounds,
                    method="L-BFGS-B",
                    options={'maxiter': max_iter, 'disp': False}
                )

                '''result = opt.minimize(
                    fun=l2_objective,
                    x0=initial_guess,
                    bounds=bounds,
                    method="trust-exact",
                    jac=jacobian_objective,
                    hess=hessian_objective
                )'''

                if (result.success and result.fun <= tol):
                    perturbed_df = pd.DataFrame([result.x], columns=initial_c_i_prime.columns)
                    for feat in self.data_interface.continuous_feature_names:
                        perturbed_df.at[0, feat] = float(round(perturbed_df.at[0, feat]))
                    perturbed_cfs.append(perturbed_df)
                    valid_perturbation_found = True
                else:
                    std_dev = np.random.uniform(0.1, 0.6)
                    if method == 'gaussian':
                        perturbation_instance.std_dev = std_dev
                    #print(f"Optimization failed for the instances \n {c_i_df} \n {initial_guess}")
                    perturbed_candidate = perturbation_instance.generate(c_i=c_i_df)
                    is_valid_perturbation = perturbation_instance.validate(c_i_df, perturbed_candidate,
                                                                           target_class, self.predict_fn_scores, 0.1)
                    if is_valid_perturbation:
                        perturbed_cfs.append(perturbed_candidate)
                        valid_perturbation_found = True

        if perturbed_cfs:
            perturbed_cfs_df = pd.concat(perturbed_cfs, ignore_index=True)
        else:
            perturbed_cfs_df = pd.DataFrame(columns=input_instance.columns)

        perturbed_cfs_df.reset_index(drop=True, inplace=True)
        return perturbed_cfs_df

    
    def get_validity_percentage(self) -> float:
        """
        Returns the percentage of generated counterfactuals that
        actually satisfy the desired outcome condition
        (`self.is_cf_valid`).

        Notes
        -----
        • Uses `self.final_cfs`, which is populated in
        `find_counterfactuals`.  
        • If no CFs were generated, returns 0.0.
        """
        # make sure we have CFs
        if not hasattr(self, "final_cfs") or self.final_cfs is None or len(self.final_cfs) == 0:
            return 0.0

        # numpy-ise and drop duplicates so duplicates don't inflate validity
        cfs_np = np.unique(np.asarray(self.final_cfs), axis=0)

        # model predictions for each unique CF
        preds = self.predict_fn_scores(cfs_np)

        # count CFs that meet the desired target condition
        valid_mask = np.array([self.is_cf_valid(p) for p in preds])
        validity_pct = 100.0 * valid_mask.mean()
        return float(validity_pct)

    def perturb_cfs(self, input_df: pd.DataFrame,
                    method: str = "gaussian", std_dev: float=0.10,
                    mean: float=0.2) -> pd.DataFrame:

        rng = np.random.default_rng()
        cont_cols = self.data_interface.continuous_feature_names
        cat_cols = self.data_interface.categorical_feature_names
        ranges = self.data_interface.get_features_range_float()[1]
        cont_lo = np.array([ranges[c][0] for c in cont_cols])
        cont_hi = np.array([ranges[c][1] for c in cont_cols])
        span = cont_hi - cont_lo

        X_cont = input_df[cont_cols].to_numpy()

        if method == "gaussian":
            noise = rng.normal(scale=std_dev, size=X_cont.shape) * span
        elif method == "random":
            noise = rng.uniform(-mean * span, mean * span, size=X_cont.shape)
        elif method == "spherical":
            dir_ = rng.normal(size=X_cont.shape)
            dir_ /= (np.linalg.norm(dir_, axis=1, keepdims=True) + 1e-12)
            r = rng.uniform(0.0, mean, size=(X_cont.shape[0], 1))
            noise = dir_ * r * span
        else:
            raise ValueError(method)

        X_cont = np.clip(X_cont + noise, cont_lo, cont_hi)

        cat_part = input_df[cat_cols].copy()
        cat_choices = {c: np.array(self.data_interface.get_features_range()[1][c])
                       for c in cat_cols}
        flip_prob = 0.01
        flip_mask = rng.random(size=cat_part.shape) < flip_prob
        for j, col in enumerate(cat_cols):
            choices = cat_choices[col]
            rnd_vals = rng.choice(choices, size=len(cat_part))
            cat_part.loc[flip_mask[:, j], col] = rnd_vals[flip_mask[:, j]]

        X_df = pd.DataFrame(X_cont, columns=cont_cols, index=None)
        X_df[cat_cols] = cat_part.values
        X_df.reset_index(drop=True, inplace=True)
        return X_df

    def compute_robustness_distance(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame,
                                    preprocessing_bins: int=10, eps=1e-12):
        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(self.label_decode_cfs(cfs),
                                                                                 perturbed_cfs,
                                                                                 num_bins=preprocessing_bins)

        intersection = np.sum(np.minimum(cfs_processed, perturbed_cfs_processed), axis=1)
        union = np.sum(cfs_processed, axis=1) + np.sum(perturbed_cfs_processed, axis=1)

        sorensen_dice_coefficient = (2 * intersection) / (union + eps)
        sorensen_dice_coefficient[np.isnan(sorensen_dice_coefficient)] = 1.0

        return sorensen_dice_coefficient

    def compute_robustness_loss_(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame,
                                 preprocessing_bins: int=10, desired_class="opposite"):
        query_instance = self.x1.copy()
        base_scores = self.predict_fn_scores(query_instance)
        base_pred = int(np.argmax(base_scores, axis=1)[0])
        if desired_class == "opposite":
            target_pred = 1 - base_pred
        else:
            target_pred = desired_class

        dist = self.compute_robustness_distance(cfs, perturbed_cfs, preprocessing_bins)
        perturbed_scores = self.predict_fn_scores(perturbed_cfs)
        perturbed_preds = np.argmax(perturbed_scores, axis=1)
        indicator_var = (perturbed_preds == target_pred).astype(np.float32)
        return float(np.mean(dist * indicator_var))

    def generate_perturbations_fast(
        self,
        input_df: pd.DataFrame,
        method: str = "gaussian",
        tol: float = 0.05,
        batch_size: int = 64,
        n_attempts: int = 5,
        std_dev: float = 0.10,
        max_radius: float = 0.5,
        force_flip_constraint: bool=True,
    ) -> pd.DataFrame:
        
        rng = np.random.default_rng()

        cont_cols = self.data_interface.continuous_feature_names
        cat_cols = self.data_interface.categorical_feature_names
        ranges = self.data_interface.get_features_range_float()[1]

        X_base = input_df[cont_cols].to_numpy()
        cont_lo = np.array([ranges[c][0] for c in cont_cols])
        cont_hi = np.array([ranges[c][1] for c in cont_cols])
        span = cont_hi - cont_lo

        cat_choices = {c: np.array(self.data_interface.get_features_range()[1][c])
                    for c in cat_cols}

        base_scores = self.predict_fn_scores(input_df)
        base_cls = base_scores.argmax(axis=1)
        base_prob = base_scores[np.arange(len(base_scores)), base_cls]

        found_mask = np.zeros(len(input_df), dtype=bool)
        out_rows = input_df.copy()

        radiuses = max_radius * (1.5 ** np.arange(n_attempts))
        sigmas = std_dev * (1.5 ** np.arange(n_attempts))

        for radius, sigma in zip(radiuses, sigmas):
            if found_mask.all():
                break  

            need_idx  = np.where(~found_mask)[0]
            
            n_need    = len(need_idx)
            repeat    = np.repeat(need_idx, batch_size)
            
            X_rep  = X_base[repeat]
            
            if method == "gaussian":
                noise = rng.normal(scale=sigma * span, size=X_rep.shape)
            elif method == "random":
                noise = rng.uniform(-radius * span, radius * span, size=X_rep.shape)
            elif method == "spherical":
                dir   = rng.normal(size=X_rep.shape)
                dir  /= np.linalg.norm(dir, axis=1, keepdims=True)
                r     = rng.uniform(0, radius * span, size=(len(dir), 1))
                noise = dir * r
            else:
                raise ValueError(method)

            X_cont = np.clip(X_rep + noise, cont_lo, cont_hi)

            cat_part = input_df.iloc[repeat][cat_cols].copy()
            flip_mask = rng.random(size=cat_part.shape) < 0.2
            for j, col in enumerate(cat_cols):
                choices = cat_choices[col]
                rnd_vals = rng.choice(choices, size=len(cat_part))
                cat_part.loc[flip_mask[:, j], col] = rnd_vals[flip_mask[:, j]]

            cand_df = pd.DataFrame(X_cont, columns=cont_cols, index=None)
            cand_df[cat_cols] = cat_part.values
            cand_df.reset_index(drop=True, inplace=True)

            cand_scores = self.predict_fn_scores(cand_df)
            cand_prob   = cand_scores[:, base_cls[repeat]]
            delta       = np.abs(cand_prob - base_prob[repeat])

            if force_flip_constraint:
                hit_mask = delta <= tol
            else:
                hit_mask = np.ones(len(delta), dtype=bool)
    
            if not hit_mask.any():
                continue

            flat_hits = np.flatnonzero(hit_mask)
            if flat_hits.size == 0:
                continue

            row_idx  = flat_hits // batch_size
            valid    = row_idx < n_need
            flat_hits = flat_hits[valid]
            row_idx   = row_idx[valid]
            if flat_hits.size == 0:
                continue

            base_rows = need_idx[row_idx]

            _, first_pos = np.unique(base_rows, return_index=True)
            hit_rows_idx = flat_hits[first_pos]
            base_rows    = base_rows[first_pos]
            chosen       = cand_df.iloc[hit_rows_idx]

            cols_to_write = cont_cols + cat_cols
            if cols_to_write:
                out_rows.loc[base_rows, cols_to_write] = chosen[cols_to_write].values
            found_mask[base_rows] = True

        return out_rows.reset_index(drop=True)

    def l2_objective(self, perturbed_df: pd.DataFrame, original_df: pd.DataFrame, predict_proba_fn: callable):
        """
        Computes the L2 loss of the model output for perturbations and the original counterfactals.

        Args:
            perturbed_df (pandas.DataFrame): The perturbed counterfactual as a DataFrame.
            original_df (pandas.DataFrame): Original counterfactual as a DataFrame.
            predict_proba_fn (callable): The prediction function that will return so-called probabilities

        Returns:
            float: The L2 loss value.
        """
        perturbed_pred = predict_proba_fn(perturbed_df)
        original_pred = predict_proba_fn(original_df)

        l2_loss = (perturbed_pred - original_pred) ** 2
        return l2_loss

    def _preprocess_for_robustness(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame, num_bins: int=10) -> tuple:

        continuous_cols = self.data_interface.continuous_feature_names
        categorical_cols = self.data_interface.categorical_feature_names

        def preprocess_continuous_features(data: pd.DataFrame):
            binarized = []
            ranges = self.data_interface.get_features_range_float()[1]

            for col in continuous_cols:
                col_min, col_max = ranges[col]
                bins = np.linspace(col_min, col_max, num=num_bins + 1)
                binned = np.digitize(data[col], bins, right=False)
                binned = np.clip(binned, 0, len(bins) - 1)
                one_hot = np.eye(len(bins))[binned]
                binarized.append(one_hot)
            return np.hstack(binarized)

        combined_data = pd.concat([cfs, perturbed_cfs])
        combined_data[categorical_cols] = combined_data[categorical_cols].astype("str")
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(combined_data[categorical_cols])

        def preprocess_categorical_features(p_data: pd.DataFrame):
            p_data[categorical_cols] = p_data[categorical_cols].astype("str")
            encoded = encoder.transform(p_data[categorical_cols])
            return encoded

        cfs_continuous = preprocess_continuous_features(cfs)
        cfs_categorical = preprocess_categorical_features(cfs)

        perturbed_cfs_continuous = preprocess_continuous_features(perturbed_cfs)
        perturbed_cfs_categorical = preprocess_categorical_features(perturbed_cfs)

        cfs_processed = np.hstack([cfs_continuous, cfs_categorical])
        perturbed_cfs_processed = np.hstack([perturbed_cfs_continuous, perturbed_cfs_categorical])

        return cfs_processed, perturbed_cfs_processed

    def _fit_bins(self, x_df: pd.DataFrame, num_bins: int=10) -> list:
        X = x_df.to_numpy(dtype=float)
        cont_idxs = x_df.columns.get_indexer(self.data_interface.continuous_feature_names)
        bin_centers = []

        for idx in cont_idxs:
            col = X[:, idx]

            if len(col) >= num_bins:
                q = np.linspace(0, 100, num_bins)
                centers = np.percentile(col, q)

            else:
                centers = np.pad(col,
                                 (0, num_bins - len(col)),
                                 mode="edge")
            bin_centers.append(centers)
        return bin_centers


    def _phi_soft(self, X_df: pd.DataFrame, num_bins=10, sigma: float=None, eps: float=1e-8) -> np.ndarray:
        """Soft, differentiable feature map for robustness:
            - Categoricals: per-group softmax (relaxes one-hot)
            - Continuous: soft RBF-binning in [0,1] (num_bins bins)
        """
        X = X_df.to_numpy(dtype=float)
        parts = []
        print(self.data_interface.continuous_feature_names)
        print(X_df.columns)
        cont_idxs = X_df.columns.get_indexer(self.data_interface.continuous_feature_names)
        cat_idxs = list(set(range(len(X_df.columns))) - set(cont_idxs))

        bin_centers = self._fit_bins(X_df, num_bins)

        cont = X[:, cont_idxs]
        cont = np.clip(cont, 0.0, 1.0)

        centers_array = np.array(bin_centers)
        cont_exp = cont[..., None]
        centers_broad = centers_array[None, ...]

        if sigma is None:
            spacings = []
            for centers in bin_centers:
                if len(centers) > 1:
                    sorted_centers = np.sort(centers)
                    avg_spacing = np.mean(np.diff(sorted_centers))
                    spacings.append(avg_spacing)
            if spacings:
                median_spacing = np.median(spacings)
                sigma = max(median_spacing * 0.6, 0.1)
            else:
                sigma = 0.2

        rbf = np.exp(- (cont_exp - centers_broad) ** 2 / (2 * sigma ** 2))
        rbf /= (rbf.sum(axis=-1, keepdims=True) + eps)
        parts.append(rbf.reshape(cont.shape[0], -1))

        for idx in cat_idxs:
            g = X[:, idx:idx+1] if isinstance(idx, (int, np.integer)) else X[:, idx]
            if g.ndim == 1:
                g = g.reshape(-1, 1)

            g_max = g.max(axis=1, keepdims=True)
            g = g - g_max
            e = np.exp(g)
            sm = e / (e.sum(axis=1, keepdims=True) + eps)
            parts.append(sm)
        return np.concatenate(parts, axis=1) if parts else X

    def compute_robustness_distance_binned_RBF(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame,
                                    num_bins: int=10, sigma=None, eps: float=1e-8):
        """
        Soft kernel-based robustness using RBF distance on soft feature representations.
        Returns per-sample similarities in [0, 1].
        """
        
        cfs_df_ohe = self.data_interface.get_ohe_min_max_normalized_data(self.label_decode_cfs(cfs))
        perturbed_cfs_df_ohe = self.data_interface.get_ohe_min_max_normalized_data(perturbed_cfs)

        if not isinstance(cfs_df_ohe, pd.DataFrame) or not isinstance(perturbed_cfs_df_ohe, pd.DataFrame):
            raise ValueError("Both `cfs` and `perturbed_cfs` must be of type pandas.DataFrame")

        if len(cfs_df_ohe) != len(perturbed_cfs_df_ohe):
            raise ValueError("Row counts of `cfs` and `perturbed_cfs` must match")
        if sigma is None:
            bin_width = 1.0 / (num_bins - 1) if num_bins > 1 else 1.0
            sigma = bin_width / 2

        p = self._phi_soft(cfs_df_ohe, num_bins=num_bins, sigma=sigma)
        q = self._phi_soft(perturbed_cfs_df_ohe, num_bins=num_bins, sigma=sigma)

        dot_products = np.sum(p * q, axis=1)
        num_features = len(self.data_interface.continuous_feature_names) + \
                    len(self.data_interface.categorical_feature_names)
        proximity_kernel = dot_products / num_features
        return proximity_kernel

    def compute_robustness_distance_RBF(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame,
                                    sigma: float | np.ndarray | None=None, gamma: float | None=None,
                                    separate_features: bool=False, atol_ohe: float=1e-6) -> float:

        cfs_df_ohe = self.data_interface.get_ohe_min_max_normalized_data(self.label_decode_cfs(cfs))
        perturbed_cfs_df_ohe = self.data_interface.get_ohe_min_max_normalized_data(perturbed_cfs)

        if not isinstance(cfs_df_ohe, pd.DataFrame):
            cfs_df_ohe = pd.DataFrame(cfs_df_ohe)
        if not isinstance(perturbed_cfs, pd.DataFrame):
            perturbed_cfs_df_ohe = pd.DataFrame(perturbed_cfs, columns=cfs_df_ohe.columns)

        perturbed_cfs_df_ohe = perturbed_cfs_df_ohe[cfs_df_ohe.columns]
        assert len(cfs_df_ohe) == len(perturbed_cfs_df_ohe), "CFs and perturbed CFs must have same number of rows"

        if not separate_features:
            X = cfs_df_ohe.values.astype(np.float32)
            Y = perturbed_cfs_df_ohe.values.astype(np.float32)
            diff = X - Y

            if sigma is None:
                stacked = np.vstack([X, Y])
                sigma = np.std(stacked, axis=0).astype(np.float32) + 1e-8

            if sigma.ndim == 0:
                squared_dist = np.sum(diff ** 2, axis=1)
                kernel_sim = np.exp(-squared_dist / (2.0 * float(sigma) ** 2))
            else:
                sigma = np.asarray(sigma, dtype=np.float32)
                squared_dist = np.sum((diff ** 2) / (2.0 * (sigma ** 2)), axis=1)
                kernel_sim = np.exp(-squared_dist)

            return kernel_sim

        cont_cols = list(self.data_interface.continuous_feature_names)
        cat_cols = list(cfs_df_ohe.columns.difference(cont_cols))

        X_cont = cfs_df_ohe[cont_cols].values.astype(np.float32) if cont_cols else None
        Y_cont = perturbed_cfs_df_ohe[cont_cols].values.astype(np.float32) if cont_cols else None

        X_cat = cfs_df_ohe[cat_cols].values.astype(np.float32) if cat_cols else None
        Y_cat = perturbed_cfs_df_ohe[cat_cols].values.astype(np.float32) if cat_cols else None

        if cont_cols:
            cont_diff = X_cont - Y_cont

            if sigma is None:
                stacked_conts = np.vstack([X_cont, Y_cont])
                sigma_conts = np.std(stacked_conts, axis=0).astype(np.float32) + 1e-8
            else:
                sigma_conts = sigma

            if sigma_conts.ndim == 0:
                squared_dist_cont = np.sum(cont_diff ** 2, axis=1)
                kernel_sim_cont = np.exp(-squared_dist_cont / (2.0 * float(sigma_conts) ** 2))
            else:
                sigma_conts = np.asarray(sigma_conts, dtype=np.float32)
                squared_dist_cont = np.sum((cont_diff ** 2) / (2.0 * sigma_conts ** 2), axis=1)
                kernel_sim_cont = np.exp(-squared_dist_cont)
        else:
            kernel_sim_cont = np.ones(len(cfs_df_ohe), dtype=np.float32)

        if cat_cols:
            not_equal = ~np.isclose(X_cat, Y_cat, atol=atol_ohe)
            hamming_dist = np.mean(not_equal, axis=1).astype(np.float32)

            if gamma is None:
                gamma = 0.2
            gamma = float(max(gamma, 1e-6))

            kernel_sim_cat = np.exp(-hamming_dist / gamma)
        else:
            kernel_sim_cat = np.ones(len(cfs_df_ohe), dtype=np.float32)

        similarity = kernel_sim_cont * kernel_sim_cat

        return similarity

    def compute_robustness_loss_RBF(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame,
                                    sigma: float | np.ndarray | None=None, gamma: float | None=None,
                                    separate_features: bool=False, atol_ohe: float=1e-6,
                                    desired_class="opposite") -> float:
        query_instance = self.x1.copy()
        base_scores = self.predict_fn_scores(query_instance)
        base_pred = int(np.argmax(base_scores, axis=1)[0])
        if desired_class == "opposite":
            target_pred = 1 - base_pred
        else:
            target_pred = desired_class

        dist = self.compute_robustness_distance_RBF(cfs, perturbed_cfs, sigma=sigma, gamma=gamma,
                                                    separate_features=separate_features, atol_ohe=atol_ohe)
        perturbed_scores = self.predict_fn_scores(perturbed_cfs)
        perturbed_preds = np.argmax(perturbed_scores, axis=1)
        indicator_var = (perturbed_preds == target_pred).astype(np.float32)
        return float(np.mean(dist * indicator_var))

    """ def compute_robustness_loss_RBF_with_bin(self, cfs: df.DataFrame, perturbed_cfs: pd.DataFrame,
                                             sigma: float=0.1, num_bins: int=10) -> float: """

    def compute_robustness_loss(self, cfs: pd.DataFrame, perturbed_cfs: pd.DataFrame, preprocessing_bins: int=10) -> float:
        """
        Computes the robustness loss using Dice-Sørensen coefficient. Adopted from 
        https://doi.org/10.48550/arXiv.2407.00843

        Args:
            cfs (pandas.DataFrame): Generated counterfactual instances.
            perturbed_cfs (pandas.DataFrame): Perturbed counterfactual instances as pandas.DataFrame.

        Returns:
            float: Computed loss for robustness
        """


        """ if not isinstance(cfs, pd.DataFrame) or not isinstance(perturbed_cfs, pd.DataFrame):
            raise ValueError(f"Both `cfs` and `perturbed_cfs` must be of type pandas.DataFrame")

        if len(cfs) != len(perturbed_cfs):
            raise ValueError(f"The number of rows in `cfs` doesn't match with the number of rows in `perturbed_cfs`") """

        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(self.label_decode_cfs(cfs), perturbed_cfs, num_bins=preprocessing_bins)

        intersection = np.sum(np.minimum(cfs_processed, perturbed_cfs_processed), axis=1)
        union = np.sum(cfs_processed, axis=1) + np.sum(perturbed_cfs_processed, axis=1)

        sorensen_dice_coefficient = (2 * intersection) / union
        sorensen_dice_coefficient[np.isnan(sorensen_dice_coefficient)] = 1.0

        return sorensen_dice_coefficient

    def compute_yloss(self, cfs, desired_range, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.model.model_type == ModelTypes.Classifier:
            predicted_value = np.array(self.predict_fn_scores(cfs))
            if self.yloss_type == 'hinge_loss':
                maxvalue = np.full((len(predicted_value)), -np.inf)
                for c in range(self.num_output_nodes):
                    if c != desired_class:
                        maxvalue = np.maximum(maxvalue, predicted_value[:, c])
                yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])
            return yloss

        elif self.model.model_type == ModelTypes.Regressor:
            predicted_value = self.predict_fn(cfs)
            if self.yloss_type == 'hinge_loss':
                yloss = np.zeros(len(predicted_value))
                for i in range(len(predicted_value)):
                    if not desired_range[0] <= predicted_value[i] <= desired_range[1]:
                        yloss[i] = min(abs(predicted_value[i] - desired_range[0]),
                                       abs(predicted_value[i] - desired_range[1]))
            return yloss

    def compute_proximity_loss(self, x_hat_unnormalized, query_instance_normalized):
        """Compute weighted distance between two vectors."""
        x_hat = self.data_interface.normalize_data(x_hat_unnormalized)
        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in self.data_interface.continuous_feature_indexes])
        product = np.multiply(
            (abs(x_hat - query_instance_normalized)[:, [self.data_interface.continuous_feature_indexes]]),
            feature_weights)
        product = product.reshape(-1, product.shape[-1])
        proximity_loss = np.sum(product, axis=1)

        # Dividing by the sum of feature weights to normalize proximity loss
        return proximity_loss / sum(feature_weights)

    def compute_sparsity_loss(self, cfs):
        """Compute weighted distance between two vectors."""
        sparsity_loss = np.count_nonzero(cfs - self.x1, axis=1)
        return sparsity_loss / len(
            self.data_interface.feature_names)  # Dividing by the number of features to normalize sparsity loss

    def generate_perturbations_simple(
        self,
        input_df: pd.DataFrame,
        method: str = "gaussian",
        std_dev: float = 0.20,
        max_radius: float = 0.8,
        cat_flip_prob: float = 0.25
    ) -> pd.DataFrame:
        """
        Simple, direct perturbation generator for robustness evaluation.
        Always returns perturbed versions (never falls back to originals).
        """
        rng = np.random.default_rng()

        cont_cols = self.data_interface.continuous_feature_names
        cat_cols = self.data_interface.categorical_feature_names
        ranges = self.data_interface.get_features_range_float()[1]

        perturbed_df = input_df.copy()

        # Perturb continuous features
        if cont_cols:
            cont_data = perturbed_df[cont_cols].to_numpy()
            cont_lo = np.array([ranges[c][0] for c in cont_cols])
            cont_hi = np.array([ranges[c][1] for c in cont_cols])
            span = cont_hi - cont_lo

            if method == "gaussian":
                noise = rng.normal(scale=std_dev * span, size=cont_data.shape)
            elif method == "random":
                noise = rng.uniform(-max_radius * span, max_radius * span, size=cont_data.shape)
            elif method == "spherical":
                direction = rng.normal(size=cont_data.shape)
                norms = np.linalg.norm(direction, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
                direction /= norms
                # Generate radius for each sample, matching span dimensions
                radius = rng.uniform(0, 1, size=(len(direction), 1)) * max_radius * span
                noise = direction * radius
            else:
                noise = rng.normal(scale=std_dev * span, size=cont_data.shape)

            perturbed_cont = np.clip(cont_data + noise, cont_lo, cont_hi)
            perturbed_df[cont_cols] = perturbed_cont

        # Perturb categorical features
        if cat_cols:
            cat_ranges = self.data_interface.get_features_range()[1]
            for col in cat_cols:
                flip_mask = rng.random(len(perturbed_df)) < cat_flip_prob
                n_flips = flip_mask.sum()
                if n_flips > 0:
                    choices = np.array(cat_ranges[col])
                    new_values = rng.choice(choices, size=n_flips)
                    perturbed_df.loc[flip_mask, col] = new_values

        return perturbed_df

    def compute_loss(self, cfs, desired_range, desired_class, perturbation_method: str,
                     preprocessing_bins=10, robustness_type=RobustnessType.DICE_SORENSEN,
                     separate_features: bool | None=None, **kwargs):
        """Computes the overall loss"""
        if robustness_type != RobustnessType.GAUSSIAN_KERNEL and separate_features is not None:
            raise ValueError("`separate_features` can only be used when `robustness_type == RobustnessType.GAUSSIAN_KERNEL`.")
        input_instance = self.label_decode(cfs)
        perturbed_cfs = self.perturb_cfs(input_instance)

        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)
        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 else np.zeros(len(cfs))
        self.sparsity_loss = self.compute_sparsity_loss(cfs) if self.sparsity_weight > 0 else np.zeros(len(cfs))

        if robustness_type == RobustnessType.DICE_SORENSEN:
            self.robustness_loss = self.compute_robustness_loss_(cfs, perturbed_cfs, preprocessing_bins=preprocessing_bins) \
                                                                 if self.robustness_weight > 0 else np.zeros(len(cfs))
        elif robustness_type == RobustnessType.GAUSSIAN_KERNEL:
            self.robustness_loss = self.compute_robustness_loss_RBF(cfs, perturbed_cfs,
                                                                    separate_features=separate_features,
                                                                    desired_class=desired_class) \
                                                                    if self.robustness_weight > 0 else np.zeros(len(cfs))
        elif robustness_type == RobustnessType.BINNED_GAUSSIAN_KERNEL:
            self.robustness_loss = self.compute_robustness_distance_binned_RBF(cfs, perturbed_cfs) if self.robustness_weight > 0 else np.zeros(len(cfs))
        else:
            raise ValueError("Three types of robustness are supported: Dice-Sorensen, Gaussian Kernel, and Binned Gaussian Kernel")
        self.loss = np.reshape(np.array(self.yloss + (self.proximity_weight * self.proximity_loss) +
                                        self.sparsity_weight * self.sparsity_loss +
                                        self.robustness_weight * self.robustness_loss), (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    def mate(self, k1, k2, features_to_vary, query_instance):
        """Performs mating and produces new offsprings"""
        # chromosome for offspring
        one_init = np.zeros(self.data_interface.number_of_features)
        for j in range(self.data_interface.number_of_features):
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.data_interface.feature_names[j]

            # random probability
            prob = random.random()

            if prob < 0.40:
                # if prob is less than 0.40, insert gene from parent 1
                one_init[j] = gp1
            elif prob < 0.80:
                # if prob is between 0.40 and 0.80, insert gene from parent 2
                one_init[j] = gp2
            else:
                # otherwise insert random gene(mutate) for maintaining diversity
                if feat_name in features_to_vary:
                    if feat_name in self.data_interface.continuous_feature_names:
                        one_init[j] = np.random.uniform(self.feature_range[feat_name][0],
                                                        self.feature_range[feat_name][1])
                    else:
                        one_init[j] = np.random.choice(self.feature_range[feat_name])
                else:
                    one_init[j] = query_instance[j]
        return one_init

    def _reset_loss_history(self):
        self.loss_history = {key: [] for key in self.loss_history}

    def _populate_loss_history(self, it, y_loss, sparsity_loss,
                               proximity_loss, robustness_loss, total_loss):
        self.loss_history["iterations"].append(it)
        self.loss_history["y_loss"].append(y_loss.mean())
        self.loss_history["sparsity_loss"].append(sparsity_loss.mean())
        self.loss_history["proximity_loss"].append(proximity_loss.mean())
        self.loss_history["robustness_loss"].append(robustness_loss if type(self.robustness_loss) == float else robustness_loss.mean())
        self.loss_history["total_loss"].append(total_loss.mean())

    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             perturbation_method, features_to_vary, maxiterations,
                             thresh, verbose, preprocessing_bins, robustness_type,
                             separate_features: bool | None=None, **kwargs):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""

        self._reset_loss_history()
        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None

        self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        self.query_instance_normalized = self.query_instance_normalized.astype('float')

        while iterations < maxiterations and self.total_CFs > 0:
            if abs(previous_best_loss - current_best_loss) <= thresh and \
                    (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                     (self.model.model_type == ModelTypes.Regressor and
                      all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                stop_cnt += 1

            else:
                stop_cnt = 0

            if stop_cnt >= 5:
                break
            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)
            population_fitness = self.compute_loss(population, desired_range, desired_class,
                                                   perturbation_method, preprocessing_bins, robustness_type,
                                                   separate_features=separate_features, **kwargs)

            population_fitness = population_fitness[population_fitness[:, 1].argsort()]
            current_best_loss = population_fitness[0][1]
            self._populate_loss_history(
                iterations, self.yloss, self.sparsity_loss,
                self.proximity_loss, self.robustness_loss, current_best_loss)

            to_pred = np.array([population[int(tup[0])]
                                for tup in population_fitness[:self.total_CFs]])

            if self.total_CFs > 0:
                if self.model.model_type == ModelTypes.Classifier:
                    cfs_preds = self._predict_fn_custom(to_pred, desired_class)
                else:
                    cfs_preds = self.predict_fn(to_pred)

            # self.total_CFS of the next generation obtained from the fittest members of current generation
            top_members = self.total_CFs
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

            # rest of the next generation obtained from top 50% of fittest members of current generation
            rest_members = self.population_size - top_members
            new_generation_2 = None
            if rest_members > 0:
                new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
                for new_gen_idx in range(rest_members):
                    parent1 = random.choice(population[:int(len(population) / 2)])
                    parent2 = random.choice(population[:int(len(population) / 2)])
                    child = self.mate(parent1, parent2, features_to_vary, query_instance)
                    new_generation_2[new_gen_idx] = child

            if new_generation_2 is not None:
                if self.total_CFs > 0:
                    population = np.concatenate([new_generation_1, new_generation_2])
                else:
                    population = new_generation_2
            else:
                raise SystemError("The number of total_Cfs is greater than the population size!")
            iterations += 1

        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        while i < self.total_CFs:

            predictions = self.predict_fn_scores(population[i])[0]

            if self.is_cf_valid(predictions):
                self.final_cfs.append(population[i])
                # checking if predictions is a float before taking the length as len() works only for array-like
                # elements. isinstance(predictions, (np.floating, float)) checks if it's any float (numpy or otherwise)
                # We do this as we take the argmax if the prediction is a vector -- like the output of a classifier
                if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                    self.cfs_preds.append(np.argmax(predictions))
                else:
                    self.cfs_preds.append(predictions)
            i += 1

        # converting to dataframe
        query_instance_df = self.label_decode(query_instance)
        query_instance_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.test_pred)
        self.final_cfs_df = self.label_decode_cfs(self.final_cfs)
        self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
            self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds
            self.round_to_precision()

        # decoding to original label
        query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse = \
            self.decode_to_original_labels(query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse)

        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            if len(self.final_cfs) == self.total_CFs:
                print('Diverse Counterfactuals found! total time taken: %02d' %
                      m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                      'Diverse Counterfactuals found for the given configuation, perhaps ',
                      'change the query instance or the features to vary...'  '; total time taken: %02d' % m,
                      'min %02d' % s, 'sec')
        return query_instance_df.reset_index(drop=True)

    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values
        """
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}

            for i in range(len(labelled_input[j])):
                if self.data_interface.feature_names[i] in self.data_interface.categorical_feature_names:
                    enc = self.labelencoder[self.data_interface.feature_names[i]]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[self.data_interface.feature_names[i]] = val[0]
                else:
                    temp[self.data_interface.feature_names[i]] = labelled_input[j][i]
            input_instance.append(temp)
        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        return input_instance_df

    def label_decode_cfs(self, cfs_arr):
        ret_df = None
        if cfs_arr is None:
            return None
        for cf in cfs_arr:
            df = self.label_decode(cf)
            if ret_df is None:
                ret_df = df
            else:
                ret_df = pd.concat([ret_df, df])
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
        return ret
