"""
Module to generate diverse counterfactual explanations based on tensorflow 2.x
"""
import copy
import random
import timeit
# To suppress TensorFlow warning about the optimizer running slowly on Apple chips.
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import numpy as np
import pandas as pd
import tensorflow as tf

from dice_ml_x import diverse_counterfactuals as exp
from dice_ml_x.counterfactual_explanations import CounterfactualExplanations
from dice_ml_x.explainer_interfaces.explainer_base import ExplainerBase


class DiceTensorFlow2Vectorized(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method
        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """
        # initiating data related parameters
        super().__init__(data_interface)
        # initializing model related variables
        self.model = model_interface
        self.model.load_model()  # loading trained model
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()
        # temp data to create some attributes like encoded feature names
        if hasattr(self.data_interface, "data_df"):
            temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        else:
            temp_ohe_data = None
        self.data_interface.create_ohe_params(temp_ohe_data)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()

        # number of output nodes of ML model
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]

        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs_var = None
        self.num_cfs = 0
        self.total_random_inits = 0

        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.optimizer = None
        self.loss_history = {
            "iterations": [],
            "y_loss": [],
            "proximity_loss": [],
            "diversity_loss": [],
            "robustness_loss": [],
            "total_loss": []
        }

        self.target_cf_class = None
        self.x1 = None
        self.stopping_threshold = 0.5
        self.converged = False

    def generate_counterfactuals(self, query_instances, total_CFs, desired_class="opposite", proximity_weight=0.5,
                                 diversity_weight=1.0, robustness_weight=0.5, categorical_penalty=0.1, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad",
                                 optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000,
                                 project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                 init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                 posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000,
                                 perturbation_method="gaussian", **kwargs):
        """Generates diverse counterfactual explanations

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the
                              outcome class of query_instance for binary classification.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to
                                 the query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical
                                    variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized
                               in data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding
                                weights as values. Default option is "inverse_mad" where the weight for a continuous feature
                                is the inverse of the Median Absolute Devidation (MAD) of the feature's values in the training
                                set; the weight for a categorical feature is equal to 1 by default.
        :param optimizer: Tensorflow optimization algorithm. Currently tested only with "tensorflow:adam".

        :param learning_rate: Learning rate for optimizer.
        :param min_iter: Min iterations to run gradient descent for.
        :param max_iter: Max iterations to run gradient descent for.
        :param project_iter: Project the gradients at an interval of these many iterations.
        :param loss_diff_thres: Minimum difference between successive loss values to check convergence.
        :param loss_converge_maxiter: Maximum number of iterations for loss_diff_thres to hold to declare convergence.
                                      Defaults to 1, but we assigned a more conservative value of 2 in the paper.
        :param verbose: Print intermediate loss value.
        :param init_near_query_instance: Boolean to indicate if counterfactuals are to be initialized near query_instance.
        :param tie_random: Used in rounding off CFs and intermediate projection.
        :param stopping_threshold: Minimum threshold for counterfactuals target class probability.
        :param posthoc_sparsity_param: Parameter for the post-hoc operation on continuous features to enhance sparsity.
        :param posthoc_sparsity_algorithm: Perform either linear or binary search. Takes "linear" or "binary".
                                           Prefer binary search when a feature range is large
                                           (for instance, income varying from 10k to 1000k) and only if the features
                                           share a monotonic relationship with predicted outcome in the model.
        :param limit_steps_ls: Defines an upper limit for the linear search step in the posthoc_sparsity_enhancement

        :return: A CounterfactualExamples object to store and visualize the resulting counterfactual explanations
                (see diverse_counterfactuals.py).

        """
        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            # if not self.data_interface.check_features_range(permitted_range):
            #     raise ValueError(
            #         "permitted range of features should be within their original range")
            # else:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        # if([total_CFs, algorithm, features_to_vary] != self.cf_init_weights):
        self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [proximity_weight, diversity_weight, robustness_weight, categorical_penalty] != self.hyperparameters:
            self.update_hyperparameters(proximity_weight, diversity_weight, robustness_weight, categorical_penalty)

        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(query_instances, desired_class, optimizer,
                                      learning_rate, min_iter, max_iter, project_iter,
                                      loss_diff_thres, loss_converge_maxiter, verbose,
                                      init_near_query_instance, tie_random, stopping_threshold,
                                      posthoc_sparsity_param, posthoc_sparsity_algorithm, limit_steps_ls,
                                      perturbation_method, **kwargs)

        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class)

        return CounterfactualExplanations(cf_examples_list=[counterfactual_explanations])

    def predict_fn(self, input_instance: tf.Tensor) -> tf.Tensor:
        """prediction function"""
        preds = self.model.get_output(input_instance)
        preds = preds[:, (self.num_output_nodes - 1):]
        return preds
    
    def predict_fn_with_grads(self, input_instance):
        return self.model.model(input_instance, training=False)

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance = self.model.transformer.transform(input_instance).to_numpy()
        return self.predict_fn(tf.constant(input_instance, dtype=tf.float32))

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            self.total_random_inits = total_CFs
            self.num_cfs = 1
        else:
            self.total_random_inits = 0
            self.num_cfs = total_CFs
        
        d = self.minx.shape[1]

        init = np.zeros((self.num_cfs, d), dtype=np.float32)
        self.cfs_var = tf.Variable(init, dtype=tf.float32)

        idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
        mask = np.zeros(d, dtype=np.float32)
        for ix in idxs:
            mask[ix] = 1.0

        self.freezer_mask = tf.constant(mask, shape=(d, ), dtype=tf.float32)

    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights):
        """Intializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]

        # define the loss parts
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        # define feature weights
        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=True)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1/normalized_mads[feature], 2)

            feature_weights_list = []
            for feature in self.data_interface.ohe_encoded_feature_names:
                if feature in feature_weights:
                    feature_weights_list.append(feature_weights[feature])
                else:
                    feature_weights_list.append(1.0)
            self.feature_weights_list = tf.constant([feature_weights_list], dtype=tf.float32)

    def update_hyperparameters(self, proximity_weight, diversity_weight, robustness_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.robustness_weight = robustness_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based TensorFLow optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)


    def do_perturbation(self):
        cfs_noised = self.cfs_var
        if self.encoded_continuous_feature_indexes:
            continuous_slice = tf.gather(cfs_noised, self.encoded_continuous_feature_indexes, axis=1)
            noise = continuous_slice * 0.3
            noise_mask = tf.zeros_like(cfs_noised)

            k = tf.shape(cfs_noised)[0]
            cont_idx = tf.constant(self.encoded_continuous_feature_indexes, dtype=tf.int32)

            I, J = tf.meshgrid(
                tf.range(k, dtype=tf.int32),
                cont_idx,
                indexing='ij'
            )

            I_flat = tf.reshape(I, [-1])
            J_flat = tf.reshape(J, [-1])
            indices = tf.stack([I_flat, J_flat], axis=1)
            updates = tf.reshape(noise, [-1])
            noise_mask = tf.tensor_scatter_nd_update(noise_mask, indices, updates)
            cfs_noised = cfs_noised + noise_mask

        if self.encoded_categorical_feature_indexes:
            for cat_cols in self.encoded_categorical_feature_indexes:
                cat_slice = tf.gather(cfs_noised, cat_cols, axis=1)
                sample_size = tf.shape(cat_slice)[0]
                num_cats = tf.shape(cat_slice)[1]

                rand_idx = tf.random.uniform(shape=(sample_size, ), minval=0, maxval=num_cats, dtype=tf.int32)
                cat_slice_perturbed = tf.one_hot(rand_idx, depth=num_cats, dtype=tf.float64)

                cat_mask = tf.zeros_like(cfs_noised)
                k = tf.shape(cfs_noised)[0]
                cat_cols_tf = tf.constant(cat_cols, dtype=tf.int32)
                I, J = tf.meshgrid(
                    tf.range(k, dtype=tf.int32),
                    cat_cols_tf,
                    indexing='ij'
                )
                I_flat = tf.reshape(I, [-1])  
                J_flat = tf.reshape(J, [-1])
                indices = tf.stack([I_flat, J_flat], axis=1)

                updates = tf.reshape(cat_slice_perturbed, [-1])
                updates = tf.cast(updates, dtype=tf.float32)
                cat_mask = tf.tensor_scatter_nd_update(cat_mask, indices, updates)
                cfs_noised = cfs_noised + cat_mask
        cfs_perturbed = tf.Variable(tf.identity(cfs_noised), trainable=True)
        
        return cfs_perturbed

    
    def optimize_perturbations(self, c_i_prime: tf.Tensor, c_i: tf.Tensor, gamma):
       #print(f"c_i_prime_numerical shape: {c_i_prime_numerical.shape}")
        with tf.GradientTape() as tape:
            tape.watch(c_i_prime)
            c_i = tf.reshape(c_i, [-1, c_i.shape[-1]])
            c_i_prime = tf.reshape(c_i_prime, [-1, c_i_prime.shape[-1]])
            pred_i = self.predict_fn_with_grads(c_i)
            pred_i_prime = self.predict_fn_with_grads(c_i_prime)
            class_loss = tf.reduce_mean((pred_i - pred_i_prime) ** 2)
            distance = tf.norm(c_i_prime - c_i, ord=2)
            loss = class_loss - gamma * distance
        grads = tape.gradient(loss, [c_i_prime])
        return loss, grads
    
    
    def _perturbation_step(self, c_i_prime, c_i, gamma, optimizer):
        with tf.GradientTape() as tape:
            loss, grads = self.optimize_perturbations(c_i_prime, c_i, gamma)
        optimizer.apply_gradients(zip(grads, [c_i_prime]))
        return loss
    
    def generate_perturbations_vectorized(self, max_iter=100, tol=1e-3, gamma=1e-2, **kwargs):
        
        c_i_prime = self.do_perturbation()
        c_i = self.cfs_var
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-2)
        prev_loss = np.inf
        for it in range(max_iter):
            loss_tensor = self._perturbation_step(c_i_prime, c_i, gamma, optimizer)
            
            loss_val = float(loss_tensor.numpy())
            
            if abs(loss_val - prev_loss) < tol: 
                
                break
            prev_loss = loss_val
        return c_i_prime
    
    def _preprocess_for_robustness(self, cfs: tf.Tensor, perturbed_cfs: tf.Tensor) -> tuple:
        
        def preprocess_continuous_features(cfs: tf.Tensor, num_bins=10) -> tf.Tensor:
            edges = tf.linspace(0.0, 1.0, num=num_bins + 1)
            all_one_hots = []

            for cont_idx in self.encoded_continuous_feature_indexes:
                col_vals = cfs[:, cont_idx]
                binned_indices = tf.searchsorted(edges, col_vals, side='left')
                binned_indices = tf.clip_by_value(binned_indices, 0, num_bins - 1)

                one_hot_col = tf.one_hot(binned_indices, depth=num_bins, dtype=tf.float64)
                all_one_hots.append(one_hot_col)
            return tf.concat(all_one_hots, axis=1)
        
        cat_cols = [col for group in self.encoded_categorical_feature_indexes for col in group]
        cat_cols = tf.constant(cat_cols, dtype=tf.int32)
          
        cfs_cats = tf.cast(tf.gather(cfs, cat_cols, axis=1), dtype=tf.float32)
        perturbed_cfs_cats = tf.cast(tf.gather(perturbed_cfs, cat_cols, axis=1), dtype=tf.float32)

        cfs_one_hot = preprocess_continuous_features(cfs)
        perturbed_cfs_one_hot = preprocess_continuous_features(perturbed_cfs)

        cfs_one_hot = tf.cast(cfs_one_hot, dtype=tf.float32)
        perturbed_cfs_one_hot = tf.cast(perturbed_cfs_one_hot, dtype=tf.float32)

        cfs_processed = tf.concat([cfs_one_hot, cfs_cats], axis=1)
        perturbed_cfs_processed = tf.concat([perturbed_cfs_one_hot, perturbed_cfs_cats], axis=1)

        return cfs_processed, perturbed_cfs_processed

    def compute_robustness_loss(self, perturbed_cfs: tf.Tensor) -> float:
        """
        Computes the robustness loss.
        Args:
            perturbed_cfs_df (pandas.DataFrame): The perturbed counterfactuals that will
            be compared against the original counterfactual instances.
        Returns:
            float: Robustness loss in scalar.
        """
        
        cfs = self.cfs_var
        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs)

        intersection = tf.reduce_sum(tf.minimum(cfs_processed, perturbed_cfs_processed), axis=1)
        intersection = np.sum(np.minimum(cfs_processed, perturbed_cfs_processed), axis=1)
        union = tf.reduce_sum(cfs_processed, axis=1) + tf.reduce_sum(perturbed_cfs_processed, axis=1)

        epsilon = 1e-8
        sorensen_dice_coefficient = (2 * intersection) / (union + epsilon)
        sorensen_dice_coefficient = tf.where(
            tf.math.is_nan(sorensen_dice_coefficient), 
            tf.ones_like(sorensen_dice_coefficient), 
            sorensen_dice_coefficient
        )
        return tf.cast(tf.reduce_mean(sorensen_dice_coefficient), dtype=tf.float32)
    

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.yloss_type == "l2_loss":
            preds = self.predict_fn(self.cfs_var)
            target = self.target_cf_class
            t = tf.ones_like(preds) * target
            y_loss = tf.reduce_mean(tf.pow(preds - t, 2))
        elif self.yloss_type == "log_loss":
            temp_logits = tf.math.log((tf.abs(self.predict_fn(self.cfs_var - 0.000001))) / (1 - tf.abs(self.predict_fn(self.cfs_var) - 0.000001)))
            preds_shape = tf.shape(temp_logits)
            labels = tf.ones(preds_shape, dtype=tf.float32) * self.target_cf_class
            y_loss_batched = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=temp_logits, labels=labels
            )
            y_loss = tf.reduce_mean(y_loss_batched)
        elif self.yloss_type == "hinge_loss":
            temp_logits = tf.math.log((tf.abs(self.predict_fn(self.cfs_var - 0.000001))) / (1 - tf.abs(self.predict_fn(self.cfs_var) - 0.000001))) 
            preds_shape = tf.shape(temp_logits)
            labels = tf.ones(preds_shape, dtype=tf.float32) * self.target_cf_class
            hinge_batch = tf.compat.v1.losses.hinge_loss(
                logits=temp_logits,
                labels=labels,
                reduction=tf.compat.v1.losses.Reduction.NONE
            )
            y_loss = tf.reduce_mean(hinge_batch)
            
        return y_loss.numpy()

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply((tf.abs(x_hat - x1)), self.feature_weights_list))

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        diff = tf.abs(self.cfs_var - self.x1)

        weighted_diff = diff * self.feature_weights_list
        row_sums = tf.reduce_sum(weighted_diff, axis=1)
        return tf.reduce_mean(row_sums)

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""

        cfs = self.cfs_var
        k = tf.shape(cfs)[0]

        if tf.equal(k, 1):
            return 0.0
        
        c_exp = tf.expand_dims(cfs, axis=1)
        c_exp2 = tf.expand_dims(cfs, axis=0)

        dist_matrix = tf.reduce_sum(tf.abs(c_exp, c_exp2) * self.feature_weights_list, axis=2)


        if submethod == "inverse_dist":
            M = 1.0 / (1.0 + dist_matrix)

            diag_eps = tf.eye(k, dtype=M.dtype) * 0.0001
            M = M + diag_eps

        elif submethod == "exponential_dist":
            M = 1.0 / tf.exp(dist_matrix)
            diag_eps = tf.eye(k, dtype=M.dtype) * 0.0001
            M = M + diag_eps
        else:
            raise ValueError(f"Unsupported submethod for DPP: {submethod}")
        
        diversity_loss = tf.linalg.det(M)
        return diversity_loss.numpy()

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        k = tf.shape(self.cfs_var)[0]
        if k == 1:
            return 0.0
        
        c_exp = tf.expand_dims(self.cfs_var, axis=1)
        c_exp2 = tf.expand_dims(self.cfs_var, axis=0)
        dist = tf.reduce_sum(tf.abs(c_exp - c_exp2) * self.feature_weights_list, axis=2)
        M = 1.0 / (1.0 + dist)

        diag_eps = tf.eye(k, dtype=M.dtype) * 0.0001
        M = M + diag_eps

        return tf.linalg.det(M).numpy()

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels
           of a categorical variable sums to one"""
        reg_sum = tf.constant(0.0, dtype=tf.float32)
        k = tf.shape(self.cfs_var)[0]
        for cat_cols in self.encoded_categorical_feature_indexes:
            start, end = cat_cols[0], cat_cols[-1] + 1
            cat_sum = tf.reduce_sum(self.cfs_var[:, start:end], axis=1)
            reg_sum += tf.reduce_sum(tf.pow(cat_sum - 1.0, 2))

        return reg_sum / tf.cast(k, tf.float32)

    def compute_loss_vectorized(self, **kwargs):
        """Computes the overall loss"""
        perturbed_cfs = self.generate_perturbations_vectorized(**kwargs)
        
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        self.robustness_loss = self.compute_robustness_loss(perturbed_cfs=perturbed_cfs) if self.robustness_weight > 0 else 0.0
        
        self.regularization_loss = self.compute_regularization_loss()
        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - \
            (self.diversity_weight * self.diversity_loss) - \
            (self.robustness_weight * self.robustness_loss) + \
            (self.categorical_penalty * self.regularization_loss)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            one_init = []
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        one_init.append(query_instance[0][i]+(n*0.01))
                    else:
                        one_init.append(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    one_init.append(query_instance[0][i])
            one_init = np.array([one_init], dtype=np.float32)
            self.cfs[n].assign(one_init)

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        cfs_np = self.cfs_var.numpy()
        k, d = cfs_np.shape
        for idx_i, v in enumerate(self.encoded_continuous_feature_indexes):
            org_cont = cfs_np[:, v] * (self.cont_maxx[idx_i] - self.cont_minx[idx_i]) +self.cont_minx[idx_i]
            org_cont = np.round(org_cont, self.cont_precisions[idx_i])
            normalized = (org_cont - self.cont_minx[idx_i]) / (self.cont_maxx[idx_i] - self.cont_minx[idx_i])
            cfs_np[:, v] = normalized

        for group in self.encoded_categorical_feature_indexes:

            cat_vals = cfs_np[:, group[0]:group[-1] + 1]
            max_idx = np.argmax(cat_vals, axis=1)

            for row_i in range(k):
                cfs_np[row_i, group[0]:group[-1] + 1] = 0.0
                cfs_np[row_i, group[0] + max_idx[row_i]] = 1.0

        if assign:
            self.cfs_var.assign(cfs_np)
            return None
        else:
            return cfs_np

    def stop_loop(self, itr, loss_diff):
        """Determines the stopping condition for gradient descent."""

        # intermediate projections
        if self.project_iter > 0 and itr > 0:
            if itr % self.project_iter == 0:
                self.round_off_cfs(assign=True)

        # do GD for min iterations
        if itr < self.min_iter:
            return False

        # stop GD if max iter is reached
        if itr >= self.max_iter:
            return True
        
        # else stop when loss diff is small & all CFs are valid (less or greater than a stopping threshold)
        if loss_diff <= self.loss_diff_thres:
            self.loss_converge_iter += 1
            if self.loss_converge_iter < self.loss_converge_maxiter:
                return False
            else:
                temp_cfs = self.round_off_cfs(assign=False)
                test_preds = [self.predict_fn(tf.constant(cf, dtype=tf.float32))[0] for cf in temp_cfs]

                if self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                elif self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds):
                    self.converged = True
                    return True
                else:
                    return False
        else:
            self.loss_converge_iter = 0
            return False
        

    def _reset_loss_history(self):
        self.loss_history = {key: [] for key in self.loss_history}

    def _populate_loss_history(self, it, y_loss, proximity_loss, diversity_loss, robustness_loss, total_loss):
        self.loss_history["iterations"].append(it)
        self.loss_history["y_loss"].append(y_loss)
        self.loss_history["diversity_loss"].append(diversity_loss)
        self.loss_history["proximity_loss"].append(proximity_loss)
        self.loss_history["robustness_loss"].append(robustness_loss)
        self.loss_history["total_loss"].append(total_loss)

    def find_counterfactuals(self, query_instance, desired_class, optimizer, learning_rate, min_iter,
                             max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
                             init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
                             posthoc_sparsity_algorithm, limit_steps_ls, perturbation_method: str, **kwargs):
        """Finds counterfactuals by gradient-descent."""
        self._reset_loss_history()
        
        query_instance = self.model.transformer.transform(query_instance).to_numpy()
        self.x1 = tf.constant(query_instance, dtype=tf.float32)

        # find the predicted value of query_instance
        test_pred = self.predict_fn(self.x1)[0][0].numpy()
        if desired_class == "opposite":
            desired_class = 1.0 - round(test_pred)
        self.target_cf_class = float(desired_class)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        # no. of iterations to wait to confirm that loss has converged
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75

        # to resolve tie - if multiple levels of an one-hot-encoded categorical variable take value 1
        self.tie_random = tie_random
        self.do_optimizer_initializations(optimizer, learning_rate)
        # running optimization steps
        start_time = timeit.default_timer()
        final_cfs_list = []
        final_preds_list = []

        if self.total_random_inits > 0:
            for run_ix in range(self.total_random_inits):
                self._initialize_single_run(query_instance, init_near_query_instance)
                run_cfs, run_preds = self._run_vectorized_optimization(
                    min_iter, max_iter, loss_diff_thres, loss_converge_maxiter,
                    verbose, **kwargs
                )
                final_cfs_list.append(run_cfs)
                final_preds_list.append(run_preds)

        else:

            self._initialize_multi_run(query_instance, init_near_query_instance)
            run_cfs, run_preds = self._run_vectorized_optimization(
                min_iter, max_iter, loss_diff_thres, loss_converge_maxiter,
                verbose, **kwargs
            )

        self.elapsed = timeit.default_timer() - start_time

        cfs_array = np.concatenate(final_cfs_list, axis=0)
        preds_array = np.concatenate(final_preds_list, axis=0)

        cfs_df = self.model.transformer.inverse_transform(self.data_interface.get_decoded_data(cfs_array))
        preds_flat = np.round(preds_array.flatten(), 3)
        cfs_df[self.data_interface.outcome_name] = preds_flat

        test_instance_df = self.model.transformer.inverse_transform(
            self.data_interface.get_decoded_data(query_instance)
        )

        test_instance_df[self.data_interface.outcome_name] = np.array([round(test_pred, 3)])

        final_cfs_df_sparse = None
        if (posthoc_sparsity_param is not None) and (posthoc_sparsity_param > 0):
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(
                cfs_df.copy(), test_instance_df, posthoc_sparsity_param,
                posthoc_sparsity_algorithm, limit_steps_ls
            )


        valid_ix = range(len(cfs_df))
        final_cfs_df = cfs_df.iloc[valid_ix].reset_index(drop=True)
        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)

        return final_cfs_df, test_instance_df, final_cfs_df_sparse
        
    def _initialize_single_run(self, query_instance, init_near):
        if init_near:
            arr = np.tile(query_instance, (1, 1))
            arr[0, :] += 0.01 * random.random()

        else:
            arr = np.random.uniform(self.minx, self.maxx, size=(1, self.minx.shape[1]))
        self.cfs_var.assign(arr.astype(np.float32))


    def _initialize_multi_run(self, query_instance, init_near):
        k = self.num_cfs
        d = self.minx.shape[1]
        if init_near:
            arr = np.tile(query_instance, (k, 1))
            for i in range(k):
                arr[i, :] += 0.01 * i
        else:
            arr = np.random.uniform(self.minx, self.maxx, size=(k, d))
        self.cfs_var.assign(arr.astype(np.float32))

    def _run_vectorized_optimization(self, min_iter, max_iter, loss_diff_thres,
                                 loss_converge_maxiter, verbose, **kwargs):
        prev_loss = 1e9
        best_loss = 1e9
        self.loss_converge_iter = 0
        cfs_best = None
        preds_best = None

        for it in range(max_iter):
            with tf.GradientTape() as tape:
                loss_value = self.compute_loss_vectorized(**kwargs)
            
            grads = tape.gradient(loss_value, [self.cfs_var])
            grads[0] *= self.freezer_mask
            self.optimizer.apply_gradients(zip(grads, [self.cfs_var]))

            clipped = tf.clip_by_value(self.cfs_var, self.minx, self.maxx)
            self.cfs_var.assign(clipped)

            current_loss_val = loss_value.numpy()  # Convert to float
            diff = abs(current_loss_val - prev_loss)
            print(f"Iter {it}: loss = {current_loss_val}, diff = {diff}")
            if diff < loss_diff_thres and it >= min_iter:
                self.loss_converge_iter += 1
                if self.loss_converge_iter >= loss_converge_maxiter:
                    if verbose:
                        print(f"Stopping at iteration {it+1} due to small loss diff.")
                    break
            else:
                self.loss_converge_iter = 0
            
            # update for next iteration
            prev_loss = current_loss_val

        cfs_np = self.cfs_var.numpy()
        preds_tf = self.predict_fn(self.cfs_var)
        preds_np = preds_tf.numpy()
        return cfs_np, preds_np