"""
Module to generate diverse counterfactual explanations based on tensorflow 2.x
"""
from dice_ml_x.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml_x.counterfactual_explanations import CounterfactualExplanations
from dice_ml_x import diverse_counterfactuals as exp
from dice_ml_x.constants import RobustnessType
import tensorflow as tf
import numpy as np
import copy
import timeit
from collections import defaultdict


# To suppress TensorFlow warning about the optimizer running slowly on Apple chips.
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


class DiceTensorFlow2(ExplainerBase):

    def __init__(self, data_interface, model_interface):
        """Init method
        :param data_interface: an interface class to access data related params.
        :param model_interface: an interface class to access trained ML model.

        """
        # initiating data related parameters
        super().__init__(data_interface)

        # Set device - default to CPU, can be overridden by user
        # Options: "cpu", "gpu", or specific device like "/GPU:0"
        # Note: TensorFlow on Mac doesn't require explicit MPS device specification
        self.device = "cpu"

        # initializing model related variables
        self.model = model_interface
        with tf.device(self.device):
            self.model.load_model()    # loading the trained model
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
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.loss_history = {
            "iterations": [],
            "y_loss": [],
            "proximity_loss": [],
            "diversity_loss": [],
            "robustness_loss": [],
            "total_loss": []
        }
        self.ohe_groups = self._get_ohe_groups(self.data_interface.categorical_feature_names,
                                               self.data_interface.ohe_encoded_feature_names)

    def _get_ohe_groups(self, cat_col_names: list[str], ohe_col_names: list[str]):
        groups = defaultdict(list)

        for i, col in enumerate(ohe_col_names):
            for cat in cat_col_names:
                if col.startswith(cat + "_"):
                    groups[cat].append(i)
                    break
        return [groups[c] for c in cat_col_names]

    def generate_counterfactuals(self, query_instances, total_CFs, desired_class="opposite", proximity_weight=0.5,
                                 diversity_weight=1.0, robustness_weight=0.5, categorical_penalty=0.1, algorithm="DiverseCF",
                                 features_to_vary="all", permitted_range=None, yloss_type="hinge_loss",
                                 diversity_loss_type="dpp_style:inverse_dist", feature_weights="inverse_mad",
                                 optimizer="tensorflow:adam", learning_rate=0.05, min_iter=500, max_iter=5000,
                                 project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                 init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                 posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000,
                                 perturbation_method="gaussian", preprocessing_bins: int=10, **kwargs):
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
                                      perturbation_method, preprocessing_bins=preprocessing_bins, **kwargs)

        counterfactual_explanations = exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class)

        return CounterfactualExplanations(cf_examples_list=[counterfactual_explanations])

    def predict_fn(self, input_instance):
        """prediction function"""
        temp_preds = self.model.get_output(input_instance).numpy()
        return np.array([preds[(self.num_output_nodes-1):] for preds in temp_preds], dtype=np.float32)

    def predict_fn_with_grads(self, input_instance):
        return self.model.model(input_instance, training=False)

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance = self.model.transformer.transform(input_instance).to_numpy()
        return self.predict_fn(tf.constant(input_instance, dtype=tf.float32))

    def get_validity_percentage(self):
        """
        Percentage of generated counterfactuals that actually flip the
        model prediction to the desired `self.target_cf_class`.

        Returns
        -------
        float
            Validity in percent (0â€’100).
        """

        cfs_np = np.array([cf.numpy() if tf.is_tensor(cf) else cf for cf in self.cfs])
        unique_cfs_np = np.unique(cfs_np, axis=0)

        preds = self.predict_fn(tf.convert_to_tensor(unique_cfs_np, dtype=tf.float32))

        if self.target_cf_class == 0:
            valid_mask = preds[:, 0] <= self.stopping_threshold
        else:
            valid_mask = preds[:, 0] >= self.stopping_threshold

        validity_percentage = 100.0 * valid_mask.mean()
        return float(validity_percentage)

    def do_cf_initializations(self, total_CFs, algorithm, features_to_vary):
        """Intializes CFs and other related variables."""

        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            # no. of times to run the experiment with random inits for diversity
            self.total_random_inits = total_CFs
            self.total_CFs = 1          # size of counterfactual set
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs  # size of counterfactual set

        # freeze those columns that need to be fixed
        if features_to_vary != self.features_to_vary:
            self.features_to_vary = features_to_vary
        self.feat_to_vary_idxs = self.data_interface.get_indexes_of_features_to_vary(features_to_vary=features_to_vary)
        self.freezer = tf.constant([1.0 if ix in self.feat_to_vary_idxs else 0.0 for ix in range(len(self.minx[0]))])

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            with tf.device(self.device):
                for _ in range(self.total_CFs):
                    one_init = np.random.uniform(self.minx, self.maxx).astype(np.float32)
                    self.cfs.append(tf.Variable(one_init, dtype=tf.float32))

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
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def perturb_cfs(self, method: str = "gaussian",
                    std_dev: float=0.10, mean: float=0.2, flip_prob: float=0.05,
                    seed: int | None=None) -> tf.Tensor:

        cfs_stacked = tf.stack(self.cfs, axis=0)

        if cfs_stacked.shape.rank == 3:
            cfs_stacked = tf.squeeze(cfs_stacked, axis=1)

        X = tf.identity(cfs_stacked)

        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])

        seed2 = tf.constant([seed, seed ^ 0x9E3779B9], dtype=tf.int32)

        cont_cols = self.encoded_continuous_feature_indexes
        ranges_float = self.data_interface.get_features_range_float()[1]
        cont_col_names = self.data_interface.ohe_encoded_feature_names

        cont_lo = tf.constant([ranges_float[cont_col_names[c]][0] for c in cont_cols], dtype=tf.float32)
        cont_hi = tf.constant([ranges_float[cont_col_names[c]][1] for c in cont_cols], dtype=tf.float32)

        span = tf.reshape(cont_hi - cont_lo, [1, -1])

        X_cont = tf.gather(X, cont_cols, axis=1)

        if method == "gaussian":
            eps = tf.random.stateless_normal(tf.shape(X_cont), seed=seed2, dtype=X_cont.dtype)
            noise = eps * std_dev * span
        elif method == "random":
            u = tf.random.stateless_uniform(tf.shape(X_cont), seed=seed2, dtype=X_cont.dtype) * 2.0 - 1.0
            noise = u * (mean * span)
        elif method == "spherical":
            dir_ = tf.random.stateless_normal(tf.shape(X_cont), seed=seed2, dtype=X_cont.dtype)
            dir_ = dir_ / (tf.norm(dir_, axis=1, keepdims=True) + 1e-12)

            n = tf.shape(X_cont)[0]
            seed_r = seed2 + tf.constant([1, 1], tf.int32)
            r = tf.random.stateless_uniform((n, 1), seed=seed_r, dtype=X_cont.dtype) * mean
            noise = dir_ * r * span
        else:
            raise ValueError(f"Unsupported method: {method}")

        X_cont_new = tf.clip_by_value(X_cont + noise, clip_value_min=cont_lo, clip_value_max=cont_hi)
        n, d_cont = tf.shape(X)[0], tf.shape(X_cont_new)[1]
        rows = tf.repeat(tf.range(n)[:, None], repeats=d_cont, axis=1)
        cols = tf.repeat(tf.constant(cont_cols)[None, :], repeats=n, axis=0)
        idx = tf.stack([tf.reshape(rows, [-1]), tf.reshape(cols, [-1])], axis=1)
        X = tf.tensor_scatter_nd_update(X, idx, tf.reshape(X_cont_new,  [-1]))
        N = tf.shape(X)[0]

        if getattr(self, "ohe_groups", None):
            for gi, group in enumerate(self.ohe_groups):
                """ cat_cols_grp = tf.constant(group, dtype=tf.int32)
                k = tf.size(cat_cols_grp)

                seed_flip = seed2 + tf.constant([10 + gi, 20 + gi], tf.int32)
                do_flip = tf.random.stateless_uniform((N,), seed=seed_flip, dtype=tf.float32) < flip_prob
                row_idx = tf.cast(tf.where(do_flip)[:, 0], tf.int32)
                m = tf.shape(row_idx)[0]

                seed_cat = seed2 + tf.constant([100 + gi, 200 + gi], tf.int32)
                new_cat = tf.random.stateless_uniform((m,), seed=seed_cat, minval=0, maxval=k, dtype=tf.int32)
                chosen_cols = tf.gather(cat_cols_grp, new_cat)

                row_rep = tf.repeat(row_idx, repeats=k)
                col_tile = tf.tile(cat_cols_grp, multiples=[m])
                idx0 = tf.stack([row_rep, col_tile], axis=1)
                X = tf.tensor_scatter_nd_update(X, idx0, tf.zeros((m * k,), dtype=X.dtype))

                idx1 = tf.stack([row_idx, chosen_cols], axis=1)
                X = tf.tensor_scatter_nd_update(X, idx1, tf.ones((m,), dtype=X.dtype)) """

                grp = tf.constant(group, dtype=tf.int32)      # [k]
                k = tf.shape(grp)[0]
                grp_vals = tf.gather(X, grp, axis=1)          # [N, k]

                seed_flip = seed2 + tf.constant([10 + gi, 20 + gi], tf.int32)
                do_flip = tf.random.stateless_uniform((N,), seed=seed_flip, dtype=tf.float32) < flip_prob
                do_flip_f = tf.cast(do_flip[:, None], X.dtype)  # [N,1]

                seed_cat = seed2 + tf.constant([100 + gi, 200 + gi], tf.int32)
                new_cat = tf.random.stateless_uniform((N,), seed=seed_cat, minval=0, maxval=k, dtype=tf.int32)  # [N]
                new_onehot = tf.one_hot(new_cat, depth=k, dtype=X.dtype)  # [N,k]

                grp_new = grp_vals * (1.0 - do_flip_f) + new_onehot * do_flip_f  # [N,k]

                X = tf.tensor_scatter_nd_update(
                    X,
                    indices=tf.stack([tf.repeat(tf.range(N), k), tf.tile(grp, [N])], axis=1),
                    updates=tf.reshape(grp_new, [-1])
                )

        return X

    def do_perturbation(self):
        cfs_stacked = tf.squeeze(tf.stack(self.cfs, axis=0), axis=1)  # (K,D)
        if self.encoded_continuous_feature_indexes:
            continuous_slice = tf.gather(cfs_stacked, self.encoded_continuous_feature_indexes, axis=1)
            noise = 0.3 * continuous_slice
            noise_mask = tf.zeros_like(cfs_stacked)
            idx = tf.constant([[i, j] for i in range(cfs_stacked.shape[0])
                               for j in self.encoded_continuous_feature_indexes], dtype=tf.int32)
            updates = tf.reshape(noise, [-1])
            noise_mask = tf.tensor_scatter_nd_update(noise_mask, idx, updates)
            cfs_stacked = cfs_stacked + noise_mask

        if self.encoded_categorical_feature_indexes:
            for cat_cols in self.encoded_categorical_feature_indexes:
                cat_slice = tf.gather(cfs_stacked, cat_cols, axis=1)
                sample_sz = tf.shape(cat_slice)[0]
                depth = tf.shape(cat_slice)[1]
                rand_idx = tf.random.uniform(shape=(sample_sz,), minval=0, maxval=depth, dtype=tf.int32)
                cat_onehot = tf.one_hot(rand_idx, depth=depth, dtype=tf.float32)

                cat_mask = tf.zeros_like(cfs_stacked)
                idx = tf.constant([[i, j] for i in range(cfs_stacked.shape[0]) for j in cat_cols], dtype=tf.int32)
                updates = tf.reshape(cat_onehot, [-1])
                cat_mask = tf.tensor_scatter_nd_update(cat_mask, idx, updates)
                cfs_stacked = cfs_stacked + cat_mask

        return tf.identity(cfs_stacked)  # (K,D)

        cfs_stacked = tf.stack(self.cfs, axis=0)
        cfs_stacked = tf.squeeze(cfs_stacked, axis=1)

        if self.encoded_continuous_feature_indexes:
            continuous_slice = tf.gather(cfs_stacked, self.encoded_continuous_feature_indexes, axis=1)
            noise = continuous_slice * 0.3
            noise_mask = tf.zeros_like(cfs_stacked)

            indices = tf.constant([[i, j] for i in range(cfs_stacked.shape[0])
                                   for j in self.encoded_continuous_feature_indexes], dtype=tf.int32)
            updates = tf.reshape(noise, [-1])
            noise_mask = tf.tensor_scatter_nd_update(noise_mask, indices, updates)
            cfs_stacked = cfs_stacked + noise_mask

        if self.encoded_categorical_feature_indexes:
            for cat_cols in self.encoded_categorical_feature_indexes:
                cat_slice = tf.gather(cfs_stacked, cat_cols, axis=1)
                sample_size = tf.shape(cat_slice)[0]
                num_cats = tf.shape(cat_slice)[1]

                rand_idx = tf.random.uniform(shape=(sample_size, ), minval=0, maxval=num_cats, dtype=tf.int32)
                cat_slice_perturbed = tf.one_hot(rand_idx, depth=num_cats, dtype=tf.float64)

                cat_mask = tf.zeros_like(cfs_stacked)
                indices = tf.constant([[i, j] for i in range(cfs_stacked.shape[0]) for j in cat_cols], dtype=tf.int32)
                updates = tf.reshape(cat_slice_perturbed, [-1])
                updates = tf.cast(updates, dtype=tf.float32)
                cat_mask = tf.tensor_scatter_nd_update(cat_mask, indices, updates)
                cfs_perturbed = cfs_stacked + cat_mask
        cfs_perturbed = tf.Variable(tf.identity(cfs_perturbed), trainable=True)
        return cfs_perturbed

    @tf.function(jit_compile=True)
    def optimize_perturbations(self, c_i_prime: tf.Tensor, c_i: tf.Tensor, gamma):
       # print(f"c_i_prime_numerical shape: {c_i_prime_numerical.shape}")
        with tf.GradientTape() as tape:
            c_i = tf.reshape(c_i, [-1, c_i.shape[-1]])
            c_i_prime = tf.reshape(c_i_prime, [-1, c_i_prime.shape[-1]])
            tape.watch(c_i_prime)
            pred_i = self.predict_fn_with_grads(c_i)
            pred_i_prime = self.predict_fn_with_grads(c_i_prime)
            class_loss = tf.reduce_mean((pred_i - pred_i_prime) ** 2)
            distance = tf.norm(c_i_prime - c_i, ord=2)
            loss = class_loss - gamma * distance
        grads = tape.gradient(loss, [c_i_prime])

        return loss, grads

    """ @tf.function(jit_compile=True)
    def generate_perturbations_vectorized(self, max_iter=100, tol=tf.constant(1e-3), gamma=1e-2, **kwargs):
        
        c_i_prime = self.do_perturbation()
        c_i = tf.stack(self.cfs, axis=0)
        optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-2)
        prev_loss = tf.constant(float('inf'))
        for _ in range(max_iter):
            loss, grads = self.optimize_perturbations(c_i_prime, c_i, gamma)
            optimizer.apply_gradients(zip(grads, [c_i_prime]))
            
            if tf.abs(loss - prev_loss) < tol:
                break
            prev_loss = loss
        return tf.stop_gradient(c_i_prime) """

    @tf.function(jit_compile=False)
    def generate_perturbations_vectorized(self, max_iter=100, tol=1e-3, gamma=1e-2, **kwargs):
        c_i_prime = self.do_perturbation()          # Tensor (K, D)
        c_i = tf.stack(self.cfs, axis=0)      # (K,1,D) or (K,D)
        
        tol = tf.convert_to_tensor(tol, dtype=tf.float32)
        lr = tf.constant(1e-2, dtype=tf.float32)
        inf = tf.constant(float('inf'), dtype=tf.float32)
        maxit = tf.convert_to_tensor(max_iter, dtype=tf.int32)

        i0, prev0, diff0 = tf.constant(0, tf.int32), inf, inf
        c0 = tf.convert_to_tensor(c_i_prime, dtype=tf.float32)

        def cond(i, prev_loss, diff, c_var):
            return tf.logical_and(i < maxit, diff > tol)

        def body(i, prev_loss, diff, c_var):
            loss, grads = self.optimize_perturbations(c_var, c_i, gamma)
            g = tf.convert_to_tensor(grads[0], dtype=c_var.dtype)
            new_c = c_var - lr * g
            loss_s = tf.squeeze(loss)
            new_df = tf.abs(loss_s - prev_loss)
            return (i + 1, loss_s, new_df, new_c)

        _, _, _, c_final = tf.while_loop(cond, body, (i0, prev0, diff0, c0))
        return tf.stop_gradient(c_final)

    def _preprocess_for_robustness(self, cfs: tf.Tensor, perturbed_cfs: tf.Tensor, num_bins: int=10) -> tuple:

        def preprocess_continuous_features(cfs: tf.Tensor, num_bins=num_bins) -> tf.Tensor:
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
        cfs = tf.squeeze(cfs, axis=1)

        cfs_cats = tf.cast(tf.gather(cfs, cat_cols, axis=1), dtype=tf.float32)
        perturbed_cfs_cats = tf.cast(tf.gather(perturbed_cfs, cat_cols, axis=1), dtype=tf.float32)

        cfs_one_hot = preprocess_continuous_features(cfs)
        perturbed_cfs_one_hot = preprocess_continuous_features(perturbed_cfs)

        cfs_one_hot = tf.cast(cfs_one_hot, dtype=tf.float32)
        perturbed_cfs_one_hot = tf.cast(perturbed_cfs_one_hot, dtype=tf.float32)

        cfs_processed = tf.concat([cfs_one_hot, cfs_cats], axis=1)
        perturbed_cfs_processed = tf.concat([perturbed_cfs_one_hot, perturbed_cfs_cats], axis=1)

        return cfs_processed, perturbed_cfs_processed

    def _phi_soft(self, X: tf.Tensor, num_bins: int = 10, sigma: float = 0.1, eps: float = 1e-8) -> tf.Tensor:
        parts = []

        if getattr(self, "encoded_continuous_feature_indexes", None):
            idx = tf.constant(self.encoded_continuous_feature_indexes, dtype=tf.int32)
            cont = tf.gather(X, idx, axis=1)
            cont = tf.clip_by_value(cont, 0.0, 1.0)

            centers = tf.linspace(0.0, 1.0, num_bins)
            cont_exp = cont[..., tf.newaxis]   # type: ignore
            centers = tf.reshape(centers, (1, 1, num_bins))
            rbf = tf.exp(- (cont_exp - centers) ** 2 / (2.0 * (sigma ** 2)))
            rbf = rbf / (tf.reduce_sum(rbf, axis=-1, keepdims=True) + eps)
            parts.append(tf.reshape(rbf, (tf.shape(cont)[0], -1)))

        if getattr(self, "encoded_categorical_feature_indexes", None):
            for grp in self.encoded_categorical_feature_indexes:
                g = tf.gather(X, grp, axis=1)
                parts.append(tf.nn.softmax(g, axis=1))

        return tf.concat(parts, axis=1) if parts else X

    def compute_robustness_loss_SDS(self, perturbed_cfs: tf.Tensor, num_bins: int = 10,
                                    sigma: float = 0.1, eps: float = 1e-8) -> tf.Tensor:
        cfs = tf.stack(self.cfs, axis=0)

        if tf.rank(cfs) == 3 and cfs.shape[1] == 1:
            cfs = tf.squeeze(cfs, axis=1)

        p = self._phi_soft(cfs, num_bins=num_bins, sigma=sigma)
        q = self._phi_soft(perturbed_cfs, num_bins=num_bins, sigma=sigma)

        num = 2.0 * tf.reduce_sum(p * q, axis=1)
        den = tf.reduce_sum(p * p, axis=1) + tf.reduce_sum(q * q, axis=1) + eps
        sdc = num / den
        return tf.reduce_mean(sdc)

    def compute_robustness_loss(self, perturbed_cfs: tf.Tensor, preprocessing_bins: int=10) -> float:
        """
        Computes the robustness loss.
        Args:
            perturbed_cfs_df (pandas.DataFrame): The perturbed counterfactuals that will
            be compared against the original counterfactual instances.
        Returns:
            float: Robustness loss in scalar.
        """
        cfs = tf.stack(self.cfs, axis=0)  # (K,1,D) or (K,D)

        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs,
                                                                                 num_bins=preprocessing_bins)
        # Sorensenâ€“Dice on processed representations (all TF ops)
        intersection = tf.reduce_sum(tf.minimum(cfs_processed, perturbed_cfs_processed), axis=1)
        union = tf.reduce_sum(cfs_processed, axis=1) + tf.reduce_sum(perturbed_cfs_processed, axis=1)
        eps = tf.constant(1e-8, dtype=intersection.dtype)
        sdc = (2.0 * intersection) / (union + eps)
        sdc = tf.where(tf.math.is_nan(sdc), tf.ones_like(sdc), sdc)
        return tf.reduce_mean(sdc)

    def compute_robustness_distance(self, perturbed_cfs: tf.Tensor, preprocessing_bins: int=10) -> float:
        cfs = tf.stack(self.cfs, axis=0)
        cfs_processed, preturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs,
                                                                                 num_bins=preprocessing_bins)
        intersection = tf.reduce_sum(tf.minimum(cfs_processed, preturbed_cfs_processed), axis=1)
        union = tf.reduce_sum(cfs_processed, axis=1) + tf.reduce_sum(preturbed_cfs_processed, axis=1)
        eps = tf.constant(1e-12, dtype=intersection.dtype)
        dice_sorensen = (2.0 * intersection) / (union + eps)
        dice_sorensen = tf.where(tf.math.is_nan(dice_sorensen), tf.ones_like(dice_sorensen), dice_sorensen)
        return dice_sorensen

    def compute_robustness_loss_(self, perturbed_cfs: tf.Tensor, preprocessing_bins: int=10,
                                 desired_class="opposite"):
        base_prob = tf.squeeze(self.predict_fn_with_grads(self.x1))
        base_label = tf.cast(base_prob > 0.5, tf.float32)
        base_label = tf.stop_gradient(base_label)

        if desired_class == "opposite":
            target_label = 1.0 - base_label
        else:
            target_label = tf.cast(desired_class, tf.float32)

        dist = self.compute_robustness_distance(perturbed_cfs, preprocessing_bins)
        pert_prob = tf.squeeze(self.predict_fn_with_grads(perturbed_cfs))  # shape [K] or [K,1]
        pert_prob = tf.reshape(pert_prob, [-1])
        gate = target_label * pert_prob + (1.0 - target_label) * (1.0 - pert_prob)
        return tf.reduce_mean(dist * gate)

    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function."""
        cfs_stacked = tf.concat(self.cfs, axis=0)
        predictions = self.model.get_output(cfs_stacked)[:, (self.num_output_nodes - 1):]

        if self.yloss_type == "l2_loss":
            yloss = tf.reduce_mean(tf.pow(predictions - self.target_cf_class, 2))
        elif self.yloss_type == "log_loss":
            temp_logits = tf.math.log((tf.abs(predictions - 0.000001)) / (1 - tf.abs(predictions - 0.000001)))
            yloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=temp_logits, labels=tf.cast(self.target_cf_class, dtype=tf.float32)))
        elif self.yloss_type == "hinge_loss":
            temp_logits = tf.math.log((tf.abs(predictions - 0.000001)) / (1 - tf.abs(predictions - 0.000001)))
            labels = tf.cast(tf.broadcast_to(self.target_cf_class, tf.shape(predictions)),
                             dtype=tf.float32)
            yloss = tf.reduce_mean(tf.compat.v1.losses.hinge_loss(logits=temp_logits, labels=labels))
        else:
            yloss = 0.0

        return yloss

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return tf.reduce_sum(tf.multiply((tf.abs(x_hat - x1)), self.feature_weights_list))

    '''def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss/tf.cast((tf.multiply(len(self.minx[0]), self.total_CFs)), dtype=tf.float32)'''

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        cfs_stacked = tf.concat(self.cfs, axis=0)
        # Broadcasting self.x1 to match the shape of cfs_stacked
        distances = tf.multiply(tf.abs(cfs_stacked - self.x1), self.feature_weights_list)
        proximity_loss = tf.reduce_mean(distances)
        return proximity_loss / tf.cast(len(self.minx[0]), dtype=tf.float32)

    '''def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = []
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.add(
                        1.0, self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_temp_entry = tf.add(det_temp_entry, 0.0001)
                    det_entries.append(det_temp_entry)

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_temp_entry = tf.divide(1.0, tf.exp(
                        self.compute_dist(self.cfs[i], self.cfs[j])))
                    det_entries.append(det_temp_entry)

        det_entries = tf.reshape(det_entries, [self.total_CFs, self.total_CFs])
        diversity_loss = tf.linalg.det(det_entries)
        return diversity_loss'''

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        cfs_stacked = tf.squeeze(tf.stack(self.cfs, axis=0), axis=1)  # Shape: (total_CFs, num_features)

        # Expand dimensions for broadcasting
        cfs1 = tf.expand_dims(cfs_stacked, 1)  # Shape: (total_CFs, 1, num_features)
        cfs2 = tf.expand_dims(cfs_stacked, 0)  # Shape: (1, total_CFs, num_features)

        # Compute pairwise distances
        pairwise_dist = tf.reduce_sum(self.feature_weights_list * tf.abs(cfs1 - cfs2), axis=2)

        if submethod == "inverse_dist":
            det_entries = 1.0 / (1.0 + pairwise_dist)
            # Add a small value to the diagonal for numerical stability
            det_entries += tf.eye(self.total_CFs, dtype=tf.float32) * 0.0001
        elif submethod == "exponential_dist":
            det_entries = 1.0 / tf.exp(pairwise_dist)
        else:
            return tf.constant(0.0)

        diversity_loss = tf.linalg.det(det_entries)
        return diversity_loss

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return tf.constant(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return tf.reduce_sum(self.dpp_style(submethod))
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    '''def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels
           of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += tf.pow((tf.reduce_sum(self.cfs[i][0, v[0]:v[-1]+1]) - 1.0), 2)

        return regularization_loss'''

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions - to ensure all levels
        of a categorical variable sums to one"""
        # tf.concat on a list of (1, num_features) tensors creates a (total_CFs, num_features) tensor.
        cfs_stacked = tf.concat(self.cfs, axis=0)

        # The squeeze operation was incorrect and is removed. We use cfs_stacked directly.
        regularization_loss = 0.0
        for v in self.encoded_categorical_feature_indexes:
            # Sum over the categorical feature columns for all CFs at once
            # cfs_stacked is 2D, so slicing and summing works as intended.
            sum_over_cat_features = tf.reduce_sum(cfs_stacked[:, v[0]:v[-1]+1], axis=1)
            regularization_loss += tf.reduce_sum(tf.pow(sum_over_cat_features - 1.0, 2))

        return regularization_loss

    def compute_loss(self, preprocessing_bins: int=10, robustness_distance_type=RobustnessType.DICE_SORENSEN, **kwargs):
        """Computes the overall loss"""
        perturbed_cfs = self.perturb_cfs("gaussian")

        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        if robustness_distance_type == RobustnessType.DICE_SORENSEN:
            self.robustness_loss = self.compute_robustness_loss_(perturbed_cfs=perturbed_cfs, preprocessing_bins=preprocessing_bins) \
                                                                if self.robustness_weight > 0 else 0.0
        elif robustness_distance_type == RobustnessType.GAUSSIAN_KERNEL:
            self.robustness_loss = 1.0 if self.robustness_weight > 0 else 0.0
        elif robustness_distance_type == RobustnessType.BINNED_GAUSSIAN_KERNEL:
            self.robustness_loss = 1.0 if self.robustness_weight > 0 else 0.0
        else:
            raise ValueError(f"Unsupported robustness distance type: {robustness_distance_type}. Supported types: \
                             `DICE_SORENSEN`, `GAUSSIAN_KERNEL`, `BINNED_GAUSSIAN_KERNEL`. Use `RobustnessType` \
                             class for distance types")
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

    '''def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[0, v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[0, v] = normalized_cont  # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[0, v[0]:v[-1]+1] == np.amax(cf[0, v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[0, v[vi]] = 1.0
                    else:
                        cf[0, v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                self.cfs[index].assign(temp_cfs[index])

        if assign:
            return None
        else:
            return temp_cfs
    
    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""

        cfs_stacked = tf.squeeze(tf.stack(self.cfs, axis=0), axis=1)  # (K,D)

        # continuous features: bin + re-normalize (vectorized)
        for i, v in enumerate(self.encoded_continuous_feature_indexes):
            org = cfs_stacked[:, v] * (self.cont_maxx[i] - self.cont_minx[i]) + self.cont_minx[i]
            org = tf.round(org * (10 ** self.cont_precisions[i])) / (10 ** self.cont_precisions[i])
            norm = (org - self.cont_minx[i]) / (self.cont_maxx[i] - self.cont_minx[i])
            # scatter back
            cfs_stacked = tf.tensor_scatter_nd_update(
                cfs_stacked,
                tf.concat([tf.reshape(tf.range(tf.shape(cfs_stacked)[0]), (-1,1)) * 0 + 0,
                        tf.reshape(tf.constant(v, dtype=tf.int32), (1,-1)).tile([tf.shape(cfs_stacked)[0],1])], axis=1),
                norm  # this trick is ugly; if it scares you, keep your original NumPy version
            )

        # categorical features: argmax one-hot (vectorized)
        for v in self.encoded_categorical_feature_indexes:
            cat = cfs_stacked[:, v[0]:v[-1]+1]
            arg = tf.argmax(cat, axis=1, output_type=tf.int32)
            one = tf.one_hot(arg, depth=cat.shape[1], dtype=cat.dtype)
            # overwrite slice
            left  = cfs_stacked[:, :v[0]]
            right = cfs_stacked[:, v[-1]+1:]
            cfs_stacked = tf.concat([left, one, right], axis=1)

        if assign:
            # split back into list of Variables without leaving TF
            for i in range(len(self.cfs)):
                self.cfs[i].assign(tf.expand_dims(cfs_stacked[i], axis=0))
            return None
        else:
            return [tf.expand_dims(cfs_stacked[i], axis=0) for i in range(len(self.cfs))]'''

    def round_off_cfs(self, assign=False):
        cfs_stacked = tf.squeeze(tf.stack(self.cfs, axis=0), axis=1)  # (K, D)

        # ----- continuous features: round & normalize, then splice back
        for i, vidx in enumerate(self.encoded_continuous_feature_indexes):
            # Allow either int or list of indices
            idxs = vidx if isinstance(vidx, (list, tuple)) else [vidx]

            cols = tf.gather(cfs_stacked, idxs, axis=1)  # (K, len(idxs))
            # de-normalize, round to precision, re-normalize
            span = tf.cast(self.cont_maxx[i] - self.cont_minx[i], tf.float32)
            base = tf.cast(self.cont_minx[i], tf.float32)
            cols_org = cols * span + base

            prec_pow = tf.cast(10 ** self.cont_precisions[i], tf.float32)
            cols_org = tf.round(cols_org * prec_pow) / prec_pow

            cols_norm = (cols_org - base) / span  # (K, len(idxs))

            left = cfs_stacked[:, :idxs[0]]
            right = cfs_stacked[:, idxs[-1] + 1:]
            cfs_stacked = tf.concat([left, cols_norm, right], axis=1)

        # ----- categorical features: argmax one-hot & splice back
        for idxs in self.encoded_categorical_feature_indexes:
            cat = cfs_stacked[:, idxs[0]:idxs[-1] + 1]
            arg = tf.argmax(cat, axis=1, output_type=tf.int32)
            one = tf.one_hot(arg, depth=cat.shape[1], dtype=cat.dtype)
            left = cfs_stacked[:, :idxs[0]]
            right = cfs_stacked[:, idxs[-1] + 1:]
            cfs_stacked = tf.concat([left, one, right], axis=1)

        if assign:
            for i in range(len(self.cfs)):
                self.cfs[i].assign(tf.expand_dims(cfs_stacked[i], axis=0))
            return None
        else:
            return [tf.expand_dims(cfs_stacked[i], axis=0) for i in range(len(self.cfs))]

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
                             posthoc_sparsity_algorithm, limit_steps_ls, perturbation_method: str,
                             preprocessing_bins: int=10, **kwargs):
        """Finds counterfactuals by gradient-descent."""
        self._reset_loss_history()

        with tf.device(self.device):
            query_instance = self.model.transformer.transform(query_instance).to_numpy()
            self.x1 = tf.constant(query_instance, dtype=tf.float32)

            # find the predicted value of query_instance
            test_pred = self.predict_fn(tf.constant(query_instance, dtype=tf.float32))[0][0]
            if desired_class == "opposite":
                desired_class = 1.0 - round(test_pred)
            self.target_cf_class = np.array([[desired_class]], dtype=np.float32)

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

            # running optimization steps
            start_time = timeit.default_timer()
            self.final_cfs = []

            # looping the find CFs depending on whether its random initialization or not
            loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1

            # variables to backup best known CFs so far in the optimization process -
            # if the CFs dont converge in max_iter iterations, then best_backup_cfs is returned.
            self.best_backup_cfs = [0]*max(self.total_CFs, loop_find_CFs)
            self.best_backup_cfs_preds = [0]*max(self.total_CFs, loop_find_CFs)
            self.min_dist_from_threshold = [100]*loop_find_CFs  # for backup CFs

            for loop_ix in range(loop_find_CFs):
                # CF init
                if self.total_random_inits > 0:
                    self.initialize_CFs(query_instance, False)
                else:
                    self.initialize_CFs(query_instance, init_near_query_instance)

                # initialize optimizer
                self.do_optimizer_initializations(optimizer, learning_rate)

                iterations = 0
                loss_diff = 1.0
                prev_loss = 0
                while self.stop_loop(iterations, loss_diff) is False:

                    # compute loss and tape the variables history
                    with tf.GradientTape() as tape:
                        loss_value = self.compute_loss(preprocessing_bins=preprocessing_bins, **kwargs)

                    # get gradients
                    grads = tape.gradient(loss_value, self.cfs)

                    # freeze features other than feat_to_vary_idxs
                    for ix in range(self.total_CFs):
                        grads[ix] *= self.freezer

                    # apply gradients and update the variables
                    self.optimizer.apply_gradients(zip(grads, self.cfs))

                    self._populate_loss_history(iterations, self.yloss, self.proximity_loss,
                                                self.diversity_loss, self.robustness_loss, loss_value)

                    cfs_stack = tf.stack([cf for cf in self.cfs], axis=0)   # (K,1,D)
                    cfs_clip = tf.clip_by_value(cfs_stack, self.minx, self.maxx)
                    for j in range(self.total_CFs):
                        self.cfs[j].assign(cfs_clip[j])

                    '''# projection step
                    for j in range(0, self.total_CFs):
                        temp_cf = self.cfs[j].numpy()
                        clip_cf = np.clip(temp_cf, self.minx, self.maxx)  # clipping
                        # to remove -ve sign before 0.0 in some cases
                        clip_cf = np.add(clip_cf, np.array(
                            [np.zeros([self.minx.shape[1]])]))
                        self.cfs[j].assign(clip_cf)'''

                    if verbose:
                        if (iterations) % 50 == 0:
                            print('step %d,  loss=%g' % (iterations+1, loss_value))

                    loss_diff = abs(loss_value-prev_loss)
                    prev_loss = loss_value
                    iterations += 1

                    '''# backing up CFs if they are valid
                    temp_cfs_stored = self.round_off_cfs(assign=False)
                    test_preds_stored = [self.predict_fn(tf.constant(cf, dtype=tf.float32)) for cf in temp_cfs_stored]

                    if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                    (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                        avg_preds_dist = np.mean([abs(pred[0][0]-self.stopping_threshold) for pred in test_preds_stored])
                        if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                            self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                            for ix in range(self.total_CFs):
                                self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                                self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])'''

                    temp_cfs_stored = self.round_off_cfs(assign=False)                 # list of (1,D) tensors
                    temp_batch = tf.concat(temp_cfs_stored, axis=0)               # (K,D)
                    preds_batch = self.predict_fn(temp_batch)                      # (K,1)
                    preds_vec = tf.squeeze(preds_batch, axis=1)                  # (K,)

                    if self.target_cf_class == 0:
                        all_valid = tf.reduce_all(preds_vec <= self.stopping_threshold)
                        avg_dist = tf.reduce_mean(tf.abs(preds_vec - self.stopping_threshold))
                    else:
                        all_valid = tf.reduce_all(preds_vec >= self.stopping_threshold)
                        avg_dist = tf.reduce_mean(tf.abs(preds_vec - self.stopping_threshold))

                    if bool(all_valid):
                        if float(avg_dist) < self.min_dist_from_threshold[loop_ix]:
                            self.min_dist_from_threshold[loop_ix] = float(avg_dist)
                            for ix in range(self.total_CFs):
                                self.best_backup_cfs[loop_ix+ix] = temp_cfs_stored[ix].numpy()
                                self.best_backup_cfs_preds[loop_ix+ix] = np.array([[float(preds_vec[ix])]])

                # rounding off final cfs - not necessary when inter_project=True
                self.round_off_cfs(assign=True)

                # storing final CFs
                for j in range(0, self.total_CFs):
                    temp = self.cfs[j].numpy()
                    self.final_cfs.append(temp)

                # max iterations at which GD stopped
                self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time
        self.cfs_preds = [self.predict_fn(tf.constant(cfs, dtype=tf.float32)) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        if ((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for ix in range(self.total_CFs):
                        self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                        self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])

        final_cfs_df = self.model.transformer.inverse_transform(
            self.data_interface.get_decoded_data(cfs))

        cfs_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds]
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        test_instance_df = self.model.transformer.inverse_transform(
            self.data_interface.get_decoded_data(query_instance))
        test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred, 3))

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and \
                'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        else:
            final_cfs_df_sparse = None
        # need to check the above code on posthoc sparsity

        # if posthoc_sparsity_param != None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
        #     final_cfs_sparse = copy.deepcopy(self.final_cfs)
        #     cfs_preds_sparse = copy.deepcopy(self.cfs_preds)
        #     self.final_cfs_sparse, self.cfs_preds_sparse = self.do_posthoc_sparsity_enhancement(
        #           self.total_CFs, final_cfs_sparse, cfs_preds_sparse, query_instance, posthoc_sparsity_param,
        #           posthoc_sparsity_algorithm, total_random_inits=self.total_random_inits)
        # else:
        #     self.final_cfs_sparse = None
        #     self.cfs_preds_sparse = None

        m, s = divmod(self.elapsed, 60)
        if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = []  # indexes of valid CFs
            for cf_ix, pred in enumerate(self.cfs_preds):
                if ((self.target_cf_class == 0 and pred < self.stopping_threshold) or
                   (self.target_cf_class == 1 and pred > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0:
                print('No Counterfactuals found for the given configuation, perhaps try with different ',
                      'values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                      ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                      'values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)
        # returning only valid CFs
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse
