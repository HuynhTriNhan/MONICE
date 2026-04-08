"""
Module to generate diverse counterfactual explanations based on PyTorch framework
"""
import copy
import random
import timeit

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F

from dice_ml_x import diverse_counterfactuals as exp
from dice_ml_x.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml_x.autoencoders.dp_s_ae import DPStandardAutoEncoder


class DicePyTorchDP(ExplainerBase):

    def __init__(self, data_interface, model_interface, dp_autoencoder: DPStandardAutoEncoder):
        """
        Init method for DP-based counterfactual generation.

        :param data_interface: data interface for feature names, bounds, etc.
        :param model_interface: classifier to evaluate CF predictions (operates in input space).
        :param dp_autoencoder: a trained DPStandardAutoEncoder (with encoder and decoder)
        :param dp_prototypes: dictionary of {class_label: prototype_z} latent centroids (ε-DP).
        """
        super().__init__(data_interface)

        self.model = model_interface
        self.dp_ae: DPStandardAutoEncoder = dp_autoencoder

        # Load classifier model (for prediction evaluation)
        self.model.load_model()

        # Set up transformer for encoding/decoding
        self.model.transformer.feed_data_params(data_interface)
        self.model.transformer.initialize_transform_func()

        # Create OHE parameters (used for transforming instances)
        temp_ohe_data = self.model.transformer.transform(
            self.data_interface.data_df.iloc[[0]]
        )
        self.data_interface.create_ohe_params(temp_ohe_data)

        # Store metadata for proximity etc.
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, \
        self.encoded_continuous_feature_indexes, \
        self.cont_minx, self.cont_maxx, self.cont_precisions = \
            self.data_interface.get_data_params_for_gradient_dice()

        # Get output dimension (number of classes)
        self.num_output_nodes = self.model.get_num_output_nodes(
            len(self.data_interface.ohe_encoded_feature_names)
        ).shape[1]

        # CF generation placeholders
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # weights for initialization
        self.loss_weights = []     # [prediction, diversity, feature-wise]
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity, diversity, categorical penalty
        self.optimizer_weights = []  # optimizer, learning rate

        self.loss_history = {
            "iterations": [],
            "y_loss": [],
            "proximity_loss": [],
            "diversity_loss": [],
            "regularization_loss": [],
            "robustness_loss": [],
            "total_loss": []
        }

    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  proximity_weight=0.5,
                                  diversity_weight=1.0,
                                  robustness_weight=0.4,
                                  categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all",
                                  permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500,
                                  max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                  init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                  posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000,
                                  perturbation_method="gaussian", **kwargs):
        """Generates diverse counterfactual explanations.

        :param query_instance: Test point of interest. A dictionary of feature names and values or a single row dataframe
        :param total_CFs: Total number of counterfactuals required.
        :param desired_class: Desired counterfactual class - can take 0 or 1. Default value is "opposite" to the outcome class
                              of query_instance for binary classification.
        :param desired_range: Not supported currently.
        :param proximity_weight: A positive float. Larger this weight, more close the counterfactuals are to the
                                 query_instance.
        :param diversity_weight: A positive float. Larger this weight, more diverse the counterfactuals are.
        :param categorical_penalty: A positive float. A weight to ensure that all levels of a categorical variable sums to 1.
        :param algorithm: Counterfactual generation algorithm. Either "DiverseCF" or "RandomInitCF".
        :param features_to_vary: Either a string "all" or a list of feature names to vary.
        :param permitted_range: Dictionary with continuous feature names as keys and permitted min-max range in list as values.
                               Defaults to the range inferred from training data. If None, uses the parameters initialized in
                               data_interface.
        :param yloss_type: Metric for y-loss of the optimization function. Takes "l2_loss" or "log_loss" or "hinge_loss".
        :param diversity_loss_type: Metric for diversity loss of the optimization function.
                                    Takes "avg_dist" or "dpp_style:inverse_dist".
        :param feature_weights: Either "inverse_mad" or a dictionary with feature names as keys and corresponding weights as
                                values. Default option is "inverse_mad" where the weight for a continuous feature is the
                                inverse of the Median Absolute Devidation (MAD) of the feature's values in the training set;
                                the weight for a categorical feature is equal to 1 by default.
        :param optimizer: PyTorch optimization algorithm. Currently tested only with "pytorch:adam".
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

        :return: A CounterfactualExamples object to store and visualize the resulting
                 counterfactual explanations (see diverse_counterfactuals.py).
        """
        # check feature MAD validity and throw warnings
        if feature_weights == "inverse_mad":
            self.data_interface.get_valid_mads(display_warnings=True, return_mads=False)

        # check permitted range for continuous features
        if permitted_range is not None:
            self.data_interface.permitted_range = permitted_range
            self.minx, self.maxx = self.data_interface.get_minx_maxx(normalized=True)
            self.cont_minx = []
            self.cont_maxx = []
            for feature in self.data_interface.continuous_feature_names:
                self.cont_minx.append(self.data_interface.permitted_range[feature][0])
                self.cont_maxx.append(self.data_interface.permitted_range[feature][1])

        if [total_CFs, algorithm, features_to_vary] != self.cf_init_weights:
            self.do_cf_initializations(total_CFs, algorithm, features_to_vary)
        if [yloss_type, diversity_loss_type, feature_weights] != self.loss_weights:
            self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights)
        if [proximity_weight, diversity_weight, robustness_weight, categorical_penalty] != self.hyperparameters:
            self.update_hyperparameters(proximity_weight, diversity_weight, robustness_weight, categorical_penalty)

        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(
                query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter,
                project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance,
                tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, limit_steps_ls,
                perturbation_method, **kwargs)

        return exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class)

    def get_model_output(self, input_instance,
                         transform_data=False, out_tensor=True):
        """get output probability of ML model"""
        return self.model.get_output(
                input_instance,
                transform_data=transform_data,
                out_tensor=out_tensor)[(self.num_output_nodes-1):]

    def predict_fn(self, input_instance):
        """prediction function"""
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        return self.get_model_output(
                input_instance, transform_data=False, out_tensor=False)

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        input_instance_np = self.model.transformer.transform(input_instance).to_numpy()[0]
        input_instance_np = input_instance_np.astype(np.float32)
        return self.predict_fn(torch.tensor(input_instance_np))

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

        # CF initialization
        if len(self.cfs) != self.total_CFs:
            self.cfs = []
            for ix in range(self.total_CFs):
                one_init = []
                for jx in range(self.minx.shape[1]):
                    one_init.append(np.random.uniform(self.minx[0][jx], self.maxx[0][jx]))
                self.cfs.append(torch.tensor(one_init).float())
                self.cfs[ix].requires_grad = True

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
            self.feature_weights_list = torch.tensor(feature_weights_list)

        # define different parts of loss function
        self.yloss_opt = torch.nn.BCEWithLogitsLoss()

    def update_hyperparameters(self, proximity_weight, diversity_weight,
                               robustness_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""

        self.hyperparameters = [proximity_weight, diversity_weight,
                                robustness_weight, categorical_penalty]
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.robustness_weight = robustness_weight
        self.categorical_penalty = categorical_penalty

    def do_optimizer_initializations(self, optimizer, learning_rate):
        """Initializes gradient-based PyTorch optimizers."""
        opt_method = optimizer.split(':')[1]

        # optimizater initialization
        if opt_method == "adam":
            self.optimizer = torch.optim.Adam(self.cfs, lr=learning_rate)
        elif opt_method == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.cfs, lr=learning_rate)


    def compute_yloss(self):
        """Computes the first part (y-loss) of the loss function on decoded CFs."""
        deltas_batch = torch.stack(self.cfs, dim=0)
        z_batch = self.rho_s + deltas_batch
        x_cfs_batch = self.dp_ae.decoder(z_batch)
        preds_batch = self.get_model_output(x_cfs_batch, transform_data=False, out_tensor=True).view(-1)

        if self.yloss_type == "l2_loss":
            target = self.target_cf_class.expand_as(preds_batch)
            return F.mse_loss(preds_batch, target)

        elif self.yloss_type == "log_loss":
            target = self.target_cf_class.expand_as(preds_batch)
            logits = torch.log(preds_batch.clamp(1e-6, 1-1e-6) / (1 - preds_batch.clamp(1e-6, 1-1e-6)))
            return F.binary_cross_entropy_with_logits(logits, target)

        elif self.yloss_type == "hinge_loss":
            labels = 2 * self.target_cf_class - 1
            logits = torch.log(preds_batch.clamp(1e-6, 1-1e-6) / (1 - preds_batch.clamp(1e-6, 1-1e-6)))
            return torch.mean(torch.relu(1 - labels * logits))

        else:
            raise ValueError(f"Unknown yloss_type: {self.yloss_type}")
        



    '''def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)'''
    
    def compute_dist(self, x_hat: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature-weighted L1 distance between x_hat and x1.

        Both x_hat and x1 may live on MPS, CPU, or CUDA—this will
        automatically cast x1 and the feature-weights to x_hat.device.
        Returns a single scalar per instance if x_hat is a 1-D vector,
        or a vector of distances if x_hat is 2-D ([batch_size, n_features]).
        """
        w = self.feature_weights_list

        diff = (x_hat - x1).abs() * w
        return diff.sum(dim=-1)
    
    def compute_proximity_loss(self):
        """
        Compute the average (feature-weighted) distance between each decoded CF
        and the original query x1 (in the model’s input space).
        """
        x_orig = self.x1.float()

        deltas_batch = torch.stack(self.cfs, dim=0)
        z_batch = self.rho_s + deltas_batch
        x_cfs_batch = self.dp_ae.decoder(z_batch)

        distances = self.compute_dist(x_cfs_batch, x_orig)
        return torch.mean(distances)
    
    def dpp_style(self, submethod, decoded_cfs):
        """Computes the DPP of a matrix using vectorized operations."""
        # decoded_cfs is a list of tensors; stack them into a single batch tensor
        
        cfs_batch = torch.stack(decoded_cfs, dim=0)
        k = self.total_CFs
        cfs1 = cfs_batch.unsqueeze(1)
        cfs2 = cfs_batch.unsqueeze(0)

        pairwise_diff = (cfs1 - cfs2).abs()
        
        weights = self.feature_weights_list
        dist_matrix = (pairwise_diff * weights).sum(dim=-1) # Shape: [k, k]

        if submethod == "inverse_dist":
            det_entries = 1.0 / (1.0 + dist_matrix)
        elif submethod == "exponential_dist":
            det_entries = 1.0 / torch.exp(dist_matrix)
        else:
            raise ValueError(f"Unknown dpp submethod: {submethod}")

        det_entries += torch.eye(k) * 0.0001
        
       
        return torch.det(det_entries)
        
    
    def do_perturbation(self, sigma: float=0.05):
        cfs_stacked = torch.stack([
            self.dp_ae.decoder(self.rho_s + delta)
            for delta in self.cfs
        ], dim=0)
    
        # 1) Continuous features: scaled noise in the same device
        if self.encoded_continuous_feature_indexes:
            cont_idxs = self.encoded_continuous_feature_indexes
            continuous_slice = cfs_stacked[:, cont_idxs]
            noise = continuous_slice * sigma
            noise_mask = torch.zeros_like(cfs_stacked)
            noise_mask[:, cont_idxs] = noise
            cfs_stacked = cfs_stacked + noise_mask

        # 2) Categorical features: random one‐hot masks on the right device
        if self.encoded_categorical_feature_indexes:
            mask = torch.zeros_like(cfs_stacked)
            for cat_cols in self.encoded_categorical_feature_indexes:
                sample_size, num_cats = cfs_stacked[:, cat_cols].shape
                rand_idx = torch.randint(0, num_cats, (sample_size,))
                cat_one_hot = torch.nn.functional.one_hot(rand_idx, num_classes=num_cats).float()
                mask[:, cat_cols] = cat_one_hot
            cfs_stacked = cfs_stacked + mask

        # Wrap as a Parameter so downstream optimizers can touch it
        cfs_perturbed = torch.nn.Parameter(cfs_stacked.clone(), requires_grad=True)
        return cfs_perturbed
    

    def generate_perturbations(self, method: str='gaussian', max_iter=100,
                           tol=1e-3, gamma=1e-2):
        orig_cfs = torch.stack([self.dp_ae.decoder(self.rho_s + delta) for delta in self.cfs], 
                            dim=0)

        # FIX: Detach orig_cfs from the main computation graph
        orig_cfs = orig_cfs.detach()

        perturbed_cfs = self.do_perturbation().detach().requires_grad_(True)
        perturbation_optimizer = torch.optim.Adam([perturbed_cfs], lr=1e-3)
            
        prev_loss = np.inf
        for _ in range(max_iter):
            with torch.no_grad():
                self.model.model.eval()
                pred_i = self.model.get_output(
                    orig_cfs,
                    transform_data=False, out_tensor=True
                )
                pred_i_prime = self.model.get_output(perturbed_cfs,
                                                    transform_data=False, out_tensor=True)
            class_loss = torch.mean((pred_i - pred_i_prime) ** 2)
            distance = torch.norm(perturbed_cfs - orig_cfs, p=2)
            loss = class_loss + gamma * distance


            perturbation_optimizer.zero_grad()
            loss.backward()

            perturbation_optimizer.step()
            if abs(loss.item() - prev_loss) < tol:
                break
            prev_loss = loss.item()
        self.model.model.train()
        return perturbed_cfs
    
    
    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function on decoded CFs."""
        k = self.total_CFs
        if k <= 1:
            return torch.tensor(0.0)

        # 1) Decode all CFs once
        decoded_cfs = [
            self.dp_ae.decoder((self.rho_s + delta).unsqueeze(0)).squeeze(0)
            for delta in self.cfs
        ]

        # 2) DPP‐style diversity
        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            # build similarity matrix on decoded CFs
            return self.dpp_style(submethod, decoded_cfs)

        # 3) Average‐distance diversity
        elif self.diversity_loss_type == "avg_dist":
            total_sim = 0.0
            count = 0
            for i in range(k):
                for j in range(i + 1, k):
                    dist_ij = self.compute_dist(decoded_cfs[i], decoded_cfs[j])
                    total_sim += 1.0 / (1.0 + dist_ij)
                    count += 1
            # normalized diversity = 1 − average similarity
            return 1.0 - (total_sim / count)

        else:
            raise ValueError(f"Unknown diversity_loss_type: {self.diversity_loss_type}")


    def compute_regularization_loss(self):
        """
        Ensures each one‐hot categorical group in the decoded CF sums to 1.
        Operates on decoded counterfactuals.
        """
        total_reg = 0.0
        deltas_batch = torch.stack(self.cfs, dim=0)
        z_batch = self.rho_s + deltas_batch
        decoded_cfs_batch = self.dp_ae.decoder(z_batch)

        for cols in self.encoded_categorical_feature_indexes:
            group_batch = decoded_cfs_batch[:, cols[0]:cols[-1] + 1]
            group_sums = torch.sum(group_batch, dim=1)
            total_reg += torch.sum((group_sums - 1.0) ** 2)
        return total_reg

    def _preprocess_for_robustness(self, cfs: torch.Tensor, perturbed_cfs: torch.Tensor) -> tuple:
        """
        Conducts preprocessing steps for robustness loss calculation i.e., converts the given
        tensors into binarized torch.Tensors
        Args:
            cfs (torch.Tensor): The counterfactual instances generated by the algorithm.
            perturbed_cfs (torch.Tensor): The perturbed counterfactual instances of cfs.

        Returns:
            tuple: A tuple that contains preprocessed original counterfactual instances and
                preprocessed perturbed instances of type torch.Tensor.
        """
        def preprocess_continuous_features(cfs: torch.Tensor, num_bins=10) -> torch.Tensor:
            edges = torch.linspace(0.0, 1.0, steps=num_bins + 1)
            all_one_hots = []
            for cont_idx in self.encoded_continuous_feature_indexes:
                col_vals = cfs[:, cont_idx].contiguous()

                binned_indices = torch.bucketize(col_vals, edges, right=False) - 1
                binned_indices = torch.clamp(binned_indices, 0, num_bins - 1)

                one_hot_col = torch.nn.functional.one_hot(binned_indices, num_classes=num_bins).float()
                all_one_hots.append(one_hot_col)
            return torch.cat(all_one_hots, dim=1)
        
        cat_cols = [col for group in self.encoded_categorical_feature_indexes for col in group]

        cfs_cats = cfs[:, cat_cols]
        perturbed_cfs_cats = perturbed_cfs[:, cat_cols]

        cfs_one_hot = preprocess_continuous_features(cfs)
        perturbed_cfs_one_hot = preprocess_continuous_features(perturbed_cfs)

        cfs_processed = torch.cat([cfs_one_hot, cfs_cats], dim=1)
        perturbed_cfs_processed = torch.cat([perturbed_cfs_one_hot, perturbed_cfs_cats], dim=1)

        return cfs_processed, perturbed_cfs_processed


    def compute_robustness_loss(self, perturbed_cfs: torch.Tensor) -> torch.Tensor:
        """
        Computes the robustness loss.

        Args:
            pertubed_cfs_df (pandas.DataFrame): The perturbed counterfactuals that will
            be compared against the original counterfactual instances.
        Returns:
            float: Robustness loss in scalar.
        """
        cfs = torch.stack([self.dp_ae.decoder(self.rho_s + delta) for delta in self.cfs])

        # from IPython.display import displaya
        # print("--------------------------------------------------------Original instances--------------------------------------------------------")
        # display(cfs_df)
        # print("--------------------------------------------------------Perturbed instances--------------------------------------------------------")
        # display(perturbed_cfs_df)

        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs)

        intersection = torch.sum(torch.min(cfs_processed, perturbed_cfs_processed), dim=1)

        union = torch.sum(cfs_processed, dim=1) + torch.sum(perturbed_cfs_processed, dim=1)

        epsilon = 1e-8
        sorensen_dice_coefficient = (2 * intersection) / (union + epsilon)
        sorensen_dice_coefficient[torch.isnan(sorensen_dice_coefficient)] = 1.0
        #test_grad = torch.autograd.grad(sorensen_dice_coefficient.mean(), self.cfs, retain_graph=True)
        #print("Gradients:", test_grad)
        return sorensen_dice_coefficient.mean()


    def compute_loss(self, method: str, **kwargs):
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss  = self.compute_diversity_loss()  if self.diversity_weight   > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss()

        # 1) run your inner robustness‐optimizer
        perturbed_cfs = self.generate_perturbations('gaussian')

        # 2) detach its graph so outer backward() doesn't revisit those edges
        perturbed_cfs = perturbed_cfs.detach()

        # 3) compute robustness loss on the *detached* tensors
        self.robustness_loss = self.compute_robustness_loss(perturbed_cfs=perturbed_cfs)

        # 4) combine all losses as before
        self.loss = (
            self.yloss
        + self.proximity_weight   * self.proximity_loss
        - self.diversity_weight   * self.diversity_loss
        - self.robustness_weight  * self.robustness_loss
        + self.categorical_penalty * self.regularization_loss
        )
        return self.loss


    # def compute_loss(self, method: str, **kwargs):
    #     """Computes the overall loss"""
    #     self.yloss = self.compute_yloss()
    #     self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
    #     self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
    #     self.regularization_loss = self.compute_regularization_loss().to(self.device, dtype=torch.float32)
    #     perturbed_cfs = self.generate_perturbations('gaussian')
    #     self.robustness_loss = self.compute_robustness_loss(perturbed_cfs=perturbed_cfs)
        
    #     self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - \
    #         (self.diversity_weight * self.diversity_loss) - \
    #         (self.robustness_weight * self.robustness_loss) + \
    #         (self.categorical_penalty * self.regularization_loss)
    #     return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """
        Initialize counterfactuals in latent space.
        """
        qi = torch.tensor(query_instance, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            self.rho_s = self.dp_ae.encoder(qi).squeeze(0)

        new_cfs = []
        for n in range(self.total_CFs):
            # Copy input and perturb allowed features
            cf_input = np.array(query_instance, copy=True)
            for i in range(len(cf_input)):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        cf_input[i] = query_instance[i] + (n * 0.01)
                    else:
                        cf_input[i] = np.random.uniform(self.minx[0][i], self.maxx[0][i])

            tensor_in = torch.tensor(cf_input, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                z_cf = self.dp_ae.encoder(tensor_in).squeeze(0)

            delta = (z_cf - self.rho_s).detach().clone().requires_grad_(True)
            new_cfs.append(delta)

        # Replace self.cfs with the new deltas
        self.cfs = new_cfs


    # def round_off_cfs(self, assign=False):
    #     """
    #     Decode each latent delta, round in input space, then re-encode.
    #     If assign==False, returns a list of rounded input vectors (np.ndarray).
    #     If assign==True, updates self.cfs in place to the new deltas and returns None.
    #     """
    #     decoded_list = []
    #     device = self.device

    #     for idx, delta in enumerate(self.cfs):
    #         # 1) decode latent → input
    #         with torch.no_grad():
    #             z = self.rho_s + delta        # latent prototype + delta
    #             x_cf = self.dp_ae.decoder(z.unsqueeze(0)).squeeze(0)  # tensor, on device

    #         # 2) move to CPU / NumPy for rounding
    #         x_np = x_cf.cpu().numpy().copy()

    #         # 2a) continuous rounding
    #         for cont_idx, feat_idxs in enumerate(self.encoded_continuous_feature_indexes):
    #             # feat_idxs may be a single index or a slice
    #             # here it's typically a list of one element
    #             i = feat_idxs if isinstance(feat_idxs, int) else feat_idxs[0]
    #             # un‐normalize → real scale
    #             real = x_np[i] * (self.cont_maxx[cont_idx] - self.cont_minx[cont_idx]) \
    #                 + self.cont_minx[cont_idx]
    #             real = round(real, self.cont_precisions[cont_idx])
    #             # re‐normalize
    #             x_np[i] = (real - self.cont_minx[cont_idx]) / (self.cont_maxx[cont_idx] - self.cont_minx[cont_idx])

    #         # 2b) categorical one‐hot rounding
    #         for cat_group in self.encoded_categorical_feature_indexes:
    #             block = x_np[cat_group[0]:cat_group[-1]+1]
    #             # pick the max index (tieBreak via self.tie_random)
    #             max_idxs = np.flatnonzero(block == block.max())
    #             pick = random.choice(max_idxs) if (len(max_idxs)>1 and self.tie_random) else max_idxs[0]
    #             onehot = np.zeros_like(block)
    #             onehot[pick] = 1.0
    #             x_np[cat_group[0]:cat_group[-1]+1] = onehot

    #         decoded_list.append(x_np)

    #         if assign:
    #             # 3) re-encode the rounded x_np back into new latent z
    #             with torch.no_grad():
    #                 x_tensor = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0)
    #                 z_new   = self.dp_ae.encoder(x_tensor).squeeze(0)
    #             # 4) overwrite deltaᵢ = z_new − ρₛ, keep requires_grad
    #             new_delta = (z_new - self.rho_s).detach().clone().requires_grad_(True)
    #             self.cfs[idx] = new_delta

    #     return None if assign else decoded_list



    def round_off_cfs(self, assign=False):
        """
        Rounds each decoded CF and (optionally) writes back the corresponding latent delta.
        If assign=False, returns (rounded_cfs, valid_idxs).
        If assign=True, updates self.cfs in-place and returns None.
        """
        rounded_cfs = []
        for idx, delta in enumerate(self.cfs):
            with torch.no_grad():
                x_cf = self.dp_ae.decoder((self.rho_s + delta).unsqueeze(0)).squeeze(0)
            x = x_cf.cpu().numpy()

            for i, cols in enumerate(self.encoded_continuous_feature_indexes):
                orig = (x[cols] * (self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                orig = np.round(orig, self.cont_precisions[i])
                x[cols] = (orig - self.cont_minx[i]) / (self.cont_maxx[i] - self.cont_minx[i])

            for cols in self.encoded_categorical_feature_indexes:
                block = x[cols[0] : cols[-1] + 1]
                winners = np.flatnonzero(block == block.max())
                pick = np.random.choice(winners) if (len(winners) > 1 and self.tie_random) else winners[0]
                for j, c in enumerate(cols):
                    x[c] = 1.0 if j == pick else 0.0

            rounded_cfs.append(x)

            if assign:
                tensor_x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    z_new = self.dp_ae.encoder(tensor_x).squeeze(0)
                new_delta = (z_new - self.rho_s).detach().clone().requires_grad_(True)
                self.cfs[idx].data.copy_(new_delta.data)

        if assign:
            return None
        
        valid_idxs = []
        for i, x in enumerate(rounded_cfs):
            p = self.predict_fn(torch.tensor(x, dtype=torch.float32)).item()
            if (self.target_cf_class == 0 and p <= self.stopping_threshold) or \
            (self.target_cf_class == 1 and p >= self.stopping_threshold):
                valid_idxs.append(i)

        return rounded_cfs, valid_idxs


    def get_validity_percentage(self):
        orig_cfs = torch.stack([self.dp_ae.decoder((self.rho_s + delta).unsqueeze(0)).squeeze(0)
                    for delta in self.cfs], dim=0)
        
        cfs_np = orig_cfs.detach().cpu().numpy()
        unique_cfs_np = np.unique(cfs_np, axis=0)
        
        predictions = [self.predict_fn(torch.tensor(cf).float()) for cf in unique_cfs_np]
        
        valid_count = 0
        for pred in predictions:
            if (self.target_cf_class == 0 and pred[0] <= self.stopping_threshold) or \
            (self.target_cf_class == 1 and pred[0] >= self.stopping_threshold):
                valid_count += 1

        validity_percentage = (valid_count / len(self.cfs)) * 100.0
        return validity_percentage

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
                rounded_cfs, _ = self.round_off_cfs(assign=False)    # type: ignore

                if not rounded_cfs:
                    return False
                
                test_preds = [self.predict_fn(torch.tensor(cf, dtype=torch.float32)).item() for cf in rounded_cfs]

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


    def find_counterfactuals(
        self, query_instance, desired_class, optimizer, learning_rate, min_iter,
        max_iter, project_iter, loss_diff_thres, loss_converge_maxiter, verbose,
        init_near_query_instance, tie_random, stopping_threshold, posthoc_sparsity_param,
        posthoc_sparsity_algorithm, limit_steps_ls, perturbation_method: str, **kwargs
    ):
        """Finds counterfactuals by gradient-descent."""
        self._reset_loss_history()
        df_trans = self.model.transformer.transform(query_instance)

        arr = df_trans.to_numpy(dtype=np.float32)
        query_np = arr[0]

        self.x1 = torch.from_numpy(query_np)
        query_instance = query_np

        test_pred = self.predict_fn(self.x1)[0]
        if desired_class == "opposite":
            desired_class = 1.0 - np.round(test_pred)
        self.target_cf_class = torch.tensor(desired_class, dtype=torch.float32)

        self.min_iter = min_iter
        self.max_iter = max_iter
        self.project_iter = project_iter
        self.loss_diff_thres = loss_diff_thres
        self.loss_converge_maxiter = loss_converge_maxiter
        self.loss_converge_iter = 0
        self.converged = False

        self.stopping_threshold = stopping_threshold
        if self.target_cf_class == 0 and self.stopping_threshold > 0.5:
            self.stopping_threshold = 0.25
        elif self.target_cf_class == 1 and self.stopping_threshold < 0.5:
            self.stopping_threshold = 0.75
        original_proximity_weight = self.proximity_weight
        self.proximity_weight = 0.01
        self.tie_random = tie_random

        start_time = timeit.default_timer()
        self.final_cfs = []

        loop_find_CFs = self.total_random_inits if self.total_random_inits > 0 else 1
        self.best_backup_cfs = [0] * max(self.total_CFs, loop_find_CFs)
        self.best_backup_cfs_preds = [0] * max(self.total_CFs, loop_find_CFs)
        self.min_dist_from_threshold = [100] * loop_find_CFs

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
            prev_loss = 0.0
            it = 0

            while not self.stop_loop(iterations, loss_diff):
                it += 1
                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropagate
                loss_value = self.compute_loss(perturbation_method, **kwargs)
                self.loss.backward()

                self._populate_loss_history(
                    it,
                    self.yloss.detach().item(),
                    self.proximity_loss.detach().item(),
                    self.diversity_loss.detach().item() if isinstance(self.diversity_loss, torch.Tensor) else self.diversity_loss,
                    self.robustness_loss.detach().item(),
                    loss_value.detach().item()
                )

                self.optimizer.step()

                for idx, delta in enumerate(self.cfs):
                    x_cf = self.dp_ae.decoder((self.rho_s + delta).unsqueeze(0)).squeeze(0)
                    frozen = set(range(x_cf.numel())) - set(self.feat_to_vary_idxs)
                    for feat in frozen:
                        x_cf[feat] = self.x1[feat]

                    with torch.no_grad():
                        z_new = self.dp_ae.encoder(x_cf.unsqueeze(0)).squeeze(0)
                    new_delta = (z_new - self.rho_s).detach().clone().requires_grad_(True)
                    self.cfs[idx].data.copy_(new_delta.data)

                # projection step on delta's latent values to keep decoded within minx/maxx
                min_x = torch.tensor(self.minx[0], dtype=torch.float32)
                max_x = torch.tensor(self.maxx[0], dtype=torch.float32)

                for ix in range(self.total_CFs):
                    with torch.no_grad():
                        z = self.rho_s + self.cfs[ix]
                        x_cf = self.dp_ae.decoder(z.unsqueeze(0)).squeeze(0)

                    x_cf_clamped = torch.max(torch.min(x_cf, max_x), min_x)

                    with torch.no_grad():
                        z_new = self.dp_ae.encoder(x_cf_clamped.unsqueeze(0)).squeeze(0)
                    
                    new_delta = (z_new - self.rho_s).detach()
                    self.cfs[ix].data.copy_(new_delta)

                if verbose and iterations % 50 == 0:
                    print(f"step {iterations+1}, loss={loss_value:.4g}")

                loss_diff = abs(loss_value - prev_loss)
                prev_loss = loss_value
                iterations += 1
                self.proximity_weight = original_proximity_weight
                # backing up CFs if they are valid
                rounded_cfs_stored, _ = self.round_off_cfs(assign=False)    # type: ignore
                test_preds_stored = [self.predict_fn(torch.tensor(cf, dtype=torch.float32)) for cf in rounded_cfs_stored]

                if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored))
                    or (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    avg_preds_dist = np.mean([abs(pred[0] - self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                        self.min_dist_from_threshold[loop_ix] = avg_preds_dist    # type: ignore
                        for ix in range(self.total_CFs):
                            self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(rounded_cfs_stored[ix])
                            self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])

            # rounding off final cfs
            self.round_off_cfs(assign=True)
            for j in range(self.total_CFs):
                with torch.no_grad():
                    z = self.rho_s + self.cfs[j]
                    x_cf = self.dp_ae.decoder(z.unsqueeze(0)).squeeze(0)
                self.final_cfs.append(x_cf.cpu().numpy())

            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time
        self.cfs_preds = [self.predict_fn(cf) for cf in self.final_cfs]

        any_invalid = (
            (self.target_cf_class == 0 and any(p[0] > self.stopping_threshold for p in self.cfs_preds)) or
            (self.target_cf_class == 1 and any(p[0] < self.stopping_threshold for p in self.cfs_preds))
        )
        if any_invalid:
            restored = []
            restored_preds = []
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for cf_offset in range(self.total_CFs):
                        i = loop_ix + cf_offset
                        restored.append(self.best_backup_cfs[i])
                        restored_preds.append(self.best_backup_cfs_preds[i])
            # Replace final_cfs and cfs_preds with these restored decoded CFs
            self.final_cfs = restored
            self.cfs_preds  = restored_preds

        for tix in range(len(self.final_cfs)):
            arr = np.array(self.final_cfs[tix], dtype=np.float32)
            self.final_cfs[tix] = arr.reshape(1, -1)
            self.cfs_preds[tix]  = np.array(self.cfs_preds[tix], dtype=np.float32).reshape(1, -1)

        cfs = np.vstack([fc[0] for fc in self.final_cfs])
        final_cfs_df = self.model.transformer.inverse_transform(
            self.data_interface.get_decoded_data(cfs)
        )

        # round predictions and attach outcome column
        flat_preds = [float(p[0]) for p in np.vstack(self.cfs_preds)]
        final_cfs_df[self.data_interface.outcome_name] = np.round(flat_preds, 3)

        test_arr = np.array(query_instance, dtype=np.float32).reshape(1, -1)
        test_instance_df = self.model.transformer.inverse_transform(
            self.data_interface.get_decoded_data(test_arr)
        )
        test_instance_df[self.data_interface.outcome_name] = np.round(float(test_pred), 3)

        # post-hoc operation on continuous features to enhance sparsity - only for public data
        if posthoc_sparsity_param is not None and posthoc_sparsity_param > 0 and 'data_df' in self.data_interface.__dict__:
            final_cfs_df_sparse = final_cfs_df.copy()
            final_cfs_df_sparse = self.do_posthoc_sparsity_enhancement(final_cfs_df_sparse,
                                                                       test_instance_df,
                                                                       posthoc_sparsity_param,
                                                                       posthoc_sparsity_algorithm,
                                                                       limit_steps_ls)
        else:
            final_cfs_df_sparse = None

        m, s = divmod(self.elapsed, 60)
        if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in self.cfs_preds))):
            self.total_CFs_found = max(loop_find_CFs, self.total_CFs)
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = []
            for cf_ix, pred in enumerate(self.cfs_preds):
                if ((self.target_cf_class == 0 and pred[0][0] < self.stopping_threshold) or
                   (self.target_cf_class == 1 and pred[0][0] > self.stopping_threshold)):
                    self.total_CFs_found += 1
                    valid_ix.append(cf_ix)

            if self.total_CFs_found == 0:
                print('No Counterfactuals found for the given configuation, ',
                      'perhaps try with different values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d)' % (self.total_CFs_found, max(loop_find_CFs, self.total_CFs)),
                      ' Diverse Counterfactuals found for the given configuation, perhaps try with different',
                      ' values of proximity (or diversity) weights or learning rate...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        if final_cfs_df_sparse is not None:
            final_cfs_df_sparse = final_cfs_df_sparse.iloc[valid_ix].reset_index(drop=True)
        # returning only valid CFs
        return final_cfs_df.iloc[valid_ix].reset_index(drop=True), test_instance_df, final_cfs_df_sparse
