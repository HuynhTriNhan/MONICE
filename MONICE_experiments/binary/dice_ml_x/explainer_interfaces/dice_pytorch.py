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

from dice_ml_x import diverse_counterfactuals as exp
from dice_ml_x.explainer_interfaces.explainer_base import ExplainerBase
from dice_ml_x.perturbation_factory import PerturbationFactory
from dice_ml_x.constants import RobustnessType
from collections import OrderedDict, defaultdict

class DicePyTorch(ExplainerBase):

    def __init__(self, data_interface, model_interface, dice_x: bool):
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
        temp_ohe_data = self.model.transformer.transform(self.data_interface.data_df.iloc[[0]])
        self.data_interface.create_ohe_params(temp_ohe_data)
        self.minx, self.maxx, self.encoded_categorical_feature_indexes, self.encoded_continuous_feature_indexes, \
            self.cont_minx, self.cont_maxx, self.cont_precisions = self.data_interface.get_data_params_for_gradient_dice()
        self.num_output_nodes = self.model.get_num_output_nodes(len(self.data_interface.ohe_encoded_feature_names)).shape[1]
        self.dice_x = dice_x
        # variables required to generate CFs - see generate_counterfactuals() for more info
        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []  # total_CFs, algorithm, features_to_vary
        self.loss_weights = []  # yloss_type, diversity_loss_type, feature_weights
        self.feature_weights_input = ''
        self.hyperparameters = [1, 1, 1]  # proximity_weight, diversity_weight, categorical_penalty
        self.optimizer_weights = []  # optimizer, learning_rate
        self.loss_history = {
            "iterations": [],
            "y_loss": [],
            "proximity_loss": [],
            "diversity_loss": [],
            "regularization_loss": [],
            "robustness_loss": [],
            "total_loss": []
        }
        self.model.model.to('cpu')
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

    def _generate_counterfactuals(self, query_instance, total_CFs,
                                  desired_class="opposite", desired_range=None,
                                  proximity_weight=0.5,
                                  diversity_weight=1.0,
                                  robustness_weight=2.0,
                                  categorical_penalty=0.1, algorithm="DiverseCF", features_to_vary="all",
                                  permitted_range=None, yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", optimizer="pytorch:adam", learning_rate=0.05, min_iter=500,
                                  max_iter=5000, project_iter=0, loss_diff_thres=1e-5, loss_converge_maxiter=1, verbose=False,
                                  init_near_query_instance=True, tie_random=False, stopping_threshold=0.5,
                                  posthoc_sparsity_param=0.1, posthoc_sparsity_algorithm="linear", limit_steps_ls=10000,
                                  perturbation_method="gaussian", preprocessing_bins=10,
                                  robustness_type=RobustnessType.DICE_SORENSEN, separate_features: bool | None=None,
                                  gate_only: bool | None=None, inline_robustness: bool | None=None, **kwargs):
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
        dice_x_params = OrderedDict()
        if self.dice_x:
            dice_x_params = OrderedDict(perturbation_method=perturbation_method, preprocessing_bins=preprocessing_bins,
                                        robustness_type=robustness_type, separate_features=separate_features,
                                        inline_robustness=inline_robustness)
        
        final_cfs_df, test_instance_df, final_cfs_df_sparse = \
            self.find_counterfactuals(
                query_instance, desired_class, optimizer, learning_rate, min_iter, max_iter,
                project_iter, loss_diff_thres, loss_converge_maxiter, verbose, init_near_query_instance,
                tie_random, stopping_threshold, posthoc_sparsity_param, posthoc_sparsity_algorithm, limit_steps_ls,
                **dice_x_params, **kwargs)

        return exp.CounterfactualExamples(
            data_interface=self.data_interface,
            final_cfs_df=final_cfs_df,
            test_instance_df=test_instance_df,
            final_cfs_df_sparse=final_cfs_df_sparse,
            posthoc_sparsity_param=posthoc_sparsity_param,
            desired_class=desired_class)

    """ def get_model_output(self, input_instance,
                         transform_data=False, out_tensor=True):
        return self.model.get_output(
                input_instance,
                transform_data=transform_data,
                out_tensor=out_tensor)[(self.num_output_nodes-1):] """

    """ def predict_fn(self, input_instance, out_tensor=False):
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        return self.get_model_output(
                input_instance, transform_data=False, out_tensor=out_tensor) """

    def get_model_output(self, input_instance, transform_data=False, out_tensor=True):
        out = self.model.get_output(input_instance, transform_data=transform_data, out_tensor=out_tensor)
        if out_tensor and torch.is_tensor(out) and out.ndim == 2 and out.shape[1] == 1:
            out = out.squeeze(1)

        return out

    def predict_fn(self, input_instance, out_tensor=True):
        if not torch.is_tensor(input_instance):
            input_instance = torch.tensor(input_instance).float()
        return self.get_model_output(input_instance, transform_data=False, out_tensor=out_tensor)

    def predict_fn_for_sparsity(self, input_instance):
        """prediction function for sparsity correction"""
        x = self.model.transformer.transform(input_instance).to_numpy().astype(np.float32)
        x_t = torch.from_numpy(x)
        device = next(self.model.model.parameters()).device
        x_t = x_t.to(device)
        pred_t = self.predict_fn(x_t, out_tensor=True)  # returns torch tensor
        pred_t = pred_t.view(-1)                        # (1,) or (n,)
        return pred_t.detach().cpu().numpy()
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
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        for i in range(self.total_CFs):
            if self.yloss_type == "l2_loss":
                temp_loss = torch.pow((self.get_model_output(self.cfs[i]) - self.target_cf_class), 2)[0]
            elif self.yloss_type == "log_loss":
                temp_logits = torch.log(
                    (abs(self.get_model_output(self.cfs[i]) - 0.000001)) /
                    (1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.BCEWithLogitsLoss()
                temp_loss = criterion(temp_logits, torch.tensor([self.target_cf_class]))
            elif self.yloss_type == "hinge_loss":
                temp_logits = torch.log(
                    (abs(self.get_model_output(self.cfs[i]) - 0.000001)) /
                    (1 - abs(self.get_model_output(self.cfs[i]) - 0.000001)))
                criterion = torch.nn.ReLU()
                all_ones = torch.ones_like(self.target_cf_class)
                labels = 2 * self.target_cf_class - all_ones
                temp_loss = all_ones - torch.mul(labels, temp_logits)
                temp_loss = torch.norm(criterion(temp_loss))

            yloss += temp_loss

        return yloss/self.total_CFs

    def compute_dist(self, x_hat, x1):
        """Compute weighted distance between two vectors."""
        return torch.sum(torch.mul((torch.abs(x_hat - x1)), self.feature_weights_list), dim=0)

    def compute_proximity_loss(self):
        """Compute the second part (distance from x1) of the loss function."""
        proximity_loss = 0.0
        for i in range(self.total_CFs):
            proximity_loss += self.compute_dist(self.cfs[i], self.x1)
        return proximity_loss/(torch.mul(len(self.minx[0]), self.total_CFs))

    def dpp_style(self, submethod):
        """Computes the DPP of a matrix."""
        det_entries = torch.ones((self.total_CFs, self.total_CFs))
        if submethod == "inverse_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        elif submethod == "exponential_dist":
            for i in range(self.total_CFs):
                for j in range(self.total_CFs):
                    det_entries[(i, j)] = 1.0/(torch.exp(self.compute_dist(self.cfs[i], self.cfs[j])))
                    if i == j:
                        det_entries[(i, j)] += 0.0001

        diversity_loss = torch.det(det_entries)
        return diversity_loss

    def perturb_cfs(self, method: str = "gaussian",
                    std_dev: float=0.10, mean: float=0.05, flip_prob: float=0.01,
                    cat_alpha: float = 0.2, cat_mode: str = "dirichlet",
                    seed: int | None=None, device: torch.device | None=None,
                    eps: float=1e-8) -> torch.Tensor:
        cfs_stacked = torch.stack(self.cfs, dim=0)

        if device is None:
            device = cfs_stacked.device
        cfs_stacked = cfs_stacked.to(device)

        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])

        g = torch.Generator(device=device)
        g.manual_seed(seed)

        X = cfs_stacked.clone()
        
        cont_cols = self.encoded_continuous_feature_indexes
        if len(cont_cols) > 0:

            ranges_float = self.data_interface.get_features_range_float()[1]
            cont_cols_names = self.data_interface.ohe_encoded_feature_names

            cont_lo = torch.tensor([ranges_float[cont_cols_names[c]][0] for c in cont_cols],
                                   device=device, dtype=torch.float32)
            cont_hi = torch.tensor([ranges_float[cont_cols_names[c]][1] for c in cont_cols],
                                   device=device, dtype=torch.float32)
            span = cont_hi - cont_lo

            X_cont = X[:, cont_cols]

            if method == "gaussian":
                noise = torch.randn(X_cont.shape, generator=g, device=device) * std_dev * span
            elif method == "random":
                u = torch.rand(X_cont.shape, generator=g, device=device) * 2.0 - 1.0
                noise = u * (mean * span)
            elif method == "spherical":
                dir_ = torch.randn(X_cont.shape, generator=g, device=device)
                dir_ = dir_ / (torch.linalg.norm(dir_, dim=1, keepdim=True) + 1e-12)
                r = torch.rand((X_cont.shape[0], 1), generator=g, device=device) * mean
                noise = dir_ * r * span
            else:
                raise ValueError(f"Unsupported method: {method}")

            X_cont = torch.clamp(X_cont + noise, min=cont_lo, max=cont_hi)
            X[:, cont_cols] = X_cont
        if getattr(self, "ohe_groups", None) and len(self.ohe_groups) > 0:
            for group in self.ohe_groups:
                idx = torch.tensor(group, device=device, dtype=torch.long)
                Xg = X.index_select(dim=1, index=idx)

                do_perturb = (torch.rand((X.shape[0], ), generator=g, device=device) < flip_prob)
                if not do_perturb.any():
                    continue

                rows = torch.where(do_perturb)[0]
                Xg_rows = Xg[rows]

                m = Xg_rows.shape[1]

                if cat_mode == "uniform":
                    u = torch.full_like(Xg_rows, 1.0 / float(m))
                elif cat_mode == "dirichlet":
                    u = torch.rand(Xg_rows.shape, generator=g,
                                   device=Xg_rows.device, dtype=Xg_rows.dtype).clamp_min(eps)
                    u = -torch.log(u)
                    u = u / (u.sum(dim=1, keepdim=True) + eps)
                else:
                    raise ValueError(f"Unsupported cat_mode: {cat_mode}")

                alpha = float(cat_alpha)
                Xg_new = (1.0 - alpha) * Xg_rows + alpha * u
                Xg_new = Xg_new / (Xg_new.sum(dim=1, keepdim=True) + eps)
                Xg[rows] = Xg_new
                X[:, idx] = Xg

        return X

        """ if len(self.ohe_groups) > 0:
            for group in self.ohe_groups:
                cat_cols_grp = torch.tensor(group, device=device, dtype=torch.long)
                do_flip = torch.rand((X.shape[0], ), generator=g, device=device) < flip_prob
                if not do_flip.any().item():
                    continue
                row_idx = torch.where(do_flip)[0] 
                k = cat_cols_grp.numel()
                new_cat = torch.randint(0, k, (row_idx.numel(),), generator=g, device=device)
                X[row_idx[:, None], cat_cols_grp] = 0.0
                chosen_cols = cat_cols_grp[new_cat]
                X[row_idx, chosen_cols] = 1.0
        return X """

    def compute_robustness_distance(self, perturbed_cfs: torch.Tensor, preprocessing_bins: int=10,
                                    ) -> torch.Tensor:
        cfs = torch.stack(self.cfs, dim=0)

        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs, num_bins=preprocessing_bins)

        intersection = torch.sum(torch.min(cfs_processed, perturbed_cfs_processed), dim=1)

        union = torch.sum(cfs_processed, dim=1) + torch.sum(perturbed_cfs_processed, dim=1)

        epsilon = 1e-8
        dice = (2 * intersection) / (union + epsilon)
        dice = torch.nan_to_num(dice, nan=1.0)

        return dice

    def compute_robustness_loss_(self, perturbed_cfs: torch.Tensor, preprocessing_bins: int=10,
                                 desired_class="opposite") -> torch.Tensor:
        base_prob = self.predict_fn(self.x1, out_tensor=True).squeeze()
        base_label = (base_prob.detach() > 0.5).float()
        if desired_class == "opposite":
            target_label = 1.0 - base_label
        else:
            target_label = torch.tensor(float(desired_class), device=perturbed_cfs.device)

        dist_vec = self.compute_robustness_distance(perturbed_cfs, preprocessing_bins)
        pert_prob = self.predict_fn(perturbed_cfs, out_tensor=True).squeeze()
        gate = target_label * pert_prob + (1.0 - target_label) * (1.0 - pert_prob)
        return torch.mean(dist_vec * gate)
    
    def compute_robustness_loss_soft(
        self,
        perturbed_cfs: torch.Tensor,
        preprocessing_bins: int = 10,
        desired_class="opposite",
        temperature: float = 1.0,   # optional: sharpen/soften gate if you use logits
        detach_target: bool = True  # optional: keep target decision fixed
    ) -> torch.Tensor:

        # 1) Determine target class WITHOUT breaking graph.
        # If you want target to be fixed (recommended), detach.
        base_score = self.predict_fn(self.x1, out_tensor=True)  # shape: (1,) or (1,1) typically
        base_score = base_score.squeeze()

        if detach_target:
            base_score = base_score.detach()

        # base_prob in [0,1]
        base_prob = base_score

        if desired_class == "opposite":
            # If base_prob > 0.5 => target=0, else target=1
            # Use a *soft* target label in [0,1] instead of hard {0,1}
            # This produces a smooth transition near 0.5.
            target_prob = 1.0 - base_prob
        else:
            # desired_class is 0 or 1
            target_prob = torch.tensor(float(desired_class), device=perturbed_cfs.device)

        # 2) Per-sample similarity/distances (must be vector, not mean)
        dist_vec = self.compute_robustness_distance(perturbed_cfs, preprocessing_bins)
        # dist_vec shape: (N,)  <-- IMPORTANT

        # 3) Soft gate: probability of being in the target class
        pert_scores = self.predict_fn(perturbed_cfs, out_tensor=True).squeeze()  # (N,)

        # If your predict_fn outputs probabilities (sigmoid), use directly:
        # P(y=1|x) = pert_scores
        # P(y=0|x) = 1 - pert_scores
        gate = target_prob * pert_scores + (1.0 - target_prob) * (1.0 - pert_scores)
        # gate in [0,1], differentiable w.r.t. pert_scores

        # 4) Robustness objective (maximize similarity for target-valid perturbations)
        return torch.mean(dist_vec * gate)

    def do_perturbation(self):
        cfs_stacked = torch.stack(self.cfs, dim=0)
        
        #cfs_perturbed = torch.nn.Parameter(cfs_stacked.clone(), requires_grad=True)
        
        if self.encoded_continuous_feature_indexes:
            
            continuous_slice = cfs_stacked[:, self.encoded_continuous_feature_indexes]
            noise = continuous_slice * 0.05
            noise_mask = torch.zeros_like(cfs_stacked)
            noise_mask[:, self.encoded_continuous_feature_indexes] = noise
            cfs_stacked = cfs_stacked + noise_mask

        if self.encoded_categorical_feature_indexes:
            for cat_cols in self.encoded_categorical_feature_indexes:
                cat_slice = cfs_stacked[:, cat_cols]
                sample_size = cat_slice.shape[0]
                num_cats = cat_slice.shape[1]

                rand_idx = torch.randint(low=0, high=num_cats, size=(sample_size, ))

                cat_slice_perturbed = torch.nn.functional.one_hot(rand_idx, num_classes=num_cats).float()

                cat_mask = torch.zeros_like(cfs_stacked)
                cat_mask[:, cat_cols] = cat_slice_perturbed
                cfs_perturbed = cfs_stacked + cat_mask
        cfs_perturbed = torch.nn.Parameter(cfs_stacked.clone(), requires_grad=True)
        return cfs_perturbed

    def generate_perturbations(self, method: str, max_iter=100,
                               tol=1e-3, gamma=1e-2):
        perturbed_cfs = self.do_perturbation()
        perturbation_optimizer = torch.optim.Adam([perturbed_cfs], lr=1e-3)

        prev_loss = np.inf
        for _ in range(max_iter):
            # Remove torch.no_grad() to maintain gradient flow
            self.model.model.eval()
            pred_i = self.model.model(torch.stack(self.cfs, dim=0))
            pred_i_prime = self.model.model(perturbed_cfs)

            class_loss = torch.mean((pred_i - pred_i_prime) ** 2)
            distance = torch.norm(perturbed_cfs - torch.stack(self.cfs, dim=0), p=2)
            loss = class_loss + gamma * distance

            perturbation_optimizer.zero_grad()
            loss.backward()

            perturbation_optimizer.step()
            if abs(loss.item() - prev_loss) < tol:
                break
            prev_loss = loss.item()
        # Return without detaching to maintain gradient flow
        return perturbed_cfs

    def compute_diversity_loss(self):
        """Computes the third part (diversity) of the loss function."""
        if self.total_CFs == 1:
            return torch.tensor(0.0)

        if "dpp" in self.diversity_loss_type:
            submethod = self.diversity_loss_type.split(':')[1]
            return self.dpp_style(submethod)
        elif self.diversity_loss_type == "avg_dist":
            diversity_loss = 0.0
            count = 0.0
            # computing pairwise distance and transforming it to normalized similarity
            for i in range(self.total_CFs):
                for j in range(i+1, self.total_CFs):
                    count += 1.0
                    diversity_loss += 1.0/(1.0 + self.compute_dist(self.cfs[i], self.cfs[j]))

            return 1.0 - (diversity_loss/count)

    def compute_regularization_loss(self):
        """Adds a linear equality constraints to the loss functions -
           to ensure all levels of a categorical variable sums to one"""
        regularization_loss = 0.0
        for i in range(self.total_CFs):
            for v in self.encoded_categorical_feature_indexes:
                regularization_loss += torch.pow((torch.sum(self.cfs[i][v[0]:v[-1]+1]) - 1.0), 2)
        return regularization_loss

    def _preprocess_for_robustness(self, cfs: torch.Tensor, perturbed_cfs: torch.Tensor, num_bins: int=10) -> tuple:
        """
        Conducts preprocessing steps fro robustness loss calculation i.e., converts the given
        dataframes into binarized torch.Tensors
        Args:
            cfs (pandas.DataFrame): The counterfactual instances generated by the algorithm.
            perturbed_cfs (pandas.DataFrame): The perturbed counterfactual instances of cfs.

        Returns:
            tuple: A tuple that contains preprocessed original counterfactual instances and
                preprocessed perturbed instances of type torch.Tensor.
        """
        def preprocess_continuous_features(cfs: torch.Tensor, num_bins=num_bins) -> torch.Tensor:
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

    def _fit_bins(self, X: torch.Tensor, num_bins: int=10) -> torch.Tensor:
        cont_idxs = torch.as_tensor(self.encoded_continuous_feature_indexes, dtype=torch.long, device=X.device)
        cont = X[:, cont_idxs]

        percentiles = torch.linspace(0, 100, num_bins, device=X.device, dtype=X.dtype)
        bin_centers = torch.quantile(cont, percentiles / 100.0, dim=0, interpolation='linear')
        return bin_centers.T

    def _phi_soft(self, X: torch.Tensor, num_bins: int=10, bin_centers: torch.Tensor=None,
                  sigma: float=0.1, eps: float=1e-8, tau: float=1.0) -> torch.Tensor:
        """
        Soft, differentiable feature map for robustness:
            - Continuous -> soft RBF bins in [0,1] for smooth, differentiable comparison
            - Categorical one-hot groups -> kept as-is (already differentiable during optimization)
        Returns: (N, D') tensor suitable for soft-Dice.
        Assumes X is already in the *model's internal normalized space* (e.g., [0,1] for cont).

        Note: One-hot encoded categorical features are NOT passed through softmax, as this would
        make all categories artificially similar. During gradient descent, one-hot vectors
        naturally become "soft" (e.g., [0.9, 0.05, 0.05]), which preserves distinctiveness.
        """
        parts = []

        if getattr(self, "encoded_continuous_feature_indexes", None):
            idx = torch.as_tensor(self.encoded_continuous_feature_indexes, dtype=torch.long, device=X.device)
            cont = X[:, idx]

            cont = cont.clamp(0.0, 1.0)
            if bin_centers is not None:
                centers = bin_centers.to(X.device)
                cont_exp = cont.unsqueeze(-1)
                centers_exp = centers.unsqueeze(0)
                rbf = torch.exp(-(cont_exp - centers_exp) ** 2 / (2.0 * sigma **2))
            else:
                centers = torch.linspace(0.0, 1.0, num_bins, device=X.device, dtype=X.dtype)
                cont_exp = cont.unsqueeze(-1)
                rbf = torch.exp(-(cont_exp - centers) ** 2 / (2.0 * sigma ** 2))

            rbf = rbf / (rbf.sum(dim=1, keepdim=True) + eps)
            parts.append(rbf.reshape(cont.size(0), -1))

        if getattr(self, "encoded_categorical_feature_indexes", None):
            for grp in self.encoded_categorical_feature_indexes:
                grp_idx = torch.as_tensor(grp, dtype=torch.long, device=X.device)
                g = X[:, grp_idx]
                g_max = g.max(dim=1, keepdim=True)[0]
                g_shifted = g - g_max
                exp_g = torch.exp(g_shifted)
                sm = exp_g / (exp_g.sum(dim=1, keepdim=True) + eps)
                parts.append(sm)
        return torch.cat(parts, dim=1) if parts else X

    def compute_robustness_loss_binned_RBF(self, perturbed_cfs: torch.Tensor, num_bins: int=10,
                                           sigma: float=0.1, eps: float=1e-8) -> torch.Tensor:
        """
        Soft-Dice robustness similarity in [0,1]; higher is better.
        Returns a scalar (mean across CFs) to plug into your total loss with +λ3 * robustness.
        """
        cfs = torch.stack(self.cfs, dim=0)
        if cfs.dim() == 3 and cfs.size(1) == 1:
            cfs = cfs.squeeze(dim=1)

        if perturbed_cfs.dim() == 3 and perturbed_cfs.size(1) == 1:
            perturbed_cfs = perturbed_cfs.squeeze(dim=1)

        all_cfs = torch.cat([cfs, perturbed_cfs], dim=0)
        bin_centers = self._fit_bins(all_cfs, num_bins=num_bins)

        p = self._phi_soft(cfs, bin_centers=bin_centers, num_bins=num_bins)
        q = self._phi_soft(perturbed_cfs, bin_centers=bin_centers, num_bins=num_bins)

        squared_distance = torch.dist(p, q, p=2) ** 2
        kernel_similarity = torch.exp(-squared_distance / (2 * sigma ** 2))
        return kernel_similarity

    def compute_robustness_distance_RBF(self, perturbed_cfs: torch.Tensor, sigma: float | torch.Tensor | None=None,
                                    gamma: float | None=None, separate_features: bool | None=None,
                                    atol: float=1e-6, eps: float=1e-8) -> torch.Tensor:
        cfs = torch.stack(self.cfs, dim=0)
        if cfs.dim() == 3 and cfs.size(1) == 1:
            cfs = cfs.squeeze(dim=1)

        if perturbed_cfs.dim() == 3 and perturbed_cfs.size(1) == 1:
            perturbed_cfs = perturbed_cfs.squeeze(dim=1)

        assert cfs.shape == perturbed_cfs.shape, f"Shape mismatch: {cfs.shape} vs {perturbed_cfs.shape}"
        k, _ = cfs.shape

        device = cfs.device
        dtype = cfs.dtype

        def rbf_similarity(X: torch.Tensor, Y: torch.Tensor, sig: float | torch.Tensor | None) -> torch.Tensor:
            diff = X - Y

            if sig is None:
                stacked = torch.cat([X, Y], dim=0)
                sig = torch.std(stacked, dim=0) + eps

            if isinstance(sig, (float, int)) or (torch.is_tensor(sig) and sig.ndim == 0):
                s = torch.as_tensor(sig, device=device, dtype=dtype)
                sq = (diff * diff).sum(dim=1)  # (k,)
                return torch.exp(-sq / (2.0 * s * s))
            else:
                s = sig.to(device=device, dtype=dtype)
                sq = (diff * diff) / (2.0 * (s * s))
                return torch.exp(-sq.sum(dim=1))

        if not separate_features:
            sim = rbf_similarity(cfs, perturbed_cfs, sigma)
            return sim

        cont_cols = list(self.encoded_continuous_feature_indexes)
        ohe_groups = getattr(self, "ohe_groups", None)

        if len(cont_cols) > 0:
            X_cont = cfs[:, cont_cols]
            Y_cont = perturbed_cfs[:, cont_cols]
            sim_cont = rbf_similarity(X_cont, Y_cont, sigma)
        else:
            sim_cont = torch.ones(k, device=device, dtype=dtype)

        if ohe_groups is not None and len(ohe_groups) > 0:
            group_dists = []
            for g in ohe_groups:
                idx = torch.as_tensor(g, device=device)
                Xg = cfs.index_select(dim=1, index=idx)
                Yg = perturbed_cfs.index_select(dim=1, index=idx)
                overlap = (Xg * Yg).sum(dim=1)
                dist_g = 1.0 - overlap
                group_dists.append(dist_g)
            ham = torch.stack(group_dists, dim=1).mean(dim=1)
        else:
            cat_cols = list(self.encoded_categorical_feature_indexes)
            if len(cat_cols) > 0:
                X_cat = cfs[:, cat_cols]
                Y_cat = perturbed_cfs[:, cat_cols]
                neq = ~torch.isclose(X_cat, Y_cat, atol=atol)
                ham = neq.to(dtype).mean(dim=1)
            else:
                ham = torch.zeros(k, device=device, dtype=dtype)

        if gamma is None:
            gamma = 0.2
        gamma_t = torch.as_tensor(max(float(gamma), 1e-6), device=device, dtype=dtype)

        sim_cat = torch.exp(-ham / gamma_t)

        sim = sim_cont * sim_cat
        return sim

    def compute_robustness_loss_RBF(self, perturbed_cfs: torch.Tensor, sigma: float | torch.Tensor | None=None,
                                    gamma: float | None=None, separate_features: bool | None=None,
                                    atol: float=1e-6, eps: float=1e-8, desired_class="opposite",
                                    gate_only: bool | None=True) -> torch.Tensor:
        base_prob = self.predict_fn(self.x1, out_tensor=True).squeeze()
        base_label = (base_prob.detach() > 0.5).float()
        if desired_class == "opposite":
            target_label = 1.0 - base_label
        else:
            target_label = torch.tensor(float(desired_class), device=perturbed_cfs.device)

        dist_vec = self.compute_robustness_distance_RBF(perturbed_cfs, sigma, gamma, separate_features,
                                                        atol, eps)
        pert_prob = self.predict_fn(perturbed_cfs, out_tensor=True).squeeze()
        gate = target_label * pert_prob + (1.0 - target_label) * (1.0 - pert_prob)
        print("dist_mean", dist_vec.mean().item(), "gate_mean", gate.mean().item(), "rob_mean", (dist_vec*gate).mean().item())
        sim = torch.mean(gate) if gate_only else torch.mean(dist_vec * gate)
        return sim

    def compute_robustness_loss(self, perturbed_cfs: torch.Tensor, preprocessing_bins: int=10) -> torch.Tensor:
        """
        Computes the robustness loss.

        Args:
            pertubed_cfs_df (pandas.DataFrame): The perturbed counterfactuals that will
            be compared against the original counterfactual instances.
        Returns:
            float: Robustness loss in scalar.
        """
        cfs = torch.stack(self.cfs, dim=0)

        # from IPython.display import display
        # print("--------------------------------------------------------Original instances--------------------------------------------------------")
        # display(cfs_df)
        # print("--------------------------------------------------------Perturbed instances--------------------------------------------------------")
        # display(perturbed_cfs_df)

        cfs_processed, perturbed_cfs_processed = self._preprocess_for_robustness(cfs, perturbed_cfs, num_bins=preprocessing_bins)

        intersection = torch.sum(torch.min(cfs_processed, perturbed_cfs_processed), dim=1)

        union = torch.sum(cfs_processed, dim=1) + torch.sum(perturbed_cfs_processed, dim=1)

        epsilon = 1e-8
        sorensen_dice_coefficient = (2 * intersection) / (union + epsilon)
        sorensen_dice_coefficient[torch.isnan(sorensen_dice_coefficient)] = 1.0
        #test_grad = torch.autograd.grad(sorensen_dice_coefficient.mean(), self.cfs, retain_graph=True)
        #print("Gradients:", test_grad)
        return sorensen_dice_coefficient.mean()

    def compute_loss(self, method: str, desired_class, preprocessing_bins: int=10,
                     robustness_type: RobustnessType=RobustnessType.DICE_SORENSEN,
                     separate_features: bool | None=None, gate_only: bool | None=None, inline_robustness: bool | None=None, **kwargs):
        """Computes the overall loss"""
        if robustness_type != RobustnessType.GAUSSIAN_KERNEL and separate_features is not None:
            raise ValueError("`separate_features` can only be used when `robustness_type == RobustnessType.GAUSSIAN_KERNEL`.")
        if robustness_type != RobustnessType.GAUSSIAN_KERNEL and gate_only is not None:
            raise ValueError("`gate_only` can only be used when `robustness_type == RobustnessType.GAUSSIAN_KERNEL`.")
        self.yloss = self.compute_yloss()
        self.proximity_loss = self.compute_proximity_loss() if self.proximity_weight > 0 else 0.0
        self.diversity_loss = self.compute_diversity_loss() if self.diversity_weight > 0 else 0.0
        self.regularization_loss = self.compute_regularization_loss()
        perturbed_cfs = self.generate_perturbations(method)
        if inline_robustness:
            if robustness_type is not RobustnessType.DICE_SORENSEN:
                perturbed_cfs = self.perturb_cfs(method=method, std_dev=kwargs.get("std_dev", 0.1), seed=42,
                                                mean=kwargs.get("mean", 0.05), flip_prob=kwargs.get("flip_prob", 0.5),
                                                cat_alpha=kwargs.get("cat_alpha", 0.7), cat_mode=kwargs.get("cat_mode", "dirichlet"))
            # perturbed_cfs = self.generate_perturbations("gaussian")
            if robustness_type == RobustnessType.DICE_SORENSEN:
                self.robustness_loss = self.compute_robustness_loss(perturbed_cfs)
            elif robustness_type == RobustnessType.GATED_DICE_SORENSEN:
                self.robustness_loss = self.compute_robustness_loss_(perturbed_cfs=perturbed_cfs, desired_class=desired_class,
                                                                    preprocessing_bins=preprocessing_bins) if self.robustness_weight > 0 else 0.0
            elif robustness_type == RobustnessType.GAUSSIAN_KERNEL:
                if gate_only == False:
                    perturbed_cfs = self.perturb_cfs(method=method)
                self.robustness_loss = self.compute_robustness_loss_RBF(perturbed_cfs=perturbed_cfs, desired_class=desired_class,
                                                                        gamma=1.0, separate_features=separate_features, gate_only=gate_only) \
                                                                        if self.robustness_weight > 0 else 0.0
            elif robustness_type == RobustnessType.BINNED_GAUSSIAN_KERNEL:
                self.robustness_loss = self.compute_robustness_loss_binned_RBF(perturbed_cfs=perturbed_cfs, num_bins=preprocessing_bins) if self.robustness_weight > 0 else 0.0
            else:
                raise ValueError("Unsupported method. Supported types: Dice-Sorensen, Gaussian Kernel, and Binned Gaussian Kernel")
        else:
            self.robustness_loss = 0.0

        self.loss = self.yloss + (self.proximity_weight * self.proximity_loss) - \
            (self.diversity_weight * self.diversity_loss) - \
            (self.robustness_weight * self.robustness_loss) + \
            (self.categorical_penalty * self.regularization_loss)
        return self.loss

    def initialize_CFs(self, query_instance, init_near_query_instance=False):
        """Initialize counterfactuals."""
        for n in range(self.total_CFs):
            for i in range(len(self.minx[0])):
                if i in self.feat_to_vary_idxs:
                    if init_near_query_instance:
                        self.cfs[n].data[i] = float(query_instance[i]+(n*0.01))
                    else:
                        self.cfs[n].data[i] = float(np.random.uniform(self.minx[0][i], self.maxx[0][i]))
                else:
                    self.cfs[n].data[i] = query_instance[i]

    def round_off_cfs(self, assign=False):
        """function for intermediate projection of CFs."""
        temp_cfs = []
        for index, tcf in enumerate(self.cfs):
            cf = tcf.detach().clone().numpy()
            for i, v in enumerate(self.encoded_continuous_feature_indexes):
                # continuous feature in orginal scale
                org_cont = (cf[v]*(self.cont_maxx[i] - self.cont_minx[i])) + self.cont_minx[i]
                org_cont = round(org_cont, self.cont_precisions[i])  # rounding off
                normalized_cont = (org_cont - self.cont_minx[i])/(self.cont_maxx[i] - self.cont_minx[i])
                cf[v] = normalized_cont  # assign the projected continuous value

            for v in self.encoded_categorical_feature_indexes:
                maxs = np.argwhere(
                    cf[v[0]:v[-1]+1] == np.amax(cf[v[0]:v[-1]+1])).flatten().tolist()
                if len(maxs) > 1:
                    if self.tie_random:
                        ix = random.choice(maxs)
                    else:
                        ix = maxs[0]
                else:
                    ix = maxs[0]
                for vi in range(len(v)):
                    if vi == ix:
                        cf[v[vi]] = 1.0
                    else:
                        cf[v[vi]] = 0.0

            temp_cfs.append(cf)
            if assign:
                for jx in range(len(cf)):
                    self.cfs[index].data[jx] = torch.tensor(temp_cfs[index])[jx]

        if assign:
            return None
        else:
            return temp_cfs

    def get_validity_percentage(self):
        cfs_np = np.array([cf.detach().cpu().numpy() for cf in self.cfs])
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
                temp_cfs = self.round_off_cfs(assign=False)
                test_preds = [self.predict_fn(cf)[0] for cf in temp_cfs]

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
                             posthoc_sparsity_algorithm, limit_steps_ls, perturbation_method: str=None,
                             preprocessing_bins: int=10,
                             robustness_type: RobustnessType=RobustnessType.DICE_SORENSEN,
                             separate_features: bool | None=None, gate_only: bool | None=None, inline_robustness: bool | None=None,
                             **kwargs):
        """Finds counterfactuals by gradient-descent."""

        self._reset_loss_history()
        query_instance_np = self.model.transformer.transform(
        query_instance).to_numpy(dtype=np.float32)[0]

        self.x1 = torch.from_numpy(query_instance_np)
        # find the predicted value of query_instance
        test_pred = self.predict_fn(torch.tensor(query_instance_np).float())[0]
        if desired_class == "opposite":
            desired_class = 1.0 - torch.round(test_pred)
        self.target_cf_class = torch.tensor(desired_class).float()

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
                self.initialize_CFs(query_instance_np, False)
            else:
                self.initialize_CFs(query_instance_np, init_near_query_instance)

            # initialize optimizer
            self.do_optimizer_initializations(optimizer, learning_rate)

            iterations = 0
            loss_diff = 1.0
            prev_loss = 0.0
            it = 0

            while self.stop_loop(iterations, loss_diff) is False:
                it += 1
                # zero all existing gradients
                self.optimizer.zero_grad()
                self.model.model.zero_grad()

                # get loss and backpropogate
                loss_value = self.compute_loss(perturbation_method, desired_class, preprocessing_bins, robustness_type,
                                               separate_features=separate_features, gate_only=gate_only, inline_robustness=inline_robustness,
                                               **kwargs)
                self.loss.backward()

                self._populate_loss_history(it, self.yloss.detach().item() if isinstance(self.yloss, torch.Tensor) else self.yloss,
                                            self.proximity_loss.detach().item() if isinstance(self.proximity_loss, torch.Tensor) else self.proximity_loss,
                                            self.diversity_loss.detach().item() if isinstance(self.diversity_loss,
                                                                                              torch.Tensor) else self.diversity_loss,
                                            self.robustness_loss.detach().item() if isinstance(self.robustness_loss, torch.Tensor) else self.robustness_loss, loss_value.detach().item())

                # freeze features other than feat_to_vary_idxs
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        if jx not in self.feat_to_vary_idxs:
                            self.cfs[ix].grad[jx] = 0.0

                # update the variables
                self.optimizer.step()

                # projection step
                for ix in range(self.total_CFs):
                    for jx in range(len(self.minx[0])):
                        self.cfs[ix].data[jx] = torch.clamp(self.cfs[ix][jx], min=self.minx[0][jx], max=self.maxx[0][jx])
                
                if verbose:
                    if (iterations) % 50 == 0:
                        print('step %d,  loss=%g' % (iterations+1, loss_value))

                loss_diff = abs(loss_value-prev_loss)
                prev_loss = loss_value
                iterations += 1

                # backing up CFs if they are valid
                temp_cfs_stored = self.round_off_cfs(assign=False)
                with torch.no_grad():
                    test_preds_stored = [self.predict_fn(cf).detach() for cf in temp_cfs_stored]
                    #test_preds_stored = [self.predict_fn(cf) for cf in temp_cfs_stored]

                if ((self.target_cf_class == 0 and all(i <= self.stopping_threshold for i in test_preds_stored)) or
                   (self.target_cf_class == 1 and all(i >= self.stopping_threshold for i in test_preds_stored))):
                    with torch.no_grad():
                        avg_preds_dist = torch.mean(
                            torch.stack([torch.abs(pred[0] - self.stopping_threshold) for pred in test_preds_stored])
                        ).item()
                        #avg_preds_dist = np.mean([abs(pred[0]-self.stopping_threshold) for pred in test_preds_stored])
                    if avg_preds_dist < self.min_dist_from_threshold[loop_ix]:
                        self.min_dist_from_threshold[loop_ix] = avg_preds_dist
                        for ix in range(self.total_CFs):
                            # self.best_backup_cfs_preds[loop_ix+ix] = test_preds_stored[ix].detach().clone()
                            # self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])
                            self.best_backup_cfs[loop_ix+ix] = copy.deepcopy(temp_cfs_stored[ix])
                            self.best_backup_cfs_preds[loop_ix+ix] = copy.deepcopy(test_preds_stored[ix])

            # rounding off final cfs - not necessary when inter_project=True
            self.round_off_cfs(assign=True)

            # storing final CFs
            for j in range(0, self.total_CFs):
                temp = self.cfs[j].detach().clone().numpy()
                self.final_cfs.append(temp)

            # max iterations at which GD stopped
            self.max_iterations_run = iterations

        self.elapsed = timeit.default_timer() - start_time

        self.cfs_preds = [self.predict_fn(cfs) for cfs in self.final_cfs]

        # update final_cfs from backed up CFs if valid CFs are not found
        if ((self.target_cf_class == 0 and any(i[0] > self.stopping_threshold for i in self.cfs_preds)) or
           (self.target_cf_class == 1 and any(i[0] < self.stopping_threshold for i in self.cfs_preds))):
            for loop_ix in range(loop_find_CFs):
                if self.min_dist_from_threshold[loop_ix] != 100:
                    for ix in range(self.total_CFs):
                        self.final_cfs[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs[loop_ix+ix])
                        self.cfs_preds[loop_ix+ix] = copy.deepcopy(self.best_backup_cfs_preds[loop_ix+ix])

        # convert to the format that is consistent with dice_tensorflow
        query_instance = np.array([query_instance_np], dtype=np.float32)
        for tix in range(max(loop_find_CFs, self.total_CFs)):
            self.final_cfs[tix] = np.array([self.final_cfs[tix]], dtype=np.float32)
            p = self.cfs_preds[tix]
            if torch.is_tensor(p):
                p = p.detach().cpu().numpy()
            self.cfs_preds[tix] = np.array([p], dtype=np.float32)
            #self.cfs_preds[tix] = np.array([self.cfs_preds[tix]], dtype=np.float32)

            # if self.final_cfs_sparse is not None:
            #     self.final_cfs_sparse[tix] = np.array([self.final_cfs_sparse[tix]], dtype=np.float32)
            #     self.cfs_preds_sparse[tix] = np.array([self.cfs_preds_sparse[tix]], dtype=np.float32)
            #
            if isinstance(self.best_backup_cfs[0], np.ndarray):  # checking if CFs are backed
                self.best_backup_cfs[tix] = np.array([self.best_backup_cfs[tix]], dtype=np.float32)
                self.best_backup_cfs_preds[tix] = np.array([self.best_backup_cfs_preds[tix]], dtype=np.float32)

        # do inverse transform of CFs to original user-fed format
        cfs = np.array([self.final_cfs[i][0] for i in range(len(self.final_cfs))])
        final_cfs_df = self.model.transformer.inverse_transform(
                self.data_interface.get_decoded_data(cfs))
        # rounding off to 3 decimal places
        cfs_preds = [np.round(preds.flatten().tolist(), 3) for preds in self.cfs_preds]
        cfs_preds = [item for sublist in cfs_preds for item in sublist]
        final_cfs_df[self.data_interface.outcome_name] = np.array(cfs_preds)

        query_instance_2d = np.array([query_instance_np])

        test_instance_df = self.model.transformer.inverse_transform(
                self.data_interface.get_decoded_data(query_instance_2d))
        tp = test_pred.detach().cpu().view(-1).numpy()
        test_instance_df[self.data_interface.outcome_name] = np.round(tp, 3)
        #test_instance_df[self.data_interface.outcome_name] = np.array(np.round(test_pred, 3))

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
            valid_ix = [ix for ix in range(max(loop_find_CFs, self.total_CFs))]  # indexes of valid CFs
            print('Diverse Counterfactuals found! total time taken: %02d' %
                  m, 'min %02d' % s, 'sec')
        else:
            self.total_CFs_found = 0
            valid_ix = []  # indexes of valid CFs
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
