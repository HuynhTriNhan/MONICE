"""
NICE (Nearest Instance Counterfactual Explanations) Implementation
This file contains a standalone implementation of the NICE algorithm for generating counterfactual explanations.
Source: https://github.com/DBrughmans/NICE
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import mode
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Optional

# =============================================================================
# Abstract Base Classes
# =============================================================================

class NumericDistance(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def measure(self):
        pass


class DistanceMetric(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def measure(self):
        pass


class RewardFunction(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_reward(self):
        pass


class optimization(ABC):
    @abstractmethod
    def optimize(self):
        pass

# =============================================================================
# Preprocessing
# =============================================================================

class OHE_minmax:
    def __init__(self,cat_feat,con_feat):
        self.cat_feat = cat_feat
        self.con_feat = con_feat

    def fit(self,X):
        if self.cat_feat != []:
            # self.OHE = OneHotEncoder(handle_unknown='ignore',sparse= False) due to sklearn version
            self.OHE = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
            self.OHE.fit(X[:,self.cat_feat])
            self.nb_OHE = self.OHE.transform(X[0:1,self.cat_feat]).shape[1]
        if self.con_feat != []:
            self.minmax = MinMaxScaler(feature_range=(-1, 1))
            self.minmax.fit(X[:,self.con_feat])

    def transform(self,X):
        if self.cat_feat == []:
            return self.minmax.transform(X[:, self.con_feat])
        elif self.con_feat == []:
            return self.OHE.transform(X[:,self.cat_feat])
        else:
            X_minmax = self.minmax.transform(X[:,self.con_feat])
            X_ohe = self.OHE.transform(X[:,self.cat_feat])
            return np.c_[X_ohe,X_minmax]

    def inverse_transform(self,X):
        if self.cat_feat == []:
            return self.minmax.inverse_transform(X)
        elif self.con_feat == []:
            return self.OHE.inverse_transform(X[:,:self.nb_OHE])
        else:
            X_con = self.minmax.inverse_transform(X[:,self.nb_OHE:])
            X_cat = self.OHE.inverse_transform(X[:,:self.nb_OHE])
            return np.c_[X_cat,X_con]

# =============================================================================
# Data Classes
# =============================================================================

class data_NICE:
    def __init__(self,X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,eps):
        self.X_train = X_train
        self.y_train = y_train
        self.cat_feat = cat_feat
        self.num_feat = num_feat
        self.predict_fn = predict_fn
        self.justified_cf = justified_cf
        self.eps = eps

        if self.num_feat == 'auto':
            self.num_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]

        self.X_train = self.num_as_float(self.X_train)

        self.train_proba = predict_fn(X_train)
        self.n_classes = self.train_proba.shape[1]
        self.X_train_class = np.argmax(self.train_proba, axis=1)

        if self.justified_cf:
            self.candidates_mask = self.y_train == self.X_train_class
        else:
            self.candidates_mask = np.ones(self.X_train.shape[0],dtype=bool)




    def num_as_float(self,X:np.ndarray)->np.ndarray:
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    def fit_to_X(self,X,target_class):
        self.X = self.num_as_float(X)
        self.X_score = self.predict_fn(self.X)
        self.X_class =  self.X_score.argmax()
        if target_class == 'other':
            self.target_class = [i for i in range(self.n_classes) if (i != self.X_class)]
        else:
            self.target_class = target_class
        self.class_mask = np.array([i in self.target_class for i in self.X_train_class])#todo check if this is correct for muliticlass
        self.mask = self.class_mask&self.candidates_mask
        self.candidates_view = self.X_train[self.mask,:].view()

class data_SEDC:
    def __init__(self,X_train,predict_fn,cat_feat,num_feat):
        self.X_train =X_train
        self.predict_fn = predict_fn
        self.cat_feat = cat_feat
        self.num_feat = num_feat

        if self.num_feat == 'auto':
            self.num_feat = [feat for feat in range(self.X_train.shape[1]) if feat not in self.cat_feat]
        self.X_train = self.num_as_float(self.X_train)

    def num_as_float(self, X):
        X[:, self.num_feat] = X[:, self.num_feat].astype(np.float64)
        return X

    def fit(self):
        self.replace_values = np.zeros(self.X_train.shape[1])
        self.replace_values[self.cat_feat] = mode(self.X_train[:, self.cat_feat],axis=0,nan_policy='omit')[0]
        self.replace_values[self.num_feat] = self.X_train[:, self.num_feat].mean(axis = 0)
        self.replace_values = self.replace_values[np.newaxis,:]

    def fit_to_X(self,X,target_class):
        self.X=X
        self.X_score = self.predict_fn(self.X)
        self.X_class = self.X_score.argmax()
        self.n_classes = self.X_score.shape[1]
        if target_class == 'other':
            self.target_class = [i for i in range(self.n_classes) if (i != self.X_class)]
        else:
            self.target_class = target_class

# =============================================================================
# Distance Metrics
# =============================================================================

class StandardDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:,num_feat].std(axis=0, dtype=np.float64)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance

class MinMaxDistance(NumericDistance):
    def __init__(self,X_train:np.ndarray,num_feat:list,eps):
        self.num_feat = num_feat
        self.scale = X_train[:, num_feat].max(axis=0) - X_train[:, num_feat].min(axis=0)
        self.scale[self.scale < eps] = eps
    def measure(self,X1,X2):
        distance = X2[:,self.num_feat].copy()
        distance = abs(distance - X1[0, self.num_feat]) / self.scale
        distance = np.sum(distance,axis=1)
        return distance


class HEOM(DistanceMetric):
    def __init__(self, data, numeric_distance:NumericDistance):
        self.data = data
        self.numeric_distance = numeric_distance(data.X_train,data.num_feat,data.eps)
    def measure(self,X1,X2):
        num_distance = self.numeric_distance.measure(X1,X2)
        cat_distance = np.sum(X2[:, self.data.cat_feat] != X1[0, self.data.cat_feat],axis=1)
        distance = num_distance + cat_distance
        return distance

class NearestNeighbour:
    def __init__(self,data,distance_metric:DistanceMetric):
        self.data = data
        self.distance_metric = distance_metric

    def find_neighbour(self,X):
        distances = self.distance_metric.measure(X,self.data.candidates_view)
        min_idx = distances.argmin()
        return self.data.candidates_view[min_idx, :].copy()[np.newaxis, :]

# =============================================================================
# Reward Functions
# =============================================================================

class SparsityReward(RewardFunction):

    def __init__(self,data,**kwargs):
        self.data = data
        pass

    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = -self.data.predict_fn(previous_CF_candidate) + self.data.predict_fn(X_prune)
        score_diff = score_prune[:, self.data.target_class]# - score_prune[:, self.data.X_class][:,np.newaxis] #multiclas
        score_diff = score_diff.max(axis = 1)
        idx_max = np.argmax(score_diff)
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate


class ProximityReward(RewardFunction):
    def __init__(self,data,distance_metric:DistanceMetric,**kwargs):
        self.data = data
        self.distance_metric = distance_metric
    def calculate_reward(self,X_prune, previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]
        distance = self.distance_metric.measure(self.data.X,X_prune)\
                   -self.distance_metric.measure(self.data.X, previous_CF_candidate)
        idx_max = np.argmax(score_diff / (distance + self.data.eps)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate

class PlausibilityReward(RewardFunction):
    def __init__(self,data, auto_encoder, **kwargs):
        self.data = data
        self.auto_encoder = auto_encoder

    def calculate_reward(self,X_prune,previous_CF_candidate):
        score_prune = self.data.predict_fn(X_prune)
        score_diff = self.data.predict_fn(X_prune)[:, self.data.target_class]\
                     - self.data.predict_fn(previous_CF_candidate)[:, self.data.target_class]#target_class for multiclass
        AE_loss_diff = self.auto_encoder(previous_CF_candidate)-self.auto_encoder(X_prune)
        idx_max = np.argmax(score_diff * (AE_loss_diff)[:,np.newaxis])
        CF_candidate = X_prune[idx_max:idx_max+1, :]
        return CF_candidate
# =============================================================================
# Optimization
# =============================================================================

class best_first(optimization):
    def __init__(self,data,reward_function:RewardFunction):
        self.reward_function = reward_function
        self.data = data

    def optimize(self,NN):
        CF_candidate = self.data.X.copy()
        stop = False
        while stop == False:
            diff = np.where(CF_candidate != NN)[1]
            X_prune = np.tile(CF_candidate, (len(diff), 1))
            for r, c in enumerate(diff):
                X_prune[r, c] = NN[0, c]
            CF_candidate = self.reward_function.calculate_reward(X_prune,CF_candidate)
            if self.data.predict_fn(CF_candidate).argmax() in self.data.target_class:
                return CF_candidate

# =============================================================================
# Main NICE Class
# =============================================================================

CRITERIA_DIS = {'HEOM':HEOM}
CRITERIA_NRM = {'std':StandardDistance,
                'minmax':MinMaxDistance}
CRITERIA_REW = {'sparsity':SparsityReward,
                'proximity':ProximityReward,
                'plausibility':PlausibilityReward}

class NICE:
    def __init__(
            self,
            predict_fn,
            X_train:np.ndarray,
            cat_feat:list,
            num_feat ='auto',
            y_train: Optional[np.ndarray]=None,
            optimization='sparsity',
            justified_cf:bool = True,
            distance_metric:str ='HEOM',
            num_normalization:str = 'minmax',
            auto_encoder = None):

        self.optimization = optimization
        self.data = data_NICE(X_train,y_train,cat_feat,num_feat,predict_fn,justified_cf,0.00000000001)
        self.distance_metric = CRITERIA_DIS[distance_metric](self.data, CRITERIA_NRM[num_normalization])
        self.nearest_neighbour = NearestNeighbour(self.data, self.distance_metric)
        if optimization != 'none':
            self.reward_function = CRITERIA_REW[optimization](
                self.data,
                distance_metric = self.distance_metric,
                auto_encoder= auto_encoder
            )
            self.optimizer = best_first(self.data,self.reward_function)


    def explain(self,X,target_class ='other'):#todo target class 'other'
        self.data.fit_to_X(X,target_class)
        NN = self.nearest_neighbour.find_neighbour(self.data.X)
        if self.optimization != 'none':
            CF = self.optimizer.optimize(NN)
            return CF
        return NN


