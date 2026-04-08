from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import dice_ml
import dice_ml_x
from nice import NICE
from scipy import stats
from monice import MONICE, AutoencoderPlausibility
from time import time
from .nice_helpers import AutoEncoderWrapper
from moc_python.moc_counterfactuals import MOCCounterfactuals
import lime
import lime.lime_tabular

class CounterfactualWrapper(ABC):
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def explain(self):
        pass
        
class MoniceWrapper(CounterfactualWrapper):
    def __init__(self, dataset, model,  **kwargs):
        predict_fn = lambda x: model.predict_proba(x)
        autoencoder_model = kwargs.get("autoencoder_model", None)
        # Wrap the autoencoder to handle output shape issues
        plausibility_model = AutoencoderPlausibility(autoencoder_model)
        self.explainer =  MONICE(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            predict_fn=predict_fn,
            plausibility_model=plausibility_model,
            cat_feats=dataset.categorical_indices,
            num_feats=dataset.continuous_indices,
            integer_feats=None,
            cost_weights=None,
            immutable_features=None,
            feature_bounds=None,
            monotonic_constraints=None,
            distance_metric='gower',
            justified_cf=True,
            eps=1e-6,
            verbose=False
        )
    @abstractmethod
    def explain(self,x):
        pass

class MoniceMultiObjectiveHEOMWrapper(CounterfactualWrapper):
    def __init__(self, dataset, model,  **kwargs):
        predict_fn = lambda x: model.predict_proba(x)
        autoencoder_model = kwargs.get("autoencoder_model", None)
        # Wrap the autoencoder to handle output shape issues
        plausibility_model = AutoencoderPlausibility(autoencoder_model)
        self.explainer =  MONICE(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            predict_fn=predict_fn,
            plausibility_model=plausibility_model,
            cat_feats=dataset.categorical_indices,
            num_feats=dataset.continuous_indices,
            integer_feats=None,
            cost_weights=None,
            immutable_features=None,
            feature_bounds=None,
            monotonic_constraints=None,
            distance_metric='HEOM',
            justified_cf=True,
            eps=1e-6,
            verbose=False
        )  
    def explain(self,x):
        try:
            cf = self.explainer.explain(
                X=x,
                target_classes='other',
                optimization=['robustness', 'sparsity', 'proximity', 'plausibility'],
                k_nearest=5,
                n_cfs=6,
                numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
            cf_key, cf_value = next(iter(cf.items()))
            return cf_value.counterfactual, cf_value.computation_time
        except Exception as e:
            cf = np.tile(np.nan, (6, x.shape[0]))
            return cf, 0
        
class MoniceMultiObjectiveGowerWrapper(MoniceWrapper):
    def explain(self,x):
        try:
            cf = self.explainer.explain(
                X=x,
                target_classes='other',
                optimization=['robustness', 'sparsity', 'proximity', 'plausibility'],
                k_nearest=5,
                n_cfs=6,
                numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
            cf_key, cf_value = next(iter(cf.items()))
            return cf_value.counterfactual, cf_value.computation_time
        except Exception as e:
            cf = np.tile(np.nan, (6, x.shape[0]))
            return cf, 0
              
class MoniceSparsProxWrapper(MoniceWrapper):
    def explain(self,x):
        try:
            cf = self.explainer.explain(
                X=x,
                target_classes='other',
                optimization=['sparsity', 'proximity'],
                k_nearest=5,
                n_cfs=6,
                numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
            cf_key, cf_value = next(iter(cf.items()))
            return cf_value.counterfactual, cf_value.computation_time
        except Exception as e:
            print(e)
            cf = np.tile(np.nan, (6, x.shape[0]))
            return cf, 0
    
class MoniceSparsPlausWrapper(MoniceWrapper):
    def explain(self,x):
        try:
            cf = self.explainer.explain(
            X=x,
            target_classes='other',
            optimization=['sparsity', 'plausibility'],
            k_nearest=5,
            n_cfs=6,
            numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
            cf_key, cf_value = next(iter(cf.items()))
            return cf_value.counterfactual, cf_value.computation_time
        except Exception as e:
            print(e)
            cf = np.tile(np.nan, (6, x.shape[0]))
            return cf, 0
    

class MoniceProxPlausWrapper(MoniceWrapper):
    def explain(self,x):
        try:
            cf = self.explainer.explain(
                X=x,
                target_classes='other',
                optimization=['proximity', 'plausibility'],
                k_nearest=5,
                n_cfs=6,
                numerical_steps=[0.2, 0.4, 0.6, 0.8, 1.0]
            )
            cf_key, cf_value = next(iter(cf.items()))
            return cf_value.counterfactual, cf_value.computation_time
        except Exception as e:
            print(e)
            cf = np.tile(np.nan, (6, x.shape[0]))
            return cf, 0
    
class DiceRandomWrapper(CounterfactualWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.n = 6
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        Xy_test = np.concatenate((dataset.X_test,dataset.y_test[:,np.newaxis]), axis= 1)
        Xy = np.concatenate((Xy,Xy_test),axis=0)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        con_feat = [dataset.feature_names[i] for i in dataset.continuous_indices]

        self.d = dice_ml.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        self.m = dice_ml.Model(model = model, backend='sklearn')
        self.explainer = dice_ml.Dice(self.d, self.m, method='random')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        start = time()
        try:
            e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
            explanation = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values
        except Exception as e:
            print(e)
            explanation = np.tile(np.nan,(self.n, x.shape[1]))
        run_time = time()-start
        if explanation.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-explanation.shape[0],x.shape[1]))
            explanation = np.concatenate((explanation,missing),axis=0)
        return explanation, run_time
    
class DiceExtendedWrapper(CounterfactualWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.n = 6
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        Xy_test = np.concatenate((dataset.X_test,dataset.y_test[:,np.newaxis]), axis= 1)
        Xy = np.concatenate((Xy,Xy_test),axis=0)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        con_feat = [dataset.feature_names[i] for i in dataset.continuous_indices]

        self.d = dice_ml_x.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        self.m = dice_ml_x.Model(model = model, backend='sklearn')
        self.explainer = dice_ml_x.Dice(self.d, self.m, method='genetic')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        print(df_x.shape)
        start = time()
        try:
            e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
            explanation = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values
        except Exception as e:
            print(e)
            explanation = np.tile(np.nan,(self.n, x.shape[1]))
        run_time = time()-start
        if explanation.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-explanation.shape[0],x.shape[1]))
            explanation = np.concatenate((explanation,missing),axis=0)
        return explanation, run_time
class GecoWrapper(CounterfactualWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.n = 6
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        Xy_test = np.concatenate((dataset.X_test,dataset.y_test[:,np.newaxis]), axis= 1)
        Xy = np.concatenate((Xy,Xy_test),axis=0)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        con_feat = [dataset.feature_names[i] for i in dataset.continuous_indices]

        self.d = dice_ml.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        self.m = dice_ml.Model(model = model, backend='sklearn')
        self.explainer = dice_ml.Dice(self.d, self.m, method='genetic')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        start = time()
        try:
            e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
            explanation = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values
        except Exception as e:
            print(e)
            explanation = np.tile(np.nan,(self.n, x.shape[1]))
        run_time = time()-start
        if explanation.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-explanation.shape[0],x.shape[1]))
            explanation = np.concatenate((explanation,missing),axis=0)
        return explanation, run_time 
class CFProtoWrapper(CounterfactualWrapper):
    def __init__(self,dataset,model, **kwargs):
        self.n = 1
        Xy = np.concatenate((dataset.X_train,dataset.y_train[:,np.newaxis]), axis= 1)
        Xy_test = np.concatenate((dataset.X_test,dataset.y_test[:,np.newaxis]), axis= 1)
        Xy = np.concatenate((Xy,Xy_test),axis=0)
        self.feature_names = dataset.feature_names + ['y']
        df = pd.DataFrame(Xy,columns= self.feature_names)
        con_feat = [dataset.feature_names[i] for i in dataset.continuous_indices]

        self.d = dice_ml.Data(dataframe = df, continuous_features = con_feat, outcome_name = 'y')
        self.m = dice_ml.Model(model = model, backend='sklearn')
        self.explainer = dice_ml.Dice(self.d, self.m, method='kdtree')

    def explain(self,x):
        df_x = pd.DataFrame(x, columns= self.feature_names[:-1])
        start = time()
        try:
            e1 = self.explainer.generate_counterfactuals(df_x, total_CFs=self.n, desired_class='opposite',verbose= False)
            explanation = e1.cf_examples_list[0].final_cfs_df.iloc[:,:-1].values
        except Exception as e:
            print(e)
            explanation = np.tile(np.nan,(self.n, x.shape[1]))
        run_time = time()-start
        if explanation.shape[0]<self.n:
            missing= np.tile(np.nan,(self.n-explanation.shape[0],x.shape[1]))
            explanation = np.concatenate((explanation,missing),axis=0)
        return explanation, run_time

class NiceWrapper(CounterfactualWrapper):
    def explain(self,x):
        start=  time()
        explanation = self.explainer.explain(x)
        run_time = time() - start
        return explanation, run_time
class NiceNoneWrapper(NiceWrapper):
    def __init__(self,dataset, model, **kwargs):
        predict_fn = lambda x: model.predict_proba(x)
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=predict_fn,
            y_train= dataset.y_train,
            cat_feat= dataset.categorical_indices,
            num_feat= dataset.continuous_indices,
            optimization='none',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True
        )
        
class NiceSparsityWrapper(NiceWrapper):
    def __init__(self, dataset, model, **kwargs):
        predict_fn = lambda x: model.predict_proba(x)
        self.explainer = NICE(
            X_train=dataset.X_train,
            predict_fn=predict_fn,
            y_train=dataset.y_train,
            cat_feat=dataset.categorical_indices,
            num_feat=dataset.continuous_indices,
            optimization='sparsity',
            justified_cf=True
        )

class NiceProximityWrapper(NiceWrapper):
    def __init__(self,dataset, model, **kwargs):
        predict_fn = lambda x: model.predict_proba(x)
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=predict_fn,
            y_train= dataset.y_train,
            cat_feat= dataset.categorical_indices,
            num_feat= dataset.continuous_indices,
            optimization='proximity',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True
        )

class NicePlausibilityWrapper(NiceWrapper):
    def __init__(self,dataset, model, **kwargs):
        autoencoder_trainer = kwargs.get("autoencoder_model", None)
        # Wrap the autoencoder to handle output shape issues
        autoencoder_wrapper = AutoEncoderWrapper(autoencoder_trainer) if autoencoder_trainer else None
        predict_fn = lambda x: model.predict_proba(x)
        self.explainer = NICE(
            X_train= dataset.X_train,
            predict_fn=predict_fn,
            y_train= dataset.y_train,
            cat_feat= dataset.categorical_indices,
            num_feat= dataset.continuous_indices,
            optimization='plausibility',
            distance_metric= 'HEOM',
            num_normalization= 'minmax',
            justified_cf=True,
            auto_encoder= autoencoder_wrapper
        )

class LIMECounterfactualWrapper(CounterfactualWrapper):
    def __init__(self, dataset, model, **kwargs):
        self.model = model
        self.feature_names = dataset.feature_names
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=dataset.X_train,
            feature_names=dataset.feature_names,
            class_names=["0", "1"],  # Binary classification
            categorical_features=dataset.categorical_indices,
            verbose=False,
            mode="classification"
        )
        
        self.categorical_indices = dataset.categorical_indices
        self.continuous_indices = dataset.continuous_indices
        self.X_train = dataset.X_train
        
    def explain(self, x):
        start = time()  
        try:
            lime_exp = self.explainer.explain_instance(
                x[0], 
                self.model.predict_proba, 
                num_features=len(self.feature_names)
            )
            
            feature_weights = dict(lime_exp.as_list())
            
            cf = self._generate_counterfactual(x[0], feature_weights)
                
            explanation = np.array(cf).reshape(1, -1)
            
        except Exception as e:
            print(e)
            explanation = np.tile(np.nan, (1, x.shape[1]))
            
        run_time = time() - start
        
        return explanation, run_time
    
    def _generate_counterfactual(self, instance, feature_weights):
        cf = instance.copy()
        original_class = self.model.predict_proba(instance.reshape(1, -1)).argmax()
        target_class = 1 - original_class  # Opposite class for binary classification
        
        sorted_features = []

        for feature_description, weight in feature_weights.items(): 
            for fname in self.feature_names:
                if fname in feature_description:
                    sorted_features.append((fname, weight))
                    break
        
        # Sort by absolute importance
        sorted_features = sorted(sorted_features, key=lambda x: abs(x[1]), reverse=True)
        
        # Generate counterfactual by modifying features
        for feature_name, weight in sorted_features:
            feature_idx = self.feature_names.index(feature_name)
            
            if feature_idx in self.categorical_indices:
                unique_values = np.unique(self.X_train[:, feature_idx])
                unique_values = unique_values[unique_values != cf[feature_idx]]
                if len(unique_values) > 0:
                    best_value = None
                    best_target_proba = -np.inf
                    
                    for val in unique_values:
                        temp_cf = cf.copy()
                        temp_cf[feature_idx] = val
                        new_proba = self.model.predict_proba(temp_cf.reshape(1, -1))
                        target_proba = new_proba[0][target_class]
                        
                        if target_proba > best_target_proba:
                            best_target_proba = target_proba
                            best_value = val
                    
                    if best_value is not None:
                        cf[feature_idx] = best_value
            else:
                std = np.std(self.X_train[:, feature_idx])
                min_val = np.min(self.X_train[:, feature_idx])
                max_val = np.max(self.X_train[:, feature_idx])
                
                current_proba = self.model.predict_proba(cf.reshape(1, -1))[0][target_class]
                
                temp_cf_inc = cf.copy()
                temp_cf_inc[feature_idx] = np.clip(cf[feature_idx] + std * 0.5, min_val, max_val)
                proba_inc = self.model.predict_proba(temp_cf_inc.reshape(1, -1))[0][target_class]
                
                temp_cf_dec = cf.copy()
                temp_cf_dec[feature_idx] = np.clip(cf[feature_idx] - std * 0.5, min_val, max_val)
                proba_dec = self.model.predict_proba(temp_cf_dec.reshape(1, -1))[0][target_class]
                
                if proba_inc > current_proba and proba_inc >= proba_dec:
                    cf[feature_idx] = temp_cf_inc[feature_idx]
                elif proba_dec > current_proba:
                    cf[feature_idx] = temp_cf_dec[feature_idx]
            
            new_proba = self.model.predict_proba(cf.reshape(1, -1))
            if np.argmax(new_proba) == target_class:
                break
            
        final_prob = self.model.predict_proba(cf.reshape(1, -1))
        final_class = np.argmax(final_prob)
        
        if final_class == target_class:
            return cf
        else:      
            return np.tile(np.nan, (1, cf.shape[0]))

class MOCWrapper(CounterfactualWrapper):
    def __init__(self, dataset, model, **kwargs):
        self.n = 6
        self.dataset = dataset
        self.model = model

        # Use preprocessed data directly - no need for additional preprocessing
        self.train_df = pd.DataFrame(
            dataset.X_train, columns=dataset.feature_names
        )

        # Convert indices to feature names (MOC expects feature names, not indices)
        self.categorical_features = [
            dataset.feature_names[i] for i in dataset.categorical_indices
        ]
        self.numerical_features = [
            dataset.feature_names[i] for i in dataset.continuous_indices
        ]

        print(f"MOC initialized with preprocessed data: {self.train_df.shape}")
        print(f"Categorical features: {self.categorical_features}")
        print(f"Numerical features: {self.numerical_features}")

    def explain(self, x):
        start = time()

        try:
            # Convert input to DataFrame
            x_df = pd.DataFrame(x, columns=self.dataset.feature_names)

            # Determine target class (opposite of current prediction)
            current_pred = self.model.predict(x)[0]
            target = 1 - current_pred  # Binary classification

            # Initialize MOC with preprocessed data
            moc = MOCCounterfactuals(
                predictor=self.model,
                x_interest=x_df.iloc[0:1],
                target=target,
                data=self.train_df,
                categorical_features=self.categorical_features,
                numerical_features=self.numerical_features,
                population_size=20,  # Smaller for faster computation
                generations=15,  # Moderate generations
                random_state=42,
            )

            moc.generate_counterfactuals(verbose=False)

            best_cfs = moc.get_best_counterfactuals(
                n=self.n, criteria="dist_target"
            )

            explanation = best_cfs[self.dataset.feature_names].values

            if explanation.shape[0] < self.n:
                missing = np.tile(
                    np.nan, (self.n - explanation.shape[0], x.shape[1])
                )
                explanation = np.concatenate((explanation, missing), axis=0)
            elif explanation.shape[0] > self.n:
                explanation = explanation[: self.n]

        except Exception as e:
            print(f"MOC error: {e}")
            explanation = np.tile(np.nan, (self.n, x.shape[1]))

        run_time = time() - start
        return explanation, run_time
