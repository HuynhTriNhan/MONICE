from experiments.core.data_loader import TabularDataLoader
from experiments.core.path_utils import path_generator
from experiments.core.data_preprocessing import PreprocessorBuilder
from experiments.models.autoencoder_trainer import AutoencoderTrainer
from experiments.models.classification_trainer import SklearnTabularModeler
from experiments.counterfactuals.cf_wrapper import MoniceMultiObjectiveGowerWrapper, MoniceMultiObjectiveHEOMWrapper, MoniceSparsProxWrapper, MoniceProxPlausWrapper, MoniceSparsPlausWrapper, \
    DiceRandomWrapper, CFProtoWrapper, GecoWrapper, \
    NiceNoneWrapper, NiceSparsityWrapper, NiceProximityWrapper, NicePlausibilityWrapper, \
    LIMECounterfactualWrapper, MOCWrapper

from time import time
import pickle
from tqdm import tqdm
import os
import numpy as np

from concurrent.futures import ThreadPoolExecutor, TimeoutError

CF_WRAPPERS = {
    'monicemultiobjectivegower': MoniceMultiObjectiveGowerWrapper,
    'monicemultiobjectiveheom': MoniceMultiObjectiveHEOMWrapper,
    'monicesparsprox': MoniceSparsProxWrapper,
    'moniceproxplaus': MoniceProxPlausWrapper,
    'monicesparsplaus': MoniceSparsPlausWrapper,
    
    'nicenone': NiceNoneWrapper,
    'nicespars': NiceSparsityWrapper,
    'niceprox': NiceProximityWrapper,
    'niceplaus': NicePlausibilityWrapper,
    
    'moc': MOCWrapper,
    'dicerandom': DiceRandomWrapper,
    'cfproto': CFProtoWrapper,
    'geco': GecoWrapper,

    'lime': LIMECounterfactualWrapper,
}
N_CF = {
    'monicemultiobjectivegower': 6,
    'monicemultiobjectiveheom': 6,
    'monicesparsprox': 6,
    'moniceproxplaus': 6,
    'monicesparsplaus': 6,
        
    'nicenone': 1,
    'nicespars': 1,
    'niceprox': 1,
    'niceplaus': 1,
    
    'moc': 6,
    'dicerandom': 6,
    'geco': 6,
    'lime': 1,
    
    'cfproto': 1, 
}
def run_explanation(cf_algo, original):
    return cf_algo.explain(original)
class CounterfactualGenerator:
    def __init__(self,  dataset_name:str, model:str, cf_name:str, verbose:bool = True):
        self.dataset_name = dataset_name
        self.model_name = model
        self.cf_name = cf_name
        self.verbose = verbose
        self.paths = path_generator(dataset_name, model, cf_name)
        self.dataset = TabularDataLoader(self.paths['dataset'])
        self.description = f'{model} // {dataset_name} // {cf_name}'
        
        self.modeler = SklearnTabularModeler(self.dataset_name, model)
        self.modeler.load()

        self.autoencoder = AutoencoderTrainer(self.dataset_name)
        self.autoencoder.load()
        
        self.cf_algo = CF_WRAPPERS[cf_name](self.dataset, self.modeler, autoencoder_model=self.autoencoder)

    def run_experiment(self, overwrite: bool = False):
        save_time = time()
        checkpoint_every = 10
        timeout_seconds = 300
        results = []

        if os.path.isfile(self.paths['results']) and not overwrite:
            with open(self.paths['results'], 'rb') as f:
                results = pickle.load(f)
                print(f"Loaded {len(results)} existing results")          
        if self.dataset_name == "mushroom" and self.cf_name in ["geco", "cfproto"]:
            # all results are nan, time = 0 
            results = []
            for row in range(self.dataset.X_explain.shape[0]):
                result = {'original': self.dataset.X_explain[row:row + 1, :]}
                result['cf'] = np.tile(result['original'], (N_CF[self.cf_name], result['original'].shape[0]))
                result['time'] = 0
                results.append(result)
            with open(self.paths['results'], 'wb') as f:
                pickle.dump(results, f)
            return

        for row in tqdm(range(len(results), self.dataset.X_explain.shape[0]), desc=self.description):
            result = {'original': self.dataset.X_explain[row:row + 1, :]}
            n_cfs = N_CF[self.cf_name]

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_explanation, self.cf_algo, result['original'])
                
                try:
                    result['cf'], result['time'] = future.result(timeout=timeout_seconds)
                    
                except TimeoutError:
                    print(f"[TIMEOUT] Row {row} exceeded {timeout_seconds}s")
                    result['cf'] = np.tile(result['original'], (n_cfs, result['original'].shape[0]))
                    result['time'] = 0
                except Exception as e:
                    print(f"[ERROR] Row {row}: {e}")
                    result['cf'] = np.tile(result['original'], (n_cfs, result['original'].shape[0]))
                    result['time'] = 0
                finally:
                    executor.shutdown(wait=True)
                
            results.append(result)
            # Save checkpoint every 10 rows or every 3 minutes
            if (row + 1) % checkpoint_every == 0 or (time() - save_time) > 180:
                with open(self.paths['results'], 'wb') as f:
                    pickle.dump(results, f)
                print(f"Checkpoint saved at row {row + 1}")
                save_time = time()

        # Final save
        with open(self.paths['results'], 'wb') as f:
            pickle.dump(results, f)
        print("All results saved.")