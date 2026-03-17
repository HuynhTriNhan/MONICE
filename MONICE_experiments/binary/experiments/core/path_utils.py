from typing import Optional

def path_generator(dataset: str, model: Optional[str] = None, cf_name: Optional[str] = None):
    base_path = f'./data/{dataset}'
    paths = {
        'dataset': f'{base_path}/data.pkl',
        'preprocessor': f'{base_path}/preprocessor.pkl',
        'autoencoder': f'{base_path}/autoencoder.pkl',
        'autoencoder_0': f'{base_path}/autoencoder_0.pkl',
        'autoencoder_1': f'{base_path}/autoencoder_1.pkl',
        'model': f'{base_path}/{model}.pkl' if model else None,
        'results': f'{base_path}/results/{model}_{cf_name}.pkl' if model and cf_name else None,
        'model_stats': f'{base_path}/model_stats_{model}.csv' if model else None,
        'autoencoder_stats': f'{base_path}/autoencoder_stats.csv',
    }
    return paths
