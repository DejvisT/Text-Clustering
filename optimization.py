import optuna
from UmapEmbeddings import UmapEmbeddings
from constants import ALGORITHM_FUNCTIONS, PARAMETER_RANGE
from sklearn.metrics import normalized_mutual_info_score
import pickle
from sklearn.manifold import TSNE
import numpy as np

umap_n_neighbors = 20
tsne_perplexity = 50
reduced_datasets = {}

reduction = 'tsne'
embedding_path = 'embeddings/bbc_BERT_500.npy'
dataset_name = 'bbc'
model_name = 'BERT'



def create_umap(umap_n_components):
    """
    Create UMAP embeddings with the specified number of components.

    Args:
        umap_n_components (int): Number of dimensions for the UMAP embeddings.

    Returns:
        numpy.ndarray: The reduced embeddings.
    """
    umap = UmapEmbeddings(embeddings_path=embedding_path)
    reduced_embeddings = umap.reduce_embeddings(n_components=umap_n_components, n_neighbors=umap_n_neighbors, min_dist=0)

    return reduced_embeddings

def create_tsne(tsne_n_components):
    """
    Create t-SNE embeddings with the specified number of components.

    Args:
        tsne_n_components (int): Number of dimensions for the t-SNE embeddings.

    Returns:
        numpy.ndarray: The reduced embeddings.
    """
    if tsne_n_components >= 4:
        tsne = TSNE(n_components=tsne_n_components, perplexity=tsne_perplexity, method='exact')
    else:
        tsne = TSNE(n_components=tsne_n_components, perplexity=tsne_perplexity)
    embeddings = np.load(embedding_path)
    reduced_data = tsne.fit_transform(embeddings[:, :-1])
    labels = embeddings[:, -1].reshape(-1, 1)
    reduced_data = np.concatenate((reduced_data, labels), axis=1)

    return reduced_data

def objective(trial, algorithm, index):
    """
    Objective function for hyperparameter optimization.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object.
        algorithm (str): The algorithm name.
        index (int): The index of the algorithm in the ALGORITHM_FUNCTIONS list.

    Returns:
        float: The score (e.g., NMI) of the algorithm with the specified hyperparameters.
    """
    param = PARAMETER_RANGE[index]
    parameters = {}
    for key, value in param.items():
        if isinstance(value['start'], int):
            parameters[key] = trial.suggest_int(key, value['start'], value['stop'])
        else:
            parameters[key] = trial.suggest_float(key, value['start'], value['stop'])

    if reduction == 'umap':
        umap_n_components = trial.suggest_categorical('umap_n_components', [2, 4, 8, 16])

        if umap_n_components in reduced_datasets.keys():
            reduced_embeddings = reduced_datasets[umap_n_components]
        else:
            reduced_embeddings = create_umap(umap_n_components)
            reduced_datasets[umap_n_components] = reduced_embeddings
    elif reduction == 'tsne':
        tsne_n_components = trial.suggest_categorical('tsne_n_components', [2, 4, 8, 16])

        if tsne_n_components in reduced_datasets.keys():
            reduced_embeddings = reduced_datasets[tsne_n_components]
        else:
            reduced_embeddings = create_tsne(tsne_n_components)
            reduced_datasets[tsne_n_components] = reduced_embeddings

    try:
        labels = ALGORITHM_FUNCTIONS[algorithm](reduced_embeddings[:, :-1], parameters)
        true_labels = reduced_embeddings[:, -1]

        nmi = normalized_mutual_info_score(true_labels, labels)

    except:
        nmi = 0

    return nmi

def optimize_algorithms(n_trials=200):
    """
    Perform hyperparameter optimization for different algorithms using Optuna.

    Args:
        n_trials (int, optional): The number of trials for each algorithm's hyperparameter optimization. Defaults to 200.
    """

    index=0
    for algorithm in ALGORITHM_FUNCTIONS.keys():

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, algorithm, index), n_trials=n_trials)
        index += 1
        best_params = study.best_params
        best_score = study.best_value
        print('Best Hyperparameters:', best_params)
        print('Best Score:', best_score)

        filename = f'Studies/{dataset_name}/{reduction}/pickles_{dataset_name}_{model_name}/{algorithm}_study_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(study, f)

optimize_algorithms()