from hdbscan import HDBSCAN
from algorithms.DBADV.DBADV import DBADV
from algorithms.DBHD.LDClusAlgo import LDClus as DBHD
from algorithms.spectacl import Spectacl as SpectralACL
from algorithms.SNNDPC.python.SNNDPC import SNNDPC
from algorithms.dpca.dpca.cluster import DensityPeakCluster
from sklearn.cluster import KMeans, DBSCAN, OPTICS, SpectralClustering, MeanShift
import numpy as np
def dpc_func(data, **param):
    """
    Perform density peak clustering on the input data using the given parameters.

    :param data: Numpy array, shape (n_samples, n_features). The data to cluster.
    :param param: Parameters for the DensityPeakCluster algorithm.
    :return: Numpy array with cluster labels for each data point.
    """
    d = DensityPeakCluster(**param, anormal=False)
    d.fit(data)
    return d.labels_

ALGORITHM_FUNCTIONS = {
    "KMeans": lambda data, params: KMeans(**params).fit_predict(data),
    "DBSCAN": lambda data, params: DBSCAN(**params).fit_predict(data),
    "HDBSCAN": lambda data, params: HDBSCAN(**params).fit(data).labels_,
    "OPTICS": lambda data, params: OPTICS(**params).fit_predict(data),
    "SpectralClustering": lambda data, params: SpectralClustering(**params).fit_predict(data),
    "MeanShift": lambda data, params: MeanShift(**params).fit_predict(data),
    "DBADV": lambda data, params: np.array(DBADV(X=data, **params)),
    "DBHD": lambda data, params: DBHD(X=data, **params),
    "SpectralACL": lambda data, params: SpectralACL(**params).fit_predict(data),
    "SNNDPC": lambda data, params: SNNDPC(data=data, **params)[1],
    "DPC": lambda data, params: dpc_func(data, **params)
}

PARAMETER_RANGE = [{'n_clusters': {'start': 2, 'stop': 6}},
                {'min_samples': {'start': 5, 'stop': 10},
                'eps': {'start': 0.1, 'stop': 1}},
                {'min_cluster_size': {'start': 5, 'stop': 195},
                'min_samples': {'start': 5, 'stop': 195}},
                {'min_samples': {'start': 5, 'stop': 195},
                'xi': {'start': 0.01, 'stop': 0.91}},
                {'n_clusters': {'start': 2, 'stop': 8}},
                {'bandwidth': {'start': 0.01, 'stop': 0.91}},
                {'perplexity': {'start': 1, 'stop': 30},
                'MinPts': {'start': 1, 'stop': 30},
                'probability': {'start': 0.997, 'stop': 0.9971}},
                {'min_cluster_size': {'start': 5, 'stop': 195},
                'rho': {'start': 0.05, 'stop': 2},
                'beta': {'start': 0.1, 'stop': 0.6}},
                {'n_clusters': {'start': 4, 'stop': 4},
                'epsilon': {'start': 0.01, 'stop': 1}},
                {'k': {'start': 5, 'stop': 50},
                'nc': {'start': 6, 'stop': 6}},
                {'density_threshold': {'start': 0.01, 'stop': 1},
                'distance_threshold': {'start': 0.01, 'stop': 1}}]

PARAMETERS = [['n_clusters'],
              ['min_samples', 'eps'],
              ['min_cluster_size', 'min_samples'],
              ['min_samples', 'xi'],
              ['n_clusters'],
              ['bandwidth'],
              ['perplexity', 'MinPts', 'probability'],
              ['min_cluster_size', 'rho', 'beta'],
              ['n_clusters', 'epsilon'],
              ['k', 'nc'],
              ['density_threshold', 'distance_threshold']]