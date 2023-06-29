from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from itertools import product
import numpy as np
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from UmapEmbeddings import UmapEmbeddings
from constants import ALGORITHM_FUNCTIONS


class ClusteringAlgorithmsThreaded:
    """
    Class for running multiple clustering algorithms on a given dataset with different parameter combinations.

    Args:
        filename (str): The name of the file containing the data in form of a numpy array.
        load_labels (bool, optional): Whether the data is labeled or not.

    Attributes:
        algorithms (list of str): Algorithms to be run.
        metrics (list of tuples): Names and functions of the evaluation metrics to compute.
        data (numpy array): Data to cluster.
        load_labels (bool, optional): Whether the data is labeled or not.
        scores (list of lists): Silhouette Score for each algorithm and parameter combination.
        metrics_list (list of lists of lists): Metric values for each algorithm, parameter combination, and metric.
        parameters (list of lists): Names of parameters for each algorithm.
    """
    def __init__(self, filename, load_labels=True):
        self.algorithms = ['KMeans', 'DBSCAN', 'HDBSCAN', 'OPTICS',
                           'SpectralClustering', 'MeanShift', 'DBADV',
                           'DBHD', 'SpectralACL', 'SNNDPC', 'DPC']

        self.metrics = [('AMI', adjusted_mutual_info_score),
                        ('ARI', adjusted_rand_score),
                        ('NMI', normalized_mutual_info_score)]
        self.data = np.load('embeddings/' + filename)
        self.load_labels = load_labels
        self.scores = [[] for _ in range(len(self.algorithms))]
        self.metrics_list = [[[] for _ in range(len(self.metrics))] for _ in range(len(self.algorithms))]

        if load_labels:
            self.labels = self.data[:, -1].astype(int)
            self.data = self.data[:, :-1]
        self.parameters = [['n_clusters'],
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

    def run_algorithms(self, param_ranges):
        """
        Run each algorithm on the data with different parameter combinations.
        Saves results as .csv files for each algorithm in directory 'csv_files'.

        :param param_ranges: list of dicts with the parameter ranges:
        {'param_name': {'start': start_val, 'stop': stop_val, 'range': range_val}}
        """
        alg_num = 0

        # Create a directory named 'csv files'
        if not os.path.exists('csv_files'):
            os.makedirs('csv_files')

        with ThreadPoolExecutor(max_workers=4) as executor:

            for algorithm, params in zip(self.algorithms, self.parameters):

                # Create a csv file for each algorithm
                filename = os.path.join('csv_files', algorithm + '.csv')
                with open(filename, 'w', newline='') as file:
                    writer = csv.writer(file)
                    if self.load_labels:
                        metric_names = [metric[0] for metric in self.metrics]
                        writer.writerow(self.parameters[alg_num] + ['score'] + metric_names)
                    else:
                        writer.writerow(self.parameters[alg_num] + ['score'])

                    print(f"Algorithm: {algorithm}   {params}")

                    param_space = []
                    alg_param_ranges = param_ranges[alg_num]
                    # Compute all parameter combinations
                    for p, param in enumerate(params):
                        print(f"{p}   {param}")
                        p_values = np.arange(alg_param_ranges[param]['start'], alg_param_ranges[param]['stop'], alg_param_ranges[param]['range'])
                        param_space.append(p_values)
                    print(f"Parameter space: {param_space}")
                    param_mesh = np.array(list(product(*param_space)))
                    print(f"{param_mesh}" + "\n")

                    names = self.parameters[alg_num]
                    values = param_mesh.tolist()
                    print(f"List of values: {values}")
                    values = [[int(x) if x == int(x) else x for x in sublist] for sublist in values]
                    list_of_dicts = [{name: value for name, value in zip(names, value_list)} for value_list in values]

                    print(f"List of dicts: {list_of_dicts}")
                    algorithm_function = ALGORITHM_FUNCTIONS[algorithm]
                    # Evaluate the algorithm with every parameter combination

                    results = list(executor.map(lambda param: self.evaluate_algorithm(algorithm_function, param), list_of_dicts))

                    j = 0
                    for result in results:
                        labels, score, metrics_values = result

                        print(list_of_dicts[j])
                        self.scores[alg_num].append(score)
                        for i, metric_val in enumerate(metrics_values):
                            self.metrics_list[alg_num][i].append(metric_val)

                        if self.load_labels:
                            writer.writerow([value for value in list_of_dicts[j].values()] + [score] + metrics_values)
                        else:
                            writer.writerow([value for value in list_of_dicts[j].values()] + [score])
                        j += 1
                alg_num += 1

    def evaluate_algorithm(self, algorithm_function, params):
        """
        Evaluate the given algorithm on the data with the given parameters.

        :param algorithm_function: Algorithm function that should be evaluated.
        :param params: Parameter combinations for which to evaluate.
        :return: The calculated metrics.
        """
        try:
            labels = algorithm_function(self.data, params)
            try:
                score = round(silhouette_score(self.data, labels), 4)
            except:
                score = -1
            metrics_values = []
            for metric in self.metrics:
                try:
                    metric_val = round(metric[1](self.labels, labels), 4)
                except:
                    metric_val = -1
                metrics_values.append(metric_val)
            return labels, score, metrics_values

        except:
            score = -1
            metrics_values = [-1] * len(self.metrics)
            return None, score, metrics_values


param_ranges = [{'n_clusters': {'start': 2, 'stop': 9, 'range': 1}},
                {'min_samples': {'start': 5, 'stop': 10, 'range': 195},
                'eps': {'start': 0.1, 'stop': 1, 'range': 0.1}},
                {'min_cluster_size': {'start': 5, 'stop': 10, 'range': 195},
                 'min_samples': {'start': 5, 'stop': 10, 'range': 195}},
                {'min_samples': {'start': 5, 'stop': 25, 'range': 5},
                 'xi': {'start': 0.1, 'stop': 0.3, 'range': 0.1}},
                {'n_clusters': {'start': 2, 'stop': 4, 'range': 1}},
                {'bandwidth': {'start': 0.2, 'stop': 0.7, 'range': 0.1}},
                {'perplexity': {'start': 1, 'stop': 5, 'range': 1},
                 'MinPts': {'start': 1, 'stop': 3, 'range': 1},
                 'probability': {'start': 0.997, 'stop': 0.9971, 'range': 0.0001}},
                {'min_cluster_size': {'start': 3, 'stop': 5, 'range': 1},
                 'rho': {'start': 0.1, 'stop': 0.2, 'range': 0.05},
                 'beta': {'start': 0.1, 'stop': 0.3, 'range': 0.1}},
                {'n_clusters':  {'start': 3, 'stop': 4, 'range': 1},
                 'epsilon':  {'start': 0.02, 'stop': 0.03, 'range': 0.01}},
                {'k': {'start': 4, 'stop': 5, 'range': 1},
                 'nc': {'start': 5, 'stop': 6, 'range': 1}},
                {'density_threshold': {'start': 8, 'stop': 9, 'range': 1},
                 'distance_threshold': {'start': 5, 'stop': 6, 'range': 1}}]


class ClusAlg(ClusteringAlgorithmsThreaded):
    def __init__(self, file, algorithms, load_labels=True):
        super().__init__(file, load_labels)
        self.algorithms = [self.algorithms[i] for i in algorithms]
        self.parameters = [self.parameters[i] for i in algorithms]

        umap = UmapEmbeddings(embeddings_path='embeddings/' + file)
        self.data = umap.reduce_embeddings(n_components=2, n_neighbors=10, min_dist=0)


algorithms = [0,1]
param_ranges = [param_ranges[i] for i in algorithms]

filename = 'bbc_tfidf_500.npy'
cl = ClusAlg(file=filename, algorithms=algorithms)
cl.run_algorithms(param_ranges)



