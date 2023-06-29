import pandas as pd
import pickle
import os
from evaluation_classifier import evaluate_clustering
import numpy as np
import umap.umap_ as umap
from constants import ALGORITHM_FUNCTIONS
def pickle_to_df(pickle_file_path):
    """
    Load a pickle file and convert it to a pandas DataFrame.

    Args:
        pickle_file_path (str): The path to the pickle file.

    Returns:
        pandas.DataFrame: The DataFrame containing the data from the pickle file.
    """
    with open(pickle_file_path, "rb") as file:
        study = pickle.load(file)

    trials = study.trials
    trial_data = []

    for trial in trials:
        params = trial.params
        value = trial.value
        trial_data.append({**params, 'NMI': value})

    df = pd.DataFrame(trial_data)
    return df




directories = ['Studies/bbc/tsne/pickles_bbc_BERT', 'Studies/bbc/tsne/pickles_bbc_word2vec',
               'Studies/bbc/tsne/pickles_bbc_tfidf', 'Studies/bbc/tsne/pickles_bbc_fasttext']
filepath = 'Studies/bbc/tsne/csvs'
def save_csv_pickles(directories, output_directory):
    """
    Combine multiple pickle files into a single CSV file for each algorithm.

    Args:
        directories (list): A list of directories containing the pickle files.
        output_directory (str): The directory where the output CSV files will be saved.

    Returns:
        None
    """
    combined_dfs = {}
    for directory in directories:
        embedding = directory.split('_')[-1]
        for filename in os.listdir(directory):
            algorithm_name = filename.split('_')[0]
            filepath = os.path.join(directory, filename)

            if filename.endswith('.pkl'):
                df = pickle_to_df(filepath)
                df['embedding'] = embedding
                if algorithm_name in combined_dfs:
                    combined_dfs[algorithm_name] = pd.concat([combined_dfs[algorithm_name], df])
                else:
                    combined_dfs[algorithm_name] = df

    for algorithm_name, df in combined_dfs.items():
        print(f"Algorithm: {algorithm_name}")
        print(df)
        print("\n")
        output_filename = f"{algorithm_name}_data.csv"
        print(output_filename)
        output_filepath = os.path.join(output_directory, output_filename)
        print(output_filepath)
        df.to_csv(output_filepath, index=False)


def save_metrics(directory, n, dataset):
    """
    Calculate metrics for top N rows of each algorithm and save them in separate CSV files.

    Args:
        directory (str): The directory containing the CSV files.
        n (int): The number of top rows to consider.
        dataset (str): The name of the dataset.

    Returns:
        None
    """
    i = 0
    for file in os.listdir(directory):
        if i in []:
            print(f"Not {file}")
            i += 1
            continue
        filepath = os.path.join(directory, file)
        if file.endswith('.csv'):
            df = pd.read_csv(filepath)
            top_n_rows = df.groupby('embedding').apply(lambda x: x.nlargest(n, 'NMI')).reset_index(drop=True)


            for index, row in top_n_rows.iterrows():
                embeddings = np.load(f'embeddings/{dataset}_' + row['embedding'] + '_500.npy')
                reduced_embeddings = umap.UMAP(n_neighbors=20, n_components=row['umap_n_components'],
                                               min_dist=0).fit_transform(embeddings[:, :-1])

                base_embeddings = np.load(f"embeddings/{dataset}_base_500.npy")[:, :-1]
                true_labels = embeddings[:, -1]

                parameters = {column: row[column] for column in df.columns[:-3]}

                pred_labels = ALGORITHM_FUNCTIONS[file.split('_')[0]](reduced_embeddings[:, :-1], parameters)
                metrics = evaluate_clustering(base_embeddings, true_labels, pred_labels)

                for metric_name, metric_value in metrics.items():
                    top_n_rows.loc[index, metric_name] = metric_value
                print(top_n_rows)
                output_filename = f"{file.split('_')[0]}_top_n_rows.csv"
                output_filepath = directory.split('_')[0] + '/top_n/' + output_filename
                top_n_rows.to_csv(output_filepath, index=False)


directory = 'Studies/bbc/tsne/csvs'
n = 3
dataset = 'bbc'
save_csv_pickles(directories, filepath)
#save_metrics(directory, n, dataset)