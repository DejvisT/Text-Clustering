import numpy as np
from TextDatasetThreaded import TextDatasetThreaded
import umap.umap_ as umap
import matplotlib.pyplot as plt
import pandas as pd
import os
from TextEmbeddingsThreaded import TextEmbeddingsThreaded


max_files_per_class = 500
data_dir = ["bbc", 'Ecommerce/ecommerceDataset.csv', "amazon_reviews/train.csv", "20news-18828"]
batch_size = 100
model = "BERTlong"
n_neighbors = 20
min_dist = 0.0
def save_embeddings(data_dir, batch_size, max_files_per_class, model, n_neighbors, min_dist):
    """
        Generate embeddings for text data and saves them. It also creates and saves plots of 2d-UMAP embeddings.

        Args:
            data_dir (list): A list of dataset directories. Each directory represents a dataset to be processed.
            batch_size (int): The batch size used for processing the data.
            max_files_per_class (int): The maximum number of files to be processed per class in the dataset.
            model (str): The type of model to use for generating embeddings. It can be one of the following:
                         'Word2Vec', 'BERT', 'tfidf', 'fasttext', 'base', 'BERTlong'.
            n_neighbors (int): The number of neighbors used for UMAP dimensionality reduction.
            min_dist (float): The minimum distance used for UMAP dimensionality reduction.

        Returns:
            None
        """
    for dataset_dir in data_dir:
        if dataset_dir == "20news-18828" or dataset_dir == "bbc":
            inp_type = 'text'
        else:
            inp_type = 'csv'

        print(inp_type)
        print(dataset_dir)
        dataset = TextDatasetThreaded(data_dir="datasets/" + dataset_dir, inp_type=inp_type, max_files_per_class=max_files_per_class)

        emb = TextEmbeddingsThreaded(dataset, batch_size=batch_size)
        if model == 'Word2Vec':
            embeddings, labels = emb.create_word2vec_embeddings()
        elif model == 'BERT':
            embeddings, labels = emb.create_embeddings()
        elif model == 'tfidf':
            embeddings, labels = emb.create_tf_idf_embeddings()
        elif model == 'fasttext':
            embeddings, labels = emb.create_fasttext_embeddings()
        elif model == 'base':
            embeddings, labels = emb.create_base_embeddings()
        elif model == 'BERTlong':
            embeddings, labels = emb.create_BERT_long()
        dataset_dir = dataset_dir.split('/')[0]
        np.save(f'{dataset_dir}_{model}_{max_files_per_class}.npy', np.concatenate((embeddings, labels.reshape(-1,1)), axis=-1))

        label_names = dataset.get_label_names()
        colors = [label_names[label] for label in labels]

        umap_data = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric='cosine', n_components=2).fit_transform(embeddings)

        scatter_plot = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, cmap='Spectral', s=20)
        handles, labels = scatter_plot.legend_elements(prop="colors")
        dataset_name = os.path.basename(dataset_dir) if os.path.dirname(dataset_dir) == '' else os.path.basename(os.path.dirname(dataset_dir))
        plt.title(f"Model: {model}      Dataset: {dataset_name}")
        plt.legend(handles, set(colors), title="Classes")

        num_classes = len(dataset.labels_names)
        filename = f'{model}_{dataset_name}'
        df = pd.read_csv('EmbeddingPlots/embeddings_log.csv')
        new_row = {'filename': filename, 'model': model, 'dataset': dataset_name, 'num_classes': num_classes, 'max_files': max_files_per_class, 'batch_size': batch_size}
        if not (df == new_row).all(1).any():
            df = df.append(new_row, ignore_index=True)
        df.to_csv('EmbeddingPlots/embeddings_log.csv', index=False)
        plt.savefig(f"EmbeddingPlots/{filename}.png", format="png")
        plt.clf()

save_embeddings(data_dir, batch_size, max_files_per_class, model, n_neighbors, min_dist)

