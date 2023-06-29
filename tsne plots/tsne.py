import matplotlib.pyplot as plt
from TextDatasetThreaded import TextDatasetThreaded
import umap.umap_ as umap
from TextEmbeddingsThreaded import TextEmbeddingsThreaded
from sklearn.manifold import TSNE
import numpy as np
import os
from Autoencoder.Autoencoder import run_autoencoder

datasets = {}

dataset = TextDatasetThreaded(data_dir="../datasets/Ecommerce/ecommerceDataset.csv", inp_type='csv', max_files_per_class=10)
datasets['Ecommerce'] = dataset
dataset = TextDatasetThreaded(data_dir="../datasets/bbc", max_files_per_class=10)
datasets['bbc'] = dataset
dataset = TextDatasetThreaded(data_dir="../datasets/20news-18828", max_files_per_class=10)
datasets['20news-18828'] = dataset

classes = {'Ecommerce': 4, 'bbc': 5, '20news': 4}
directory_path = '../embeddings'
file_list = os.listdir(directory_path)
for file_name in file_list:
    if file_name.endswith('.npy') and not file_name.startswith('20news'):
        file_path = os.path.join(directory_path, file_name)
        data = np.load(file_path)
        embeddings = data[:, :-1]
        labels = data[:, -1]
        print(file_name)
        for data in datasets.items():
            name, dataset = data
            if name == file_name.split('_')[0]:

                label_names = dataset.get_label_names()

                colors = [label_names[label] for label in labels]

        reduction = 'tsne'

        if reduction == 'umap':
            reduced_data = umap.UMAP(n_neighbors=25, min_dist=0.0, metric='cosine', n_components=2).fit_transform(embeddings)
        elif reduction == 'tsne':
            tsne = TSNE(n_components=2, perplexity=50)
            reduced_data = tsne.fit_transform(embeddings)
        elif reduction == 'autoencoder':
            inp_size = embeddings.shape[1]
            embeddings = embeddings.astype(np.float32)
            labels = labels.astype(int)
            if 0 not in set(labels):
                labels = labels - 1
            indices = np.random.permutation(len(embeddings))
            embeddings = embeddings[indices]
            labels = labels[indices]
            num_classes = classes[file_name.split('_')[0]]
            reduced_data, labels = run_autoencoder(embeddings, labels, inp_size=inp_size, num_classes=num_classes)


        scatter_plot = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Spectral', s=20)
        handles, labels = scatter_plot.legend_elements(prop="colors")

        model = file_name.split('_')[1]
        dataset_name = file_name.split('_')[0]
        plt.title(f"Model: {model}      Dataset: {dataset_name}")
        plt.legend(handles, set(colors), title="Classes")
        filename = f"{model}_{dataset_name}"
        #plt.savefig(f"../Autoencoder/{filename}.png", format="png")
        plt.show()





