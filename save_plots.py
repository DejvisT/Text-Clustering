import matplotlib.pyplot as plt
from TextDatasetThreaded import TextDatasetThreaded
import umap.umap_ as umap
from TextEmbeddingsThreaded import TextEmbeddingsThreaded
from sklearn.manifold import TSNE


if __name__ == 'main':
    inptype = 'csv'
    dataset = TextDatasetThreaded(data_dir="datasets/Ecommerce/ecommerceDataset.csv", inp_type=inptype, max_files_per_class=500)
    dataset_name = 'Ecommerce'

    emb = TextEmbeddingsThreaded(dataset, batch_size=100)
    embeddings, labels = emb.create_fasttext_embeddings()
    model_name = 'tfidf'
    label_names = dataset.get_label_names()
    colors = [label_names[label] for label in labels]

    reduction = 'umap'

    if reduction == 'umap':
        reduced_data = umap.UMAP(n_neighbors=25, min_dist=0.0, metric='cosine', n_components=2).fit_transform(embeddings)
    elif reduction == 'tsne':
        tsne = TSNE(n_components=2)
        reduced_data = tsne.fit_transform(embeddings)

    scatter_plot = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Spectral', s=20)
    handles, labels = scatter_plot.legend_elements(prop="colors")

    plt.title(f"Model: {model_name}     Dataset: {dataset_name}")
    plt.legend(handles, set(colors), title="Classes")
    plt.show()





