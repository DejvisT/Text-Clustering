import numpy as np
import umap.umap_ as umap


class UmapEmbeddings:
    """
    Class for creating UMAP embeddings from high text embeddings.

    Args:
        embeddings_path (str): Path to the embeddings file.
        filename (str, optional): Name of the file to save the reduced embeddings (if save_file is True).
                                          Defaults to None.
        load_labels (bool, optional): Indicates whether to load labels from the embeddings file.
                                              Defaults to True.
        save_file (bool, optional): Indicates whether to save the reduced embeddings to a file.
                                            Defaults to False.
    """
    def __init__(self, embeddings_path, filename=None, load_labels=True, save_file=False):
        self.filename = filename
        self.embeddings = np.load(embeddings_path)
        self.load_labels = load_labels
        self.save_file = save_file

    def reduce_embeddings(self, n_components, n_neighbors, min_dist):

        if self.load_labels:
            labels = self.embeddings[:, -1].reshape(-1, 1)
            self.embeddings = self.embeddings[:, :-1]

        umap_embedding = umap.UMAP(n_components=n_components,
                                   n_neighbors=n_neighbors,
                                   min_dist=min_dist,
                                   metric='cosine').fit_transform(self.embeddings)

        if self.load_labels:
            umap_embedding = np.concatenate((umap_embedding, labels), axis=1)
        if self.save_file:
            if self.load_labels:
                np.save("embeddings/" + self.filename, np.concatenate((umap_embedding, labels), axis=1))
            else:
                np.save("embeddings/" + self.filename, umap_embedding)

        return umap_embedding






