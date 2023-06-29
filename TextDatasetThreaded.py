from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import os
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


class TextDatasetThreaded(Dataset):
    """
    A PyTorch dataset for loading text files in parallel using threads.

    Args:
        data_dir (str): The directory containing the text files to load.
        load_labels (bool, optional): Whether to load labels from the text files. Defaults to True.
        num_workers (int, optional): The number of threads to use for loading text files. Defaults to 4.
        inp_type (str, optional): The type of input. Can be either text or csv. Defaults to text.
        max_files_per_class (int, optional): The max number of files to load from each class. Defaults to 100.

    Attributes:
        data_dir (str): The directory containing the text files to load.
        max_files_per_class (int): The max number of files to load from each class. Defaults to 100.

        model (BertModel): The BERT model used to encode the text.
        max_seq_length (int): The maximum sequence length for each text.
        texts (list): The loaded texts.
        load_labels (bool): Whether to load labels from the directory names.
        labels_names (dict): A dictionary mapping label indices to label names.
        labels (list): The labels corresponding to each text.
    """
    def __init__(self, data_dir, load_labels=True, num_workers=4, inp_type='text', max_files_per_class=100):
        self.data_dir = data_dir
        self.max_files_per_class = max_files_per_class

        if inp_type == 'text':
            self.texts = []
            self.load_labels = load_labels
            self.labels_names = {}
            self.labels = []
            self.load_texts(num_workers)
        elif inp_type == 'csv':
            self.data = pd.read_csv(data_dir)
            self.data = sample_csv(self.data, self.max_files_per_class)
            self.texts = self.data.iloc[:, 0].tolist()
            labels = self.data.iloc[:, 1].tolist()
            label_map = {label: i for i, label in enumerate(self.data.iloc[:, 1].unique())}
            self.labels = [label_map[label] for label in labels]
            self.labels_names = {i: label for i, label in enumerate(self.data.iloc[:, 1].unique())}


    def load_texts(self, num_workers):
        """
        Loads all the text files in the directory using multiple threads.

        :param num_workers: number of threads used for loading texts.
        """
        categories = [
            "alt.atheism",
            "comp.graphics",
            "rec.autos",
            "sci.crypt",
            "talk.politics.misc"
        ]
        # Load all files in a directory and assign the subdirectory name as label if there are labels
        print("Loading texts -----------------------")
        i = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            label_num = 0
            for root, _, files in os.walk(self.data_dir):
                subdir = os.path.basename(root)
                # For the 20news dataset only use a subset of the classes in categories
                if self.data_dir != "datasets/20news-18828" or subdir in categories:
                    if len(files) > self.max_files_per_class:
                        files = files[:self.max_files_per_class]
                    for file in files:
                        file_path = os.path.join(root, file)
                        future = executor.submit(self.load_text, file_path, i, label_num)
                        i += 1

                        self.texts.append(future)
                    label_num += 1 if self.load_labels else 0
        self.texts = [f.result() for f in self.texts]

        print("Finish loading ----------------------")
        print(f"Labels {self.labels_names}  {self.labels}")

    def load_text(self, file_path, index, label_num):
        """
        Loads a single text file and returns the text.

        :param file_path: The path of the file to be loaded.
        :param index: The index of the file being loaded.
        :param label_num: The current label index.
        :return: The loaded text.
        """
        with open(file_path, 'r', encoding='ISO-8859-1') as f:
            text = f.read().strip()
            if index % 100 == 0:
                print(f"Loaded {index} texts")

            if self.load_labels:
                label = os.path.basename(os.path.dirname(file_path))
                self.labels_names[label_num] = label
                self.labels.append(label_num)
                label_num += 1
            return text

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

    def get_label_names(self):
        return self.labels_names


def sample_csv(data, max_files_per_class):
    """
        Samples a specified number of files per class from a given DataFrame.

        Args:
            data (pandas.DataFrame): The input DataFrame containing text data and corresponding labels.
            max_files_per_class (int): The maximum number of files to sample per class.

        Returns:
            pandas.DataFrame: A sampled DataFrame containing the specified number of files per class.
    """
    unique_labels = data['label'].unique()

    sampled_df = pd.DataFrame()

    for label in unique_labels:
        subset = data[data['label'] == label]
        sampled_subset = subset.head(max_files_per_class)
        sampled_df = sampled_df.append(sampled_subset)

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df