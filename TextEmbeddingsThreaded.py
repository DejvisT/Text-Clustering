import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from gensim import corpora, models, matutils
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from string import punctuation
import fasttext.util
from sklearn.feature_extraction.text import TfidfVectorizer
class TextEmbeddingsThreaded:
    """
    A class for generating text embeddings using various embedding models.

    Args:
        dataset (TextDatasetThreaded): The dataset object containing the texts and labels.
        batch_size (int): The batch size for creating embeddings.
        shuffle (bool, optional): Whether to shuffle the dataset during embedding creation. Defaults to False.
    """
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)
        self.batch_list = []
        self.labels_list = []
        self.batch_size = batch_size

    def create_embeddings(self):
        """
        Create BERT embeddings for the texts in the dataset.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        all_embeddings = []
        all_labels = []
        num = 0
        for texts, labels in self.dataloader:
            encoded_texts = tokenizer.batch_encode_plus(
                list(texts),
                max_length=500,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding='longest',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            input_ids = encoded_texts['input_ids']

            attention_masks = encoded_texts['attention_mask']

            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_masks)

            embeddings = outputs['last_hidden_state']
            mean_embeddings = torch.mean(embeddings, dim=1)
            all_embeddings.append(mean_embeddings)
            all_labels.append(labels)
            num += 1

        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        return all_embeddings, all_labels

    def create_sentence_transformer_embeddings(self):
        """
        Create SentenceTransformer embeddings for the texts in the dataset.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        all_embeddings = []
        all_labels = []
        sent_trans_model = SentenceTransformer("paraphrase-mpnet-base-v2")
        for texts, labels in self.dataloader:
            embeddings = sent_trans_model.encode(texts, show_progress_bar=False)
            all_embeddings.append(embeddings)
            all_labels.append(labels)

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = torch.cat(all_labels, dim=0).numpy()

        return all_embeddings, all_labels

    def create_word2vec_embeddings(self):
        """
        Create word2vec embeddings for the texts in the dataset.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        model_name = 'word2vec-google-news-300'
        model = api.load(model_name)

        texts = self.dataset.texts
        tokenized_texts = [text.lower().split() for text in texts]

        text_vectors = []
        labels = []
        for text, label in zip(tokenized_texts, self.dataset.labels):
            word_vectors = [model.get_vector(word) for word in text if word in model.key_to_index]
            if len(word_vectors) > 0:
                text_vector = sum(word_vectors) / len(word_vectors)
                text_vectors.append(text_vector)
                labels.append(label)

        embeddings = np.array(text_vectors)

        return embeddings, np.array(labels)

    def create_tf_idf_embeddings(self):
        """
        Create TF-IDF embeddings for the texts in the dataset.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        texts = self.dataset.texts
        labels = self.dataset.labels

        tokenized_texts = [word_tokenize(text.lower()) for text in texts]

        stop_words = set(stopwords.words('english'))
        filtered_texts = [[word for word in text if word not in stop_words] for text in tokenized_texts]

        lemmatizer = WordNetLemmatizer()
        lemmatized_texts = [[lemmatizer.lemmatize(word) for word in text] for text in filtered_texts]

        dictionary = corpora.Dictionary(lemmatized_texts)

        corpus = [dictionary.doc2bow(text) for text in lemmatized_texts]

        tfidf = models.TfidfModel(corpus)

        corpus_tfidf = tfidf[corpus]

        return matutils.corpus2dense(corpus_tfidf, num_terms=len(dictionary)).T, np.array(labels)

    def create_fasttext_embeddings(self):
        """
        Create fastText embeddings for the texts in the dataset.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        ft = fasttext.load_model('cc.en.300.bin')

        embeddings = []
        for texts, labels in self.dataloader:
            batch_embeddings = []
            preprocessed_texts = [preprocess_text(text) for text in texts]

            for text in preprocessed_texts:
                embedding = ft.get_sentence_vector(text)
                batch_embeddings.append(embedding)
            embeddings.append(np.array(batch_embeddings))

        embeddings = np.vstack(embeddings)

        return embeddings, np.array(self.dataset.labels)

    def create_base_embeddings(self):
        """
        Create TF-IDF embeddings using the base implementation.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding
        """
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(self.dataset.texts).toarray()

        return embeddings, np.array(self.dataset.labels)

    def create_BERT_long(self):
        """
        Create BERT embeddings for long texts in the dataset by splitting them into chunks.

        Returns:
            np.ndarray, np.ndarray: The generated embeddings and their corresponding labels.
        """
        model = BertModel.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

        all_embeddings = []
        all_labels = []

        max_length = 1000
        chunks = max_length // 500
        for texts, labels in self.dataloader:
            encoded_texts = []
            attention_masks = []

            for text in texts:
                # Tokenize the text
                tokens = tokenizer.encode(text, add_special_tokens=True, truncation=True)

                # Pad or truncate the tokens to the desired max length
                if len(tokens) <= max_length:
                    padded_tokens = tokens + [tokenizer.pad_token_id] * (max_length - len(tokens))
                    attn_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                else:
                    padded_tokens = tokens[:max_length]
                    attn_mask = [1] * max_length

                encoded_texts.append(padded_tokens)
                attention_masks.append(attn_mask)

            input_ids = torch.tensor(encoded_texts)
            attention_masks = torch.tensor(attention_masks)

            split_inp_ids = torch.chunk(input_ids, chunks=chunks, dim=1)
            split_attention_masks = torch.chunk(attention_masks, chunks=chunks, dim=1)

            model.eval()

            emb = []
            for inp_ids, attn_mask in zip(split_inp_ids, split_attention_masks):
                with torch.no_grad():
                    outputs = model(inp_ids, attention_mask=attn_mask)
                    embeddings = outputs['last_hidden_state']
                    mean_embeddings = torch.mean(embeddings, dim=1)
                    emb.append(mean_embeddings)

            mean_embeddings = torch.cat(emb, dim=0)

            all_embeddings.append(mean_embeddings)
            all_labels.append(labels)


        all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
        reshaped_embeddings = all_embeddings.reshape(-1, chunks, 1024)
        averaged_embeddings = np.mean(reshaped_embeddings, axis=1)
        all_labels = torch.cat(all_labels, dim=0).numpy()

        return averaged_embeddings, all_labels

def preprocess_text(text):
    """
        Preprocess the input text by removing stop words and punctuation.

        Args:
            text (str): The input text.

        Returns:
            str: The preprocessed text.
        """
    stop_words = set(stopwords.words('english'))
    # Tokenize the text into words
    words = nltk.word_tokenize(text)

    # Remove stop words and punctuation
    filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word not in punctuation]

    # Join the filtered words back into a string
    processed_text = ' '.join(filtered_words)

    return processed_text

