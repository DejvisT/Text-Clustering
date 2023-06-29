from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer, f1_score


def evaluate_clustering(data, true_labels, cluster_labels, base=False):
    """
        Evaluate the performance of clustering using a support vector machine (SVM) classifier.

        Args:
            data (numpy.ndarray): The input data for evaluation.
            true_labels (numpy.ndarray): The true labels of the data.
            cluster_labels (numpy.ndarray): The predicted cluster labels of the data.
            base (bool, optional): Indicates whether the clustering is based on base embeddings or not. Defaults to False.

        Returns:
            dict: A dictionary containing the evaluation metrics.
                - 'f1_score': The average F1-score.
                - 'accuracy': The average accuracy.
                - 'precision': The average precision.
                - 'recall': The average recall.
    """
    if not base:
        cluster_labels = cluster_labels.reshape(-1, 1)
        combined_features = np.concatenate((data, cluster_labels), axis=1)

    svm_classifier = SVC()

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro'),
        'recall': make_scorer(recall_score, average='macro'),
        'f1_score': make_scorer(f1_score, average='macro')
    }

    kf = KFold(n_splits=5, shuffle=True)

    if not base:
        scores = cross_validate(svm_classifier, combined_features, true_labels, cv=kf, scoring=scoring)
    else:
        scores = cross_validate(svm_classifier, data, true_labels, cv=kf, scoring=scoring)
    average_f1_score = np.mean(scores['test_f1_score'])
    average_accuracy = np.mean(scores['test_accuracy'])
    average_precision = np.mean(scores['test_precision'])
    average_recall = np.mean(scores['test_recall'])

    print("Average F1-score:", average_f1_score)
    print("Average Accuracy:", average_accuracy)
    print("Average Precision:", average_precision)
    print("Average Recall:", average_recall)

    return {'f1_score': average_f1_score, 'accuracy': average_accuracy, 'precision': average_precision, 'recall': average_recall}



