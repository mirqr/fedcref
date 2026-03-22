import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# recall clusters labels are arbitrary, so we need to find the best matching 
# between predicted (y_pred) and true labels (y) to compute accuracy
def unsupervised_clustering_accuracy(y, y_pred):
    """Unsupervised clustering accuracy via optimal assignment (Hungarian algorithm).
    Returns accuracy and (pred_label -> true_label) assignment pairs.
    """
    assert len(y) == len(y_pred)

    # returns the sorted unique values AND the indices that reconstruct 
    # the original array. So if y_pred = [5, 5, 9, 9], then y_pred_labels = [5, 9] and y_pred_mapped = [0, 0, 1, 1].
    y_true_labels, y_true = np.unique(y, return_inverse=True)
    y_pred_labels, y_pred_mapped = np.unique(y_pred, return_inverse=True)

    # reward[i, j] = how many samples are in predicted cluster i AND true cluster j. 
    # It counts co-occurrences. np.add.at does this without a loop.
    n = max(len(y_true_labels), len(y_pred_labels))
    reward = np.zeros((n, n), dtype=np.int64)
    np.add.at(reward, (y_pred_mapped, y_true), 1)

    # Finds the assignment of predicted clusters → true clusters 
    # that maximizes total reward (by minimizing max - reward). This is the optimal label permutation.
    row_ind, col_ind = linear_sum_assignment(reward.max() - reward)

    # Sum of correctly matched samples divided by total. 
    # assignments maps back to the original label values (e.g. [[5, 1], [9, 0]] meaning "predicted cluster 5 = true cluster 1").
    accuracy = reward[row_ind, col_ind].sum() / len(y)
    assignments = np.column_stack([y_pred_labels[row_ind], y_true_labels[col_ind]])

    return accuracy, assignments


def errs(x1, x2, kind='eucl'):
    if kind == 'eucl':
        r = np.sqrt(np.sum((x1 - x2)**2, axis=1))
    elif kind == 'abs':
        r = np.mean(np.abs(x1 - x2), axis=1)
    elif kind == 'mse':
        r = np.mean((x1 - x2)**2, axis=1)
    else:
        raise ValueError('kind not recognized')
    return r


def predict_cluster_proximity_centers(data, centers, metric='euclidean'):
    if data.shape[1] != centers.shape[1]:
        raise ValueError('data and centers must have same shape')
    m = cdist(data, centers, metric=metric)
    return np.argmin(m, axis=1)


def get_sha256_hash(data):
    return hashlib.sha256(data.tobytes()).hexdigest()
