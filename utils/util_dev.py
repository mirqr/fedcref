import hashlib

import numpy as np  
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

# given an array of integer, return a map with lowest integer to 0, second lowest to 1, etc

def map_to_0_1(arr):
    arr_origin = arr.copy()
    sorted_unique_arr = np.sort(np.unique(arr))
    dic_map = dict(zip(sorted_unique_arr, range(len(sorted_unique_arr))))
    mapped_arr = np.array([dic_map[i] for i in arr_origin])
    return dic_map, mapped_arr

def revert_map(dic_map, arr):
    try:
        reverted = np.array([k for k,v in dic_map.items()])[arr]
        return reverted
    except:
        print('arr', arr)
        print('dic_map', dic_map)
        raise # re-raise the last exception




def unsupervised_clustering_accuracy(y, y_pred): # y must be 0,1,....like a cluster
    """Unsupervised Clustering Accuracy
    """
    #print('len y_pred', len(y_pred))
    #print('len y', len(y))
    assert len(y_pred) == len(y)
    y_pred_origin = y_pred.copy()
    dic_map, y_pred = map_to_0_1(y_pred) # converti le label vere in 0,1,2 ecc
    
    u = np.unique(np.concatenate((y, y_pred))) # take all the possible values
    n_clusters = len(u) # number of clusters (unique values)
    mapping = dict(zip(u, range(n_clusters))) # map each value to a number
    reward_matrix = np.zeros((n_clusters, n_clusters), dtype=np.int64) # matrix of zeros
    for y_pred_, y_ in zip(y_pred, y): # for each pair of values
        if y_ in mapping: # if the value is in the mapping
            reward_matrix[mapping[y_pred_], mapping[y_]] += 1  # add 1 to the matrix 
    cost_matrix = reward_matrix.max() - reward_matrix # invert the matrix (max - matrix) so that the hungarian algorithm can work 

    row_assign, col_assign = linear_sum_assignment(cost_matrix)
    
    #print('cost matrix')
    #print(cost_matrix)

    # Construct optimal assignments matrix
    row_assign = row_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape
    col_assign = col_assign.reshape((-1, 1))  # (n,) to (n, 1) reshape

    optimal_reward = reward_matrix[row_assign, col_assign].sum() * 1.0
    
    row_assign = revert_map(dic_map, row_assign)
    assignments = np.concatenate((row_assign, col_assign), axis=1)

    return optimal_reward / y_pred.size, assignments 



def errs(x1,x2, kind='eucl'):
    if kind == 'eucl':
        r = np.sqrt(np.sum((x1 - x2)**2, axis=1)) # l2 norm. ALSO np.linalg.norm(x1 - x2, ord=2, axis=1) # l2 norm
    elif kind == 'abs':
        r = np.mean(np.abs(x1 - x2), axis=1) #  mean absolute error
    elif kind == 'mse':
        r = np.mean((x1 - x2)**2, axis=1) #  mse
    else:
        raise ValueError('kind not recognized')
    return r

def predict_cluster_proximity_centers(data, centers, metric='euclidean'):
    # check same shape 
    if data.shape[1] != centers.shape[1]:
        raise ValueError('data and centers must have same shape')
    
    # matrix of distances between each point and each centroid

    m = cdist(data, centers, metric=metric)
    # if metric == 'cosine': # cosine similarity mi sembra gia inverito
    y_cc = np.argmin(m, axis=1)
    return y_cc


def get_sha256_hash(data):
        return hashlib.sha256(data.tobytes()).hexdigest()