import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from joblib import load, dump

from joblib import Memory
cachedir = '.cachejoblib/'
memory = Memory(cachedir, verbose=0)

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.datasets import mnist, fashion_mnist
from extra_keras_datasets import emnist,kmnist





# load mnist or fashion mnist dataset with if on string
#@memory.cache
def get_dataset(dataset_name: str, flatten_and_normalize=False):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset_name == 'mnist_and_fmnist_merged': #  20 classes
        (x_train0, y_train0), (x_test0, y_test0) = tf.keras.datasets.mnist.load_data()
        (x_train1, y_train1), (x_test1, y_test1) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = np.concatenate((x_train0, x_train1))
        y_train = np.concatenate((y_train0, y_train1+10)) # shift labels by 10
        x_test = np.concatenate((x_test0, x_test1))
        y_test = np.concatenate((y_test0, y_test1+10)) # shift labels by 10
    elif dataset_name == 'emnist_digits': #  
        (x_train, y_train), (x_test, y_test) = emnist.load_data(type='digits')
    elif dataset_name == 'emnist_letters': #
        (x_train, y_train), (x_test, y_test) = emnist.load_data(type='letters')
    elif dataset_name == 'emnist_byclass':
        (x_train, y_train), (x_test, y_test) = emnist.load_data(type='byclass')
    elif dataset_name == 'kmnist': 
        (x_train, y_train), (x_test, y_test) = kmnist.load_data(type='kmnist')
    elif dataset_name == 'kmnist-49':
        (x_train, y_train), (x_test, y_test) = kmnist.load_data(type='k49') 
        unique, counts = np.unique(y_train, return_counts=True)
        # remove classes with less than 6000 samples
        classes_to_remove = unique[counts < 6000]
        x_train = np.delete(x_train, np.where(np.isin(y_train, classes_to_remove)), axis=0)
        y_train = np.delete(y_train, np.where(np.isin(y_train, classes_to_remove)), axis=0)
        x_test = np.delete(x_test, np.where(np.isin(y_test, classes_to_remove)), axis=0)
        y_test = np.delete(y_test, np.where(np.isin(y_test, classes_to_remove)), axis=0)

        unique, counts = np.unique(y_train, return_counts=True)
        print(dict(zip(unique, counts)))

    elif dataset_name == 'femnist-iid': 
        emnist_dict = load('emnist_dict.joblib')
        # {client : (x_train, y_train, x_test, y_test)}
        # merge all clients in one x_train, y_train, x_test, y_test and shuffle
        x_train, y_train, x_test, y_test = [], [], [], []
        for cl_id, (x_train_tmp, y_train_tmp, x_test_tmp, y_test_tmp) in emnist_dict.items():
            x_train.append(x_train_tmp)
            y_train.append(y_train_tmp)
            x_test.append(x_test_tmp)
            y_test.append(y_test_tmp)
        # Concatenate all at once
        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)
        # flatten
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784) 

        # shuffle
        #x_train, y_train = shuffle(x_train, y_train, random_state=42)
        flatten_and_normalize = False # TODO 
    else:
        raise('Bad dataset name', dataset_name)
    
    # normalization hardcoded for mnist-like datasets
    if flatten_and_normalize:
        # flatten and normalize
        n_features = np.prod(x_train.shape[1:])
        x_train = x_train.reshape(x_train.shape[0], n_features) / 255.0
        x_test = x_test.reshape(x_test.shape[0], n_features) / 255.0

    return x_train, y_train, x_test, y_test



# return like a list of (x, y) tuples, one for each client
@memory.cache(ignore=['x_train'])
def get_system(num_clients, x_train, y_train, num_min_class, num_max_class, unique_classes=None,
               min_samples_per_class=500, max_samples_per_class=600, replace_sample=False, seed=None):
    
    if seed is not None:
        np.random.seed(seed)

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("Feature and label data must have the same number of samples.")

    # Create a copy of the training data to avoid in-place modifications
    x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
    if unique_classes is None:
        unique_classes = np.unique(y_train_copy)

    list_chosen_classes = []
    clients_data = []

    for _ in range(num_clients):
        # Ensure unique class combinations for each client
        unique_combination_found = False
        while not unique_combination_found:
            num_classes = np.random.randint(num_min_class, num_max_class + 1) 
            chosen_classes = np.random.choice(unique_classes, num_classes, replace=False)
            chosen_classes = np.sort(chosen_classes)

            if not np.any([np.array_equal(chosen_classes, c) for c in list_chosen_classes]):
                unique_combination_found = True
                list_chosen_classes.append(chosen_classes)
        #print('chosen_classes', chosen_classes)

        # Sample number of samples per class
        # returns for each class (size) a number of samples
        num_samples = np.random.randint(min_samples_per_class, max_samples_per_class, size=num_classes)  # more correct max_samples_per_class +1

        # Initialize empty arrays for x and y
        client_x, client_y = [], []

        # Sample data for each class
        for i, cls in enumerate(chosen_classes):
            num = num_samples[i] # number of samples for this class
            idx = np.where(y_train_copy == cls)[0] # indices of samples for this class

            if len(idx) < num:
                #raise ValueError(f"Not enough samples in class {cls} to meet the sampling requirements!")
                print(f"Not enough samples in class {cls} to meet the sampling requirements!")
                print("refill the dataset")
                
                # refill the dataset
                x_train_copy, y_train_copy = x_train.copy(), y_train.copy()
                idx = np.where(y_train_copy == cls)[0]

            #print('Try to sample', num, 'samples from class', cls, 'with', len(idx), 'samples')
            chosen_idx = np.random.choice(idx, num, replace=replace_sample)
            client_x.append(x_train_copy[chosen_idx])
            client_y.append(y_train_copy[chosen_idx])

            # Remove the chosen samples if not sampling with replacement
            if not replace_sample:
                x_train_copy = np.delete(x_train_copy, chosen_idx, axis=0)
                y_train_copy = np.delete(y_train_copy, chosen_idx, axis=0)

        # Concatenate the chosen samples
        client_x = np.concatenate(client_x, axis=0)
        client_y = np.concatenate(client_y, axis=0)

        # Shuffle the data
        client_x, client_y = shuffle(client_x, client_y, random_state=seed)
        clients_data.append((client_x, client_y))
        

    return clients_data


def _stack_user_emnist(all_ds, super_user_index, user_start, user_end):
    dict_new = {}
    x_train, y_train, x_test, y_test = [], [], [], []
    for client in range(user_start, user_end): # user_start included, user_end not included
        x_train_tmp, y_train_tmp, x_test_tmp, y_test_tmp = all_ds[client]
        x_train.append(x_train_tmp)
        y_train.append(y_train_tmp)
        x_test.append(x_test_tmp)
        y_test.append(y_test_tmp)
    # Concatenate all at once
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    print(f"Stacked {user_start} to {user_end-1} users")
    print(f"super_user_index: {super_user_index}", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    dict_new[super_user_index] = (x_train, y_train, x_test, y_test)
    return dict_new

def _devs_emnist(emnist_dict, users_to_stack):
    # return a new dict with the stacked users by class samples
    final_dict = {}
    for i in range(0, len(emnist_dict) - users_to_stack, users_to_stack):
        mm = _stack_user_emnist(emnist_dict, i//users_to_stack, i, i+users_to_stack)
        final_dict.update(mm)
    return final_dict

def get_system_femnist(num_clients, num_min_class, num_max_class, unique_classes=None,
               min_samples_per_class=500, max_samples_per_class=600, replace_sample=False, seed=None):
    
    # load emnist only digits dataset as a dictionary of  {client : (x_train, y_train, x_test, y_test)}
    emnist_dict = load('emnist_dict.joblib')
    # its already normalized but not flattened
    for cl_id, (x_train, y_train, x_test, y_test) in emnist_dict.items():
        # reshape from 28x28 to 784
        x_train = x_train.reshape(-1, 784)
        x_test = x_test.reshape(-1, 784) 
        emnist_dict[cl_id] = (x_train, y_train, x_test, y_test)
    
    users_to_stack = 55 # computer how many users to stack in order to have ~500 samples per class.
    emnist_dict_merged = _devs_emnist(emnist_dict, 55)

    # TODO min_samples_per_class, max_samples_per_class not used. Improve
    
    if seed is not None:
        np.random.seed(seed)

    unique_classes = 10
    
    list_chosen_classes = []
    clients_data = []

    for client_id in range(num_clients):
        # Ensure unique class combinations for each client
        unique_combination_found = False
        while not unique_combination_found:
            num_classes = np.random.randint(num_min_class, num_max_class + 1) 
            chosen_classes = np.random.choice(unique_classes, num_classes, replace=False)
            chosen_classes = np.sort(chosen_classes)

            if not np.any([np.array_equal(chosen_classes, c) for c in list_chosen_classes]):
                unique_combination_found = True
                list_chosen_classes.append(chosen_classes)
        #print('chosen_classes', chosen_classes)

        #chosen_classes
        x_train, y_train, _, _ = emnist_dict_merged[client_id]
        # take only the chosen classes
        idx = np.isin(y_train, chosen_classes)
        client_x = x_train[idx]
        client_y = y_train[idx] 
        # shuffle
        client_x, client_y = shuffle(client_x, client_y, random_state=seed)

        #print(x_train.shape, y_train.shape)
        # print unique classes and count
        unique, counts = np.unique(client_y, return_counts=True)
        print(dict(zip(unique, counts)))
        clients_data.append((client_x, client_y))

    return clients_data




def get_to_share(perc, local_sample_class, num_clients_having_class):
    p = perc
    q = local_sample_class
    n = num_clients_having_class
    sample_to_share = (p * q) / (n + p - n * p) 
    print(f"perc={p}, local_sample_class={q}, num_clients_having_class={n}, sample_to_share={sample_to_share}")
    return int(sample_to_share)


def print_uniques(y, string = ''):
    unique, counts = np.unique(y, return_counts=True)
    print(string, dict(zip(unique, counts)))


class client_obj:
    def __init__(self, x, y):
        x = x.copy()
        y = y.copy()
        self.name = "_".join(map(str, sorted(np.unique(y))))
        # a dictionary with key=class and value= subset of x with class class
        self.dict_classes = {cl: x[y==cl] for cl in np.unique(y)}

    def change_samples(self, cl, x):
        self.dict_classes[cl] = x

    def append_samples(self, cl, x):
        self.dict_classes[cl] = np.concatenate([self.dict_classes[cl], x])
        print(f"cl={cl}, len={len(self.dict_classes[cl])}")
    
    def contains(self, cl):
        return cl in self.dict_classes
    
    def get_first_samples(self, cl, n):
        x_cl_share = self.dict_classes[cl][:n]
        x_cl_remaining = self.dict_classes[cl][n:]
        return x_cl_share, x_cl_remaining
    
    def get_x_y(self): # verify if it is correct
        x = np.concatenate(list(self.dict_classes.values()))
        y = np.concatenate([np.full(len(x), cl) for cl, x in self.dict_classes.items()])
        return x, y
    


def introduce_overlap_new(clients_data, classes, overlap):
    p = overlap
    clients_start = {}
    clients_end = {}
    for x, y in clients_data:
        cl = client_obj(x, y)
        clients_start[cl.name] = cl
        cl = client_obj(x, y) # create a new object
        clients_end[cl.name] = cl
    
    for cl in classes:
        clients_having_cl = [c for c in clients_start.values() if c.contains(cl)]

        num_clients_having_class = len(clients_having_cl)

        # local_sample_class = average number of samples of class cl in clients_having_cl
        local_sample_class = np.mean([len(c.dict_classes[cl]) for c in clients_having_cl])

        #print(f"cl={cl}, num_clients_having_class={num_clients_having_class}, local_sample_class={local_sample_class}")
        sample_to_share = get_to_share(p, local_sample_class, num_clients_having_class) # PER CLIENT CLASS

        common_x, common_y = [], []
        for name, c in clients_start.items():
            if c.contains(cl):
                x_cl_share, x_remaining = c.get_first_samples(cl, sample_to_share)
            
                # substitute x,y with x_new, y_new
                clients_end[name].change_samples(cl, x_remaining)
                
                #print_uniques(y_new, "new y")
                common_x.append(x_cl_share)

                #print("----")
        
        common_x = np.concatenate(common_x)
        #print(f"cl={cl}, common_x={len(common_x)}")
        #print("----")
        for name, c in clients_end.items():
            if c.contains(cl):
                c.append_samples(cl, common_x)
        

    clients_data_new = []
    for name, c in clients_end.items():
        x, y = c.get_x_y()
        clients_data_new.append((x, y))
        
    return clients_data_new

