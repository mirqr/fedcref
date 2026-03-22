
import multiprocessing
import os
import random
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"    

import numpy as np
import pandas as pd
import ray
import tensorflow as tf
from joblib import Memory

cachedir = ".cachejoblib/"
cachedir = "/dev/shm/joblib_cache"  # Use in-memory storage for caching
memory = Memory(location=cachedir, verbose=2)



import flwr as fl
from pebble import concurrent
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.cluster import normalized_mutual_info_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from keras_dec import DeepEmbeddingClustering, cluster_acc
from utils.fl_client import *
from utils.fl_server import *
from utils.util_data import get_dataset
from utils.util_dev import *
from utils.util_models import *


@concurrent.process(context=multiprocessing.get_context('spawn'))
def fit_dec_ray(x_train, y_train, data_name):
    return fit_dec_static_kmnist(x_train, y_train, data_name, iter_max=1e4)


# static fit dec 
@memory.cache(ignore=['x_train'])
def fit_dec_static_kmnist(x_train, y_train, data_name, iter_max):
    layerwise_pretrain_iters=5000; finetune_iters=10000; overwrite_pretrain=True; overwrite=True
    if data_name == 'kmnist-49':
        # given the classes in y_train, take all that samples from o
        x_train_all, y_train_all, x_test_all, y_test_all = get_dataset(data_name, flatten_and_normalize=True)
        cls = np.unique(y_train)
        x_train_real = x_train_all[np.isin(y_train_all, cls)]
        y_train_real = y_train_all[np.isin(y_train_all, cls)]
        # print uniques count
        unique, counts = np.unique(y_train_real, return_counts=True)
        print(dict(zip(unique, counts)))
        X = x_train_real
        Y = y_train_real
    else:
        X = x_train
        Y = y_train
    X_to_predict = x_train
    # hash for y
    y_hash = get_sha256_hash(Y)
    kk = len(np.unique(Y))
    c = DeepEmbeddingClustering(n_clusters=kk, input_dim=784, dev_name=y_hash, path_base='saved/', overwrite_pretrain=overwrite_pretrain)
    c.initialize(X, finetune_iters=finetune_iters, layerwise_pretrain_iters=layerwise_pretrain_iters, save_autoencoder=True)
    c.cluster(X, y=Y, iter_max=iter_max, overwrite=overwrite)


    q = c.DEC.predict(X_to_predict, verbose=0)
    y_clust = q.argmax(1)
    return y_clust

# static fit dec 
@memory.cache(ignore=['x_train'])
def fit_dec_static(x_train, y_train, data_name):
    layerwise_pretrain_iters=5000; finetune_iters=10000; iter_max=1e4; overwrite_pretrain=False; overwrite=False
    X = x_train
    Y = y_train
    # hash for y
    y_hash = get_sha256_hash(Y)
    kk = len(np.unique(Y))
    c = DeepEmbeddingClustering(n_clusters=kk, input_dim=784, dev_name=y_hash, path_base='saved/', overwrite_pretrain=overwrite_pretrain)
    c.initialize(X, finetune_iters=finetune_iters, layerwise_pretrain_iters=layerwise_pretrain_iters, save_autoencoder=True)
    c.cluster(X, y=Y, iter_max=iter_max, overwrite=overwrite)
    q = c.DEC.predict(X, verbose=0)
    y_clust = q.argmax(1)
    return y_clust

#@memory.cache
def train_single_autoencoder(x):
    model = get_model_autoencoder()
    model.fit(x, x, 
        epochs=20, shuffle=True, verbose=0,
                #validation_split=0.0, 
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=4),]
                ) # changed from val_loss after removing validation_split

    return model.get_weights()

def _train_autoencoders(x_data: dict):
    models = {}
    for lab, x in x_data.items():
        models[lab] = train_single_autoencoder(x)
    return models

@ray.remote
def train_autoencoders_ray(x_data: dict):
    return _train_autoencoders(x_data)

@ray.remote
def predict_single_autoencoder_parallel_ray(dev_name, y , models_w, datasets, devs_stop: dict):
    y_hash = get_sha256_hash(y)
    cache = {}
    print('predict_single_autoencoder_parallel', dev_name)
    for lab, mod_w in models_w:
        m = get_model_autoencoder()
        m.set_weights(mod_w)
        for dev2, device_datasets in datasets.items():
            if devs_stop[dev2] and devs_stop[dev_name]:
                print('skip', dev2, dev_name)
                continue
            for lab2, x2 in device_datasets.items():
                cache_key = f"{dev_name}-{lab}_{dev2}-{lab2}_model_={m.get_model_hash()}" # TODO attenzione
                pred = m.predict(x2, verbose=0, batch_size=128)
                errors = np.mean(np.square(pred - x2), axis=(1)) # #TODO carefull. must be the same as in the model
                cache[cache_key] = errors.tolist()
    return cache




class DevManager:
    def __init__(self, devs=None):
        self.devices = {}
        for dev in devs:
            print('add device', dev.name)
            self.add_device(dev)
        
        self.selected = {}
        # self.selected take 20 devices
        self.list_split = [self.devices]
        self.selected = {k: self.devices[k] for k in list(self.devices)[:20]}

        self.num_communities = []
        self.num_isolated = []

    def dec_all_ray(self):
        all_futures = []
        for dev in self.devices.values():
            future = fit_dec_ray(dev.x_train, dev.y_train, dev.data_name)
            # future = fit_dec_ray.remote(dev.x_train, dev.y_train) # ray_version
            all_futures.append((dev, future))
        for dev, future in all_futures:
            dev.y_clust = future.result() 
            #dev.y_clust = ray.get(future)
            print('fit_dec_ray', dev.name)

    
    def split(self):
        list_split = []
        for devices in self.list_split:
            # split devices in 2
            list_split.append({k: devices[k] for k in list(devices)[:int(len(devices)/2)]})
            list_split.append({k: devices[k] for k in list(devices)[int(len(devices)/2):]})
        for devices in list_split:
            print('split', len(devices))
        self.list_split = list_split


    def add_device(self, dev):
        self.devices[dev.name] = dev

    def get_device(self, dev_name):
        return self.devices[dev_name]
    
    def get_dev_number(self):
        return len(self.devices)

    def get_num_ideal_communities(self):
        # classes present at least in 2 devices
        dict_class_num = {} # class -> num devices with that class
        for dev in self.devices.values():
            for lab in np.unique(dev.y_train):
                dict_class_num[lab] = dict_class_num.get(lab, 0) + 1
        return len([k for k, v in dict_class_num.items() if v >= 2])

    
    def get_dev_cluster_number(self):
        return sum([dev.KK for dev in self.devices.values()])
    
    def _collect_all_current_datasets(self, selected = None):
        datasets = {}
        ds = selected if selected is not None else self.devices
        for dev in ds.values(): # self. ...
            datasets[dev.name] = {}  # Initialize a sub-dictionary for each device
            for lab in np.unique(dev.y_clust):
                datasets[dev.name][lab] = dev.x_dic_clust[lab]
        return datasets
    
    def _collect_all_current_models(self, selected = None):
        all_models_w = {}
        ds = selected if selected is not None else self.devices
        for dev in ds.values():
            all_models_w[dev.name] = {}
            for lab in np.unique(dev.y_clust):
                all_models_w[dev.name][lab] = dev.mods_clust[lab].get_weights()
        return all_models_w


    
    def reclustered_percentage(self):
        return np.mean([dev.reclustered_times for dev in self.devices.values()])
    

    
    def set_num_isolated(self, num_isolated):
        self.num_isolated.append(num_isolated)

    def set_num_communities(self, num_communities):
        self.num_communities.append(num_communities)
    
    def avg_auto_ari(self):
        return np.mean([dev.auto_ari for dev in self.devices.values()])
    
    def avg_auto_acc(self):
        return np.mean([dev.auto_acc for dev in self.devices.values()])
    

    def should_stop(self, values, max_percentage_diff=10, consecutive_limit=3):
        # Return False immediately if there are not enough values
        if len(values) < consecutive_limit + 1 :
            print('not enough values to stop', values)
            return False

        consecutive_count = 0  # Initialize the counter

        # iterate on the last consecutive_limit values
        
        for i in range(len(values)-consecutive_limit, len(values)):
            percentage_diff = abs(values[i] - values[i-1]) / max(values[i-1], 1) * 100
            print("Percentage difference between {} and {} is {:.2f}%".format(values[i-1], values[i], percentage_diff))

            if percentage_diff <= max_percentage_diff:
                consecutive_count += 1
                if consecutive_count >= consecutive_limit :  # Adjust for zero-indexing
                    return True  # Stop condition met
            else:
                consecutive_count = 0  # Reset the counter if condition not met

        return False  # Continue if condition never met

        
    def write_all_y_clust(self):
        for dev in self.devices.values():
            # write on a file all the y_clust
            filename = f'clusts/{dev.data_name}/y_clust_{dev.name}.npy'
            np.save(filename, dev.y_clust)
            print('write', filename)
    
    def read_all_y_clust(self):
        for dev in self.devices.values():
            # write on a file all the y_clust
            filename = f'clusts/{dev.data_name}/y_clust_{dev.name}.npy'
            dev.y_clust = np.load(filename)
            print('read', filename)
    

    def get_accuracy_avg(self, round_dig = 3):
        acc = []
        for dev in self.devices.values():
            acc.append(dev.acc)
        acc_mean = np.mean(acc)
        return round(acc_mean, round_dig)
        
    
    # TRAIN
    def train_local_autoencoders_parallel_ray(self):
        print('train_all')
        all_futures = []

        lll = [dev for dev in self.devices.values() if not dev.stop_train ]
        print('--------------------------------------------------train_all', len(lll))
        for dev in lll: #self.devices.values():
            x_data = dev.prepare_data_for_training(train_only_labs=None)
            future = train_autoencoders_ray.remote(x_data)
            all_futures.append((dev, future))
        
        for dev, future in all_futures:
            models = {}
            mod_weights = ray.get(future)
            for lab, weights in mod_weights.items():
                models[lab] = get_model_autoencoder()
                models[lab].set_weights(weights)
            dev.assign_trained_local_models(models)

    def stop_stable_devices(self):
        for dev in self.devices.values():
            if dev.is_stable():
                dev.stop_train = True
                print('STOP TRAIN', dev.name)

    # END TRAIN

    def predict_local_parallel_ray(self):
        for selected in self.list_split:
            datasets = self._collect_all_current_datasets(selected)
            datasets = ray.put(datasets) # its an ID object, can be passed to remote functions
            # take 2 first elements of datasets dictionary
            all_futures = []
            
            # dictionary of devices name - stable
            devs_stop = {dev.name: dev.stop_train for dev in selected.values()}

            # Collect futures for all devices first
            for dev in selected.values(): # devices TODO
                y = dev.y_clust 
                models_w = [(i, model.get_weights()) for i, model in dev.mods_clust.items()]
                future = predict_single_autoencoder_parallel_ray.remote(dev.name, y, models_w, datasets, devs_stop)
                all_futures.append((dev, future))
            for dev, future in all_futures:
                cache = ray.get(future)
                dev.dev_cache.update(cache)
            print('predict split DONE')
        
        print('predict_local_parallel DONE')
    


    def associate_devs(self, percentile=75, th=0.3, vv=True, use_only_min=True, use_perfect=False):
        devices_list = list(self.devices.values()) 
        # initialize association_clust
        for dev in devices_list:
            dev.association_clust = {}
            dev.association_perfect = {}

        
        for i, dev1 in enumerate(devices_list):
            for j, dev2 in enumerate(devices_list[i+1:]): 
                self.association_two_perfect(dev1, dev2)
        
        if use_perfect: # new july
            for dev in devices_list:
                dev.association_clust = dev.association_perfect.copy()
                for class2, list2 in dev1.association_clust.items():
                    print()

            print(dev1.association_clust.keys())


            return

        # take max 20 devices
        for selected in self.list_split:
            devices_list = list(selected.values())
            for i, dev1 in enumerate(devices_list):
                for j, dev2 in enumerate(devices_list[i+1:]):
                    if dev1.is_stable() and dev2.is_stable():
                        pass
                    self.associate_two_devs_new(dev1, dev2, percentile=percentile, th=th, vv=vv, use_only_min=use_only_min)

    def association_two_perfect(self, dev1, dev2):
        for key1 in dev1.x_dic_true.keys(): 
            for key2 in dev2.x_dic_true.keys():
                if key1 == key2:
                    dev1.association_perfect.setdefault(key1, []).append((dev2.name, key2))
                    dev2.association_perfect.setdefault(key2, []).append((dev1.name, key1))
                
  


    def associate_two_devs_new(self, dev1, dev2, percentile=75, th=0.3, vv=True, use_only_min=True):
        # skip if dev1 and dev2 are the same
        precomputed_errors = {}
        m12 = np.zeros((len(dev1.x_dic_clust.keys()), len(dev2.x_dic_clust.keys())))
        m21 = np.zeros((len(dev2.x_dic_clust.keys()), len(dev1.x_dic_clust.keys())))
        

        for key1 in dev1.x_dic_clust.keys():
            for key2 in dev2.x_dic_clust.keys():
                x_1 = dev1.x_dic_clust[key1]
                mod_1 = dev1.mods_clust[key1]
                
                x_2 = dev2.x_dic_clust[key2]
                mod_2 = dev2.mods_clust[key2]

                
                err_1_mod_1_sampleswise = dev1.compute_reconstruction_errors_new(dev1.name, key1, mod_1, dev1.name, key1, x_1)
                err_1_mod_2_sampleswise = dev2.compute_reconstruction_errors_new(dev2.name, key2, mod_2, dev1.name, key1, x_1)
                # distance sampleswise
                dist_1 = np.abs(err_1_mod_1_sampleswise - err_1_mod_2_sampleswise)
                # normalize 0,1
                dist_1 = (dist_1 - np.min(dist_1)) / (np.max(dist_1) - np.min(dist_1))
                

                err_2_mod_2_sampleswise = dev2.compute_reconstruction_errors_new(dev2.name, key2, mod_2, dev2.name, key2, x_2)
                err_2_mod_1_sampleswise = dev1.compute_reconstruction_errors_new(dev1.name, key1, mod_1, dev2.name, key2, x_2)

                # distance sampleswise
                dist_2 = np.abs(err_2_mod_2_sampleswise - err_2_mod_1_sampleswise)
                # normalize 0,1
                dist_2 = (dist_2 - np.min(dist_2)) / (np.max(dist_2) - np.min(dist_2))



                condition = np.percentile(dist_1, percentile) < th and np.percentile(dist_2, percentile) < th

                from scipy import stats

                # Two-sample Kolmogorov-Smirnov test
                # default two-sided: The null hypothesis is that the two distributions are identical
                ks_stat1, ks_p_value1 = stats.ks_2samp(err_1_mod_1_sampleswise, err_1_mod_2_sampleswise) 
                
                ks_stat2, ks_p_value2 = stats.ks_2samp(err_2_mod_2_sampleswise, err_2_mod_1_sampleswise)

                if condition:
                    m12[key1, key2] = np.percentile(dist_1, percentile)
                    m21[key2, key1] = np.percentile(dist_2, percentile)
                    possibile_class1 = dev1.cluster_to_class(key1)
                    possibile_class2 = dev2.cluster_to_class(key2)

                    dev1.association_clust.setdefault(possibile_class1, []).append((dev2.name, possibile_class2))
                    dev2.association_clust.setdefault(possibile_class2, []).append((dev1.name, possibile_class1))
                    break # skip to next key1 # TODO
        


    # to avoid too much parallelism, do one federated model at a time
    def fed_communities_all_safe(self, devs, start_port=4000):
        device_dict = {dev.name: dev for dev in devs}
        lst = list(range(4000, 8000, 40))

        # Initialize federated models for each device
        for dev in devs:
            dev.fed_models = {}

        # Identify communities with more than one device
        communities = [comm for comm in Dev.find_communities_all(devs) if len(comm) > 1]

        # example community: [('dev1', 0), ('dev2', 0), ('dev3', 0)]

        # Create servers for each community
        for server_port, comm in enumerate(communities, start=1):
            comm = comm[:30]
            my_create_server_subproc.remote("localhost", port=str(start_port + server_port), num_clients=len(comm), num_rounds=15)

            time.sleep(5)
            futures = {}
            for dev_name, label in comm:
                dev = device_dict[dev_name]
                cluster = dev.class_to_cluster(label)
                x_train_portion = dev.x_dic_clust[cluster]
                key = f"{dev_name}_{label}"
                futures[key] = start_flower_client.remote(x_train_portion, label, dev_name, address="localhost", port=str(start_port + server_port))


            for dev_name, label in comm:
                dev = device_dict[dev_name]
                key = f"{dev_name}_{label}"
                dev.fed_models[label] = ray.get(futures[key])
            print('community', server_port, 'of', len(communities), 'DONE')


        # Compile one federated model per community
        all_fed_models = []
        for comm in communities:
            dev_name, label = comm[0]  # Taking first device in each community
            dev = device_dict[dev_name]
            weights = dev.fed_models[label]

            model = get_model_autoencoder()
            model.set_weights(weights)
            model.set_model_name(f"Fed{label} {dev_name}")
            all_fed_models.append(model)

        return all_fed_models



    def fed_communities_all(self, devs, start_port=4000):
        device_dict = {dev.name: dev for dev in devs}
        lst = list(range(4000, 8000, 40))

        # Initialize federated models for each device
        for dev in devs:
            dev.fed_models = {}

        # Identify communities with more than one device
        communities = [comm for comm in Dev.find_communities_all(devs) if len(comm) > 1]

        # example community: [('dev1', 0), ('dev2', 0), ('dev3', 0)]

        # Create servers for each community
        for server_port, comm in enumerate(communities, start=1):
            comm = comm[:30]
            my_create_server_subproc.remote("localhost", port=str(start_port + server_port), num_clients=len(comm), num_rounds=15)

        time.sleep(5)
        futures = {}
        for server_port, comm in enumerate(communities, start=1):
            comm = comm[:30]
            for dev_name, label in comm:
                dev = device_dict[dev_name]
                cluster = dev.class_to_cluster(label)
                x_train_portion = dev.x_dic_clust[cluster]
                key = f"{dev_name}_{label}"
                futures[key] = start_flower_client.remote(x_train_portion, label, dev_name, address="localhost", port=str(start_port + server_port))


        # Retrieve and set federated models
        for comm in communities:
            comm = comm[:30]
            for dev_name, label in comm:
                dev = device_dict[dev_name]
                key = f"{dev_name}_{label}"
                dev.fed_models[label] = ray.get(futures[key])
            print('community', server_port, 'of', len(communities), 'DONE')


        # Compile one federated model per community
        all_fed_models = []
        for comm in communities:
            dev_name, label = comm[0]  # Taking first device in each community
            dev = device_dict[dev_name]
            weights = dev.fed_models[label]

            model = get_model_autoencoder()
            model.set_weights(weights)
            model.set_model_name(f"Fed{label} {dev_name}")
            all_fed_models.append(model)

        return all_fed_models




class Dev:
    def __init__(self, data_name: str, x_train, y_train, inliers=None):
        # inliers from y_train, to list
        inliers = np.unique(y_train) # list and sort
        inliers = sorted(inliers.tolist())
        
        print('init inliers',inliers)
        
        self.data_name = data_name

        inliers_shuffled = inliers.copy()
        random.shuffle(inliers_shuffled)
        self.name = '_'.join([str(i) for i in inliers_shuffled])

        self.KK= len(inliers)
        self.x_train = x_train
        self.y_train = y_train
        self.filenameclust = 'clust_dec_y/'+self.data_name+'/y_clust_'+self.name
    
        self.fed_models = {}
        self.mods_clust = {}
        self.reclustered_times = 0

        self.dev_cache = {}
        self.y_clust_history = []
        self.auto_ari = 0
        self.auto_acc = 0 # accuracy from two consecutive y_clust
        self.stop_train = False
        self.acc_history = []

    def get_local_cache(self, key):
        return self.dev_cache.get(key, None)
    
    def save_local_cache(self, key, value):
        self.dev_cache[key] = value


    def fit_dec_new(self):
        y_clust = fit_dec_static(self.x_train, self.y_train)
        self.y_clust = y_clust

    def fit_dec(self, layerwise_pretrain_iters=5000,finetune_iters=10000,iter_max=1e4, overwrite_pretrain=False, overwrite=False):
        X = self.x_train
        Y = self.y_train
        self.c = DeepEmbeddingClustering(n_clusters=self.KK, input_dim=784, dev_name=self.name, path_base='saved/'+self.data_name, overwrite_pretrain=overwrite_pretrain)
        self.c.initialize(X, finetune_iters=finetune_iters, layerwise_pretrain_iters=layerwise_pretrain_iters, save_autoencoder=True)
        self.c.cluster(X, y=Y, iter_max=iter_max, overwrite=overwrite)
        self.y_clust = self.predict_cluster(self.x_train)

        # create directory if not exists
        os.makedirs(os.path.dirname(self.filenameclust), exist_ok=True)

        # save y_clust using data_name and name
        np.save(self.filenameclust, self.y_clust)

        
        
    def reload_temp(self):
        # reload clust
        self.y_clust = np.load(self.filenameclust+'.npy')


    def is_stable(self):
        # last two y_clust are similar (high unsupervised accuracy)
        return self.auto_acc > 0.85
    

    def simulate_clustering(self, dirtiness=0.0):
        # Determine the unique classes and assign them cluster numbers
        y = self.y_train
        unique_classes = np.unique(y)
        num_clusters = len(unique_classes)
        class_to_cluster = {cls: i for i, cls in enumerate(unique_classes)}

        # Initialize the cluster labels
        y_clust = np.zeros_like(y)

        # Map the true labels to cluster labels
        for cls, cluster in class_to_cluster.items():
            y_clust[y == cls] = cluster

        # Introduce 'dirtiness' in clustering
        for i in range(len(y_clust)):
            if np.random.rand() < dirtiness:
                # Get all clusters except the correct one
                other_clusters = [c for c in range(num_clusters) if c != y_clust[i]]

                # Assign to a random cluster from the other clusters
                y_clust[i] = np.random.choice(other_clusters)

        self.y_clust = y_clust
    


    def simulate_clustering_proximity(self, dirtiness=0.0):
        y = self.y_train
        # Determine the unique classes and assign them cluster numbers
        unique_classes = np.unique(y)
        num_clusters = len(unique_classes)
        class_to_cluster = {cls: i for i, cls in enumerate(unique_classes)}

        # Initialize the cluster labels
        y_clust = np.zeros_like(y)

        # Map the true labels to cluster labels
        for cls, cluster in class_to_cluster.items():
            y_clust[y == cls] = cluster

        # Introduce 'dirtiness' in clustering
        for i in range(len(y_clust)):
            if np.random.rand() < dirtiness:
                # Exclude the correct cluster
                other_clusters = [c for c in range(num_clusters) if c != y_clust[i]]

                # Create a probability distribution based on proximity
                proximity_weights = [1 / (1 + abs(cls - y[i])) for cls in other_clusters]
                total_weight = sum(proximity_weights)
                probabilities = [weight / total_weight for weight in proximity_weights]

                # Choose a new cluster based on the proximity-based weights
                y_clust[i] = np.random.choice(other_clusters, p=probabilities)

        self.y_clust = y_clust
        


    @property
    def y_clust(self):
        return self._y_clust
    
    @y_clust.getter
    def y_clust(self):
        return self._y_clust

    @y_clust.setter
    def y_clust(self, value):
        self._y_clust = value # the actual variable 
        self.acc, self.mapping = unsupervised_clustering_accuracy(self.y_clust, self.y_train) # MAPPING FUNZIONA bene solo se l'accuratezza e' buona
        self.acc_history.append(self.acc)
        self.x_dic_clust = {} # check if keep it
        for clust in np.unique(self.y_clust):
            self.x_dic_clust[clust] = self.get_x_train_for_y(self.y_clust, clust)
        self.y_clust_history.append(self.y_clust)
        if len(self.y_clust_history) >= 2:
            # take the last 2 added to history
            last = self.y_clust_history[-1]
            prev = self.y_clust_history[-2]
            # ari and nmi and unsup_acc between last and prev (all symmetric)
            ari = adjusted_rand_score(prev, last)
            nmi = normalized_mutual_info_score(prev, last)
            acc_cluster,_  = unsupervised_clustering_accuracy(prev, last)
            # print third decimal
            print("---------> dev", self.name, "ari", round(ari, 3), "nmi", round(nmi, 3), "acc", round(acc_cluster, 3))
            self.auto_ari = ari
            self.auto_acc = acc_cluster
            # append to file named with dev name the acc and ari 
            os.makedirs('metrics', exist_ok=True)
            with open('metrics/ari_nmi_acc_'+self.name+'.txt', 'a') as f:
                f.write(str(ari) + ' ' + str(nmi) + ' ' + str(acc_cluster) + '\n')
                # and self.acc
                f.write(str(self.acc) + '\n')


    # y is how to split x_train, idx is the index of the split
    def get_x_train_for_y(self, y, idx): 
        assert len(self.x_train) == len(y)
        return self.x_train[y == idx]
    
    def acc_and_crosstab(self):
        print("ACC ------------->", self.acc)
        print(pd.crosstab(self.y_clust, self.y_train, rownames=['y_clust'], colnames=['y_true']))


    @property
    def y_train(self):
        return self._y_train
    
    @y_train.getter
    def y_train(self):
        return self._y_train
    
    @y_train.setter
    def y_train(self, value):
        self._y_train = value
        self.x_dic_true = {}
        for lab in np.unique(self.y_train):
            self.x_dic_true[lab] = self.get_x_train_for_y(self.y_train, lab)


    # TRAINING

    def prepare_data_for_training(self, train_only_labs=None):
        y = self.y_clust
        labels_all = np.unique(y)
        labels_to_train = labels_all if train_only_labs is None else train_only_labs
        assert set(labels_to_train).issubset(set(labels_all))

        x_data = {}
        for lab in labels_to_train:
            x_data[lab] = self.get_x_train_for_y(y, lab)

        print('train_local_autoencoders', self.name, labels_to_train)
        return x_data
    
    def train_dev_autoencoders(self, train_only_labs=None):

        x_data = self.prepare_data_for_training(train_only_labs=train_only_labs)

        models = {}
        for lab in x_data.keys(): # keys = labels_to_train
            x = x_data[lab]
            weights = train_single_autoencoder(x) 
            models[lab] = get_model_autoencoder()
            models[lab].set_weights(weights)

        return models
    

        
    def assign_trained_local_models(self, models):
        for clust, model in models.items():
            model.set_model_name("M" + str(self.cluster_to_class(clust)))
            self.mods_clust[clust] = model # models can be a subset of all clusters
    # END TRAINING

    
    def update_mods_clust(self):
        updated = []
        for clas, mod_weights in self.fed_models.items():
            clust = self.class_to_cluster(clas)
            print('dev', self.name, 'update lab', clas, 'clust', clust)

            model = get_model_autoencoder()
            model.set_weights(mod_weights)
            model.set_model_name("FL" + str(clas))
            self.mods_clust[clust] = model
            updated.append(clust)
        return updated
        
    def predict_errors_local_models(self, reset=False):
        print('self.mods_clust', self.mods_clust)
        for clust, model in self.mods_clust.items():
            print('Model for cluster', clust, 'is', model.model_name)
            
        

        self.errors_avg = self.process_models(self.y_clust, self.mods_clust, reset=reset, compute_average=True)
        self.errors_samplewise = self.process_models(self.y_clust, self.mods_clust, reset=reset, compute_average=False)



    def process_models(self, y, models, reset, compute_average):
        unique_labels = list(np.unique(y))
        assert len(models) == len(unique_labels), "Mismatch between number of models and unique labels, {} vs {}".format(len(models), len(unique_labels))
        assert set(models.keys()) == set(unique_labels) # Ensure that models are provided for all unique labels

        results = {}
        for label in unique_labels:
            x_data = self.get_x_train_for_y(y, label)
            model = models[label]
            reconstruction_errors = self.compute_reconstruction_errors_new(model_device_name=self.name, model_label=label, model=model, 
                                                                          data_device_name=self.name, data_label=label, data=x_data, reset=reset)
            
            # Compute the average error or get sample-wise errors based on the `compute_average` flag
            if compute_average:
                results[label] = np.mean(reconstruction_errors) # should be same as model.evaluate(x_data, x_data, batch_size=128, verbose=0)
            else:
                results[label] = reconstruction_errors
        return results
    

    # dev1 is the device that has the models, dev2 is the device that has the data
    def compute_reconstruction_errors_new(self, model_device_name, model_label, model, data_device_name, data_label, data, reset=False):
        if self.name != model_device_name:
            raise Exception('model_device_name must be the same as self.name')
        cache_key = f"{model_device_name}-{model_label}_{data_device_name}-{data_label}_model_={model.get_model_hash()}"
        errors_samplewise = self._compute_reconstruction_errors(model, data, cache_key, reset=reset)
        return errors_samplewise


    def _compute_reconstruction_errors(self, model, x, cache_key, reset=False):
        errs = self.get_local_cache(cache_key)
        if errs is None or reset:
            print(f'Not found in cache: {cache_key}')
            pred = model.predict(x, verbose=0, batch_size=128)
            errs = np.mean(np.square(pred - x), axis=(1)) # #TODO carefull. must be the same as in the model
            self.save_local_cache(cache_key, errs.tolist())
        else:
            errs = np.array(errs)  # converting back to numpy array if it's stored as a list in cache
        return errs



    def get_clusters_links(self):
        # count how many links per clusters
        clusters_links = {}
        for clust in self.x_dic_clust.keys():
            links = self.association_clust.get(clust, None) 
            if links is None: 
                clusters_links[clust] = 0
            else:
                clusters_links[clust] = len(links)
        return clusters_links


            



    def recluster(self, list_dev_other, cluster_key=[0,1], apply=False, list_auto=None):
        # create list_auto from dev_other.mod_clusts. mod_clusts is a dictionary of autoencoders, take all of them
        self.reclustered_times += 1
        if list_auto is None:
            list_auto = []
            for dev_other in list_dev_other:
                list_auto.extend(dev_other.mods_clust.values())
        
        print('Trying num list_auto', len(list_auto))

        if cluster_key is None:
            cluster_key = list(range(self.KK))
        # assert cluster_key is valid
        assert len(cluster_key) <= self.KK
        assert len(cluster_key) == len(np.unique(cluster_key))
        assert np.max(cluster_key) < self.KK
        assert np.min(cluster_key) >= 0

        y_clust_new = np.copy(self.y_clust)
        subset_indices = np.where(np.isin(self.y_clust, cluster_key))[0] #[0] to get the array from the tuple. #recluster only the first len(cluster_key) clusters


        x_train_partial = self.x_train[subset_indices]
        y_clust_partial = Dev.recursive_clustering(x_train_partial, np.arange(x_train_partial.shape[0]), len(cluster_key), list_auto=list_auto)
        # map 0,1,2 to cluster_key
        y_clust_partial = np.array([cluster_key[i] for i in y_clust_partial])
        
        y_clust_new[subset_indices] = y_clust_partial

        acc_old = self.acc 
        if apply:
            self.y_clust = y_clust_new # remember y_clust setter updates acc and mapping
        

        # print difference in accuracy
        acc_new, mapping_new = unsupervised_clustering_accuracy(y_clust_new, self.y_train) 
        print('acc_new', round(acc_new, 3), 'acc_old', round(acc_old, 3))
        # increase/decrease percentage
        perc = (acc_new - acc_old) / acc_old * 100
        print('percentage', round(perc, 3))


        return y_clust_new, acc_new, mapping_new


    @staticmethod
    def recursive_clustering(samples, indices, KK, list_auto, list_best = None, y_clust=None, errors=None):
        if KK == 0 or len(samples) == 0:
            if len(samples) > 0:
                return Dev.recursive_clustering(samples, indices, len(list_best), list_best, [], y_clust)
            return y_clust

        if y_clust is None:  # Initialization in the first call
            list_best = []
            y_clust = -1 * np.ones(samples.shape[0])
            # to integer
            y_clust = y_clust.astype(int)

        if errors is None:
            errors = np.zeros((samples.shape[0], len(list_auto)))
            for i, auto in enumerate(list_auto):
                pred = auto.predict(samples, verbose=0, batch_size=128)  # shape = (n_samples, 784)
                # loss mean squared error
                loss = np.mean(np.square(pred - samples), axis=(1)) 
                errors[:, i] = loss # shape = (n_samples, len(list_auto))

        #For each sample, determine which autoencoder gives the smallest reconstruction error
        # shape = (n_samples,) # best_auto_for_sample[i] = j means the j-th autoencoder is the best for the i-th sample        
        best_auto_for_sample = np.argmin(errors, axis=1) 
        
        #Tally the number of samples for which each autoencoder is the best
        tally = np.bincount(best_auto_for_sample, minlength=len(list_auto)) # shape = (len(list_auto),) # tally[i] = j means the i-th autoencoder is the best for j samples
        # print names autoencoders
        ll = [auto.model_name for auto in list_auto]
        # do couple (name, tally)
        ll = list(zip(ll, tally))
        # sort by tally
        ll = sorted(ll, key=lambda x: x[1], reverse=True)

        # Return the autoencoder which is best for the most number of samples
        best_auto_idx = np.argmax(tally)

        # Assign samples that find this autoencoder as best to the current cluster
        best_samples_mask = np.argmin(errors, axis=1) == best_auto_idx
        y_clust[indices[best_samples_mask]] = KK - 1  # Assign these samples to the current cluster
        
        # Remove the best autoencoder from the list
        list_best.append(list_auto[best_auto_idx])
        list_auto.pop(best_auto_idx)

    
        errors = np.delete(errors, best_auto_idx, axis=1)
        errors = np.delete(errors, best_samples_mask, axis=0)

        # Recursive call for the remaining samples
        return Dev.recursive_clustering(samples[~best_samples_mask], indices[~best_samples_mask], KK-1, list_auto, list_best, y_clust, errors)

    
    @staticmethod
    def generic_recluster(KK, x_train, list_auto):
        print('generic_recluster', KK, len(list_auto), x_train.shape)
        errors = np.zeros((x_train.shape[0], len(list_auto)))

        # Compute reconstruction errors for each autoencoder
        for i, auto in enumerate(list_auto):
            pred = auto.predict(x_train, verbose=0)  # shape = (n_samples, 784)
            # loss mean squared error
            loss = np.mean(np.square(pred - x_train), axis=(1))

            errors[:, i] = loss 

        if len(list_auto) == KK: # are they equivalent? credo di si
            # Directly assign each sample to the cluster of the autoencoder with the minimum reconstruction error
            y_clust = np.argmin(errors, axis=1)
        else: 
            # Perform the tallying logic to get the top KK autoencoders
            ranks = np.argsort(errors, axis=1) # each row (sample): sorted indices of errors (the ranks of the autoencoders)
            
            tally = Dev.tally_weighted(ranks, len(list_auto))

            top_indices = np.argsort(tally)[-KK:] # best indices, the last KK elements
            y_clust = np.argmin(errors[:, top_indices], axis=1)

        return y_clust
    
    @staticmethod
    def tally_simple(ranks, num_autoencoders):
        tally = np.zeros(num_autoencoders)
        for i in range(num_autoencoders):
            tally[i] = np.sum(ranks == i) # tally is a vector of length len(list_auto), where each element is the number of samples that have the i-th autoencoder as the best
        # tally[0] = 3 means autoencoder0 is the best for 3 samples
        return tally

    @staticmethod
    def tally_weighted(ranks, num_autoencoders):
        tally = np.zeros(num_autoencoders)
        for i in range(num_autoencoders):
            for rank in range(num_autoencoders):
                tally[i] += np.sum(ranks == i) * (num_autoencoders - rank)
        return tally
   

    def cluster_to_class(self, clust): #  # MAPPING FUNZIONA bene solo se l'accuratezza e' buona
        self.mapping # k x 2 array, first column is class, second is cluster
        # iterate rows
        for row in self.mapping:
            if row[1] == clust:
                return row[0]
        return -1 # not found
    
    def class_to_cluster(self, lab): #  # MAPPING FUNZIONA bene solo se l'accuratezza e' buona
        self.mapping
        # iterate rows
        for row in self.mapping:
            if row[0] == lab:
                return row[1]
        return -1 # not found



    # questo e' sulle classi probabili, aggiungi un field d.association_clust (veramente cluster) dovrebbe funzionare lo stesso
    @staticmethod
    def find_communities(devices, start_device_name, start_dataset):
        device_dict = {dev.name: dev for dev in devices}  # Map device names to device objects for easy lookup
        visited = set()
        community = []
        
        def dfs(current_device_name, current_dataset): 
            if (current_device_name, current_dataset) in visited:
                return
            visited.add((current_device_name, current_dataset))
            community.append((current_device_name, current_dataset))

            current_dev = device_dict[current_device_name] # recover the device object from the name

            for assoc_device_name, assoc_dataset in current_dev.association_clust.get(current_dataset, []):
                dfs(assoc_device_name, assoc_dataset)

        dfs(start_device_name, start_dataset)
        return community

    @staticmethod
    def find_communities_all(devices):
        communities = []
        visited = set()
        for dev in devices:
            for dataset in dev.x_dic_true.keys(): # (TODO sicuro non dev.x_dic_clust.keys()?
                if (dev.name, dataset) in visited:
                    continue
                community = Dev.find_communities(devices, dev.name, dataset) # list of tuples (dev_name, dataset)
                communities.append(community)
                visited.update(community)
        return communities


    
    
    # ASSOCIATION DICT: class1 -> list of tuples (other_dev, class2)
    def wrong_association(self):
        d1 = self.association_perfect
        d2 = self.association_clust
        wrong = 0
        for class2, list2 in d2.items():
            if class2 in d1: 
                for tumple2 in list2:
                    if tumple2 not in d1[class2]:
                        wrong += 1
        # divide by 2 because each wrong is counted twice
        return wrong / 2 
    
    
    def count_association(self):
        d2 = self.association_clust
        total = 0
        for class2, list2 in d2.items():
            total += len(list2)
        # divide by 2 because each association is counted twice
        return total / 2
    
    def multiple_communities(self, devs, remove=False):
        self.association_clust # class1 -> list of tuples (other_dev, class2)
        # find when there are more distinc class2 in the list of tuples
        dcount = {}
        for class1, list1 in self.association_clust.items():
            # list1 is a list of tuples (other_dev, class2)
            # check how many distinct values of class2
            dcount[class1] = len(set([t[1] for t in list1]))
        # print if more than 1
        for class1, count in dcount.items():
            if count > 1:
                print('multiple_communities', self.name, class1, count)
                if remove:
                    self.remove_association_clust(class1, devs)


    def count_different_comm_per_cluster(self):
        dcount = {}
        cc = 0
        for class1, list1 in self.association_clust.items():
            # list1 is a list of tuples (other_dev, class2)
            # check how many distinct values of class2
            dcount[class1] = len(set([t[1] for t in list1]))
        return dcount

    def remove_association_clust(self, lab1, devs):
        dev1 = self
        device_dict = {dev.name: dev for dev in devs}

        # remove from dev1
        list_my_comm = dev1.association_clust[lab1] # list of tuples (other_dev, class2)
        dev1.association_clust.pop(lab1) # remove from dev1.association_clust[lab1]
        # now need to remove from other_dev.association_clust[class2] the tuples with self.name
        for other_dev_name, lab2 in list_my_comm:
            # remove from other_dev.association_clust[class2] the tuples with self.name
            other_dev = device_dict[other_dev_name]
            # check if other_dev.association_clust[lab2] exists
            if lab2 not in other_dev.association_clust:
                continue

            list_other_comm = other_dev.association_clust[lab2]
            list_other_comm = [t for t in list_other_comm if t[0] != dev1.name] # keep only tuples with other_dev_name != self.name
            other_dev.association_clust[lab2] = list_other_comm



    




