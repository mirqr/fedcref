import time

import numpy as np
import ray

from utils.fl_client import *
from utils.fl_server import *
from utils.util_models import get_model_autoencoder

from dev import Dev, fit_dec_ray, train_autoencoders_ray, predict_single_autoencoder_parallel_ray


class DevManager:
    # ── Initialization ──────────────────────────────────────
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

    # ── DEC clustering ───────────────────────────────────────
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


    # ── Device registry ─────────────────────────────────────
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


    # ── Convergence ──────────────────────────────────────────
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


    # ── I/O ──────────────────────────────────────────────────
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


    # ── Metrics ──────────────────────────────────────────────
    def get_accuracy_avg(self, round_dig = 3):
        acc = []
        for dev in self.devices.values():
            acc.append(dev.acc)
        acc_mean = np.mean(acc)
        return round(acc_mean, round_dig)


    # ── Training ─────────────────────────────────────────────
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

    # ── Prediction ───────────────────────────────────────────
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



    # ── Association ──────────────────────────────────────────
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

                # Two-sample Kolmogorov-Smirnov test
                # default two-sided: The null hypothesis is that the two distributions are identical
                from scipy import stats
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



    # ── Federated averaging ──────────────────────────────────
    # to avoid too much parallelism, do one federated model at a time
    def fed_communities_all_safe(self, devs, start_port=4000):
        device_dict = {dev.name: dev for dev in devs}

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
        #lst = list(range(4000, 8000, 40))

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
