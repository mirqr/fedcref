"""
Experiment utilities for federated learning clustering experiments.

This module contains helper functions for:
- Configuration management
- Data loading and distribution
- Device initialization
- Training and prediction
- Logging and metrics
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

import numpy as np
import wandb

from dev import Dev
from dev_manager import DevManager
from utils.util_data import *


# Constants
DEFAULT_RECLUSTER_THRESHOLD = 10
DEFAULT_ASSOCIATION_PERCENTILE = 75
LOG_DIR = "log_main"


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Dictionary containing configuration settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is not valid JSON
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✓ Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        print(f"✗ Error: Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)


def validate_settings(settings: Dict[str, Any]) -> None:
    """
    Validate experiment settings.

    Args:
        settings: Dictionary of experiment settings

    Raises:
        ValueError: If settings are invalid
    """
    required_keys = [
        'seed', 'dataset_name', 'num_clients', 'num_min_class', 'num_max_class',
        'min_samples_per_class', 'max_samples_per_class', 'cluster_kind',
        'association_threshold'
    ]

    for key in required_keys:
        if key not in settings:
            raise ValueError(f"Missing required setting: {key}")

    if settings['num_clients'] <= 0:
        raise ValueError("num_clients must be positive")

    if not 0 <= settings['association_threshold'] <= 1:
        raise ValueError("association_threshold must be between 0 and 1")

    if settings['cluster_kind'] not in ['oracle', 'dec', 'dirty_uniform', 'dirty_proximity']:
        raise ValueError(f"Invalid cluster_kind: {settings['cluster_kind']}")


def setup_logging(dataset_name: str, start_time: str) -> str:
    """
    Set up logging directory and return log filename.

    Args:
        dataset_name: Name of the dataset
        start_time: Experiment start time string

    Returns:
        Path to log file
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    filename = os.path.join(LOG_DIR, f"listcomm_{dataset_name}_{start_time}.txt")
    return filename


def load_and_distribute_data(settings: Dict[str, Any]) -> Tuple[List[Tuple], List]:
    """
    Load dataset and distribute among clients.

    Args:
        settings: Experiment settings dictionary

    Returns:
        Tuple of (clients_data, classes)
    """
    dataset_name = settings['dataset_name']
    num_clients = settings['num_clients']
    num_min_class = settings['num_min_class']
    num_max_class = settings['num_max_class']
    min_samples_per_class = settings['min_samples_per_class']
    max_samples_per_class = settings['max_samples_per_class']
    seed = settings['seed']
    overlap = settings.get('overlap', 0.0)

    print(f"Loading dataset: {dataset_name}")

    if dataset_name == 'femnist-noniid':
        clients_data = get_system_femnist(
            num_clients, num_min_class, num_max_class,
            min_samples_per_class, max_samples_per_class, seed=seed
        )
        classes = None
    else:
        x_train, y_train, x_test, y_test = get_dataset(dataset_name, flatten_and_normalize=True)
        classes = np.unique(y_train)

        clients_data = get_system(
            num_clients, x_train, y_train, num_min_class, num_max_class,
            unique_classes=classes, min_samples_per_class=min_samples_per_class,
            max_samples_per_class=max_samples_per_class, seed=seed
        )

    if overlap > 0 and classes is not None:
        clients_data = introduce_overlap_new(clients_data, classes, overlap)
        print(f"✓ Introduced {overlap} overlap")

    print(f"✓ Data distributed to {len(clients_data)} clients")
    return clients_data, classes


def initialize_devices(clients_data: List[Tuple], dataset_name: str,
                       cluster_kind: str, dirtiness_max: float) -> List[Dev]:
    """
    Initialize device objects and apply clustering.

    Args:
        clients_data: List of (x, y) tuples for each client
        dataset_name: Name of the dataset
        cluster_kind: Type of clustering to apply
        dirtiness_max: Maximum dirtiness for simulated clustering

    Returns:
        List of initialized Dev objects
    """
    print(f"Initializing {len(clients_data)} devices with {cluster_kind} clustering...")

    devs = []
    for x, y in clients_data:
        d = Dev(dataset_name, x, y)
        devs.append(d)

    for d in devs:
        if cluster_kind == 'oracle':
            d.simulate_clustering(dirtiness=0)
        elif cluster_kind == 'dirty_uniform':
            dirtiness = np.random.uniform(dirtiness_max, dirtiness_max)
            d.simulate_clustering(dirtiness=dirtiness)
        elif cluster_kind == 'dirty_proximity':
            dirtiness = np.random.uniform(0.3, dirtiness_max)
            d.simulate_clustering_proximity(dirtiness=dirtiness)
        elif cluster_kind != 'dec':
            raise ValueError(f"Unknown cluster kind: {cluster_kind}")

        if cluster_kind != 'dec':
            d.acc_and_crosstab()

    print("✓ Devices initialized")
    return devs


def log_iteration_start(filename: str, device_manager: DevManager) -> str:
    """Log iteration start information."""
    start_time_iteration = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    try:
        with open(filename, 'a') as file:
            file.write(f"time {start_time_iteration}\n")
            file.write(f"Num clients {device_manager.get_dev_number()}\n")
    except IOError as e:
        print(f"Warning: Could not write to log file: {e}")
    return start_time_iteration


def log_initial_metrics(device_manager: DevManager, iteration: int) -> None:
    """Log initial metrics for iteration 0."""
    wandb.log({
        "iteration": iteration,
        "accuracy avg": device_manager.get_accuracy_avg(),
        "init accuracy avg": device_manager.get_accuracy_avg(),
        "init isolated clusters": device_manager.get_dev_cluster_number(),
        "communities ideal": device_manager.get_num_ideal_communities(),
        "communities found": 0,
        "isolated clusters": device_manager.get_dev_cluster_number(),
        "wrong associations": 0,
        "wrong associations perc": 0,
        "self_acc avg": 0,
        "active clients": device_manager.get_dev_number(),
        "train time": 0
    })
    device_manager.set_num_isolated(device_manager.get_dev_cluster_number())
    device_manager.set_num_communities(0)


def train_and_predict(device_manager: DevManager, devs: List[Dev],
                     filename: str, start_time_iteration: str) -> None:
    """Train local autoencoders and perform predictions."""
    # Train
    device_manager.train_local_autoencoders_parallel_ray()

    
    train_end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    train_duration = (datetime.strptime(train_end_time, "%Y-%m-%d-%H-%M-%S") -
                     datetime.strptime(start_time_iteration, "%Y-%m-%d-%H-%M-%S")).total_seconds()
    print(f"✓ Training completed ({train_duration:.1f}s)")

    try:
        with open(filename, 'a') as file:
            file.write(f"train time {train_duration}\n")
    except IOError:
        pass

    wandb.log({"train time": train_duration})

    # Predict
    print("Running predictions...")
    device_manager.predict_local_parallel_ray()
    predict_end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    predict_duration = (datetime.strptime(predict_end_time, "%Y-%m-%d-%H-%M-%S") -
                       datetime.strptime(train_end_time, "%Y-%m-%d-%H-%M-%S")).total_seconds()
    print(f"✓ Predictions completed ({predict_duration:.1f}s)")

    try:
        with open(filename, 'a') as file:
            file.write(f"predict time {predict_duration}\n")
    except IOError:
        pass

    # Compute errors
    for d in devs:
        d.predict_errors_local_models(reset=False)


def compute_associations(device_manager: DevManager, devs: List[Dev],
                        association_threshold: float) -> Tuple[int, int, float]:
    """
    Compute device associations and count errors.

    Returns:
        Tuple of (total_wrong, total_count, percentage_wrong)
    """
    device_manager.associate_devs(percentile=DEFAULT_ASSOCIATION_PERCENTILE, th=association_threshold)

    total_wrong = sum(d.wrong_association() for d in devs)
    total_count = sum(d.count_association() for d in devs)
    percentage_wrong = (total_wrong / total_count * 100) if total_count > 0 else 0

    for d in devs:
        print(d.name)
        d.association_clust

    print(f"✓ Associations: {total_wrong}/{total_count} wrong ({percentage_wrong:.1f}%)")
    return total_wrong, total_count, percentage_wrong


def find_and_log_communities(devs: List[Dev], device_manager: DevManager,
                             filename: str) -> Tuple[List, List, List]:
    """
    Find communities and log results.

    Returns:
        Tuple of (all_communities, multi_member_communities, isolated_clients)
    """
    for d in devs:
        d.multiple_communities(devs, remove=False)

    list_comm = Dev.find_communities_all(devs)
    list_comm_gt1 = [cmm for cmm in list_comm if len(cmm) > 1]
    list_single = [cmm for cmm in list_comm if len(cmm) == 1]

    device_manager.set_num_isolated(len(list_single))
    device_manager.set_num_communities(len(list_comm_gt1))

    print(f"✓ Communities: {len(list_comm_gt1)} multi-member, {len(list_single)} isolated")

    for cmm in list_comm:
        print(cmm)

    try:
        with open(filename, 'a') as file:
            file.write(f"len list_comm: {len(list_comm)}\n")
            file.write(f"len list_comm = 1: {len(list_single)}\n")
            file.write(f"len list_comm > 1: {len(list_comm_gt1)}\n")
            for community in list_comm:
                file.write(f"{community}\n")
    except IOError:
        pass

    return list_comm, list_comm_gt1, list_single


def perform_reclustering(device_manager: DevManager, devs: List[Dev]) -> int:
    """
    Perform reclustering on devices.

    Returns:
        Number of participating devices
    """
    participants = 0
    device_manager.stop_stable_devices()

    for devss in device_manager.list_split:
        dd = list(devss.values())
        all_fed_models = device_manager.fed_communities_all_safe(dd, start_port=4500)

        for dev in dd:
            if dev.is_stable():
                continue

            participants += 1
            dev2_list = [dev]
            all_fed = [dev.mods_clust[clas] for clas in dev.mods_clust.keys()]
            all_fed.extend(all_fed_models)

            acc_old = dev.acc
            y_clust_new, acc_new, mapping = dev.recluster(dev2_list, None, apply=True, list_auto=all_fed)

    print(f"✓ Reclustered {participants} devices")
    return participants


def log_iteration_results(filename: str, start_time_iteration: str,
                         reclustered_mean: float, total_wrong: int) -> None:
    """Log results for the current iteration."""
    end_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    duration = (datetime.strptime(end_time, "%Y-%m-%d-%H-%M-%S") -
               datetime.strptime(start_time_iteration, "%Y-%m-%d-%H-%M-%S")).total_seconds()

    try:
        with open(filename, 'a') as f:
            f.write(f"reclustered_mean {reclustered_mean}\n")
            f.write(f"Total wrong associations: {total_wrong}\n")
            f.write(f"time iteration {duration}\n")
            f.write("------------------\n")
    except IOError:
        pass


def check_convergence(device_manager: DevManager, reclustered_mean: float,
                     recluster_threshold: float = DEFAULT_RECLUSTER_THRESHOLD) -> bool:
    """
    Check if experiment has converged.

    Returns:
        True if experiment should stop
    """
    if reclustered_mean >= recluster_threshold:
        print(f"✓ Converged: reclustered_mean ({reclustered_mean}) >= {recluster_threshold}")
        return True

    isolated_stable = device_manager.should_stop(
        device_manager.num_isolated, max_percentage_diff=15, consecutive_limit=3
    )
    communities_stable = device_manager.should_stop(
        device_manager.num_communities, max_percentage_diff=10, consecutive_limit=3
    )

    if isolated_stable and communities_stable:
        print("✓ Converged: metrics are stable")
        return True

    return False
