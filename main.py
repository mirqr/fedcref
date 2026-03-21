"""
Federated Learning Cluster Discovery - Main Experiment Runner

This script orchestrates the federated learning clustering experiment.
Run with: python main.py [options]
See --help for available options.
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, Any

# Environment configuration (must be before TensorFlow import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import logging
import numpy as np
import wandb

from dev import DevManager
from my_config import settings
from utils import experiment_utils as exp

# Suppress framework logging
logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger("absl").setLevel(logging.ERROR)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Federated Learning Clustering Experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--seed', type=int, help='Random seed')
    parser.add_argument('--dataset', type=str,
                       choices=['mnist', 'fashion_mnist', 'emnist_digits', 'emnist_letters',
                               'kmnist', 'cifar10', 'femnist-noniid'],
                       help='Dataset to use')
    parser.add_argument('--num-clients', type=int, help='Number of clients')
    parser.add_argument('--cluster-kind', type=str,
                       choices=['oracle', 'dec', 'dirty_uniform', 'dirty_proximity'],
                       help='Clustering method')
    parser.add_argument('--wandb-mode', type=str, default='disabled',
                       choices=['online', 'offline', 'disabled'],
                       help='Weights & Biases mode')
    return parser.parse_args()


def merge_settings(args: argparse.Namespace, base_settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge settings from multiple sources with priority:
    1. Base settings
    2. Config file
    3. Command line arguments
    """
    experiment_settings = base_settings.copy()

    # Default overrides
    defaults = {
        'seed': 42,
        'dataset_name': 'emnist_digits',
        'association_threshold': 0.20,
        'cluster_kind': 'dirty_uniform',
        'dirtiness_max': 0.3,
        'overlap': 0.0,
        'experiment_group': 'default',
        'num_max_class': 5,
        'num_clients': 25,
    }
    experiment_settings.update(defaults)

    # Apply config file
    if args.config:
        config_settings = exp.load_config_file(args.config)
        experiment_settings.update(config_settings)

    # Apply CLI arguments
    if args.seed is not None:
        experiment_settings['seed'] = args.seed
    if args.dataset is not None:
        experiment_settings['dataset_name'] = args.dataset
    if args.num_clients is not None:
        experiment_settings['num_clients'] = args.num_clients
    if args.cluster_kind is not None:
        experiment_settings['cluster_kind'] = args.cluster_kind

    return experiment_settings


def run_iteration_zero(device_manager: DevManager, filename: str) -> None:
    """Run initial iteration 0 (baseline metrics)."""
    exp.log_iteration_start(filename, device_manager)
    exp.log_initial_metrics(device_manager, iteration=0)


def run_iteration(device_manager: DevManager, devs: list, filename: str,
                 iteration: int, association_threshold: float) -> bool:
    """
    Run a single iteration of the experiment.

    Returns:
        True if experiment should continue, False if converged
    """
    print(f"\n{'='*60}")
    print(f"Iteration {iteration}")
    print(f"Clients: {device_manager.get_dev_number()}")
    print(f"{'='*60}")

    start_time = exp.log_iteration_start(filename, device_manager)
    wandb.log({"iteration": iteration})

    # Train and predict
    exp.train_and_predict(device_manager, devs, filename, start_time)

    # Compute associations
    total_wrong, total_count, wrong_perc = exp.compute_associations(
        device_manager, devs, association_threshold
    )

    # Find communities
    list_comm, list_comm_gt1, list_single = exp.find_and_log_communities(
        devs, device_manager, filename
    )

    # Log to W&B
    wandb.log({
        "wrong associations": total_wrong,
        "wrong associations perc": wrong_perc,
        "communities ideal": device_manager.get_num_ideal_communities(),
        "communities found": len(list_comm_gt1),
        "isolated clusters": len(list_single)
    })

    # Recluster
    participants = exp.perform_reclustering(device_manager, devs)
    reclustered_mean = device_manager.reclustered_percentage()

    # Log results
    exp.log_iteration_results(filename, start_time, reclustered_mean, total_wrong)

    # Log final metrics
    wandb.log({
        "accuracy avg": device_manager.get_accuracy_avg(),
        "self_acc avg": device_manager.avg_auto_acc(),
        "active clients": participants
    })

    print(f"Accuracy: {device_manager.get_accuracy_avg():.4f}")

    # Check convergence
    return not exp.check_convergence(device_manager, reclustered_mean)


def main(programmatic_settings: Dict[str, Any] = None) -> None:
    """
    Main experiment function.

    Args:
        programmatic_settings: Optional dict to override settings (for programmatic use)
    """
    # Parse arguments and merge settings
    args = parse_arguments()
    experiment_settings = merge_settings(args, settings)

    # Apply programmatic overrides
    if programmatic_settings:
        print("Applying programmatic settings overrides...")
        experiment_settings.update(programmatic_settings)
        return

    # Validate
    try:
        exp.validate_settings(experiment_settings)
    except ValueError as e:
        print(f"✗ Invalid settings: {e}")
        sys.exit(1)

    # Extract key settings
    seed = experiment_settings['seed']
    dataset_name = experiment_settings['dataset_name']
    cluster_kind = experiment_settings['cluster_kind']
    dirtiness_max = experiment_settings['dirtiness_max']
    association_threshold = experiment_settings['association_threshold']
    experiment_group = experiment_settings.get('experiment_group', 'default')

    # Set random seed
    np.random.seed(seed)

    # Load and distribute data
    print("\n" + "="*60)
    print("FEDERATED LEARNING CLUSTER DISCOVERY")
    print("="*60)
    clients_data, classes = exp.load_and_distribute_data(experiment_settings)

    # Initialize devices
    devs = exp.initialize_devices(clients_data, dataset_name, cluster_kind, dirtiness_max)
    device_manager = DevManager(devs[:])

    # Apply DEC if needed
    if cluster_kind == 'dec':
        print("Applying DEC clustering...")
        device_manager.dec_all_ray()

    print("✓ Initialization complete\n")

    # Initialize W&B
    wandb.init(
        project="FED_Cluster-november",
        mode=args.wandb_mode,
        group=experiment_group,
        config=experiment_settings
    )
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")

    # Set up logging
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = exp.setup_logging(dataset_name, start_time)

    # Run experiment
    run_iteration_zero(device_manager, filename)

    iteration = 1
    while True:
        should_continue = run_iteration(
            device_manager, devs, filename, iteration,
            association_threshold
        )

        if not should_continue:
            break

        iteration += 1

    print(f"\n{'='*60}")
    print(f"✓ Experiment completed after {iteration} iterations")
    print(f"{'='*60}\n")
    wandb.finish()


if __name__ == "__main__":
    main()
