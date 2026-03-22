"""
Federated Learning Cluster Discovery - Main Experiment Runner

This script orchestrates the federated learning clustering experiment.
"""

import os
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


def main(config: Dict[str, Any] = None) -> None:
    """
    Main experiment function.

    Args:
        config: Optional dict to override settings from my_config.py
    """
    # Merge settings: my_config.py base, then optional overrides
    experiment_settings = settings.copy()
    if config:
        experiment_settings.update(config)

    # Validate
    try:
        exp.validate_settings(experiment_settings)
    except ValueError as e:
        print(f"Invalid settings: {e}")
        return

    # Extract key settings
    seed = experiment_settings['seed']
    dataset_name = experiment_settings['dataset_name']
    cluster_kind = experiment_settings['cluster_kind']
    dirtiness_max = experiment_settings['dirtiness_max']
    association_threshold = experiment_settings['association_threshold']
    experiment_group = experiment_settings.get('experiment_group')
    wandb_mode = experiment_settings.get('wandb_mode', 'offline') # can be 'online', 'offline', or 'disabled'


    # Initialize W&B
    wandb.init(
        project="FedCRef-2026",
        mode=wandb_mode, 
        group=experiment_group,
        config=experiment_settings
    )
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")

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


    # Set up logging
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    filename = exp.setup_logging(dataset_name, start_time)

    # Run experiment
    run_iteration_zero(device_manager, filename)

    iteration = 1
    while True:
        should_continue = run_iteration(
            device_manager, devs, filename, iteration, association_threshold
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
