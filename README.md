# FL Cluster Claude

A Federated Learning (FL) clustering system for discovering client communities in distributed environments. This project investigates how to identify groups of federated learning clients with similar data distributions using deep learning models and reconstruction error analysis.

## Overview

This system enables targeted federated learning strategies by:
- Clustering FL clients based on data similarity without direct data sharing
- Using reconstruction error analysis to find client associations
- Applying Deep Embedding Clustering (DEC) for unsupervised learning
- Identifying communities through autoencoder-based feature extraction

## Project Structure

```
fl_cluster_claude/
├── main.py                 # Main entry point and experiment orchestration
├── dev.py                  # Core Dev class (FL client) and DevManager
├── keras_dec.py            # Deep Embedding Clustering implementation
├── my_config.py            # Centralized configuration settings
├── zzz_config              # Additional configuration
│
├── utils/                  # Utility modules
│   ├── util_data.py        # Dataset loading and client data distribution
│   ├── util_models.py      # Autoencoder architectures (flat & convolutional)
│   ├── util_dev.py         # Helper functions for clustering and metrics
│   ├── util_graph.py       # Graph utilities for client associations
│   ├── fl_client.py        # Flower-based FL client implementation
│   ├── fl_server.py        # Flower-based FL server setup
│   └── fl_test.py          # Testing utilities
│
├── saved/                  # Saved models (git-ignored)
└── log_main/               # Experiment logs (git-ignored)
```

## Key Components

### Core Classes

- **Dev** (`dev.py`): Represents a federated learning client/device
  - Handles clustering (DEC, oracle, dirty uniform, proximity-based)
  - Manages autoencoder training and prediction
  - Tracks reconstruction errors and client associations
  - Computes clustering accuracy metrics (ARI, NMI)

- **DevManager** (`dev.py`): Orchestrates multiple devices
  - Parallel training using Ray
  - Association analysis across clients
  - Community detection via graph analysis

- **DeepEmbeddingClustering** (`keras_dec.py`): DEC algorithm implementation
  - Custom clustering layer
  - Pretraining and fine-tuning
  - KMeans initialization

### Utilities

- **util_data.py**: Dataset loading (MNIST, Fashion-MNIST, EMNIST, KMNIST, CIFAR10) and non-IID client data distribution
- **util_models.py**: Autoencoder models with caching and inference optimization
- **util_dev.py**: Clustering accuracy, distance metrics, hash generation
- **util_graph.py**: Graph building for client association networks

## Technologies

- **Deep Learning**: TensorFlow/Keras
- **Distributed Computing**: Ray, Pebble, Flower (flwr)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data Science**: NumPy, Pandas, Scikit-learn
- **Caching**: Joblib

## Workflow

1. **Initialization**
   - Load dataset (MNIST, Fashion-MNIST, EMNIST, etc.)
   - Distribute data among clients (non-IID with class subsets)
   - Create Dev objects for each client

2. **Clustering Phase**
   Choose one strategy:
   - **Oracle**: Ground truth baseline mapping
   - **DEC**: Deep Embedding Clustering (unsupervised)
   - **Dirty Uniform**: Simulated clustering noise (uniform)
   - **Dirty Proximity**: Simulated clustering noise (proximity-based)

3. **Iterative Community Discovery**
   ```
   For each iteration:
   ├── Train local autoencoders (parallel via Ray)
   ├── Predict on all clients' data using all models
   ├── Compute reconstruction errors (cached)
   ├── Associate clients based on error thresholds
   ├── Identify communities (connected components)
   ├── Log metrics (accuracy, associations, communities)
   └── Check convergence/stability
   ```

4. **Association Mechanism**
   - Each client trains autoencoders for each cluster in its data
   - Reconstruction error used as similarity metric
   - Associates with clients whose data has low reconstruction error
   - Builds weighted association graph by cluster labels

5. **Community Detection**
   - Groups clients into communities based on mutual associations
   - Tracks: number of communities, isolated clusters, association accuracy
   - Convergence: stable clustering accuracy or max iterations reached

## Configuration

Edit `my_config.py` to set experiment parameters:

```python
settings = {
    'seed': 42,
    'dataset_name': 'emnist_digits',  # mnist, fashion_mnist, emnist_digits, kmnist, cifar10
    'num_clients': 20,
    'cluster_kind': 'oracle',  # oracle, dec, dirty_uniform, dirty_proximity
    'association_threshold': 0.15,  # percentile threshold for associations
    'dirtiness_max': 0.3,  # max clustering noise level
    'overlap': 0.0,  # class overlap between clients
}
```

## Usage

### Basic Experiment

Run with default settings:
```bash
python main.py
```

### Command Line Arguments

Override settings via command line:
```bash
# Run with specific dataset and seed
python main.py --dataset mnist --seed 123

# Run with custom number of clients
python main.py --num-clients 30 --cluster-kind oracle

# Use configuration file
python main.py --config config_example.json

# Enable W&B tracking
python main.py --wandb-mode online

# Combine multiple options
python main.py --dataset fashion_mnist --num-clients 20 --seed 42 --wandb-mode offline
```

Available command line arguments:
- `--config PATH`: Load settings from JSON configuration file
- `--seed INT`: Random seed for reproducibility
- `--dataset NAME`: Dataset choice (mnist, fashion_mnist, emnist_digits, emnist_letters, kmnist, cifar10, femnist-noniid)
- `--num-clients INT`: Number of federated learning clients
- `--cluster-kind KIND`: Clustering method (oracle, dec, dirty_uniform, dirty_proximity)
- `--wandb-mode MODE`: W&B tracking (online, offline, disabled)

### Configuration Files

Create a JSON configuration file to set experiment parameters:

```json
{
  "seed": 42,
  "dataset_name": "emnist_digits",
  "num_clients": 25,
  "cluster_kind": "dirty_uniform",
  "association_threshold": 0.20,
  "dirtiness_max": 0.3,
  "overlap": 0.0,
  "experiment_group": "my_experiment",
  "num_max_class": 5
}
```

Then run:
```bash
python main.py --config my_config.json
```

See `config_example.json` for a complete example.

**Priority order** (later overrides earlier):
1. Default settings (my_config.py)
2. Configuration file (--config)
3. Command line arguments
4. Programmatic overrides (if called from Python)

### With Wandb Tracking

The code integrates with Weights & Biases for experiment tracking. Metrics logged include:
- Clustering accuracy, ARI, NMI
- Number of communities found
- Association counts and accuracy
- Training time per iteration

Control W&B mode with `--wandb-mode`:
- `disabled` (default): No tracking
- `offline`: Track locally
- `online`: Track and sync to W&B servers

### Output

- **Logs**: `log_main/` - Text logs with per-iteration metrics
- **Models**: `saved/` - Saved autoencoders and DEC models by dataset
- **Wandb**: Online experiment tracking dashboard (if enabled)

## Key Metrics

- **Clustering Accuracy**: Agreement between predicted and true clusters
- **ARI (Adjusted Rand Index)**: Similarity measure for cluster assignments
- **NMI (Normalized Mutual Information)**: Information-theoretic clustering metric
- **Communities Found**: Number of detected client communities
- **Association Accuracy**: Correctness of client-to-client associations

## Datasets Supported

- MNIST (digit recognition)
- Fashion-MNIST (clothing items)
- EMNIST Digits (extended digit dataset)
- EMNIST Letters (letter recognition)
- KMNIST (Japanese characters)
- CIFAR10 (natural images)

## Research Focus

This project explores:
- Client clustering and community detection in federated learning
- Deep Embedding Clustering for unsupervised learning
- Association-based grouping through reconstruction error analysis
- Autoencoder-based feature extraction and similarity metrics
- Non-IID data distribution in federated settings

## Requirements

Install dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn ray flwr wandb joblib pebble matplotlib scipy extra-keras-datasets
```

## Notes

- Caching directories (`.cachejoblib/`, `saved/`, `log_main/`, `wandb/`) are git-ignored
- Ray is used for distributed training and parallel prediction
- Models are cached by hash for reproducibility
- File locks ensure safe parallel access to cached inferences

## License

Research project - check with maintainers for usage terms.
