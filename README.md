# FL Cluster

This is the code for the paper:

**Federated clustering: An unsupervised cluster-wise training for decentralized data distributions**
Future Generation Computer Systems, Volume 178, May 2026, 108294

[![Paper](https://img.shields.io/badge/FGCS-2026-blue)](https://www.sciencedirect.com/science/article/pii/S0167739X25005886)
[![arXiv](https://img.shields.io/badge/arXiv-2408.10664-b31b1b)](https://arxiv.org/abs/2408.10664)

## Overview

![FL system schematic](figures/figures-fl-groups.png)

*N=4 clients, each holding a local dataset with K_i unique data distributions (dotted lines show local -imperfect- cluster splits). The goal is to identify the set of global distributions U (K_G=4, shown as distinct shapes) across all clients.*

The system discovers client communities in federated learning environments through reconstruction error analysis. Clients train local autoencoders on their data partitions without sharing raw data. Reconstruction errors across clients serve as a similarity signal — low error indicates similar data distributions. An association graph is built from these errors and decomposed into communities, which drive cluster-wise federated training across iterations.

## Project Structure

```
fl_cluster/
├── main.py                 # Entry point
├── dev.py                  # Dev (FL client) class
├── dev_manager.py          # DevManager class
├── keras_dec.py            # Deep Embedding Clustering
├── my_config.py            # All experiment settings
│
└── utils/
    ├── util_data.py        # Dataset loading and non-IID distribution
    ├── util_models.py      # Autoencoder architectures and caching
    ├── util_dev.py         # Clustering metrics and helpers
    ├── util_graph.py       # Graph utilities for client associations
    ├── fl_client.py        # Flower FL client
    └── fl_server.py        # Flower FL server
```

## Key Components

### Core Classes

- **Dev** (`dev.py`): Represents a federated learning client/device
  - Handles clustering (DEC, oracle, dirty uniform, proximity-based)
  - Manages autoencoder training and prediction
  - Tracks reconstruction errors and client associations
  - Computes clustering accuracy metrics (ARI, NMI)

- **DevManager** (`dev_manager.py`): Orchestrates multiple devices
  - Parallel training using Ray
  - Association analysis across clients
  - Community detection via graph analysis

- **DeepEmbeddingClustering** (`keras_dec.py`): DEC algorithm implementation
  - Custom clustering layer
  - Pretraining and fine-tuning
  - KMeans initialization

### Utilities

- **util_data.py**: Dataset loading (MNIST, Fashion-MNIST, EMNIST, KMNIST) and non-IID client data distribution
- **util_models.py**: Autoencoder models with caching and inference optimization
- **util_dev.py**: Clustering accuracy, distance metrics, hash generation
- **util_graph.py**: Graph building for client association networks

## Libraries

- **DL**: TensorFlow/Keras
- **Distributed**: Ray, Pebble, Flower (flwr)
- **Experiment Tracking**: Weights & Biases (wandb)
- **Data**: NumPy, Pandas, Scikit-learn
- **Caching**: Joblib

## Workflow

**Setup:** data is distributed across clients in a non-IID fashion. Each client holds samples from a subset of classes. An initial local clustering splits each client's data into K_i local clusters (via oracle, DEC, or simulated noise).

Then the following loop runs until convergence:

1. **Train** — each client trains one autoencoder per local cluster, learning a compact representation of that cluster's data distribution.

2. **Associate** — models are exchanged all-to-all: each model is tested on every cluster of every other client. Two clusters are considered similar if a model trained on one can reconstruct the other with low error — comparable to the self-reconstruction error. Links are formed between clusters that pass this threshold.

3. **Community detection** — the association links form a graph. Connected components identify groups of clusters (across different clients) that share the same underlying data distribution.

4. **Federated averaging** — within each community, clients aggregate their local models via federated averaging, producing one shared global model per community.

5. **Recluster** — all federated models are broadcast to all clients. Each client uses them to re-assign its local samples to clusters, effectively correcting the local partition using global knowledge.

Repeat from step 1 until cluster assignments stabilize.

## Configuration

All settings are in `my_config.py`. Edit it before running:

```python
settings = {
    'seed': 42,
    'dataset_name': 'mnist',
    'num_clients': 10,
    'cluster_kind': 'oracle',       # oracle, dec, dirty_uniform, dirty_proximity
    'association_threshold': 0.25,
    'dirtiness_max': 0.5,
    'wandb_mode': 'online',         # online, offline, disabled
    ...
}
```

## Usage

```bash
python main.py
```

To override settings programmatically:

```python
from main import main
main(config={'dataset_name': 'emnist_digits', 'num_clients': 25})
```

## Setup

```bash
source req-env.sh
```

This creates a conda environment (`fedcref`) and installs all dependencies.

## Output

- `log_main/` — per-iteration text logs
- `saved/` — cached autoencoders and DEC models
- Weights & Biases dashboard (if `wandb_mode` is `online` or `offline`)

## AI tools

AI was used for code cleanup and documentation only. All research, algorithms, and implementation are original work by the authors.

## License

Research project — check with maintainers for usage terms.
