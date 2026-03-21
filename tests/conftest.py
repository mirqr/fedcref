"""
Pytest configuration and shared fixtures.
"""

import os
import sys
import tempfile
import json
import pytest
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@pytest.fixture
def sample_config():
    """Provide a valid sample configuration."""
    return {
        'seed': 42,
        'dataset_name': 'mnist',
        'num_clients': 5,
        'num_min_class': 2,
        'num_max_class': 4,
        'min_samples_per_class': 100,
        'max_samples_per_class': 200,
        'association_threshold': 0.25,
        'cluster_kind': 'oracle',
        'dirtiness_max': 0.5,
        'overlap': 0.0
    }


@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_data():
    """Generate small sample data for testing."""
    np.random.seed(42)
    x_train = np.random.rand(100, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(100,))
    x_test = np.random.rand(20, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, size=(20,))
    return x_train, y_train, x_test, y_test


@pytest.fixture
def sample_clients_data():
    """Generate sample client data distributions."""
    np.random.seed(42)
    clients_data = []
    for i in range(3):
        x = np.random.rand(50, 784).astype(np.float32)
        y = np.random.randint(0, 5, size=(50,))
        clients_data.append((x, y))
    return clients_data


@pytest.fixture
def temp_log_dir():
    """Create a temporary log directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
