"""
Integration tests for end-to-end workflows.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dev import Dev, DevManager
from utils import experiment_utils, util_data


class TestEndToEndWorkflow:
    """Test complete experiment workflows."""

    def test_minimal_experiment_flow(self, sample_config, monkeypatch):
        """Test a minimal end-to-end experiment flow."""
        # Mock data loading to avoid downloads
        def mock_get_dataset(*args, **kwargs):
            x_train = np.random.rand(200, 784).astype(np.float32)
            y_train = np.random.randint(0, 10, size=(200,))
            x_test = np.random.rand(40, 784).astype(np.float32)
            y_test = np.random.randint(0, 10, size=(40,))
            return x_train, y_train, x_test, y_test

        def mock_get_system(*args, **kwargs):
            clients_data = []
            for i in range(sample_config['num_clients']):
                x = np.random.rand(30, 784).astype(np.float32)
                y = np.random.randint(0, 5, size=(30,))
                clients_data.append((x, y))
            return clients_data

        monkeypatch.setattr(util_data, 'get_dataset', mock_get_dataset)
        monkeypatch.setattr(util_data, 'get_system', mock_get_system)

        # Step 1: Load and distribute data
        clients_data, classes = experiment_utils.load_and_distribute_data(sample_config)
        assert len(clients_data) == sample_config['num_clients']

        # Step 2: Initialize devices
        devs = experiment_utils.initialize_devices(
            clients_data,
            dataset_name=sample_config['dataset_name'],
            cluster_kind=sample_config['cluster_kind'],
            dirtiness_max=sample_config['dirtiness_max']
        )
        assert len(devs) == sample_config['num_clients']

        # Step 3: Create device manager
        device_manager = DevManager(devs)
        assert device_manager.get_dev_number() == sample_config['num_clients']

        # Step 4: Check initial metrics
        avg_acc = device_manager.get_accuracy_avg()
        assert 0.0 <= avg_acc <= 1.0

        cluster_count = device_manager.get_dev_cluster_number()
        assert cluster_count > 0

    def test_config_to_devices_pipeline(self, sample_config, monkeypatch):
        """Test pipeline from config validation to device initialization."""
        # Mock data functions
        def mock_get_dataset(*args, **kwargs):
            x_train = np.random.rand(150, 784).astype(np.float32)
            y_train = np.random.randint(0, 10, size=(150,))
            x_test = np.random.rand(30, 784).astype(np.float32)
            y_test = np.random.randint(0, 10, size=(30,))
            return x_train, y_train, x_test, y_test

        def mock_get_system(*args, **kwargs):
            return [(np.random.rand(25, 784).astype(np.float32),
                    np.random.randint(0, 5, size=(25,)))
                   for _ in range(sample_config['num_clients'])]

        monkeypatch.setattr(util_data, 'get_dataset', mock_get_dataset)
        monkeypatch.setattr(util_data, 'get_system', mock_get_system)

        # Validate config
        experiment_utils.validate_settings(sample_config)

        # Load data
        clients_data, classes = experiment_utils.load_and_distribute_data(sample_config)

        # Initialize devices
        devs = experiment_utils.initialize_devices(
            clients_data,
            sample_config['dataset_name'],
            sample_config['cluster_kind'],
            sample_config['dirtiness_max']
        )

        # Verify end state
        assert all(isinstance(dev, Dev) for dev in devs)
        assert all(dev.y_clust is not None for dev in devs)


class TestCommunityDetection:
    """Test community detection workflows."""

    def test_find_communities_basic(self, sample_clients_data):
        """Test basic community detection."""
        devs = []
        for x, y in sample_clients_data:
            dev = Dev('mnist', x, y)
            dev.simulate_clustering(dirtiness=0.0)
            dev.acc_and_crosstab()
            devs.append(dev)

        device_manager = DevManager(devs)

        # Find communities
        all_communities, multi_communities, isolated = experiment_utils.find_and_log_communities(
            devs, device_manager, '/tmp/test_log.txt'
        )

        # Check results
        assert isinstance(all_communities, list)
        assert isinstance(multi_communities, list)
        assert isinstance(isolated, list)

        # Total should equal number of clients
        total_clients = sum(len(comm) for comm in all_communities)
        assert total_clients == len(devs)


class TestConvergenceFlow:
    """Test convergence detection in experiment flow."""

    def test_convergence_detection_stable_state(self, sample_clients_data):
        """Test convergence detection with stable metrics."""
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        device_manager = DevManager(devs)

        # Initialize metrics
        device_manager.set_num_isolated(5)
        device_manager.set_num_communities(2)

        # Keep metrics stable for multiple iterations
        for _ in range(3):
            device_manager.set_num_isolated(5)
            device_manager.set_num_communities(2)

        # High reclustered_mean should trigger convergence
        converged = experiment_utils.check_convergence(device_manager, reclustered_mean=15.0)
        assert converged is True

    def test_convergence_detection_unstable_state(self, sample_clients_data):
        """Test that unstable metrics don't trigger convergence."""
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        device_manager = DevManager(devs)

        # Varying metrics
        device_manager.set_num_isolated(3)
        device_manager.set_num_communities(2)
        device_manager.set_num_isolated(5)
        device_manager.set_num_communities(3)

        # Low reclustered_mean with unstable metrics
        converged = experiment_utils.check_convergence(device_manager, reclustered_mean=5.0)
        assert converged is False


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    def test_invalid_config_rejected(self):
        """Test that invalid configurations are rejected early."""
        invalid_config = {
            'seed': 42,
            'dataset_name': 'mnist',
            'num_clients': -5,  # Invalid
            'association_threshold': 0.25,
            'cluster_kind': 'oracle'
        }

        with pytest.raises(ValueError):
            experiment_utils.validate_settings(invalid_config)

    def test_empty_client_data_handling(self):
        """Test handling of empty client data."""
        empty_clients_data = []

        devs = experiment_utils.initialize_devices(
            empty_clients_data,
            dataset_name='mnist',
            cluster_kind='oracle',
            dirtiness_max=0.0
        )

        assert len(devs) == 0

        device_manager = DevManager(devs)
        assert device_manager.get_dev_number() == 0


class TestReproducibility:
    """Test reproducibility of experiments."""

    def test_same_seed_same_clustering(self):
        """Test that same seed produces same clustering."""
        np.random.seed(42)
        x = np.random.rand(100, 784).astype(np.float32)
        y = np.random.randint(0, 5, size=(100,))

        # First run
        dev1 = Dev('mnist', x.copy(), y.copy())
        dev1.simulate_clustering(dirtiness=0.3)
        clustering1 = dev1.y_clust.copy()

        # Reset random seed
        np.random.seed(42)

        # Second run
        dev2 = Dev('mnist', x.copy(), y.copy())
        dev2.simulate_clustering(dirtiness=0.3)
        clustering2 = dev2.y_clust.copy()

        # Should be identical
        np.testing.assert_array_equal(clustering1, clustering2)
