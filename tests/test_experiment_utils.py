"""
Unit tests for experiment_utils module.
"""

import os
import sys
import json
import tempfile
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import experiment_utils


class TestConfigManagement:
    """Test configuration loading and validation."""

    def test_load_config_file_success(self, temp_config_file, sample_config):
        """Test successful config file loading."""
        loaded_config = experiment_utils.load_config_file(temp_config_file)
        assert loaded_config == sample_config

    def test_load_config_file_not_found(self):
        """Test error handling for missing config file."""
        with pytest.raises(SystemExit):
            experiment_utils.load_config_file('nonexistent_config.json')

    def test_load_config_file_invalid_json(self):
        """Test error handling for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name

        try:
            with pytest.raises(SystemExit):
                experiment_utils.load_config_file(temp_path)
        finally:
            os.remove(temp_path)

    def test_validate_settings_success(self, sample_config):
        """Test validation with valid settings."""
        # Should not raise any exceptions
        experiment_utils.validate_settings(sample_config)

    def test_validate_settings_missing_key(self, sample_config):
        """Test validation fails with missing required key."""
        incomplete_config = sample_config.copy()
        del incomplete_config['seed']

        with pytest.raises(ValueError, match="Missing required setting"):
            experiment_utils.validate_settings(incomplete_config)

    def test_validate_settings_invalid_num_clients(self, sample_config):
        """Test validation fails with invalid num_clients."""
        invalid_config = sample_config.copy()
        invalid_config['num_clients'] = 0

        with pytest.raises(ValueError, match="num_clients must be positive"):
            experiment_utils.validate_settings(invalid_config)

    def test_validate_settings_invalid_threshold(self, sample_config):
        """Test validation fails with invalid association threshold."""
        invalid_config = sample_config.copy()
        invalid_config['association_threshold'] = 1.5

        with pytest.raises(ValueError, match="association_threshold must be between"):
            experiment_utils.validate_settings(invalid_config)

    def test_validate_settings_invalid_cluster_kind(self, sample_config):
        """Test validation fails with invalid cluster kind."""
        invalid_config = sample_config.copy()
        invalid_config['cluster_kind'] = 'invalid_method'

        with pytest.raises(ValueError, match="Invalid cluster_kind"):
            experiment_utils.validate_settings(invalid_config)


class TestLogging:
    """Test logging utilities."""

    def test_setup_logging(self, temp_log_dir):
        """Test log directory and file creation."""
        # Temporarily override LOG_DIR
        original_log_dir = experiment_utils.LOG_DIR
        experiment_utils.LOG_DIR = temp_log_dir

        try:
            filename = experiment_utils.setup_logging('mnist', '2024-01-01-12-00-00')

            # Check file path format
            assert 'listcomm_mnist_2024-01-01-12-00-00.txt' in filename
            assert filename.startswith(temp_log_dir)

            # Check directory was created
            assert os.path.exists(temp_log_dir)
        finally:
            experiment_utils.LOG_DIR = original_log_dir


class TestDataLoading:
    """Test data loading and distribution."""

    def test_load_and_distribute_data_basic(self, sample_config, monkeypatch):
        """Test basic data loading and distribution."""
        # Mock the data loading functions to avoid actual dataset downloads
        import numpy as np
        from utils import util_data

        def mock_get_dataset(*args, **kwargs):
            x_train = np.random.rand(1000, 784).astype(np.float32)
            y_train = np.random.randint(0, 10, size=(1000,))
            x_test = np.random.rand(200, 784).astype(np.float32)
            y_test = np.random.randint(0, 10, size=(200,))
            return x_train, y_train, x_test, y_test

        def mock_get_system(*args, **kwargs):
            clients_data = []
            for i in range(sample_config['num_clients']):
                x = np.random.rand(100, 784).astype(np.float32)
                y = np.random.randint(0, 5, size=(100,))
                clients_data.append((x, y))
            return clients_data

        monkeypatch.setattr(util_data, 'get_dataset', mock_get_dataset)
        monkeypatch.setattr(util_data, 'get_system', mock_get_system)

        clients_data, classes = experiment_utils.load_and_distribute_data(sample_config)

        # Check that we got the right number of clients
        assert len(clients_data) == sample_config['num_clients']

        # Check that each client has data
        for x, y in clients_data:
            assert len(x) > 0
            assert len(y) > 0
            assert len(x) == len(y)


class TestDeviceInitialization:
    """Test device initialization functions."""

    def test_initialize_devices_oracle(self, sample_clients_data):
        """Test device initialization with oracle clustering."""
        devs = experiment_utils.initialize_devices(
            sample_clients_data,
            dataset_name='mnist',
            cluster_kind='oracle',
            dirtiness_max=0.0
        )

        # Check we got the right number of devices
        assert len(devs) == len(sample_clients_data)

        # Check each device is initialized
        for dev in devs:
            assert dev.x is not None
            assert dev.y is not None
            assert dev.y_clust is not None

    def test_initialize_devices_dirty_uniform(self, sample_clients_data):
        """Test device initialization with dirty uniform clustering."""
        devs = experiment_utils.initialize_devices(
            sample_clients_data,
            dataset_name='mnist',
            cluster_kind='dirty_uniform',
            dirtiness_max=0.3
        )

        assert len(devs) == len(sample_clients_data)

        for dev in devs:
            assert dev.x is not None
            assert dev.y is not None

    def test_initialize_devices_invalid_cluster_kind(self, sample_clients_data):
        """Test that invalid cluster kind raises error."""
        with pytest.raises(ValueError, match="Unknown cluster kind"):
            experiment_utils.initialize_devices(
                sample_clients_data,
                dataset_name='mnist',
                cluster_kind='invalid_kind',
                dirtiness_max=0.3
            )


class TestConvergence:
    """Test convergence checking logic."""

    def test_check_convergence_by_threshold(self, sample_clients_data):
        """Test convergence when reclustered_mean exceeds threshold."""
        from dev import DevManager, Dev

        # Create minimal device manager
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        device_manager = DevManager(devs)

        # Test convergence with high reclustered_mean
        result = experiment_utils.check_convergence(device_manager, reclustered_mean=15.0)
        assert result is True

    def test_no_convergence_low_threshold(self, sample_clients_data):
        """Test no convergence when reclustered_mean is low."""
        from dev import DevManager, Dev

        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        device_manager = DevManager(devs)

        # Test no convergence with low reclustered_mean
        result = experiment_utils.check_convergence(device_manager, reclustered_mean=5.0)
        assert result is False


class TestAssociations:
    """Test association computation."""

    def test_compute_associations_basic(self, sample_clients_data):
        """Test basic association computation."""
        from dev import DevManager, Dev

        # Create devices with predictions
        devs = []
        for x, y in sample_clients_data:
            dev = Dev('mnist', x, y)
            dev.simulate_clustering(dirtiness=0)
            dev.acc_and_crosstab()
            devs.append(dev)

        device_manager = DevManager(devs)

        # This test verifies the function runs without error
        # Actual association logic requires trained models
        total_wrong, total_count, percentage = experiment_utils.compute_associations(
            device_manager, devs, association_threshold=0.25
        )

        assert isinstance(total_wrong, int)
        assert isinstance(total_count, int)
        assert isinstance(percentage, (int, float))
        assert 0 <= percentage <= 100
