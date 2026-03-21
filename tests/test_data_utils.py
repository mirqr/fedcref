"""
Unit tests for data utility functions.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import util_data


class TestDatasetLoading:
    """Test dataset loading functions."""

    @pytest.mark.slow
    def test_get_dataset_mnist(self):
        """Test loading MNIST dataset (integration test)."""
        x_train, y_train, x_test, y_test = util_data.get_dataset('mnist', flatten_and_normalize=True)

        # Check shapes
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(x_train.shape) == 2  # Flattened
        assert x_train.shape[1] == 784  # 28*28

        # Check normalization
        assert x_train.min() >= 0.0
        assert x_train.max() <= 1.0

        # Check labels
        assert y_train.min() >= 0
        assert y_train.max() <= 9

    def test_get_dataset_invalid(self):
        """Test that invalid dataset name raises error."""
        with pytest.raises(Exception):
            util_data.get_dataset('invalid_dataset_name')


class TestDataDistribution:
    """Test data distribution functions."""

    def test_get_system_basic(self):
        """Test basic data distribution to clients."""
        # Create small synthetic dataset
        np.random.seed(42)
        x_train = np.random.rand(500, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, size=(500,))
        classes = np.unique(y_train)

        # Distribute to 3 clients
        clients_data = util_data.get_system(
            num_clients=3,
            x_train=x_train,
            y_train=y_train,
            num_min_class=2,
            num_max_class=4,
            unique_classes=classes,
            min_samples_per_class=20,
            max_samples_per_class=50,
            seed=42
        )

        # Check we got 3 clients
        assert len(clients_data) == 3

        # Check each client has data
        for x, y in clients_data:
            assert len(x) > 0
            assert len(y) > 0
            assert len(x) == len(y)

            # Check class distribution
            unique_classes = np.unique(y)
            assert 2 <= len(unique_classes) <= 4

    def test_get_system_reproducibility(self):
        """Test that same seed produces same distribution."""
        np.random.seed(42)
        x_train = np.random.rand(300, 784).astype(np.float32)
        y_train = np.random.randint(0, 5, size=(300,))
        classes = np.unique(y_train)

        # First distribution
        clients_data_1 = util_data.get_system(
            num_clients=2,
            x_train=x_train,
            y_train=y_train,
            num_min_class=2,
            num_max_class=3,
            unique_classes=classes,
            min_samples_per_class=20,
            max_samples_per_class=40,
            seed=42
        )

        # Second distribution with same seed
        clients_data_2 = util_data.get_system(
            num_clients=2,
            x_train=x_train,
            y_train=y_train,
            num_min_class=2,
            num_max_class=3,
            unique_classes=classes,
            min_samples_per_class=20,
            max_samples_per_class=40,
            seed=42
        )

        # Check distributions are identical
        assert len(clients_data_1) == len(clients_data_2)
        for (x1, y1), (x2, y2) in zip(clients_data_1, clients_data_2):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)

    def test_get_system_data_integrity(self):
        """Test that data distribution maintains data integrity."""
        np.random.seed(42)
        x_train = np.random.rand(200, 784).astype(np.float32)
        y_train = np.random.randint(0, 5, size=(200,))
        classes = np.unique(y_train)

        clients_data = util_data.get_system(
            num_clients=3,
            x_train=x_train,
            y_train=y_train,
            num_min_class=2,
            num_max_class=3,
            unique_classes=classes,
            min_samples_per_class=10,
            max_samples_per_class=30,
            seed=42
        )

        # Check no data corruption
        for x, y in clients_data:
            # Check data types
            assert x.dtype == np.float32
            assert y.dtype in [np.int32, np.int64, int]

            # Check no NaN or inf values
            assert not np.isnan(x).any()
            assert not np.isinf(x).any()


class TestOverlap:
    """Test data overlap functionality."""

    def test_introduce_overlap_basic(self):
        """Test introducing overlap between clients."""
        np.random.seed(42)

        # Create clients with distinct classes
        clients_data = []
        for i in range(3):
            x = np.random.rand(50, 784).astype(np.float32)
            # Client i has classes [i, i+1]
            y = np.random.randint(i, i + 2, size=(50,))
            clients_data.append((x, y))

        classes = np.array([0, 1, 2, 3])

        # Introduce overlap
        clients_data_overlap = util_data.introduce_overlap_new(clients_data, classes, overlap=0.2)

        # Check structure is maintained
        assert len(clients_data_overlap) == len(clients_data)

        for (x_orig, y_orig), (x_new, y_new) in zip(clients_data, clients_data_overlap):
            # With overlap, we should have more or equal data
            assert len(x_new) >= len(x_orig)
            assert len(y_new) >= len(y_orig)

    def test_introduce_overlap_zero(self):
        """Test that zero overlap doesn't change data."""
        np.random.seed(42)

        clients_data = []
        for i in range(2):
            x = np.random.rand(30, 784).astype(np.float32)
            y = np.random.randint(i, i + 2, size=(30,))
            clients_data.append((x, y))

        classes = np.array([0, 1, 2])

        # No overlap
        clients_data_no_overlap = util_data.introduce_overlap_new(clients_data, classes, overlap=0.0)

        # Should be identical
        for (x_orig, y_orig), (x_new, y_new) in zip(clients_data, clients_data_no_overlap):
            np.testing.assert_array_equal(x_orig, x_new)
            np.testing.assert_array_equal(y_orig, y_new)


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_empty_data_handling(self):
        """Test handling of edge cases with minimal data."""
        np.random.seed(42)
        x_train = np.random.rand(50, 784).astype(np.float32)
        y_train = np.random.randint(0, 3, size=(50,))
        classes = np.unique(y_train)

        # Try to create more clients than reasonable
        # Should still work but with small data per client
        clients_data = util_data.get_system(
            num_clients=5,
            x_train=x_train,
            y_train=y_train,
            num_min_class=1,
            num_max_class=2,
            unique_classes=classes,
            min_samples_per_class=2,
            max_samples_per_class=10,
            seed=42
        )

        # Should still create clients even with limited data
        assert len(clients_data) > 0

        for x, y in clients_data:
            if len(x) > 0:  # Some clients might be empty with this constraint
                assert len(x) == len(y)
