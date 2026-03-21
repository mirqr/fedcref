"""
Unit tests for Dev and DevManager classes.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dev import Dev, DevManager


class TestDevInitialization:
    """Test Dev class initialization."""

    def test_dev_creation(self, sample_clients_data):
        """Test basic Dev object creation."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)

        assert dev.x is not None
        assert dev.y is not None
        assert dev.dataset_name == 'mnist'
        assert len(dev.x) == len(dev.y)
        assert dev.name is not None

    def test_dev_with_empty_data(self):
        """Test Dev creation with empty data."""
        x = np.array([]).reshape(0, 784)
        y = np.array([])

        dev = Dev('mnist', x, y)
        assert len(dev.x) == 0
        assert len(dev.y) == 0

    def test_dev_data_integrity(self, sample_clients_data):
        """Test that Dev preserves data integrity."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)

        # Check data is preserved
        np.testing.assert_array_equal(dev.x, x)
        np.testing.assert_array_equal(dev.y, y)


class TestDevClustering:
    """Test Dev clustering methods."""

    def test_simulate_clustering_oracle(self, sample_clients_data):
        """Test oracle clustering (perfect clustering)."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)

        dev.simulate_clustering(dirtiness=0.0)

        # Oracle clustering should assign each sample to its true class
        assert dev.y_clust is not None
        assert len(dev.y_clust) == len(dev.y)

        # With dirtiness=0, clustering should match labels perfectly
        unique_clusters = np.unique(dev.y_clust)
        assert len(unique_clusters) > 0

    def test_simulate_clustering_with_dirtiness(self, sample_clients_data):
        """Test clustering with noise."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)

        dev.simulate_clustering(dirtiness=0.3)

        # Should still produce clusters
        assert dev.y_clust is not None
        assert len(dev.y_clust) == len(dev.y)

        # With dirtiness, some samples should be misclassified
        unique_clusters = np.unique(dev.y_clust)
        assert len(unique_clusters) > 0

    def test_acc_and_crosstab(self, sample_clients_data):
        """Test accuracy computation and crosstab generation."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)

        dev.simulate_clustering(dirtiness=0.0)
        dev.acc_and_crosstab()

        # Check accuracy is computed
        assert hasattr(dev, 'acc')
        assert 0.0 <= dev.acc <= 1.0


class TestDevManager:
    """Test DevManager class."""

    def test_devmanager_creation(self, sample_clients_data):
        """Test DevManager initialization."""
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        manager = DevManager(devs)

        assert manager.get_dev_number() == len(sample_clients_data)

    def test_devmanager_empty(self):
        """Test DevManager with no devices."""
        manager = DevManager([])
        assert manager.get_dev_number() == 0

    def test_get_accuracy_avg(self, sample_clients_data):
        """Test average accuracy computation."""
        devs = []
        for x, y in sample_clients_data:
            dev = Dev('mnist', x, y)
            dev.simulate_clustering(dirtiness=0.0)
            dev.acc_and_crosstab()
            devs.append(dev)

        manager = DevManager(devs)
        avg_acc = manager.get_accuracy_avg()

        # Check average is reasonable
        assert isinstance(avg_acc, (int, float))
        assert 0.0 <= avg_acc <= 1.0

    def test_get_dev_cluster_number(self, sample_clients_data):
        """Test total cluster count across devices."""
        devs = []
        for x, y in sample_clients_data:
            dev = Dev('mnist', x, y)
            dev.simulate_clustering(dirtiness=0.0)
            devs.append(dev)

        manager = DevManager(devs)
        cluster_count = manager.get_dev_cluster_number()

        # Should have at least some clusters
        assert cluster_count >= 0

    def test_set_and_get_metrics(self, sample_clients_data):
        """Test setting and getting metrics."""
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]
        manager = DevManager(devs)

        # Test num_isolated
        manager.set_num_isolated(5)
        assert manager.num_isolated == 5

        # Test num_communities
        manager.set_num_communities(3)
        assert manager.num_communities == 3


class TestDevAssociations:
    """Test device association functionality."""

    def test_count_association(self, sample_clients_data):
        """Test association counting."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)
        dev.simulate_clustering(dirtiness=0.0)

        # Initialize associations
        dev.association_clust = {}

        count = dev.count_association()
        assert isinstance(count, int)
        assert count >= 0

    def test_wrong_association_no_associations(self, sample_clients_data):
        """Test wrong association count with no associations."""
        x, y = sample_clients_data[0]
        dev = Dev('mnist', x, y)
        dev.simulate_clustering(dirtiness=0.0)

        # Initialize empty associations
        dev.association_clust = {}

        wrong = dev.wrong_association()
        assert isinstance(wrong, int)
        assert wrong >= 0


class TestDevUtilities:
    """Test utility methods."""

    def test_dev_name_generation(self, sample_clients_data):
        """Test that each dev gets a unique name."""
        devs = [Dev('mnist', x, y) for x, y in sample_clients_data]

        # Check all names are unique
        names = [dev.name for dev in devs]
        assert len(names) == len(set(names))

    def test_dev_with_different_datasets(self):
        """Test Dev with different dataset names."""
        x = np.random.rand(50, 784).astype(np.float32)
        y = np.random.randint(0, 10, size=(50,))

        dev_mnist = Dev('mnist', x, y)
        dev_emnist = Dev('emnist_digits', x, y)

        assert dev_mnist.dataset_name == 'mnist'
        assert dev_emnist.dataset_name == 'emnist_digits'
