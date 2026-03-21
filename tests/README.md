# Test Suite Documentation

This directory contains the comprehensive test suite for the FL Cluster Claude project.

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── test_experiment_utils.py # Tests for experiment utility functions
├── test_data_utils.py       # Tests for data loading and distribution
├── test_dev.py             # Tests for Dev and DevManager classes
├── test_integration.py      # End-to-end integration tests
└── README.md               # This file
```

## Running Tests

### Quick Start

```bash
# Run all fast tests (recommended for development)
./run_tests.sh fast

# Run all tests including slow ones
./run_tests.sh all

# Run with coverage report
./run_tests.sh coverage
```

### Using pytest directly

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_experiment_utils.py

# Run specific test class
pytest tests/test_dev.py::TestDevInitialization

# Run specific test
pytest tests/test_dev.py::TestDevInitialization::test_dev_creation

# Run tests matching a pattern
pytest -k "test_config"

# Run with verbose output
pytest -v tests/

# Exclude slow tests
pytest -m "not slow" tests/
```

## Test Categories

### Unit Tests
- **test_experiment_utils.py**: Configuration, logging, data loading helpers
- **test_data_utils.py**: Dataset loading, distribution, and overlap
- **test_dev.py**: Device initialization, clustering, and associations

### Integration Tests
- **test_integration.py**: End-to-end workflows, community detection, convergence

### Slow Tests
Tests marked with `@pytest.mark.slow` download real datasets and may take longer:
- MNIST/EMNIST dataset loading tests

To skip slow tests: `pytest -m "not slow" tests/`

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_config`: Valid experiment configuration
- `temp_config_file`: Temporary JSON config file
- `sample_data`: Small synthetic dataset
- `sample_clients_data`: Pre-distributed client data
- `temp_log_dir`: Temporary directory for logs

## Writing New Tests

### Test Naming Convention
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

### Example Test

```python
def test_my_feature(sample_config):
    """Test description."""
    # Arrange
    expected = 42

    # Act
    result = my_function(sample_config)

    # Assert
    assert result == expected
```

### Marking Tests

```python
import pytest

@pytest.mark.slow
def test_dataset_download():
    """This test downloads real data and is slow."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """This is an integration test."""
    pass
```

## Coverage

Generate coverage report:

```bash
pytest --cov=. --cov-report=html --cov-report=term tests/
```

View HTML report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

Tests should pass before merging code. Key checks:
- All fast tests pass
- No new warnings
- Code coverage maintained or improved

## Troubleshooting

### Import Errors
If you get import errors, ensure the project root is in PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### TensorFlow Warnings
TensorFlow warnings are suppressed in test configuration. If you see warnings, check `conftest.py`.

### Slow Test Performance
- Use `-m "not slow"` to skip dataset downloads
- Use `-n auto` with pytest-xdist for parallel execution:
  ```bash
  pytest -n auto tests/
  ```

### Cache Issues
If tests fail due to corrupted caches:
```bash
rm -rf .cachejoblib/ saved/
pytest tests/
```

## Test Philosophy

1. **Fast by default**: Most tests use synthetic data
2. **Isolated**: Tests don't depend on each other
3. **Deterministic**: Use fixed seeds for reproducibility
4. **Comprehensive**: Cover happy paths, edge cases, and errors
5. **Maintainable**: Clear naming and documentation

## Contributing

When adding new features:
1. Write tests first (TDD approach recommended)
2. Ensure all tests pass: `pytest tests/`
3. Add docstrings to test functions
4. Update this README if adding new test categories

## Performance Benchmarks

Approximate test execution times on a standard machine:

- Fast tests: ~5-10 seconds
- All tests (excluding slow): ~10-20 seconds
- All tests (including slow): ~30-60 seconds (dataset downloads)
- With coverage: +20% overhead

## Questions?

See the main project CLAUDE.md for architecture details and contact information.
