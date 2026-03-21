"""
Test script to verify EMNIST dataset download and loading.
This isolates the dataset download to ensure it works before running the full experiment.
"""

import os
import sys

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 60)
print("EMNIST Dataset Download Test")
print("=" * 60)

print("\n1. Checking dataset cache location...")
keras_datasets_dir = os.path.expanduser('~/.keras/datasets')
print(f"   Cache directory: {keras_datasets_dir}")

print("\n2. Attempting to load EMNIST digits dataset...")
try:
    from extra_keras_datasets import emnist
    print("   ✓ extra_keras_datasets imported successfully")

    print("\n3. Loading EMNIST digits (this may take a moment)...")
    (x_train, y_train), (x_test, y_test) = emnist.load_data(type='digits')

    print("   ✓ Dataset loaded successfully!")

    print("\n4. Dataset Information:")
    print(f"   Training samples: {x_train.shape[0]}")
    print(f"   Test samples: {x_test.shape[0]}")
    print(f"   Image shape: {x_train.shape[1:]} (height x width)")
    print(f"   Number of classes: {len(set(y_train.flatten()))}")
    print(f"   Training data type: {x_train.dtype}")
    print(f"   Label range: {y_train.min()} to {y_train.max()}")

    print("\n5. Verifying data integrity...")
    assert x_train.shape[0] > 0, "No training samples found"
    assert x_test.shape[0] > 0, "No test samples found"
    assert len(x_train.shape) == 3, "Invalid image dimensions"
    print("   ✓ Data integrity check passed")

    print("\n" + "=" * 60)
    print("SUCCESS: EMNIST dataset is working correctly!")
    print("=" * 60)
    print("\nYou can now run main.py safely.")

except ImportError as e:
    print(f"\n   ✗ ERROR: Missing dependency")
    print(f"   {e}")
    print("\n   Please install: pip install extra-keras-datasets")
    sys.exit(1)

except Exception as e:
    print(f"\n   ✗ ERROR: Failed to load dataset")
    print(f"   {type(e).__name__}: {e}")
    print("\n   Troubleshooting:")
    print("   1. Delete corrupted cache: rm ~/.keras/datasets/emnist*")
    print("   2. Check internet connection")
    print("   3. Re-run this script to download fresh data")
    sys.exit(1)
