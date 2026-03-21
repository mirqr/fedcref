#!/bin/bash
# Test runner script for FL Cluster Claude

set -e

echo "======================================"
echo "FL Cluster Claude - Test Suite"
echo "======================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    print_warning "pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Parse command line arguments
MODE=${1:-"fast"}

case $MODE in
    "fast")
        print_info "Running fast tests (excluding slow tests)..."
        pytest -m "not slow" tests/
        ;;
    "all")
        print_info "Running all tests..."
        pytest tests/
        ;;
    "unit")
        print_info "Running unit tests only..."
        pytest -m "unit" tests/ || pytest tests/test_experiment_utils.py tests/test_data_utils.py tests/test_dev.py
        ;;
    "integration")
        print_info "Running integration tests only..."
        pytest -m "integration" tests/ || pytest tests/test_integration.py
        ;;
    "coverage")
        print_info "Running tests with coverage report..."
        pytest --cov=. --cov-report=html --cov-report=term tests/
        print_success "Coverage report generated in htmlcov/index.html"
        ;;
    "verbose")
        print_info "Running tests with verbose output..."
        pytest -vv tests/
        ;;
    *)
        echo "Usage: $0 {fast|all|unit|integration|coverage|verbose}"
        echo ""
        echo "Options:"
        echo "  fast        - Run fast tests only (default, excludes slow tests)"
        echo "  all         - Run all tests including slow ones"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  coverage    - Run tests with coverage report"
        echo "  verbose     - Run tests with verbose output"
        exit 1
        ;;
esac

print_success "Tests completed!"
