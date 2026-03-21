#!/bin/bash
# set -e # Exit immediately if a command exits with a non-zero status.


#  The create || update pattern makes it idempotent — 
# first run creates, subsequent runs update if the yml changed.

conda env create -f req-environment.yml -y
conda activate fedcref
uv pip install -r requirements.txt



# run: source req-env.sh