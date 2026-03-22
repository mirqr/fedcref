#!/bin/bash

conda env create -f req-environment.yml -y
conda activate fedcref
uv pip install -r requirements.txt

# run: source req-env.sh