#!/bin/bash
# This script removes all output from a Jupyter notebook
# Generated using copilot
# Usage: source clean_ipynb.sh notebook.ipynb

# Check if the file exists
if [ ! -f $1 ]; then
    echo "File $1 does not exist"
    exit 1
fi

# Check if the file is a Jupyter notebook
if [ "${1: -6}" != ".ipynb" ]; then
    echo "File $1 is not a Jupyter notebook"
    exit 1
fi

# Remove all output from the notebook
python -m nbconvert --clear-output --inplace $1
# jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace $1
