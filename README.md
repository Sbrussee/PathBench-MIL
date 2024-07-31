<div style="text-align: center;">
 <img src="PathBench-logo-gecentreerd.png" alt="PathBench Logo" width="550" height="400">
</div>

# PathBench


PathBench is a Python package designed to facilitate benchmarking, experimentation, and optimization of Multiple Instance Learning (MIL) and Semi-Supervised Learning (SSL) methods for computational histopathology. It provides tools for conducting experiments, evaluating performance, visualizing results, and optimizing hyperparameters. PathBench is built on top of SlideFlow for handling Whole Slide images, Lightly for SSL methods, and integrates Optuna for hyperparameter optimization.

## Features

- Benchmark MIL and SSL methods for computational histopathology.
- Experiment with different parameter configurations and datasets.
- Support hyperparameter optimization.
- Visualize experiment results.
- Flexibility to integrate custom feature extractors, MIL models, and SSL models.

## Package Structure

- pathbench/
  - pathbench/
    - __init__.py
    - experiment.py
    - benchmark.py
    - hpo.py
    - ssl.py
    - visualization.py
    - tensorboard/
      - __init__.py
  - tests/
    - ...
  - docs/
    - ...
  - requirements.txt
  - README.md
  - LICENSE
  - setup.py

## Installation

You can install PathBench and its dependencies using pip:

```bash
pip install pathbench
